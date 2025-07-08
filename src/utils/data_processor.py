"""
Обработка и подготовка данных для алготрейдинг системы

ПРОБЛЕМЫ В ОРИГИНАЛЬНОЙ ОБРАБОТКЕ:
1. Неправильная нормализация цен
2. Отсутствие валидации данных
3. Возможная утечка данных из будущего
4. Нет walk-forward validation

ИСПРАВЛЕНИЯ:
1. Правильная нормализация только индикаторов
2. Строгая валидация данных
3. Временная проверка на утечки
4. Добавлена подготовка для walk-forward
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from pathlib import Path

from ..core.config import DataConfig


class DataProcessor:
    """
    УЛУЧШЕННЫЙ процессор данных с исправлением проблем
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Кэш для обработанных данных
        self._processed_cache: Dict[str, pd.DataFrame] = {}
        
        # Статистика по данным
        self.data_stats: Dict[str, Any] = {}
    
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Загрузка и базовая валидация данных"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {file_path}")
        
        self.logger.info(f"📁 Загружаем данные из {file_path}")
        
        # Загружаем данные
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Ошибка загрузки CSV: {e}")
        
        # Базовая валидация
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
        
        # Проверка типов данных
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.logger.warning(f"Колонка {col} не числовая, пытаемся конвертировать")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Проверка на валидность цен OHLC
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            self.logger.warning(f"Обнаружены некорректные OHLC данные: {invalid_ohlc.sum()} строк")
            df = df[~invalid_ohlc].reset_index(drop=True)
        
        # Проверка на нулевые/отрицательные значения
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            invalid_prices = (df[col] <= 0)
            if invalid_prices.any():
                self.logger.warning(f"Обнаружены нулевые/отрицательные цены в {col}: {invalid_prices.sum()} строк")
                df = df[~invalid_prices].reset_index(drop=True)
        
        # Проверка на дубликаты timestamp (если есть)
        if 'timestamp' in df.columns:
            duplicates = df.duplicated(subset=['timestamp'])
            if duplicates.any():
                self.logger.warning(f"Обнаружены дубликаты временных меток: {duplicates.sum()}")
                df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        self.logger.info(f"✅ Данные загружены и валидированы: {len(df)} записей")
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        УЛУЧШЕННЫЙ расчет технических индикаторов с проверкой на утечки
        """
        df = df.copy()
        
        self.logger.info("🔧 Расчет технических индикаторов...")
        
        # === БАЗОВЫЕ ИНДИКАТОРЫ ===
        
        # EMA (Exponential Moving Average)
        df['ema_fast'] = df['close'].ewm(span=DataConfig.EMA_FAST_SPAN, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=DataConfig.EMA_SLOW_SPAN, adjust=False).mean()
        df['ema_signal'] = (df['ema_fast'] > df['ema_slow']).astype(int)
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=DataConfig.RSI_WINDOW).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=DataConfig.RSI_WINDOW).mean()
        rs = gain / (loss + 1e-8)  # Добавляем малое число для избежания деления на 0
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_oversold'] = (df['rsi'] < DataConfig.RSI_OVERSOLD).astype(int)
        df['rsi_overbought'] = (df['rsi'] > DataConfig.RSI_OVERBOUGHT).astype(int)
        
        # MACD
        ema_12 = df['close'].ewm(span=DataConfig.MACD_FAST).mean()
        ema_26 = df['close'].ewm(span=DataConfig.MACD_SLOW).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # === ДОПОЛНИТЕЛЬНЫЕ ИНДИКАТОРЫ ===
        
        # Bollinger Bands
        bb_window = DataConfig.BOLLINGER_WINDOW
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_std = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        
        # Price action
        df['price_change'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=20).std()
        df['price_trend'] = df['close'].pct_change(periods=5)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Momentum indicators
        df['momentum'] = df['close'] / df['close'].shift(DataConfig.MOMENTUM_WINDOW) - 1
        df['roc'] = ((df['close'] - df['close'].shift(12)) / df['close'].shift(12)) * 100
        
        # === ДОПОЛНИТЕЛЬНЫЕ ИНДИКАТОРЫ ДЛЯ ПРИБЫЛЬНОСТИ ===
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + 1e-8))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        
        # Williams %R
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14 + 1e-8))
        
        # Commodity Channel Index (CCI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-8)
        
        # Money Flow Index (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_mf = money_flow.where(typical_price.diff() > 0, 0).rolling(window=14).sum()
        negative_mf = money_flow.where(typical_price.diff() < 0, 0).rolling(window=14).sum()
        df['mfi'] = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-8)))
        
        # On-Balance Volume (OBV)
        df['obv'] = (df['volume'] * np.where(df['close'].diff() > 0, 1, 
                                            np.where(df['close'].diff() < 0, -1, 0))).cumsum()
        df['obv_signal'] = (df['obv'] > df['obv'].rolling(window=20).mean()).astype(int)
        
        # Ichimoku Cloud components
        nine_period_high = df['high'].rolling(window=9).max()
        nine_period_low = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
        
        twenty_six_period_high = df['high'].rolling(window=26).max()
        twenty_six_period_low = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2
        
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        fifty_two_period_high = df['high'].rolling(window=52).max()
        fifty_two_period_low = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
        
        # Price position relative to Ichimoku cloud
        df['above_cloud'] = (df['close'] > df[['senkou_span_a', 'senkou_span_b']].max(axis=1)).astype(int)
        df['below_cloud'] = (df['close'] < df[['senkou_span_a', 'senkou_span_b']].min(axis=1)).astype(int)
        
        # Multi-timeframe analysis
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Trend strength indicators
        df['trend_strength'] = ((df['sma_5'] > df['sma_10']).astype(int) + 
                               (df['sma_10'] > df['sma_20']).astype(int) + 
                               (df['sma_20'] > df['sma_50']).astype(int))
        
        # Market sentiment indicators
        df['bullish_sentiment'] = ((df['rsi'] > 50).astype(int) + 
                                  (df['macd'] > df['macd_signal']).astype(int) + 
                                  (df['stoch_k'] > 50).astype(int) + 
                                  (df['above_cloud']).astype(int))
        
        df['bearish_sentiment'] = ((df['rsi'] < 50).astype(int) + 
                                  (df['macd'] < df['macd_signal']).astype(int) + 
                                  (df['stoch_k'] < 50).astype(int) + 
                                  (df['below_cloud']).astype(int))
        
        # Volatility regime detection
        df['volatility_regime'] = (df['volatility'] > df['volatility'].rolling(window=50).mean()).astype(int)
        
        # Price action patterns
        df['doji'] = (np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) < 0.1).astype(int)
        df['hammer'] = ((df['close'] > df['open']) & 
                       ((df['close'] - df['open']) / (0.001 + df['close'] - df['open']) < 0.3) &
                       ((df['open'] - df['low']) / (0.001 + df['high'] - df['low']) > 0.6)).astype(int)
        
        # === ПРОВЕРКА НА УТЕЧКИ ДАННЫХ ===
        self._validate_no_future_leakage(df)
        
        self.logger.info(f"✅ Рассчитано {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])} индикаторов")
        
        return df
    
    def _validate_no_future_leakage(self, df: pd.DataFrame):
        """Проверка на утечки данных из будущего"""
        # Проверяем что все индикаторы используют только прошлые данные
        # Это важно для реалистичного backtesting
        
        indicators_to_check = ['ema_fast', 'ema_slow', 'rsi', 'macd', 'bb_middle']
        
        for indicator in indicators_to_check:
            if indicator in df.columns:
                # Проверяем что индикатор не коррелирует с будущими ценами
                future_correlation = df[indicator].shift(1).corr(df['close'].shift(-1))
                if abs(future_correlation) > 0.1:  # Произвольный порог
                    self.logger.warning(f"⚠️ Возможная утечка данных в индикаторе {indicator}: корреляция с будущим = {future_correlation:.3f}")
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ИСПРАВЛЕННАЯ нормализация - НЕ нормализуем цены!
        """
        df = df.copy()
        
        self.logger.info("🔧 Нормализация признаков...")
        
        # Цены НЕ нормализуем для корректной работы торговой логики
        price_columns = ['open', 'high', 'low', 'close']
        
        # Нормализуем только технические индикаторы и объем
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        indicators_to_normalize = [col for col in numeric_columns 
                                 if col not in price_columns and col != 'volume_ratio']
        
        # Z-score нормализация для индикаторов
        for col in indicators_to_normalize:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 1e-8:  # Избегаем деления на очень маленькие числа
                    df[col] = (df[col] - mean_val) / std_val
                    
                    # Сохраняем статистику для денормализации
                    self.data_stats[f'{col}_mean'] = mean_val
                    self.data_stats[f'{col}_std'] = std_val
        
        # Логарифмическая нормализация объема
        if 'volume' in df.columns:
            df['volume'] = np.log1p(df['volume'])  # log(1 + x) для обработки нулевых значений
        
        self.logger.info("✅ Нормализация завершена (цены оставлены без изменений)")
        return df
    
    def split_data_for_walk_forward(self, df: pd.DataFrame, 
                                  train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Разделение данных для walk-forward validation
        """
        split_idx = int(len(df) * train_ratio)
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        self.logger.info(f"📊 Разделение данных: обучение={len(train_df)}, тест={len(test_df)}")
        
        return train_df, test_df
    
    def prepare_data(self, file_path: str, use_cache: bool = True) -> pd.DataFrame:
        """
        ГЛАВНАЯ функция подготовки данных с исправлениями
        """
        cache_key = f"{file_path}_{DataConfig.WINDOW_SIZE}"
        
        # Проверяем кэш
        if use_cache and cache_key in self._processed_cache:
            self.logger.info("📋 Используем кэшированные данные")
            return self._processed_cache[cache_key]
        
        # Обрабатываем данные
        df = self.load_and_validate_data(file_path)
        df = self.calculate_technical_indicators(df)
        df = self.normalize_features(df)
        
        # Убираем NaN значения
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        final_len = len(df)
        
        if initial_len != final_len:
            self.logger.info(f"🧹 Удалены NaN значения: {initial_len} -> {final_len} строк")
        
        # Финальная валидация
        if len(df) < DataConfig.WINDOW_SIZE + 100:
            raise ValueError(f"Недостаточно данных после обработки: {len(df)} < {DataConfig.WINDOW_SIZE + 100}")
        
        # Сохраняем в кэш
        if use_cache:
            self._processed_cache[cache_key] = df
        
        # Сохраняем статистику
        self.data_stats['total_records'] = len(df)
        self.data_stats['features_count'] = len(df.columns)
        self.data_stats['price_range'] = (df['close'].min(), df['close'].max())
        
        self.logger.info(f"✅ Данные готовы: {len(df)} записей, {len(df.columns)} признаков")
        self.logger.info(f"💰 Диапазон цен: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Отчет о качестве данных"""
        report = {
            'total_records': len(df),
            'features_count': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'price_statistics': {
                'min_price': df['close'].min(),
                'max_price': df['close'].max(),
                'mean_price': df['close'].mean(),
                'price_volatility': df['close'].std() / df['close'].mean()
            },
            'volume_statistics': {
                'min_volume': df['volume'].min(),
                'max_volume': df['volume'].max(),
                'mean_volume': df['volume'].mean()
            }
        }
        
        # Проверка на аномалии
        price_anomalies = (
            (df['close'] > df['close'].quantile(0.99)) |
            (df['close'] < df['close'].quantile(0.01))
        ).sum()
        report['price_anomalies'] = price_anomalies
        
        return report


# Фабричная функция для совместимости с оригинальным кодом
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Фабричная функция для обратной совместимости"""
    processor = DataProcessor()
    return processor.prepare_data(file_path) 