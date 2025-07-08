"""
🔧 РАСШИРЕННЫЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
Полный набор индикаторов для анализа криптовалютного рынка
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AdvancedTechnicalIndicators:
    """Продвинутые технические индикаторы для алготрейдинга"""
    
    def __init__(self):
        self.indicators_cache = {}
    
    # === ТРЕНДОВЫЕ ИНДИКАТОРЫ ===
    
    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """Экспоненциальное скользящее среднее"""
        return data.ewm(span=period, adjust=False).mean()
    
    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """Простое скользящее среднее"""
        return data.rolling(window=period).mean()
    
    def wma(self, data: pd.Series, period: int) -> pd.Series:
        """Взвешенное скользящее среднее"""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    def tema(self, data: pd.Series, period: int) -> pd.Series:
        """Triple Exponential Moving Average"""
        ema1 = self.ema(data, period)
        ema2 = self.ema(ema1, period)
        ema3 = self.ema(ema2, period)
        return 3 * ema1 - 3 * ema2 + ema3
    
    def hull_ma(self, data: pd.Series, period: int) -> pd.Series:
        """Hull Moving Average"""
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = self.wma(data, half_period)
        wma_full = self.wma(data, period)
        raw_hma = 2 * wma_half - wma_full
        return self.wma(raw_hma, sqrt_period)
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """Average Directional Index"""
        tr = self.true_range(high, low, close)
        plus_dm = (high.diff() > 0) & (high.diff() > low.diff().abs()) * high.diff()
        minus_dm = (low.diff() < 0) & (low.diff().abs() > high.diff()) * low.diff().abs()
        
        tr_smooth = tr.ewm(alpha=1/period).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period).mean()
        
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period).mean()
        
        return {
            'ADX': adx,
            'DI_plus': plus_di,
            'DI_minus': minus_di
        }
    
    # === ОСЦИЛЛЯТОРЫ ===
    
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'K': k_percent,
            'D': d_percent
        }
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        ma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - ma) / (0.015 * mad)
    
    def cmo(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Chande Momentum Oscillator"""
        delta = data.diff()
        sum_up = delta.where(delta > 0, 0).rolling(window=period).sum()
        sum_down = (-delta.where(delta < 0, 0)).rolling(window=period).sum()
        return 100 * (sum_up - sum_down) / (sum_up + sum_down)
    
    # === ОБЪЕМНЫЕ ИНДИКАТОРЫ ===
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        price_change = close.diff()
        volume_change = volume.where(price_change > 0, -volume).where(price_change != 0, 0)
        return volume_change.cumsum()
    
    def ad_line(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Если high == low
        return (clv * volume).cumsum()
    
    def cmf(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """Chaikin Money Flow"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        money_flow_volume = clv * volume
        return money_flow_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    # === ВОЛАТИЛЬНОСТЬ ===
    
    def bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev),
            'width': (sma + (std * std_dev)) - (sma - (std * std_dev)),
            'position': (data - sma) / (std * std_dev)
        }
    
    def keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                        period: int = 20, multiplier: float = 2) -> Dict[str, pd.Series]:
        """Keltner Channels"""
        ema = self.ema(close, period)
        atr = self.atr(high, low, close, period)
        
        return {
            'middle': ema,
            'upper': ema + (multiplier * atr),
            'lower': ema - (multiplier * atr)
        }
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr = self.true_range(high, low, close)
        return tr.rolling(window=period).mean()
    
    def true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """True Range"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # === ПАТТЕРНЫ И ФРАКТАЛЫ ===
    
    def fractal_highs(self, high: pd.Series, period: int = 5) -> pd.Series:
        """Фрактальные максимумы"""
        def is_fractal_high(data, idx):
            if idx < period or idx >= len(data) - period:
                return False
            center = data.iloc[idx]
            for i in range(1, period + 1):
                if data.iloc[idx - i] >= center or data.iloc[idx + i] >= center:
                    return False
            return True
        
        fractals = pd.Series(index=high.index, dtype=float)
        for i in range(period, len(high) - period):
            if is_fractal_high(high, i):
                fractals.iloc[i] = high.iloc[i]
        
        return fractals
    
    def fractal_lows(self, low: pd.Series, period: int = 5) -> pd.Series:
        """Фрактальные минимумы"""
        def is_fractal_low(data, idx):
            if idx < period or idx >= len(data) - period:
                return False
            center = data.iloc[idx]
            for i in range(1, period + 1):
                if data.iloc[idx - i] <= center or data.iloc[idx + i] <= center:
                    return False
            return True
        
        fractals = pd.Series(index=low.index, dtype=float)
        for i in range(period, len(low) - period):
            if is_fractal_low(low, i):
                fractals.iloc[i] = low.iloc[i]
        
        return fractals
    
    def support_resistance_levels(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                                 lookback: int = 50, touch_threshold: float = 0.02) -> Dict[str, List[float]]:
        """Уровни поддержки и сопротивления"""
        current_price = close.iloc[-1]
        price_range = high.tail(lookback).max() - low.tail(lookback).min()
        
        # Находим локальные экстремумы
        highs = self.fractal_highs(high.tail(lookback + 10), 3).dropna()
        lows = self.fractal_lows(low.tail(lookback + 10), 3).dropna()
        
        # Группируем близкие уровни
        resistance_levels = []
        support_levels = []
        
        if not highs.empty:
            for level in highs.values:
                touches = 0
                for price in high.tail(lookback):
                    if abs(price - level) <= price_range * touch_threshold:
                        touches += 1
                if touches >= 2:  # Минимум 2 касания
                    resistance_levels.append(level)
        
        if not lows.empty:
            for level in lows.values:
                touches = 0
                for price in low.tail(lookback):
                    if abs(price - level) <= price_range * touch_threshold:
                        touches += 1
                if touches >= 2:
                    support_levels.append(level)
        
        return {
            'resistance': sorted(list(set(resistance_levels)), reverse=True),
            'support': sorted(list(set(support_levels)))
        }
    
    # === КОМПЛЕКСНЫЕ ИНДИКАТОРЫ ===
    
    def ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Ichimoku Kinko Hyo"""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close shifted back 26 periods
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, float]:
        """Pivot Points (дневные уровни)"""
        prev_high = high.iloc[-2] if len(high) > 1 else high.iloc[-1]
        prev_low = low.iloc[-2] if len(low) > 1 else low.iloc[-1] 
        prev_close = close.iloc[-2] if len(close) > 1 else close.iloc[-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    # === МАРКЕТ СТРУКТУРА ===
    
    def market_structure(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> Dict[str, any]:
        """Анализ рыночной структуры"""
        
        # Тренд по MA
        ma_short = self.ema(close, 10)
        ma_long = self.ema(close, 50)
        trend = np.where(ma_short > ma_long, 1, -1)
        
        # Волатильность
        volatility = close.pct_change().rolling(window=period).std() * np.sqrt(288)  # Для 5-мин данных
        
        # Импульс
        momentum = close.pct_change(period)
        
        # Higher Highs, Lower Lows
        swing_highs = self.fractal_highs(high, 5).dropna()
        swing_lows = self.fractal_lows(low, 5).dropna()
        
        structure_strength = 0
        if len(swing_highs) >= 2:
            if swing_highs.iloc[-1] > swing_highs.iloc[-2]:
                structure_strength += 1
        if len(swing_lows) >= 2:
            if swing_lows.iloc[-1] > swing_lows.iloc[-2]:
                structure_strength += 1
        
        return {
            'trend': trend.iloc[-1] if len(trend) > 0 else 0,
            'volatility': volatility.iloc[-1] if len(volatility) > 0 else 0,
            'momentum': momentum.iloc[-1] if len(momentum) > 0 else 0,
            'structure_strength': structure_strength - 1,  # -1 to 1
            'consolidation': 1 if volatility.iloc[-1] < volatility.rolling(50).mean().iloc[-1] else 0
        }
    
    # === ОСНОВНАЯ ФУНКЦИЯ ===
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление всех технических индикаторов"""
        
        logger.info("🔧 Вычисление расширенного набора технических индикаторов...")
        
        result = df.copy()
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        open_price = df['open']
        
        try:
            # === ТРЕНДОВЫЕ ИНДИКАТОРЫ ===
            result['ema_5'] = self.ema(close, 5)
            result['ema_10'] = self.ema(close, 10)
            result['ema_20'] = self.ema(close, 20)
            result['ema_50'] = self.ema(close, 50)
            result['ema_200'] = self.ema(close, 200)
            
            result['sma_10'] = self.sma(close, 10)
            result['sma_20'] = self.sma(close, 20)
            result['sma_50'] = self.sma(close, 50)
            
            result['hull_ma_14'] = self.hull_ma(close, 14)
            result['tema_21'] = self.tema(close, 21)
            
            # ADX
            adx_data = self.adx(high, low, close)
            result['adx'] = adx_data['ADX']
            result['di_plus'] = adx_data['DI_plus']
            result['di_minus'] = adx_data['DI_minus']
            
            # === ОСЦИЛЛЯТОРЫ ===
            result['rsi_14'] = self.rsi(close, 14)
            result['rsi_21'] = self.rsi(close, 21)
            
            stoch = self.stochastic(high, low, close)
            result['stoch_k'] = stoch['K']
            result['stoch_d'] = stoch['D']
            
            result['williams_r'] = self.williams_r(high, low, close)
            result['cci'] = self.cci(high, low, close)
            result['cmo'] = self.cmo(close)
            
            # === ОБЪЕМНЫЕ ИНДИКАТОРЫ ===
            result['obv'] = self.obv(close, volume)
            result['ad_line'] = self.ad_line(high, low, close, volume)
            result['cmf'] = self.cmf(high, low, close, volume)
            result['vwap'] = self.vwap(high, low, close, volume)
            
            # === ВОЛАТИЛЬНОСТЬ ===
            bb = self.bollinger_bands(close)
            result['bb_upper'] = bb['upper']
            result['bb_middle'] = bb['middle']
            result['bb_lower'] = bb['lower']
            result['bb_width'] = bb['width']
            result['bb_position'] = bb['position']
            
            kc = self.keltner_channels(high, low, close)
            result['kc_upper'] = kc['upper']
            result['kc_middle'] = kc['middle']
            result['kc_lower'] = kc['lower']
            
            result['atr'] = self.atr(high, low, close)
            result['true_range'] = self.true_range(high, low, close)
            
            # === ICHIMOKU ===
            ichimoku = self.ichimoku(high, low, close)
            for key, value in ichimoku.items():
                result[f'ichimoku_{key}'] = value
            
            # === ДОПОЛНИТЕЛЬНЫЕ ФИЧИ ===
            # Ценовые паттерны
            result['doji'] = np.where(np.abs(close - open_price) <= (high - low) * 0.1, 1, 0)
            result['hammer'] = np.where(
                (np.minimum(close, open_price) - low) > (high - np.maximum(close, open_price)) * 2, 1, 0
            )
            
            # Gaps
            result['gap_up'] = np.where(low > high.shift(1), 1, 0)
            result['gap_down'] = np.where(high < low.shift(1), 1, 0)
            
            # Returns
            result['return_1'] = close.pct_change(1)
            result['return_5'] = close.pct_change(5)
            result['return_10'] = close.pct_change(10)
            result['return_20'] = close.pct_change(20)
            
            # Volume ratios
            result['volume_sma_ratio'] = volume / volume.rolling(20).mean()
            result['volume_ema_ratio'] = volume / self.ema(volume, 10)
            
            # Price position in range
            result['price_position'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min())
            
            # Очищаем NaN значения
            result = result.fillna(method='ffill').fillna(0)
            
            logger.info(f"✅ Добавлено {len(result.columns) - len(df.columns)} технических индикаторов")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка при вычислении индикаторов: {e}")
            return df


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Удобная функция для добавления всех индикаторов"""
    indicators = AdvancedTechnicalIndicators()
    return indicators.calculate_all_indicators(df)


if __name__ == "__main__":
    # Демонстрация использования
    logging.basicConfig(level=logging.INFO)
    
    # Пример использования
    print("🔧 Модуль технических индикаторов готов к использованию!")
    print("Для использования: from src.utils.technical_indicators import add_advanced_features") 