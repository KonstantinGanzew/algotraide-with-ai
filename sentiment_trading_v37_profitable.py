"""
🚀 PROFITABLE SENTIMENT ТОРГОВАЯ СИСТЕМА V3.7
Улучшенная стратегия для достижения 20% годовых (14400 за 2 года)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ProfitableConfig:
    """Оптимизированная конфигурация для прибыльной торговли"""
    
    # Управление рисками - агрессивнее для 20% годовых
    INITIAL_BALANCE = 10000
    RISK_PER_TRADE = 0.08  # Увеличено с 0.02
    MAX_POSITION_SIZE = 0.6  # Увеличено с 0.25
    STOP_LOSS = 0.05  # Ослаблен с 0.03
    TAKE_PROFIT = 0.15  # Увеличен с 0.08
    TRAILING_STOP = 0.03  # Новый параметр
    
    # Веса источников данных - больше веса техническому анализу
    TECHNICAL_WEIGHT = 0.7
    SENTIMENT_WEIGHT = 0.15
    MOMENTUM_WEIGHT = 0.1
    VOLATILITY_WEIGHT = 0.05
    
    # Настройки модели
    WINDOW_SIZE = 72  # Увеличено окно
    TOTAL_TIMESTEPS = 10000  # Больше обучения
    LEARNING_RATE = 5e-5  # Более консервативное обучение
    
    # Параметры анализа настроений
    SENTIMENT_THRESHOLD = 0.15
    SENTIMENT_MULTIPLIER = 1.3
    
    # Новые параметры для улучшения
    MIN_VOLUME_RATIO = 1.2  # Минимальное отношение объема
    TREND_CONFIRMATION_PERIODS = 3
    RSI_OVERSOLD = 25
    RSI_OVERBOUGHT = 75
    
    # Фильтры качества сигнала
    MIN_SIGNAL_STRENGTH = 0.6
    MAX_DAILY_TRADES = 3


class EnhancedDataLoader:
    """Улучшенный загрузчик данных с дополнительными индикаторами"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_bitcoin_data(self) -> pd.DataFrame:
        """Загрузка исторических данных Bitcoin"""
        print(f"📊 Загрузка данных из {self.data_path}...")
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['datetime'] = df['timestamp']
        
        print(f"📅 Период: {df['datetime'].min()} - {df['datetime'].max()}")
        print(f"📈 Цены: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"📊 Записей: {len(df)}")
        return df
    
    def add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление расширенного набора технических индикаторов"""
        print("🔧 Расчет расширенных технических индикаторов...")
        
        # Базовые
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(24).std()
        
        # Множественные временные рамки для MA
        for period in [5, 10, 12, 20, 26, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # MACD семейство (после создания EMA)
        df['macd_12_26'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd_12_26'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd_12_26'] - df['macd_signal']
        
        # RSI с разными периодами
        for period in [14, 21, 30]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands с разными периодами
        for period in [20, 50]:
            bb_middle = df['close'].rolling(period).mean()
            bb_std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
            df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / bb_middle
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # Объемные индикаторы
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['price_volume'] = df['close'] * df['volume']
        df['vwap'] = df['price_volume'].rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Моментум индикаторы
        for period in [3, 5, 10, 20, 50]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
        
        # Уровни поддержки/сопротивления
        for period in [10, 20, 50]:
            df[f'resistance_{period}'] = df['high'].rolling(period).max()
            df[f'support_{period}'] = df['low'].rolling(period).min()
            df[f'price_position_{period}'] = (df['close'] - df[f'support_{period}']) / (df[f'resistance_{period}'] - df[f'support_{period}'])
        
        # Волатильность
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_21'] = self._calculate_atr(df, 21)
        
        # Trend strength
        df['trend_strength'] = abs(df['close'] - df['sma_50']) / df['sma_50']
        
        # Market regime indicators
        df['bull_bear_ratio'] = (df['close'] > df['sma_50']).astype(int).rolling(20).mean()
        df['volatility_regime'] = (df['volatility'] > df['volatility'].rolling(50).mean()).astype(int)
        
        print(f"✅ Добавлено {len([col for col in df.columns if col not in ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']])} индикаторов")
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Расчет Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()


class SmartSentimentGenerator:
    """Умный генератор настроений с корреляцией к реальным движениям"""
    
    def generate_smart_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Генерация умных данных настроений"""
        print("🧠 Генерация умных данных настроений...")
        
        # Анализируем структуру рынка
        df['price_change_1h'] = df['close'].pct_change(1)
        df['price_change_4h'] = df['close'].pct_change(4)
        df['price_change_24h'] = df['close'].pct_change(24)
        
        # Волатильность и тренд
        df['short_vol'] = df['returns'].rolling(12).std()
        df['long_vol'] = df['returns'].rolling(48).std()
        df['vol_ratio'] = df['short_vol'] / df['long_vol']
        
        # Умное настроение основано на реальных паттернах
        np.random.seed(42)
        n = len(df)
        
        # Базовое настроение коррелирует с momentum
        momentum_signal = df['momentum_20'].fillna(0)
        trend_signal = np.where(df['close'] > df['sma_50'], 0.3, -0.3)
        volume_signal = np.clip((df['volume_ratio'].fillna(1) - 1) * 0.5, -0.3, 0.3)
        
        # Добавляем реалистичные шумы и события
        noise = np.random.normal(0, 0.08, n)
        
        # События происходят реже, но сильнее
        events = np.random.choice([0, 1], size=n, p=[0.97, 0.03])
        event_strength = np.random.normal(0, 0.6, n) * events
        
        # Применяем затухание событий
        for i in range(1, len(event_strength)):
            if events[i-1] == 1:
                event_strength[i] += event_strength[i-1] * 0.7
        
        # Комбинируем все сигналы
        base_sentiment = (
            momentum_signal * 0.4 +
            trend_signal * 0.3 +
            volume_signal * 0.2 +
            noise * 0.1
        ) + event_strength
        
        base_sentiment = np.clip(base_sentiment, -1, 1)
        
        # Создаем различные источники с разными характеристиками
        df['sentiment_twitter'] = np.clip(base_sentiment + np.random.normal(0, 0.06, n), -1, 1)
        df['sentiment_reddit'] = np.clip(base_sentiment + np.random.normal(0, 0.08, n), -1, 1)
        df['sentiment_news'] = np.clip(base_sentiment + np.random.normal(0, 0.05, n), -1, 1)
        df['sentiment_social_volume'] = np.abs(base_sentiment) * 1200 + np.random.exponential(400, n)
        
        # Агрегированное настроение
        df['overall_sentiment'] = (
            df['sentiment_twitter'] * 0.5 +
            df['sentiment_reddit'] * 0.3 +
            df['sentiment_news'] * 0.2
        )
        
        # Сила сигнала настроения
        df['sentiment_strength'] = np.abs(df['overall_sentiment'])
        df['sentiment_consistency'] = df['overall_sentiment'].rolling(6).std()
        
        print(f"✅ Сгенерированы умные данные настроений для {n} точек")
        return df


class ProfitableFeatureExtractor(BaseFeaturesExtractor):
    """Улучшенный Feature Extractor для прибыльной торговли"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        self.seq_len = ProfitableConfig.WINDOW_SIZE
        self.input_features = observation_space.shape[1] if observation_space.shape else 100
        
        # Многослойная обработка признаков
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Улучшенный LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1 if 2 > 1 else 0.0
        )
        
        # Attention механизм
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.Tanh()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = observations.shape
        
        # Feature processing на последнем наблюдении
        last_obs = observations[:, -1, :]
        feature_out = self.feature_net(last_obs)
        
        # LSTM обработка
        lstm_out, _ = self.lstm(observations)
        
        # Attention на LSTM выходе
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_features = attended_out[:, -1, :]
        
        # Объединение и финальная обработка
        combined = torch.cat([lstm_features, feature_out], dim=1)
        output = self.fusion_net(combined)
        
        return output


class ProfitableTradingEnv(gym.Env):
    """Улучшенное торговое окружение для достижения 20% годовых"""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = ProfitableConfig.WINDOW_SIZE
        
        # Подготовка признаков
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numeric_columns if col not in 
                               ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        
        n_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, n_features),
            dtype=np.float32
        )
        
        self._prepare_data()
        self._reset_state()
    
    def _prepare_data(self):
        """Улучшенная подготовка данных"""
        print("🔧 Подготовка данных для прибыльной торговли...")
        
        # Заполнение NaN и обработка выбросов
        feature_data = self.df[self.feature_columns].copy()
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        feature_data = feature_data.replace([np.inf, -np.inf], 0)
        
        # Продвинутая нормализация с учетом выбросов
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(feature_data)
        
        # Более мягкое ограничение выбросов
        normalized_data = np.clip(normalized_data, -3, 3)
        
        self.normalized_df = pd.DataFrame(normalized_data, columns=self.feature_columns, index=feature_data.index)
        
        print(f"✅ Подготовлено {len(self.feature_columns)} признаков для {len(self.df)} точек")
    
    def _reset_state(self):
        """Сброс состояния с дополнительными метриками"""
        self.current_step = self.window_size
        self.balance = ProfitableConfig.INITIAL_BALANCE
        self.btc_amount = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.trailing_stop_price = 0.0
        
        self.total_trades = 0
        self.profitable_trades = 0
        self.daily_trades = 0
        self.last_trade_day = None
        
        self.portfolio_history = [float(ProfitableConfig.INITIAL_BALANCE)]
        self.trades_history = []
        
        self.max_drawdown = 0.0
        self.peak_value = ProfitableConfig.INITIAL_BALANCE
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
    
    def reset(self, seed=None, options=None):
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Получение улучшенного наблюдения"""
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        obs = self.normalized_df.iloc[start_idx:end_idx].values
        
        if len(obs) < self.window_size:
            padding = np.tile(obs[0] if len(obs) > 0 else np.zeros(len(self.feature_columns)), 
                            (self.window_size - len(obs), 1))
            obs = np.vstack([padding, obs])
        
        return obs.astype(np.float32)
    
    def _get_current_price(self) -> float:
        """Получение текущей цены"""
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.iloc[self.current_step]['close']
    
    def _get_current_datetime(self) -> str:
        """Получение текущего времени"""
        if self.current_step >= len(self.df):
            return str(self.df.iloc[-1]['datetime'])
        return str(self.df.iloc[self.current_step]['datetime'])
    
    def _calculate_signal_strength(self) -> float:
        """Расчет силы торгового сигнала"""
        if self.current_step >= len(self.df):
            return 0.0
        
        current_data = self.df.iloc[self.current_step]
        
        # Технические сигналы
        rsi_signal = 0.0
        if 'rsi_14' in current_data.index:
            rsi = current_data['rsi_14']
            if rsi < ProfitableConfig.RSI_OVERSOLD:
                rsi_signal = (ProfitableConfig.RSI_OVERSOLD - rsi) / ProfitableConfig.RSI_OVERSOLD
            elif rsi > ProfitableConfig.RSI_OVERBOUGHT:
                rsi_signal = -(rsi - ProfitableConfig.RSI_OVERBOUGHT) / (100 - ProfitableConfig.RSI_OVERBOUGHT)
        
        # MACD сигнал
        macd_signal = 0.0
        if 'macd_histogram' in current_data.index:
            macd_hist = current_data['macd_histogram']
            if not pd.isna(macd_hist):
                macd_signal = np.tanh(macd_hist * 10)  # Normalize
        
        # Bollinger Bands сигнал
        bb_signal = 0.0
        if 'bb_position_20' in current_data.index:
            bb_pos = current_data['bb_position_20']
            if not pd.isna(bb_pos):
                if bb_pos < 0.2:
                    bb_signal = 0.3  # Oversold
                elif bb_pos > 0.8:
                    bb_signal = -0.3  # Overbought
        
        # Объемный сигнал
        volume_signal = 0.0
        if 'volume_ratio' in current_data.index:
            vol_ratio = current_data['volume_ratio']
            if not pd.isna(vol_ratio) and vol_ratio > ProfitableConfig.MIN_VOLUME_RATIO:
                volume_signal = min((vol_ratio - 1) * 0.2, 0.3)
        
        # Сигнал настроения
        sentiment_signal = 0.0
        if 'overall_sentiment' in current_data.index:
            sentiment = current_data['overall_sentiment']
            if not pd.isna(sentiment) and abs(sentiment) > ProfitableConfig.SENTIMENT_THRESHOLD:
                sentiment_signal = sentiment * ProfitableConfig.SENTIMENT_MULTIPLIER * 0.3
        
        # Агрегированный сигнал
        total_signal = (
            rsi_signal * 0.25 +
            macd_signal * 0.25 +
            bb_signal * 0.2 +
            volume_signal * 0.15 +
            sentiment_signal * 0.15
        )
        
        return np.clip(total_signal, -1, 1)
    
    def _check_daily_trade_limit(self) -> bool:
        """Проверка лимита сделок в день"""
        current_date = self._get_current_datetime()[:10]  # YYYY-MM-DD
        
        if self.last_trade_day != current_date:
            self.daily_trades = 0
            self.last_trade_day = current_date
        
        return self.daily_trades < ProfitableConfig.MAX_DAILY_TRADES
    
    def _update_trailing_stop(self, current_price: float):
        """Обновление trailing stop"""
        if self.btc_amount > 0 and self.entry_price > 0:
            if self.trailing_stop_price == 0:
                self.trailing_stop_price = current_price * (1 - ProfitableConfig.TRAILING_STOP)
            else:
                # Обновляем только если цена растет
                new_trailing_stop = current_price * (1 - ProfitableConfig.TRAILING_STOP)
                if new_trailing_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_trailing_stop
    
    def _check_exit_conditions(self, current_price: float) -> str:
        """Проверка условий выхода из позиции"""
        if self.btc_amount <= 0 or self.entry_price <= 0:
            return 'none'
        
        price_change = (current_price - self.entry_price) / self.entry_price
        
        # Стоп-лосс
        if price_change <= -ProfitableConfig.STOP_LOSS:
            return 'stop_loss'
        
        # Тейк-профит
        if price_change >= ProfitableConfig.TAKE_PROFIT:
            return 'take_profit'
        
        # Trailing stop
        if self.trailing_stop_price > 0 and current_price <= self.trailing_stop_price:
            return 'trailing_stop'
        
        return 'none'
    
    def _execute_trade(self, action: int, current_price: float) -> Dict[str, Any]:
        """Улучшенная логика выполнения торговых операций"""
        signal_strength = self._calculate_signal_strength()
        trade_result = {'executed': False, 'type': None, 'signal_strength': signal_strength}
        
        # Обновляем trailing stop
        self._update_trailing_stop(current_price)
        
        # Проверка принудительного выхода
        exit_condition = self._check_exit_conditions(current_price)
        if exit_condition != 'none' and self.btc_amount > 0:
            revenue = self.btc_amount * current_price
            commission = revenue * 0.0015  # Более реалистичная комиссия
            profit = revenue - self.btc_amount * self.entry_price - commission
            
            if profit > 0:
                self.profitable_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
            
            self.balance += revenue - commission
            self.btc_amount = 0.0
            self.entry_price = 0.0
            self.trailing_stop_price = 0.0
            self.daily_trades += 1
            
            trade_result.update({
                'executed': True, 'type': f'SELL_{exit_condition.upper()}',
                'profit': profit, 'price': current_price,
                'datetime': self._get_current_datetime()
            })
            self.total_trades += 1
            self.trades_history.append(trade_result)
            
            return trade_result
        
        # Проверка лимита сделок в день
        if not self._check_daily_trade_limit():
            return trade_result
        
        # Фильтр по силе сигнала
        if abs(signal_strength) < ProfitableConfig.MIN_SIGNAL_STRENGTH:
            return trade_result
        
        # Торговые действия
        if action == 1 and self.balance > 500 and signal_strength > 0:  # Buy
            # Динамический размер позиции на основе силы сигнала
            base_position_size = ProfitableConfig.RISK_PER_TRADE
            signal_multiplier = 1 + abs(signal_strength) * 0.5
            position_size = min(base_position_size * signal_multiplier, ProfitableConfig.MAX_POSITION_SIZE)
            
            # Уменьшаем размер после убытков
            if self.consecutive_losses > 2:
                position_size *= 0.5
            
            investment = self.balance * position_size
            amount = investment / current_price
            commission = investment * 0.0015
            
            if investment + commission <= self.balance:
                self.btc_amount += amount
                self.balance -= investment + commission
                self.entry_price = current_price
                self.entry_step = self.current_step
                self.trailing_stop_price = 0.0  # Сброс trailing stop
                self.daily_trades += 1
                
                trade_result.update({
                    'executed': True, 'type': 'BUY',
                    'amount': amount, 'price': current_price,
                    'investment': investment, 'position_size': position_size,
                    'datetime': self._get_current_datetime()
                })
                
        elif action == 2 and self.btc_amount > 0 and signal_strength < -0.3:  # Sell (manual)
            revenue = self.btc_amount * current_price
            commission = revenue * 0.0015
            profit = revenue - self.btc_amount * self.entry_price - commission
            
            if profit > 0:
                self.profitable_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
            
            self.balance += revenue - commission
            self.btc_amount = 0.0
            self.entry_price = 0.0
            self.trailing_stop_price = 0.0
            self.daily_trades += 1
            
            trade_result.update({
                'executed': True, 'type': 'SELL_MANUAL',
                'profit': profit, 'price': current_price,
                'datetime': self._get_current_datetime()
            })
        
        if trade_result['executed']:
            self.total_trades += 1
            self.trades_history.append(trade_result)
        
        return trade_result
    
    def _calculate_portfolio_value(self) -> float:
        """Расчет стоимости портфеля"""
        current_price = self._get_current_price()
        return self.balance + self.btc_amount * current_price
    
    def _calculate_reward(self) -> float:
        """Улучшенная функция награды"""
        current_portfolio = self._calculate_portfolio_value()
        prev_portfolio = self.portfolio_history[-1]
        
        # Базовая награда от изменения портфеля
        portfolio_change = (current_portfolio - prev_portfolio) / prev_portfolio
        base_reward = portfolio_change * 200  # Увеличенный масштаб
        
        # Обновление максимальной просадки
        if current_portfolio > self.peak_value:
            self.peak_value = current_portfolio
        current_drawdown = (self.peak_value - current_portfolio) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Бонусы и штрафы
        if portfolio_change > 0.01:  # Награда за прибыль
            base_reward += 10
        
        if current_drawdown > 0.1:  # Штраф за просадку
            base_reward -= current_drawdown * 200
        
        if self.consecutive_losses > 3:  # Штраф за серию убытков
            base_reward -= self.consecutive_losses * 5
        
        # Бонус за эффективность сделок
        if self.total_trades > 0:
            win_rate = self.profitable_trades / self.total_trades
            if win_rate > 0.6:
                base_reward += 20
            elif win_rate < 0.4:
                base_reward -= 15
        
        return base_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Шаг симуляции с улучшенной логикой"""
        current_price = self._get_current_price()
        
        # Выполнение действия
        trade_result = self._execute_trade(action, current_price)
        
        # Обновление состояния
        self.current_step += 1
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_history.append(portfolio_value)
        
        # Расчет награды
        reward = self._calculate_reward()
        
        # Проверка завершения
        done = (
            self.current_step >= len(self.df) - 1 or
            portfolio_value <= ProfitableConfig.INITIAL_BALANCE * 0.3 or
            self.max_drawdown > 0.5 or
            self.consecutive_losses > 8
        )
        
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'btc_amount': self.btc_amount,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'max_drawdown': self.max_drawdown,
            'signal_strength': trade_result.get('signal_strength', 0),
            'current_price': current_price,
            'datetime': self._get_current_datetime(),
            'trade_result': trade_result,
            'consecutive_losses': self.consecutive_losses,
            'daily_trades': self.daily_trades
        }
        
        return self._get_observation(), reward, done, False, info


def main():
    """Главная функция для прибыльной торговой системы"""
    print("🚀 PROFITABLE SENTIMENT ТОРГОВАЯ СИСТЕМА V3.7")
    print("🎯 ЦЕЛЬ: 20% ГОДОВЫХ (14400 ЗА 2 ГОДА)")
    print("=" * 75)
    
    # 1. Загрузка данных
    print("\n📊 ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("-" * 50)
    
    data_loader = EnhancedDataLoader("data/BTC_5_2w.csv")
    bitcoin_df = data_loader.load_bitcoin_data()
    
    # 2. Добавление расширенных индикаторов
    print("\n🔧 ЭТАП 2: РАСШИРЕННЫЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ")
    print("-" * 50)
    bitcoin_df = data_loader.add_advanced_indicators(bitcoin_df)
    
    # 3. Умная генерация настроений
    print("\n🧠 ЭТАП 3: УМНАЯ ГЕНЕРАЦИЯ НАСТРОЕНИЙ")
    print("-" * 50)
    sentiment_generator = SmartSentimentGenerator()
    combined_df = sentiment_generator.generate_smart_sentiment(bitcoin_df)
    
    # Финальная очистка данных
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"✅ Финальный датасет: {len(combined_df)} записей, {len(combined_df.columns)} признаков")
    
    # 4. Создание улучшенного окружения
    print("\n🎮 ЭТАП 4: СОЗДАНИЕ УЛУЧШЕННОГО ОКРУЖЕНИЯ")
    print("-" * 50)
    env = ProfitableTradingEnv(combined_df)
    vec_env = DummyVecEnv([lambda: env])
    print(f"✅ Окружение готово к прибыльной торговле")
    
    # 5. Создание улучшенной модели
    print("\n🧠 ЭТАП 5: СОЗДАНИЕ УЛУЧШЕННОЙ МОДЕЛИ")
    print("-" * 50)
    
    policy_kwargs = dict(
        features_extractor_class=ProfitableFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[512, 256, 128],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=ProfitableConfig.LEARNING_RATE,
        n_steps=2048,
        batch_size=64,
        n_epochs=6,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.15,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    print("✅ Улучшенная модель создана")
    
    # 6. Интенсивное обучение
    print("\n🎓 ЭТАП 6: ИНТЕНСИВНОЕ ОБУЧЕНИЕ")
    print("-" * 50)
    model.learn(total_timesteps=ProfitableConfig.TOTAL_TIMESTEPS)
    print("✅ Интенсивное обучение завершено")
    
    # 7. Тестирование прибыльности
    print("\n💰 ЭТАП 7: ТЕСТИРОВАНИЕ ПРИБЫЛЬНОСТИ")
    print("-" * 50)
    
    obs, _ = env.reset()
    results = []
    trades_log = []
    
    print("💼 Начинаем прибыльную торговлю...")
    
    for step in range(min(5000, len(combined_df) - env.window_size - 1)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        
        results.append({
            'step': step,
            'datetime': info['datetime'],
            'portfolio_value': info['portfolio_value'],
            'current_price': info['current_price'],
            'balance': info['balance'],
            'btc_amount': info['btc_amount'],
            'total_trades': info['total_trades'],
            'profitable_trades': info['profitable_trades'],
            'max_drawdown': info['max_drawdown'],
            'signal_strength': info['signal_strength'],
            'consecutive_losses': info['consecutive_losses']
        })
        
        # Логируем важные сделки
        if info['trade_result']['executed']:
            trades_log.append(info['trade_result'])
            trade_type = info['trade_result']['type']
            price = info['trade_result']['price']
            datetime = info['trade_result']['datetime']
            signal = info['signal_strength']
            print(f"💎 {trade_type} ${price:.2f} сигнал:{signal:.2f} {datetime}")
        
        if done:
            break
    
    # 8. Анализ прибыльности
    print("\n📊 ЭТАП 8: АНАЛИЗ ПРИБЫЛЬНОСТИ")
    print("-" * 50)
    
    if results:
        final_result = results[-1]
        
        initial_value = ProfitableConfig.INITIAL_BALANCE
        final_value = final_result['portfolio_value']
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Экстраполяция на 2 года
        days_tested = len(results) / 12  # Примерно 5-минутные интервалы
        years_tested = days_tested / 365
        annualized_return = (final_value / initial_value) ** (1 / years_tested) - 1 if years_tested > 0 else 0
        two_year_projection = initial_value * ((1 + annualized_return) ** 2)
        two_year_profit = two_year_projection - initial_value
        
        total_trades = final_result['total_trades']
        profitable_trades = final_result['profitable_trades']
        win_rate = (profitable_trades / max(total_trades, 1)) * 100
        max_drawdown = final_result['max_drawdown'] * 100
        
        # Анализ эффективности
        portfolio_values = [r['portfolio_value'] for r in results]
        returns = [portfolio_values[i] / portfolio_values[i-1] - 1 for i in range(1, len(portfolio_values))]
        volatility = np.std(returns) * np.sqrt(len(returns)) if returns else 0
        sharpe_ratio = (np.mean(returns) / volatility) if volatility > 0 else 0
        
        # Buy & Hold сравнение
        start_price = results[0]['current_price']
        end_price = final_result['current_price']
        bnh_return = (end_price - start_price) / start_price * 100
        
        print("🎯 РЕЗУЛЬТАТЫ ПРИБЫЛЬНОЙ СТРАТЕГИИ")
        print("=" * 70)
        print(f"💰 Начальный капитал: ${initial_value:,.2f}")
        print(f"💰 Финальная стоимость: ${final_value:,.2f}")
        print(f"📈 Доходность за период: {total_return:+.2f}%")
        print(f"📈 Годовая доходность: {annualized_return*100:+.2f}%")
        print(f"🎯 Прогноз за 2 года: ${two_year_projection:,.2f}")
        print(f"💎 Прибыль за 2 года: ${two_year_profit:,.2f}")
        print(f"🏆 Цель 14400: {'✅ ДОСТИГНУТА' if two_year_profit >= 14400 else '❌ НЕ ДОСТИГНУТА'}")
        print(f"📊 Buy & Hold: {bnh_return:+.2f}%")
        print(f"🔄 Всего сделок: {total_trades}")
        print(f"✅ Прибыльных: {profitable_trades} ({win_rate:.1f}%)")
        print(f"📉 Макс. просадка: {max_drawdown:.2f}%")
        print(f"⚡ Коэф. Шарпа: {sharpe_ratio:.3f}")
        
        print("\n🎉 ЗАКЛЮЧЕНИЕ")
        print("=" * 50)
        
        if two_year_profit >= 14400 and win_rate > 55 and max_drawdown < 25:
            print("🟢 ОТЛИЧНО! Цель 20% годовых ДОСТИГНУТА!")
            print(f"💰 Ожидаемая прибыль за 2 года: ${two_year_profit:,.2f}")
        elif two_year_profit >= 10000:
            print("🟡 ХОРОШО! Близко к цели, требует оптимизации")
        elif total_return > bnh_return:
            print("🔶 ПРОГРЕСС! Превосходит пассивную стратегию")
        else:
            print("🔴 ТРЕБУЕТ ДОРАБОТКИ!")
        
        print(f"\n🚀 Улучшенная стратегия показала {annualized_return*100:+.2f}% годовых")
        print("✨ Оптимизация для достижения 20% годовых завершена!")
    
    else:
        print("❌ Недостаточно данных для анализа")


if __name__ == "__main__":
    main() 