"""
🚀 SENTIMENT ТОРГОВАЯ СИСТЕМА V3.3 - OPTIMIZED EDITION
Улучшенная система с оптимизированным риск-менеджментом и стратегией
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
import re
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class OptimizedSentimentConfig:
    """Оптимизированная конфигурация для системы V3.3"""
    
    # Улучшенное управление рисками
    INITIAL_BALANCE = 10000
    RISK_PER_TRADE = 0.015  # Снижен риск на сделку
    MAX_POSITION_SIZE = 0.3  # Максимальный размер позиции
    STOP_LOSS = 0.05  # 5% стоп-лосс
    TAKE_PROFIT = 0.15  # 15% тейк-профит
    
    # Оптимизированные веса источников данных
    TECHNICAL_WEIGHT = 0.5      # Увеличен вес технических индикаторов
    SENTIMENT_WEIGHT = 0.25     # Умеренный вес настроений
    ON_CHAIN_WEIGHT = 0.15      # Снижен вес on-chain данных
    MACRO_WEIGHT = 0.1          # Минимальный вес макро
    
    # Настройки анализа настроений
    SENTIMENT_THRESHOLD = 0.15   # Повышен порог значимости
    SENTIMENT_MULTIPLIER = 1.2   # Снижен мультипликатор
    SENTIMENT_WINDOW = 24
    NEWS_IMPACT_DECAY = 0.85
    
    # Параметры модели
    WINDOW_SIZE = 48
    TOTAL_TIMESTEPS = 15000
    LEARNING_RATE = 2e-4
    
    # Новые параметры оптимизации
    MIN_TRADES_THRESHOLD = 5     # Минимум сделок для активации
    PROFIT_THRESHOLD = 0.02      # Минимальная прибыль для закрытия
    VOLATILITY_ADJUSTMENT = True # Адаптация к волатильности


# Используем классы из V3.2 с минимальными изменениями
class SentimentAnalyzer:
    """Анализатор настроений для социальных сетей и новостей"""
    
    def __init__(self):
        self.positive_words = [
            'bullish', 'moon', 'pump', 'buy', 'hold', 'diamond', 'hands',
            'rocket', 'green', 'profit', 'gains', 'up', 'rise', 'surge',
            'breakout', 'bull', 'strong', 'support', 'resistance', 'btfd'
        ]
        
        self.negative_words = [
            'bearish', 'dump', 'sell', 'crash', 'down', 'red', 'loss',
            'fear', 'panic', 'drop', 'fall', 'weak', 'breakdown', 'bear',
            'correction', 'dip', 'liquidation', 'fud', 'rekt', 'bag'
        ]
        
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'blockchain',
            'defi', 'nft', 'altcoin', 'hodl', 'satoshi', 'whale'
        ]
    
    def analyze_text_sentiment(self, text: str) -> float:
        """Анализ настроения текста"""
        if not text:
            return 0.0
        
        text = text.lower()
        crypto_relevance = sum(1 for word in self.crypto_keywords if word in text)
        if crypto_relevance == 0:
            return 0.0
        
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)
        
        text_length = len(text.split())
        if text_length == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / max(text_length, 1)
        sentiment_score *= min(crypto_relevance / 3, 1.0)
        
        return np.clip(sentiment_score, -1.0, 1.0)
    
    def generate_social_sentiment_data(self, n_points: int) -> pd.DataFrame:
        """Генерация симулированных данных социальных настроений"""
        print("📱 Генерация оптимизированных данных настроений...")
        
        np.random.seed(42)
        timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
        
        # Более реалистичные паттерны настроений
        base_sentiment = np.random.normal(0, 0.2, n_points)
        
        # События с меньшей частотой, но большим воздействием
        events = np.random.choice([0, 1], size=n_points, p=[0.97, 0.03])
        event_impact = np.random.normal(0, 0.6, n_points) * events
        
        # Затухание влияния событий
        for i in range(1, len(event_impact)):
            if events[i-1] == 1:
                event_impact[i] += event_impact[i-1] * OptimizedSentimentConfig.NEWS_IMPACT_DECAY
        
        total_sentiment = base_sentiment + event_impact
        total_sentiment = np.clip(total_sentiment, -1, 1)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'twitter_sentiment': total_sentiment + np.random.normal(0, 0.05, n_points),
            'reddit_sentiment': total_sentiment + np.random.normal(0, 0.08, n_points),
            'news_sentiment': total_sentiment + np.random.normal(0, 0.1, n_points),
            'social_volume': np.abs(total_sentiment) * 800 + np.random.exponential(400, n_points),
            'mentions_count': np.abs(total_sentiment) * 80 + np.random.poisson(40, n_points),
            'influencer_sentiment': total_sentiment * 1.1 + np.random.normal(0, 0.15, n_points)
        })
        
        # Нормализация
        for col in ['twitter_sentiment', 'reddit_sentiment', 'news_sentiment', 'influencer_sentiment']:
            df[col] = np.clip(df[col], -1, 1)
        
        print(f"✅ Сгенерировано {len(df)} записей оптимизированных социальных данных")
        return df


class OnChainAnalyzer:
    """Анализатор on-chain метрик"""
    
    def generate_onchain_data(self, n_points: int) -> pd.DataFrame:
        """Генерация on-chain данных"""
        print("⛓️ Генерация оптимизированных on-chain данных...")
        
        np.random.seed(43)
        timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
        
        price_trend = np.linspace(45000, 65000, n_points) + np.random.normal(0, 1500, n_points)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'active_addresses': 750000 + price_trend / 120 + np.random.normal(0, 40000, n_points),
            'transaction_count': 230000 + price_trend / 250 + np.random.normal(0, 25000, n_points),
            'hash_rate': 180e18 + price_trend * 0.8e15 + np.random.normal(0, 8e18, n_points),
            'difficulty': 22e12 + price_trend * 0.8e9 + np.random.normal(0, 0.8e12, n_points),
            'exchange_inflow': np.random.exponential(800, n_points),
            'exchange_outflow': np.random.exponential(950, n_points),
            'whale_transactions': np.random.poisson(40, n_points),
            'new_addresses': 320000 + price_trend / 350 + np.random.normal(0, 15000, n_points)
        })
        
        df['net_exchange_flow'] = df['exchange_inflow'] - df['exchange_outflow']
        df['address_growth_rate'] = df['new_addresses'].pct_change().rolling(24).mean()
        df['network_value'] = df['active_addresses'] * price_trend / 1200000
        
        print(f"✅ Сгенерировано {len(df)} записей оптимизированных on-chain данных")
        return df


class MacroEconomicAnalyzer:
    """Анализатор макроэкономических данных"""
    
    def generate_macro_data(self, n_points: int) -> pd.DataFrame:
        """Генерация макроэкономических данных"""
        print("🌍 Генерация оптимизированных макроэкономических данных...")
        
        np.random.seed(44)
        timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'dxy_index': 103 + np.cumsum(np.random.normal(0, 0.05, n_points)),
            'vix_index': 20 + np.abs(np.cumsum(np.random.normal(0, 0.3, n_points))),
            'gold_price': 1900 + np.cumsum(np.random.normal(0, 3, n_points)),
            'sp500_index': 4000 + np.cumsum(np.random.normal(0, 8, n_points)),
            'fed_rate': 5.25 + np.cumsum(np.random.normal(0, 0.005, n_points)) / 100,
            'inflation_rate': 3.2 + np.cumsum(np.random.normal(0, 0.02, n_points)) / 100,
            'unemployment_rate': 3.7 + np.cumsum(np.random.normal(0, 0.01, n_points)) / 100
        })
        
        # Ограничения
        df['vix_index'] = np.clip(df['vix_index'], 12, 70)
        df['fed_rate'] = np.clip(df['fed_rate'], 0, 8)
        df['inflation_rate'] = np.clip(df['inflation_rate'], -1, 12)
        df['unemployment_rate'] = np.clip(df['unemployment_rate'], 2.5, 12)
        
        # Производные показатели
        df['risk_appetite'] = (80 - df['vix_index']) / 80
        df['dollar_strength'] = (df['dxy_index'] - 100) / 8
        df['macro_sentiment'] = (df['risk_appetite'] - df['dollar_strength']) / 2
        
        print(f"✅ Сгенерировано {len(df)} записей оптимизированных макро данных")
        return df


class OptimizedFeatureExtractor(BaseFeaturesExtractor):
    """Оптимизированный Feature Extractor"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        if observation_space.shape is not None:
            self.seq_len = observation_space.shape[0]
            self.input_features = observation_space.shape[1]
        else:
            self.seq_len = OptimizedSentimentConfig.WINDOW_SIZE
            self.input_features = 50
        
        # Упрощенная архитектура для стабильности
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Улучшенный LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_features,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Финальная сеть
        self.fusion_net = nn.Sequential(
            nn.Linear(96 + 64, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Dropout(0.2),
            nn.Linear(192, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = observations.shape
        
        last_obs = observations[:, -1, :]
        
        # Обработка признаков
        feature_out = self.feature_net(last_obs)
        
        # LSTM
        lstm_out, _ = self.lstm(observations)
        lstm_features = lstm_out[:, -1, :]
        
        # Объединение
        combined = torch.cat([lstm_features, feature_out], dim=1)
        output = self.fusion_net(combined)
        
        return output


def combine_optimized_data_sources(price_data: pd.DataFrame, 
                                 sentiment_data: pd.DataFrame,
                                 onchain_data: pd.DataFrame, 
                                 macro_data: pd.DataFrame) -> pd.DataFrame:
    """Оптимизированное объединение данных"""
    print("🔄 Объединение оптимизированных источников данных...")
    
    combined = price_data.copy()
    
    # Добавляем данные с весами
    for col in sentiment_data.columns:
        if col != 'timestamp':
            combined[f'sentiment_{col}'] = sentiment_data[col].values * OptimizedSentimentConfig.SENTIMENT_WEIGHT
    
    for col in onchain_data.columns:
        if col != 'timestamp':
            combined[f'onchain_{col}'] = onchain_data[col].values * OptimizedSentimentConfig.ON_CHAIN_WEIGHT
    
    for col in macro_data.columns:
        if col != 'timestamp':
            combined[f'macro_{col}'] = macro_data[col].values * OptimizedSentimentConfig.MACRO_WEIGHT
    
    # Улучшенные агрегированные показатели
    combined['overall_sentiment'] = (
        combined['sentiment_twitter_sentiment'] * 0.4 +
        combined['sentiment_reddit_sentiment'] * 0.3 +
        combined['sentiment_news_sentiment'] * 0.2 +
        combined['sentiment_influencer_sentiment'] * 0.1
    )
    
    combined['network_strength'] = (
        combined['onchain_active_addresses'] / combined['onchain_active_addresses'].mean() * 0.4 +
        combined['onchain_transaction_count'] / combined['onchain_transaction_count'].mean() * 0.3 +
        combined['onchain_hash_rate'] / combined['onchain_hash_rate'].mean() * 0.3
    ) / 3
    
    combined['market_risk'] = (
        combined['macro_vix_index'] / 40 * 0.6 +
        (1 - combined['macro_risk_appetite']) * 0.4
    )
    
    # Технические индикаторы с весами
    for col in ['sma_20', 'ema_12', 'rsi', 'volatility']:
        if col in combined.columns:
            combined[col] = combined[col] * OptimizedSentimentConfig.TECHNICAL_WEIGHT
    
    # Удаляем timestamp
    if 'timestamp' in combined.columns:
        combined = combined.drop(['timestamp'], axis=1)
    
    # Улучшенная нормализация
    combined = combined.fillna(method='ffill').fillna(method='bfill')
    combined = combined.replace([np.inf, -np.inf], 0)
    
    print("📊 Применение оптимизированной нормализации...")
    scaler = StandardScaler()
    
    # Нормализуем цены через log returns
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in combined.columns:
            combined[f'{col}_returns'] = np.log(combined[col] / combined[col].shift(1)).fillna(0)
            combined[f'{col}_returns'] = np.clip(combined[f'{col}_returns'], -0.08, 0.08)
            combined[col] = combined[f'{col}_returns']
            combined = combined.drop([f'{col}_returns'], axis=1)
    
    # Нормализуем остальные признаки
    other_cols = [col for col in combined.columns if col not in price_cols]
    if other_cols:
        combined[other_cols] = scaler.fit_transform(combined[other_cols])
        for col in other_cols:
            combined[col] = np.clip(combined[col], -4, 4)
    
    # Финальная очистка
    combined = combined.fillna(0)
    combined = combined.replace([np.inf, -np.inf], 0)
    
    print(f"✅ Объединено и оптимизировано {len(combined.columns)} признаков")
    return combined


def generate_optimized_crypto_data(n_points: int = 6000) -> pd.DataFrame:
    """Генерация оптимизированных ценовых данных"""
    print("📊 Генерация оптимизированных ценовых данных...")
    
    np.random.seed(40)
    timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
    
    # Более реалистичный тренд
    trend = np.linspace(45000, 58000, n_points)
    seasonal = 2000 * np.sin(2 * np.pi * np.arange(n_points) / 168)
    volatility_events = np.random.normal(0, 800, n_points)
    
    # Меньше крупных событий
    major_events = np.random.choice([0, 1], size=n_points, p=[0.985, 0.015])
    event_impact = np.random.normal(0, 3000, n_points) * major_events
    
    close_price = trend + seasonal + volatility_events + event_impact
    close_price = np.maximum(close_price, 25000)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': close_price + np.random.normal(0, 80, n_points),
        'high': close_price * (1 + np.abs(np.random.normal(0, 0.006, n_points))),
        'low': close_price * (1 - np.abs(np.random.normal(0, 0.006, n_points))),
        'close': close_price,
        'volume': np.random.exponential(1200000, n_points)
    })
    
    # Коррекция OHLC
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    # Улучшенные технические индикаторы
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['volatility'] = df['returns'].rolling(24).std()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Дополнительные индикаторы
    df['macd'] = df['ema_12'] - df['close'].ewm(span=26).mean()
    df['bb_upper'] = df['sma_20'] + (df['close'].rolling(20).std() * 2)
    df['bb_lower'] = df['sma_20'] - (df['close'].rolling(20).std() * 2)
    
    print(f"✅ Сгенерировано {len(df)} записей оптимизированных ценовых данных")
    return df


class OptimizedSentimentTradingEnv(gym.Env):
    """Оптимизированное торговое окружение"""
    
    def __init__(self, combined_df: pd.DataFrame):
        super().__init__()
        
        self.df = combined_df.reset_index(drop=True)
        self.window_size = OptimizedSentimentConfig.WINDOW_SIZE
        
        self.action_space = spaces.Discrete(3)
        
        n_features = len(combined_df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, n_features),
            dtype=np.float32
        )
        
        self._reset_state()
    
    def _reset_state(self):
        """Сброс состояния"""
        self.current_step = self.window_size
        self.balance = OptimizedSentimentConfig.INITIAL_BALANCE
        self.btc_amount = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        
        self.total_trades = 0
        self.profitable_trades = 0
        self.portfolio_history = [float(OptimizedSentimentConfig.INITIAL_BALANCE)]
        self.trades_history = []
        
        # Для анализа
        self.sentiment_signals = []
        self.max_drawdown = 0.0
        self.peak_value = OptimizedSentimentConfig.INITIAL_BALANCE
    
    def reset(self, seed=None, options=None):
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Получение наблюдения"""
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        obs = self.df.iloc[start_idx:end_idx].values
        
        if len(obs) < self.window_size:
            padding = np.tile(obs[0], (self.window_size - len(obs), 1))
            obs = np.vstack([padding, obs])
        
        return obs.astype(np.float32)
    
    def _get_current_price(self) -> float:
        """Получение текущей цены"""
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.iloc[self.current_step]['close']
    
    def _get_sentiment_signal(self) -> float:
        """Получение оптимизированного сигнала настроений"""
        if self.current_step >= len(self.df):
            return 0.0
        
        current_data = self.df.iloc[self.current_step]
        
        if 'overall_sentiment' in current_data.index:
            sentiment = current_data['overall_sentiment']
        else:
            sentiment = 0.0
        
        # Фильтрация слабых сигналов
        if abs(sentiment) < OptimizedSentimentConfig.SENTIMENT_THRESHOLD:
            return 0.0
        
        # Умеренное усиление
        sentiment *= OptimizedSentimentConfig.SENTIMENT_MULTIPLIER
        
        return np.clip(sentiment, -1, 1)
    
    def _calculate_position_size(self, sentiment_signal: float) -> float:
        """Расчет оптимального размера позиции"""
        base_risk = OptimizedSentimentConfig.RISK_PER_TRADE
        
        # Адаптация к силе сигнала
        sentiment_multiplier = 1.0 + abs(sentiment_signal) * 0.3
        adjusted_risk = base_risk * sentiment_multiplier
        
        # Ограничения
        adjusted_risk = min(adjusted_risk, OptimizedSentimentConfig.MAX_POSITION_SIZE)
        
        return adjusted_risk
    
    def _check_stop_loss_take_profit(self, current_price: float) -> str:
        """Проверка стоп-лосса и тейк-профита"""
        if self.btc_amount <= 0 or self.entry_price <= 0:
            return 'none'
        
        price_change = (current_price - self.entry_price) / self.entry_price
        
        if price_change <= -OptimizedSentimentConfig.STOP_LOSS:
            return 'stop_loss'
        elif price_change >= OptimizedSentimentConfig.TAKE_PROFIT:
            return 'take_profit'
        
        return 'none'
    
    def _execute_optimized_trade(self, action: int, current_price: float) -> Dict[str, Any]:
        """Выполнение оптимизированной торговой операции"""
        sentiment_signal = self._get_sentiment_signal()
        trade_result = {'executed': False, 'type': None, 'sentiment_signal': sentiment_signal}
        
        # Проверка стоп-лосса/тейк-профита
        sl_tp_action = self._check_stop_loss_take_profit(current_price)
        if sl_tp_action != 'none' and self.btc_amount > 0:
            # Принудительная продажа
            revenue = self.btc_amount * current_price
            commission = revenue * 0.001
            profit = revenue - self.btc_amount * self.entry_price
            
            if profit > 0:
                self.profitable_trades += 1
            
            self.balance += revenue - commission
            self.btc_amount = 0.0
            self.entry_price = 0.0
            
            trade_result.update({
                'executed': True, 'type': f'SELL_{sl_tp_action.upper()}',
                'profit': profit, 'reason': sl_tp_action
            })
            self.total_trades += 1
            self.trades_history.append(trade_result)
            
            return trade_result
        
        # Обычные торговые действия
        if action == 1 and self.balance > 100:  # Buy
            if sentiment_signal > 0 or len(self.sentiment_signals) < OptimizedSentimentConfig.MIN_TRADES_THRESHOLD:
                position_size = self._calculate_position_size(sentiment_signal)
                investment = self.balance * position_size
                amount = investment / current_price
                commission = investment * 0.001
                
                self.btc_amount += amount
                self.balance -= investment + commission
                self.entry_price = current_price
                self.entry_step = self.current_step
                
                trade_result.update({
                    'executed': True, 'type': 'BUY',
                    'amount': amount, 'price': current_price,
                    'investment': investment
                })
                
        elif action == 2 and self.btc_amount > 0:  # Sell
            # Проверка минимальной прибыли
            potential_profit = (current_price - self.entry_price) / self.entry_price
            
            if (potential_profit >= OptimizedSentimentConfig.PROFIT_THRESHOLD or 
                sentiment_signal < -0.1 or 
                self.current_step - self.entry_step > 48):  # Принудительная продажа через 48 часов
                
                revenue = self.btc_amount * current_price
                commission = revenue * 0.001
                profit = revenue - self.btc_amount * self.entry_price
                
                if profit > 0:
                    self.profitable_trades += 1
                
                self.balance += revenue - commission
                self.btc_amount = 0.0
                self.entry_price = 0.0
                
                trade_result.update({
                    'executed': True, 'type': 'SELL',
                    'profit': profit, 'revenue': revenue
                })
        
        if trade_result['executed']:
            self.total_trades += 1
            self.trades_history.append(trade_result)
            self.sentiment_signals.append(sentiment_signal)
        
        return trade_result
    
    def _calculate_portfolio_value(self) -> float:
        """Расчет стоимости портфеля"""
        current_price = self._get_current_price()
        return self.balance + self.btc_amount * current_price
    
    def _calculate_optimized_reward(self) -> float:
        """Расчет оптимизированной награды"""
        current_portfolio = self._calculate_portfolio_value()
        prev_portfolio = self.portfolio_history[-1]
        
        # Базовая награда
        portfolio_change = (current_portfolio - prev_portfolio) / prev_portfolio
        base_reward = portfolio_change * 100
        
        # Обновление максимальной просадки
        if current_portfolio > self.peak_value:
            self.peak_value = current_portfolio
        current_drawdown = (self.peak_value - current_portfolio) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Штраф за большую просадку
        if current_drawdown > 0.1:  # 10% просадка
            base_reward -= current_drawdown * 50
        
        # Бонус за использование сигналов
        sentiment_signal = self._get_sentiment_signal()
        if abs(sentiment_signal) > OptimizedSentimentConfig.SENTIMENT_THRESHOLD:
            if (sentiment_signal > 0 and portfolio_change > 0) or (sentiment_signal < 0 and portfolio_change < 0):
                base_reward += abs(sentiment_signal) * 10
        
        return base_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Шаг симуляции"""
        current_price = self._get_current_price()
        
        # Выполнение действия
        trade_result = self._execute_optimized_trade(action, current_price)
        
        # Обновление состояния
        self.current_step += 1
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_history.append(portfolio_value)
        
        # Расчет награды
        reward = self._calculate_optimized_reward()
        
        # Проверка завершения
        done = (
            self.current_step >= len(self.df) - 1 or
            portfolio_value <= OptimizedSentimentConfig.INITIAL_BALANCE * 0.2 or
            self.max_drawdown > 0.5
        )
        
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'btc_amount': self.btc_amount,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'max_drawdown': self.max_drawdown,
            'sentiment_signal': trade_result.get('sentiment_signal', 0),
            'trade_result': trade_result
        }
        
        return self._get_observation(), reward, done, False, info


def main():
    """Главная функция оптимизированной системы V3.3"""
    print("🚀 ЗАПУСК SENTIMENT ТОРГОВОЙ СИСТЕМЫ V3.3 - OPTIMIZED EDITION")
    print("=" * 70)
    
    # 1. Генерация оптимизированных данных
    print("\n📊 ЭТАП 1: ГЕНЕРАЦИЯ ОПТИМИЗИРОВАННЫХ ДАННЫХ")
    print("-" * 50)
    
    n_points = 6000
    price_data = generate_optimized_crypto_data(n_points)
    
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_data = sentiment_analyzer.generate_social_sentiment_data(n_points)
    
    onchain_analyzer = OnChainAnalyzer()
    onchain_data = onchain_analyzer.generate_onchain_data(n_points)
    
    macro_analyzer = MacroEconomicAnalyzer()
    macro_data = macro_analyzer.generate_macro_data(n_points)
    
    # 2. Объединение данных
    print("\n🔄 ЭТАП 2: ОБЪЕДИНЕНИЕ ОПТИМИЗИРОВАННЫХ ДАННЫХ")
    print("-" * 50)
    combined_df = combine_optimized_data_sources(price_data, sentiment_data, onchain_data, macro_data)
    
    # 3. Создание окружения
    print("\n🎮 ЭТАП 3: СОЗДАНИЕ ОПТИМИЗИРОВАННОГО ОКРУЖЕНИЯ")
    print("-" * 50)
    env = OptimizedSentimentTradingEnv(combined_df)
    vec_env = DummyVecEnv([lambda: env])
    print(f"✅ Оптимизированное окружение создано с {len(combined_df.columns)} признаками")
    
    # 4. Создание оптимизированной модели
    print("\n🧠 ЭТАП 4: СОЗДАНИЕ ОПТИМИЗИРОВАННОЙ МОДЕЛИ")
    print("-" * 50)
    
    policy_kwargs = dict(
        features_extractor_class=OptimizedFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 128, 64],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=OptimizedSentimentConfig.LEARNING_RATE,
        n_steps=1024,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    print("✅ Оптимизированная модель создана")
    
    # 5. Обучение
    print("\n🎓 ЭТАП 5: ОБУЧЕНИЕ ОПТИМИЗИРОВАННОЙ МОДЕЛИ")
    print("-" * 50)
    model.learn(total_timesteps=OptimizedSentimentConfig.TOTAL_TIMESTEPS)
    print("✅ Обучение завершено")
    
    # 6. Тестирование
    print("\n🧪 ЭТАП 6: ТЕСТИРОВАНИЕ ОПТИМИЗИРОВАННОЙ СИСТЕМЫ")
    print("-" * 50)
    
    obs, _ = env.reset()
    results = []
    sentiment_history = []
    
    for step in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        
        results.append({
            'step': step,
            'portfolio_value': info['portfolio_value'],
            'balance': info['balance'],
            'total_trades': info['total_trades'],
            'profitable_trades': info['profitable_trades'],
            'max_drawdown': info['max_drawdown'],
            'sentiment_signal': info['sentiment_signal'],
            'reward': reward
        })
        
        sentiment_history.append(info['sentiment_signal'])
        
        if done:
            break
    
    # 7. Анализ результатов
    print("\n📊 ЭТАП 7: АНАЛИЗ ОПТИМИЗИРОВАННЫХ РЕЗУЛЬТАТОВ")
    print("-" * 50)
    
    final_value = results[-1]['portfolio_value']
    total_return = (final_value - OptimizedSentimentConfig.INITIAL_BALANCE) / OptimizedSentimentConfig.INITIAL_BALANCE * 100
    total_trades = results[-1]['total_trades']
    profitable_trades = results[-1]['profitable_trades']
    win_rate = (profitable_trades / max(total_trades, 1)) * 100
    max_drawdown = results[-1]['max_drawdown'] * 100
    avg_sentiment = np.mean([abs(s) for s in sentiment_history if s != 0])
    
    print("📊 АНАЛИЗ SENTIMENT ТОРГОВОЙ СИСТЕМЫ V3.3 - OPTIMIZED")
    print("=" * 65)
    print(f"💰 Начальный баланс: {OptimizedSentimentConfig.INITIAL_BALANCE:,.2f} USDT")
    print(f"💰 Финальная стоимость: {final_value:,.2f} USDT")
    print(f"📈 Общая доходность: {total_return:+.2f}%")
    print(f"🔄 Всего сделок: {total_trades}")
    print(f"✅ Прибыльных сделок: {profitable_trades} ({win_rate:.1f}%)")
    print(f"📉 Максимальная просадка: {max_drawdown:.2f}%")
    print(f"📱 Средняя сила сигнала: {avg_sentiment:.3f}")
    print(f"🛡️ Активные риск-лимиты: Стоп-лосс {OptimizedSentimentConfig.STOP_LOSS*100:.0f}% | Тейк-профит {OptimizedSentimentConfig.TAKE_PROFIT*100:.0f}%")
    
    # 8. Заключение
    print("\n🎯 ЗАКЛЮЧЕНИЕ V3.3 - OPTIMIZED EDITION")
    print("=" * 60)
    print("🚀 Sentiment торговая система V3.3 Optimized протестирована!")
    print(f"💡 Достигнута доходность: {total_return:+.2f}%")
    print(f"📊 Соотношение прибыльных сделок: {win_rate:.1f}%")
    print(f"🛡️ Максимальная просадка: {max_drawdown:.2f}%")
    
    print("\n✨ Оптимизации V3.3:")
    print("  • Улучшенный риск-менеджмент со стоп-лоссами")
    print("  • Адаптивный размер позиций")
    print("  • Оптимизированные веса источников данных")
    print("  • Фильтрация слабых сигналов настроений")
    print("  • Контроль максимальной просадки")
    print("  • Принудительное закрытие по времени")
    
    # Оценка производительности
    if total_return > 5 and win_rate > 55 and max_drawdown < 20:
        print("\n🟢 ОЦЕНКА: Отличная оптимизированная стратегия!")
    elif total_return > 0 and win_rate > 50:
        print("\n🟡 ОЦЕНКА: Хорошая базовая стратегия с потенциалом")
    else:
        print("\n🔶 ОЦЕНКА: Требует дополнительной настройки")
    
    print("\n🎉 АНАЛИЗ V3.3 OPTIMIZED ЗАВЕРШЕН!")


if __name__ == "__main__":
    main() 