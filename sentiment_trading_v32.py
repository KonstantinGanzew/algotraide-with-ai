"""
🚀 SENTIMENT ТОРГОВАЯ СИСТЕМА V3.2
Интеграция анализа настроений из социальных сетей и макроэкономических данных
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


class SentimentConfig:
    """Конфигурация для системы с анализом настроений"""
    
    # Источники данных
    SOCIAL_MEDIA_WEIGHT = 0.3      # Вес социальных данных
    ON_CHAIN_WEIGHT = 0.2          # Вес on-chain метрик
    MACRO_WEIGHT = 0.1             # Вес макроэкономических данных
    TECHNICAL_WEIGHT = 0.4         # Вес технических индикаторов
    
    # Анализ настроений
    SENTIMENT_WINDOW = 24          # Окно для агрегации настроений
    SENTIMENT_THRESHOLD = 0.1      # Порог значимости настроения
    NEWS_IMPACT_DECAY = 0.9        # Затухание влияния новостей
    
    # Торговые параметры
    WINDOW_SIZE = 48
    INITIAL_BALANCE = 10000
    RISK_PER_TRADE = 0.025
    SENTIMENT_MULTIPLIER = 1.5     # Усиление сигналов при сильных настроениях
    
    # Обучение
    TOTAL_TIMESTEPS = 10000
    LEARNING_RATE = 3e-4


class SentimentAnalyzer:
    """Анализатор настроений для социальных сетей и новостей"""
    
    def __init__(self):
        # Словари для анализа настроений
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
        """Анализ настроения текста (упрощенный алгоритм)"""
        if not text:
            return 0.0
        
        text = text.lower()
        
        # Проверяем наличие крипто-тематики
        crypto_relevance = sum(1 for word in self.crypto_keywords if word in text)
        if crypto_relevance == 0:
            return 0.0  # Не относится к криптовалютам
        
        # Подсчитываем позитивные и негативные слова
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)
        
        # Нормализуем по длине текста
        text_length = len(text.split())
        if text_length == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / max(text_length, 1)
        
        # Усиливаем сигнал для высокой крипто-релевантности
        sentiment_score *= min(crypto_relevance / 3, 1.0)
        
        return np.clip(sentiment_score, -1.0, 1.0)
    
    def generate_social_sentiment_data(self, n_points: int) -> pd.DataFrame:
        """Генерация симулированных данных социальных настроений"""
        print("📱 Генерация данных социальных настроений...")
        
        np.random.seed(42)
        timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
        
        # Базовое настроение с трендами
        base_sentiment = np.random.normal(0, 0.3, n_points)
        
        # Добавляем события (новости, объявления)
        events = np.random.choice([0, 1], size=n_points, p=[0.95, 0.05])
        event_impact = np.random.normal(0, 0.8, n_points) * events
        
        # Затухание влияния событий
        for i in range(1, len(event_impact)):
            if events[i-1] == 1:
                event_impact[i] += event_impact[i-1] * SentimentConfig.NEWS_IMPACT_DECAY
        
        # Общее настроение
        total_sentiment = base_sentiment + event_impact
        total_sentiment = np.clip(total_sentiment, -1, 1)
        
        # Создаем различные метрики
        df = pd.DataFrame({
            'timestamp': timestamps,
            'twitter_sentiment': total_sentiment + np.random.normal(0, 0.1, n_points),
            'reddit_sentiment': total_sentiment + np.random.normal(0, 0.15, n_points),
            'news_sentiment': total_sentiment + np.random.normal(0, 0.2, n_points),
            'social_volume': np.abs(total_sentiment) * 1000 + np.random.exponential(500, n_points),
            'mentions_count': np.abs(total_sentiment) * 100 + np.random.poisson(50, n_points),
            'influencer_sentiment': total_sentiment * 1.2 + np.random.normal(0, 0.2, n_points)
        })
        
        # Нормализация
        for col in ['twitter_sentiment', 'reddit_sentiment', 'news_sentiment', 'influencer_sentiment']:
            df[col] = np.clip(df[col], -1, 1)
        
        print(f"✅ Сгенерировано {len(df)} записей социальных данных")
        return df


class OnChainAnalyzer:
    """Анализатор on-chain метрик блокчейна"""
    
    def generate_onchain_data(self, n_points: int) -> pd.DataFrame:
        """Генерация симулированных on-chain данных"""
        print("⛓️ Генерация on-chain данных...")
        
        np.random.seed(43)
        timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
        
        # Базовые тренды
        price_trend = np.linspace(45000, 65000, n_points) + np.random.normal(0, 2000, n_points)
        
        # On-chain метрики коррелированные с ценой
        df = pd.DataFrame({
            'timestamp': timestamps,
            'active_addresses': 800000 + price_trend / 100 + np.random.normal(0, 50000, n_points),
            'transaction_count': 250000 + price_trend / 200 + np.random.normal(0, 30000, n_points),
            'hash_rate': 200e18 + price_trend * 1e15 + np.random.normal(0, 10e18, n_points),
            'difficulty': 25e12 + price_trend * 1e9 + np.random.normal(0, 1e12, n_points),
            'exchange_inflow': np.random.exponential(1000, n_points),
            'exchange_outflow': np.random.exponential(1200, n_points),
            'whale_transactions': np.random.poisson(50, n_points),
            'new_addresses': 350000 + price_trend / 300 + np.random.normal(0, 20000, n_points)
        })
        
        # Производные метрики
        df['net_exchange_flow'] = df['exchange_inflow'] - df['exchange_outflow']
        df['address_growth_rate'] = df['new_addresses'].pct_change().rolling(24).mean()
        df['network_value'] = df['active_addresses'] * price_trend / 1000000
        
        print(f"✅ Сгенерировано {len(df)} записей on-chain данных")
        return df


class MacroEconomicAnalyzer:
    """Анализатор макроэкономических данных"""
    
    def generate_macro_data(self, n_points: int) -> pd.DataFrame:
        """Генерация симулированных макроэкономических данных"""
        print("🌍 Генерация макроэкономических данных...")
        
        np.random.seed(44)
        timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
        
        # Макроэкономические индикаторы
        df = pd.DataFrame({
            'timestamp': timestamps,
            'dxy_index': 103 + np.cumsum(np.random.normal(0, 0.1, n_points)),  # Dollar index
            'vix_index': 20 + np.abs(np.cumsum(np.random.normal(0, 0.5, n_points))),  # Volatility
            'gold_price': 1900 + np.cumsum(np.random.normal(0, 5, n_points)),
            'sp500_index': 4000 + np.cumsum(np.random.normal(0, 10, n_points)),
            'fed_rate': 5.25 + np.cumsum(np.random.normal(0, 0.01, n_points)) / 100,
            'inflation_rate': 3.2 + np.cumsum(np.random.normal(0, 0.05, n_points)) / 100,
            'unemployment_rate': 3.7 + np.cumsum(np.random.normal(0, 0.02, n_points)) / 100
        })
        
        # Ограничиваем значения в разумных пределах
        df['vix_index'] = np.clip(df['vix_index'], 10, 80)
        df['fed_rate'] = np.clip(df['fed_rate'], 0, 10)
        df['inflation_rate'] = np.clip(df['inflation_rate'], -2, 15)
        df['unemployment_rate'] = np.clip(df['unemployment_rate'], 2, 15)
        
        # Производные показатели
        df['risk_appetite'] = (100 - df['vix_index']) / 100  # Аппетит к риску
        df['dollar_strength'] = (df['dxy_index'] - 100) / 10
        df['macro_sentiment'] = (df['risk_appetite'] - df['dollar_strength']) / 2
        
        print(f"✅ Сгенерировано {len(df)} записей макроэкономических данных")
        return df


class SentimentFeatureExtractor(BaseFeaturesExtractor):
    """Feature Extractor с учетом настроений и альтернативных данных"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        if observation_space.shape is not None:
            self.seq_len = observation_space.shape[0]
            self.input_features = observation_space.shape[1]
        else:
            self.seq_len = SentimentConfig.WINDOW_SIZE
            self.input_features = 50
        
        # Упрощенная архитектура для стабильности
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Простой LSTM для временных зависимостей
        self.lstm = nn.LSTM(
            input_size=self.input_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Объединяющая сеть
        self.fusion_net = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim, features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = observations.shape
        
        # Берем последнее наблюдение для статических признаков
        last_obs = observations[:, -1, :]
        
        # Обрабатываем признаки
        feature_out = self.feature_net(last_obs)
        
        # LSTM для временных зависимостей
        lstm_out, _ = self.lstm(observations)
        lstm_features = lstm_out[:, -1, :]  # Берем последний output
        
        # Объединяем все признаки
        combined_features = torch.cat([lstm_features, feature_out], dim=1)
        output = self.fusion_net(combined_features)
        
        return output


def combine_all_data_sources(price_data: pd.DataFrame, 
                           sentiment_data: pd.DataFrame,
                           onchain_data: pd.DataFrame, 
                           macro_data: pd.DataFrame) -> pd.DataFrame:
    """Объединение всех источников данных"""
    print("🔄 Объединение всех источников данных...")
    
    # Объединяем по времени
    combined = price_data.copy()
    
    # Добавляем данные настроений
    sentiment_features = sentiment_data.drop(['timestamp'], axis=1)
    for col in sentiment_features.columns:
        combined[f'sentiment_{col}'] = sentiment_features[col].values
    
    # Добавляем on-chain данные
    onchain_features = onchain_data.drop(['timestamp'], axis=1)
    for col in onchain_features.columns:
        combined[f'onchain_{col}'] = onchain_features[col].values
    
    # Добавляем макроэкономические данные
    macro_features = macro_data.drop(['timestamp'], axis=1)
    for col in macro_features.columns:
        combined[f'macro_{col}'] = macro_features[col].values
    
    # Агрегированные показатели
    combined['overall_sentiment'] = (
        combined['sentiment_twitter_sentiment'] * 0.3 +
        combined['sentiment_reddit_sentiment'] * 0.3 +
        combined['sentiment_news_sentiment'] * 0.2 +
        combined['sentiment_influencer_sentiment'] * 0.2
    )
    
    combined['network_health'] = (
        combined['onchain_active_addresses'] / combined['onchain_active_addresses'].mean() +
        combined['onchain_transaction_count'] / combined['onchain_transaction_count'].mean()
    ) / 2
    
    combined['macro_risk'] = (
        combined['macro_vix_index'] / 50 +  # Нормализуем VIX
        (1 - combined['macro_risk_appetite'])
    ) / 2
    
    # Удаляем timestamp
    if 'timestamp' in combined.columns:
        combined = combined.drop(['timestamp'], axis=1)
    
    # Обработка NaN и бесконечных значений
    combined = combined.fillna(method='ffill').fillna(method='bfill')
    combined = combined.replace([np.inf, -np.inf], 0)
    
    # Нормализация данных для стабильности обучения
    print("📊 Нормализация данных...")
    from sklearn.preprocessing import StandardScaler
    
    # Отделяем ценовые колонки, которые не нужно нормализовать одинаково
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    other_cols = [col for col in combined.columns if col not in price_cols]
    
    # Нормализуем price данные отдельно (log-returns для цен)
    for col in ['open', 'high', 'low', 'close']:
        if col in combined.columns:
            combined[f'{col}_normalized'] = np.log(combined[col] / combined[col].shift(1)).fillna(0)
            combined[f'{col}_normalized'] = np.clip(combined[f'{col}_normalized'], -0.1, 0.1)
    
    # Нормализуем volume
    if 'volume' in combined.columns:
        combined['volume_normalized'] = (combined['volume'] - combined['volume'].mean()) / combined['volume'].std()
        combined['volume_normalized'] = np.clip(combined['volume_normalized'], -3, 3)
    
    # Стандартизируем остальные признаки
    if other_cols:
        scaler = StandardScaler()
        combined[other_cols] = scaler.fit_transform(combined[other_cols])
        # Ограничиваем значения для стабильности
        for col in other_cols:
            combined[col] = np.clip(combined[col], -5, 5)
    
    # Заменяем исходные ценовые колонки нормализованными
    cols_to_drop = []
    for col in price_cols:
        if col in combined.columns and f'{col}_normalized' in combined.columns:
            combined[col] = combined[f'{col}_normalized']
            cols_to_drop.append(f'{col}_normalized')
    
    combined = combined.drop(cols_to_drop, axis=1)
    
    # Финальная проверка на NaN и Inf
    combined = combined.fillna(0)
    combined = combined.replace([np.inf, -np.inf], 0)
    
    print(f"✅ Объединено и нормализовано {len(combined.columns)} признаков")
    return combined


def generate_comprehensive_crypto_data(n_points: int = 8000) -> pd.DataFrame:
    """Генерация комплексных данных с ценой"""
    print("📊 Генерация базовых ценовых данных...")
    
    np.random.seed(40)
    timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
    
    # Генерация реалистичной цены Bitcoin
    trend = np.linspace(45000, 62000, n_points)
    seasonal = 3000 * np.sin(2 * np.pi * np.arange(n_points) / 168)  # Недельная сезонность
    volatility_events = np.random.normal(0, 1000, n_points)
    
    # Добавляем крупные движения (новости, события)
    major_events = np.random.choice([0, 1], size=n_points, p=[0.98, 0.02])
    event_impact = np.random.normal(0, 5000, n_points) * major_events
    
    close_price = trend + seasonal + volatility_events + event_impact
    close_price = np.maximum(close_price, 20000)  # Минимальная цена
    
    # OHLCV данные
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': close_price + np.random.normal(0, 100, n_points),
        'high': close_price * (1 + np.abs(np.random.normal(0, 0.008, n_points))),
        'low': close_price * (1 - np.abs(np.random.normal(0, 0.008, n_points))),
        'close': close_price,
        'volume': np.random.exponential(1500000, n_points)
    })
    
    # Коррекция high/low
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    # Технические индикаторы
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
    
    print(f"✅ Сгенерировано {len(df)} записей ценовых данных")
    return df


class SentimentTradingEnv(gym.Env):
    """Торговое окружение с анализом настроений и альтернативными данными"""
    
    def __init__(self, combined_df: pd.DataFrame):
        super().__init__()
        
        self.df = combined_df.reset_index(drop=True)
        self.window_size = SentimentConfig.WINDOW_SIZE
        
        # Пространства
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        
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
        self.balance = SentimentConfig.INITIAL_BALANCE
        self.btc_amount = 0.0
        self.entry_price = 0.0
        
        self.total_trades = 0
        self.profitable_trades = 0
        self.portfolio_history = [float(SentimentConfig.INITIAL_BALANCE)]
        self.trades_history = []
        
        # Для анализа сигналов
        self.sentiment_signals = []
        self.macro_signals = []
    
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
        """Получение сигнала настроений"""
        if self.current_step >= len(self.df):
            return 0.0
        
        current_data = self.df.iloc[self.current_step]
        
        # Агрегированный сигнал настроений
        if 'overall_sentiment' in current_data.index:
            sentiment = current_data['overall_sentiment']
        else:
            sentiment = 0.0
        
        # Усиление сильных сигналов
        if abs(sentiment) > SentimentConfig.SENTIMENT_THRESHOLD:
            sentiment *= SentimentConfig.SENTIMENT_MULTIPLIER
        
        return np.clip(sentiment, -1, 1)
    
    def _execute_sentiment_aware_trade(self, action: int, current_price: float) -> Dict[str, Any]:
        """Выполнение торговой операции с учетом настроений"""
        sentiment_signal = self._get_sentiment_signal()
        
        # Корректируем размер позиции на основе силы сигнала настроений
        sentiment_multiplier = 1.0 + abs(sentiment_signal) * 0.5
        adjusted_risk = SentimentConfig.RISK_PER_TRADE * sentiment_multiplier
        adjusted_risk = min(adjusted_risk, 0.05)  # Максимум 5%
        
        trade_result = {'executed': False, 'type': None, 'sentiment_signal': sentiment_signal}
        
        if action == 1 and self.balance > 100:  # Buy
            # Покупаем больше при позитивном настроении
            if sentiment_signal > 0:
                investment = self.balance * adjusted_risk
            else:
                investment = self.balance * SentimentConfig.RISK_PER_TRADE * 0.5  # Меньше при негативе
            
            amount = investment / current_price
            commission = investment * 0.001
            
            self.btc_amount += amount
            self.balance -= investment + commission
            self.entry_price = current_price
            
            trade_result.update({
                'executed': True, 'type': 'BUY',
                'amount': amount, 'price': current_price,
                'investment': investment,
                'sentiment_adjusted': True
            })
            
        elif action == 2 and self.btc_amount > 0:  # Sell
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
                'amount': self.btc_amount, 'price': current_price,
                'revenue': revenue, 'profit': profit
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
    
    def _calculate_reward(self) -> float:
        """Расчет награды с учетом настроений"""
        current_portfolio = self._calculate_portfolio_value()
        prev_portfolio = self.portfolio_history[-1]
        
        # Базовая награда
        portfolio_change = (current_portfolio - prev_portfolio) / prev_portfolio
        base_reward = portfolio_change * 100
        
        # Бонус за правильное использование сигналов настроений
        sentiment_signal = self._get_sentiment_signal()
        if len(self.sentiment_signals) > 0:
            last_sentiment = self.sentiment_signals[-1]
            if abs(last_sentiment) > SentimentConfig.SENTIMENT_THRESHOLD:
                # Бонус за следование сильным сигналам настроений
                sentiment_bonus = abs(last_sentiment) * 0.5
                if (last_sentiment > 0 and portfolio_change > 0) or (last_sentiment < 0 and portfolio_change > 0):
                    base_reward += sentiment_bonus
        
        return base_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Шаг симуляции с учетом настроений"""
        current_price = self._get_current_price()
        
        # Выполнение действия
        trade_result = self._execute_sentiment_aware_trade(action, current_price)
        
        # Обновление состояния
        self.current_step += 1
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_history.append(portfolio_value)
        
        # Расчет награды
        reward = self._calculate_reward()
        
        # Проверка завершения
        done = (
            self.current_step >= len(self.df) - 1 or
            portfolio_value <= SentimentConfig.INITIAL_BALANCE * 0.1
        )
        
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'btc_amount': self.btc_amount,
            'total_trades': self.total_trades,
            'sentiment_signal': trade_result.get('sentiment_signal', 0),
            'trade_result': trade_result
        }
        
        return self._get_observation(), reward, done, False, info


def main():
    """Главная функция системы с анализом настроений V3.2"""
    print("🚀 ЗАПУСК SENTIMENT ТОРГОВОЙ СИСТЕМЫ V3.2")
    print("=" * 60)
    
    # 1. Генерация данных
    print("\n📊 ЭТАП 1: ГЕНЕРАЦИЯ КОМПЛЕКСНЫХ ДАННЫХ")
    print("-" * 40)
    
    n_points = 6000
    price_data = generate_comprehensive_crypto_data(n_points)
    
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_data = sentiment_analyzer.generate_social_sentiment_data(n_points)
    
    onchain_analyzer = OnChainAnalyzer()
    onchain_data = onchain_analyzer.generate_onchain_data(n_points)
    
    macro_analyzer = MacroEconomicAnalyzer()
    macro_data = macro_analyzer.generate_macro_data(n_points)
    
    # 2. Объединение данных
    print("\n🔄 ЭТАП 2: ОБЪЕДИНЕНИЕ ИСТОЧНИКОВ ДАННЫХ")
    print("-" * 40)
    combined_df = combine_all_data_sources(price_data, sentiment_data, onchain_data, macro_data)
    
    # 3. Создание окружения
    print("\n🎮 ЭТАП 3: СОЗДАНИЕ SENTIMENT ОКРУЖЕНИЯ")
    print("-" * 40)
    env = SentimentTradingEnv(combined_df)
    vec_env = DummyVecEnv([lambda: env])
    print(f"✅ Окружение создано с {len(combined_df.columns)} признаками")
    
    # 4. Создание модели
    print("\n🧠 ЭТАП 4: СОЗДАНИЕ SENTIMENT МОДЕЛИ")
    print("-" * 40)
    
    policy_kwargs = dict(
        features_extractor_class=SentimentFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 128],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=SentimentConfig.LEARNING_RATE,
        n_steps=1024,
        batch_size=32,
        n_epochs=5,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    print("✅ Sentiment модель создана")
    
    # 5. Обучение
    print("\n🎓 ЭТАП 5: ОБУЧЕНИЕ МОДЕЛИ")
    print("-" * 40)
    model.learn(total_timesteps=SentimentConfig.TOTAL_TIMESTEPS)
    print("✅ Обучение завершено")
    
    # 6. Тестирование
    print("\n🧪 ЭТАП 6: ТЕСТИРОВАНИЕ")
    print("-" * 40)
    
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
            'sentiment_signal': info['sentiment_signal'],
            'reward': reward
        })
        
        sentiment_history.append(info['sentiment_signal'])
        
        if done:
            break
    
    # 7. Анализ результатов
    print("\n📊 ЭТАП 7: АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("-" * 40)
    
    final_value = results[-1]['portfolio_value']
    total_return = (final_value - SentimentConfig.INITIAL_BALANCE) / SentimentConfig.INITIAL_BALANCE * 100
    total_trades = results[-1]['total_trades']
    avg_sentiment = np.mean([abs(s) for s in sentiment_history])
    
    print("📊 АНАЛИЗ SENTIMENT ТОРГОВОЙ СИСТЕМЫ V3.2")
    print("=" * 55)
    print(f"💰 Начальный баланс: {SentimentConfig.INITIAL_BALANCE:,.2f} USDT")
    print(f"💰 Финальная стоимость: {final_value:,.2f} USDT")
    print(f"📈 Общая доходность: {total_return:+.2f}%")
    print(f"🔄 Всего сделок: {total_trades}")
    print(f"📱 Средняя сила сигнала настроений: {avg_sentiment:.3f}")
    print(f"🔗 Источники данных: Социальные сети + On-chain + Макроэкономика")
    
    # 8. Заключение
    print("\n🎯 ЗАКЛЮЧЕНИЕ V3.2")
    print("=" * 50)
    print("🚀 Sentiment торговая система V3.2 успешно протестирована!")
    print(f"💡 Достигнута доходность: {total_return:+.2f}%")
    print(f"📊 Обработано источников данных: 4 (цена + настроения + on-chain + макро)")
    print("\n✨ Новые возможности V3.2:")
    print("  • Анализ настроений из социальных сетей")
    print("  • On-chain метрики блокчейна")
    print("  • Макроэкономические индикаторы")
    print("  • Multi-modal архитектура нейронной сети")
    print("  • Адаптивный размер позиций на основе настроений")
    print("  • Комплексная система сигналов")
    
    if total_return > 0:
        print("\n🟢 ОЦЕНКА: Прибыльная sentiment-driven стратегия!")
    else:
        print("\n🔶 ОЦЕНКА: Требует дальнейшей оптимизации")
    
    print("\n🎉 АНАЛИЗ V3.2 ЗАВЕРШЕН!")


if __name__ == "__main__":
    main() 