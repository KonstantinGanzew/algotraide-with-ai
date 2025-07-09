"""
🚀 SENTIMENT ТОРГОВАЯ СИСТЕМА V3.4 - HISTORICAL DATA EDITION
Тестирование на реальных исторических данных Bitcoin
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


class HistoricalConfig:
    """Конфигурация для исторических данных"""
    
    # Управление рисками
    INITIAL_BALANCE = 10000
    RISK_PER_TRADE = 0.02
    MAX_POSITION_SIZE = 0.25
    STOP_LOSS = 0.03  # 3% стоп-лосс
    TAKE_PROFIT = 0.08  # 8% тейк-профит
    
    # Веса источников данных
    TECHNICAL_WEIGHT = 0.6
    SENTIMENT_WEIGHT = 0.2
    ON_CHAIN_WEIGHT = 0.1
    MACRO_WEIGHT = 0.1
    
    # Настройки модели
    WINDOW_SIZE = 48
    TOTAL_TIMESTEPS = 5000
    LEARNING_RATE = 1e-4
    
    # Параметры анализа настроений
    SENTIMENT_THRESHOLD = 0.2
    SENTIMENT_MULTIPLIER = 1.1


class HistoricalDataLoader:
    """Загрузчик реальных исторических данных"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_bitcoin_data(self) -> pd.DataFrame:
        """Загрузка исторических данных Bitcoin"""
        print(f"📊 Загрузка исторических данных Bitcoin из {self.data_path}...")
        
        # Загружаем данные
        df = pd.read_csv(self.data_path)
        
        # Преобразуем timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['datetime'] = df['timestamp']
        
        print(f"📅 Период данных: {df['datetime'].min()} - {df['datetime'].max()}")
        print(f"📈 Диапазон цен: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"📊 Количество записей: {len(df)}")
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление технических индикаторов к историческим данным"""
        print("🔧 Расчет технических индикаторов...")
        
        # Основные индикаторы
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(24).std()
        
        # Скользящие средние
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std_dev = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Support/Resistance levels
        df['price_high_20'] = df['high'].rolling(20).max()
        df['price_low_20'] = df['low'].rolling(20).min()
        df['price_range'] = (df['price_high_20'] - df['price_low_20']) / df['close']
        
        print(f"✅ Добавлено {len([col for col in df.columns if col not in ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']])} технических индикаторов")
        return df


class HistoricalSentimentGenerator:
    """Генератор симулированных данных настроений для исторических данных"""
    
    def __init__(self):
        self.sentiment_patterns = {
            'bull_run': {'trend': 0.3, 'volatility': 0.2, 'events': 0.1},
            'bear_market': {'trend': -0.2, 'volatility': 0.3, 'events': 0.15},
            'sideways': {'trend': 0.0, 'volatility': 0.15, 'events': 0.05}
        }
    
    def generate_sentiment_for_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Генерация реалистичных данных настроений на основе исторических цен"""
        print("📱 Генерация данных настроений на основе исторических движений...")
        
        # Анализируем историческое поведение цены
        df['price_trend'] = df['close'].rolling(48).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
        df['price_volatility'] = df['returns'].rolling(48).std()
        
        # Определяем рыночный режим
        market_regime = []
        for i, row in df.iterrows():
            price_trend_val = row['price_trend']
            price_vol_val = row['price_volatility']
            
            if pd.isna(price_trend_val) or pd.isna(price_vol_val):
                market_regime.append('sideways')
            elif price_trend_val > 0.1 and price_vol_val < 0.05:
                market_regime.append('bull_run')
            elif price_trend_val < -0.05:
                market_regime.append('bear_market')
            else:
                market_regime.append('sideways')
        
        df['market_regime'] = market_regime
        
        # Генерируем данные настроений
        n_points = len(df)
        np.random.seed(42)
        
        # Базовое настроение зависит от движения цены
        base_sentiment = df['price_trend'].fillna(0) * 2  # Усиливаем связь с ценой
        base_sentiment = np.clip(base_sentiment, -1, 1)
        
        # Добавляем шум
        noise = np.random.normal(0, 0.1, n_points)
        
        # События влияют на настроение
        events = np.random.choice([0, 1], size=n_points, p=[0.95, 0.05])
        event_impact = np.random.normal(0, 0.4, n_points) * events
        
        # Применяем затухание событий
        for i in range(1, len(event_impact)):
            if events[i-1] == 1:
                event_impact[i] += event_impact[i-1] * 0.8
        
        # Комбинируем все компоненты
        total_sentiment = base_sentiment + noise + event_impact
        total_sentiment = np.clip(total_sentiment, -1, 1)
        
        # Создаем различные источники настроений
        df['sentiment_twitter'] = total_sentiment + np.random.normal(0, 0.05, n_points)
        df['sentiment_reddit'] = total_sentiment + np.random.normal(0, 0.08, n_points)
        df['sentiment_news'] = total_sentiment + np.random.normal(0, 0.06, n_points)
        df['sentiment_social_volume'] = np.abs(total_sentiment) * 1000 + np.random.exponential(500, n_points)
        
        # Нормализация
        for col in ['sentiment_twitter', 'sentiment_reddit', 'sentiment_news']:
            df[col] = np.clip(df[col], -1, 1)
        
        # Агрегированное настроение
        df['overall_sentiment'] = (
            df['sentiment_twitter'] * 0.4 +
            df['sentiment_reddit'] * 0.3 +
            df['sentiment_news'] * 0.3
        )
        
        print(f"✅ Сгенерированы данные настроений для {n_points} временных точек")
        return df


class HistoricalFeatureExtractor(BaseFeaturesExtractor):
    """Feature Extractor для исторических данных"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        if observation_space.shape is not None:
            self.seq_len = observation_space.shape[0]
            self.input_features = observation_space.shape[1]
        else:
            self.seq_len = HistoricalConfig.WINDOW_SIZE
            self.input_features = 50
        
        # Сеть для обработки признаков
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # LSTM для временных зависимостей
        self.lstm = nn.LSTM(
            input_size=self.input_features,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Финальная сеть
        self.fusion_net = nn.Sequential(
            nn.Linear(96 + 64, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim, features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = observations.shape
        
        # Обработка последнего наблюдения
        last_obs = observations[:, -1, :]
        feature_out = self.feature_net(last_obs)
        
        # LSTM для временных зависимостей
        lstm_out, _ = self.lstm(observations)
        lstm_features = lstm_out[:, -1, :]
        
        # Объединение
        combined = torch.cat([lstm_features, feature_out], dim=1)
        output = self.fusion_net(combined)
        
        return output


class HistoricalTradingEnv(gym.Env):
    """Торговое окружение для исторических данных"""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = HistoricalConfig.WINDOW_SIZE
        
        # Подготовка данных для ML (исключаем строковые колонки)
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
        """Подготовка данных для обучения"""
        print("🔧 Подготовка данных для обучения...")
        
        # Нормализация данных
        scaler = StandardScaler()
        feature_data = self.df[self.feature_columns].fillna(0)
        feature_data = feature_data.replace([np.inf, -np.inf], 0)
        
        normalized_data = scaler.fit_transform(feature_data)
        
        # Ограничиваем значения для стабильности
        normalized_data = np.clip(normalized_data, -5, 5)
        
        # Убеждаемся, что feature_columns это список строк
        feature_cols = [str(col) for col in self.feature_columns]
        self.normalized_df = pd.DataFrame(normalized_data, columns=feature_cols)
        
        print(f"✅ Подготовлено {len(self.feature_columns)} признаков для {len(self.df)} временных точек")
    
    def _reset_state(self):
        """Сброс состояния окружения"""
        self.current_step = self.window_size
        self.balance = HistoricalConfig.INITIAL_BALANCE
        self.btc_amount = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        
        self.total_trades = 0
        self.profitable_trades = 0
        self.portfolio_history = [float(HistoricalConfig.INITIAL_BALANCE)]
        self.trades_history = []
        
        self.max_drawdown = 0.0
        self.peak_value = HistoricalConfig.INITIAL_BALANCE
    
    def reset(self, seed=None, options=None):
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Получение наблюдения"""
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        obs = self.normalized_df.iloc[start_idx:end_idx].values
        
        if len(obs) < self.window_size:
            padding = np.tile(obs[0], (self.window_size - len(obs), 1))
            obs = np.vstack([padding, obs])
        
        return obs.astype(np.float32)
    
    def _get_current_price(self) -> float:
        """Получение текущей цены Bitcoin"""
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.iloc[self.current_step]['close']
    
    def _get_current_datetime(self) -> str:
        """Получение текущего времени"""
        if self.current_step >= len(self.df):
            return str(self.df.iloc[-1]['datetime'])
        return str(self.df.iloc[self.current_step]['datetime'])
    
    def _get_sentiment_signal(self) -> float:
        """Получение сигнала настроений"""
        if self.current_step >= len(self.df):
            return 0.0
        
        current_data = self.df.iloc[self.current_step]
        
        if 'overall_sentiment' in current_data.index:
            sentiment = current_data['overall_sentiment']
        else:
            return 0.0
        
        # Фильтрация слабых сигналов
        if abs(sentiment) < HistoricalConfig.SENTIMENT_THRESHOLD:
            return 0.0
        
        return np.clip(sentiment * HistoricalConfig.SENTIMENT_MULTIPLIER, -1, 1)
    
    def _check_stop_loss_take_profit(self, current_price: float) -> str:
        """Проверка стоп-лосса и тейк-профита"""
        if self.btc_amount <= 0 or self.entry_price <= 0:
            return 'none'
        
        price_change = (current_price - self.entry_price) / self.entry_price
        
        if price_change <= -HistoricalConfig.STOP_LOSS:
            return 'stop_loss'
        elif price_change >= HistoricalConfig.TAKE_PROFIT:
            return 'take_profit'
        
        return 'none'
    
    def _execute_trade(self, action: int, current_price: float) -> Dict[str, Any]:
        """Выполнение торговой операции"""
        sentiment_signal = self._get_sentiment_signal()
        trade_result = {'executed': False, 'type': None, 'sentiment_signal': sentiment_signal}
        
        # Проверка стоп-лосса/тейк-профита
        sl_tp_action = self._check_stop_loss_take_profit(current_price)
        if sl_tp_action != 'none' and self.btc_amount > 0:
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
                'profit': profit, 'price': current_price,
                'datetime': self._get_current_datetime()
            })
            self.total_trades += 1
            self.trades_history.append(trade_result)
            
            return trade_result
        
        # Обычные торговые действия
        if action == 1 and self.balance > 100:  # Buy
            position_size = HistoricalConfig.RISK_PER_TRADE
            if sentiment_signal > 0:
                position_size *= (1 + abs(sentiment_signal) * 0.5)
            
            position_size = min(position_size, HistoricalConfig.MAX_POSITION_SIZE)
            
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
                'investment': investment,
                'datetime': self._get_current_datetime()
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
        """Расчет награды"""
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
        if current_drawdown > 0.15:
            base_reward -= current_drawdown * 100
        
        return base_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Шаг симуляции"""
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
            portfolio_value <= HistoricalConfig.INITIAL_BALANCE * 0.2 or
            self.max_drawdown > 0.6
        )
        
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'btc_amount': self.btc_amount,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'max_drawdown': self.max_drawdown,
            'sentiment_signal': trade_result.get('sentiment_signal', 0),
            'current_price': current_price,
            'datetime': self._get_current_datetime(),
            'trade_result': trade_result
        }
        
        return self._get_observation(), reward, done, False, info


def main():
    """Главная функция для тестирования на исторических данных"""
    print("🚀 ЗАПУСК SENTIMENT ТОРГОВОЙ СИСТЕМЫ V3.4 - HISTORICAL DATA EDITION")
    print("=" * 75)
    
    # 1. Загрузка исторических данных
    print("\n📊 ЭТАП 1: ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ")
    print("-" * 50)
    
    # Загружаем данные меньшего размера для быстрого тестирования
    data_loader = HistoricalDataLoader("data/BTC_5_2w.csv")
    bitcoin_df = data_loader.load_bitcoin_data()
    
    # 2. Добавление технических индикаторов
    print("\n🔧 ЭТАП 2: РАСЧЕТ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ")
    print("-" * 50)
    bitcoin_df = data_loader.add_technical_indicators(bitcoin_df)
    
    # 3. Генерация данных настроений
    print("\n📱 ЭТАП 3: ГЕНЕРАЦИЯ ДАННЫХ НАСТРОЕНИЙ")
    print("-" * 50)
    sentiment_generator = HistoricalSentimentGenerator()
    combined_df = sentiment_generator.generate_sentiment_for_historical_data(bitcoin_df)
    
    # Убираем NaN значения
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✅ Подготовлен датасет с {len(combined_df)} записями и {len(combined_df.columns)} признаками")
    
    # 4. Создание окружения
    print("\n🎮 ЭТАП 4: СОЗДАНИЕ ИСТОРИЧЕСКОГО ОКРУЖЕНИЯ")
    print("-" * 50)
    env = HistoricalTradingEnv(combined_df)
    vec_env = DummyVecEnv([lambda: env])
    print(f"✅ Окружение создано для периода {combined_df['datetime'].min()} - {combined_df['datetime'].max()}")
    
    # 5. Создание модели
    print("\n🧠 ЭТАП 5: СОЗДАНИЕ МОДЕЛИ ДЛЯ ИСТОРИЧЕСКИХ ДАННЫХ")
    print("-" * 50)
    
    policy_kwargs = dict(
        features_extractor_class=HistoricalFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 128, 64],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=HistoricalConfig.LEARNING_RATE,
        n_steps=1024,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    print("✅ Модель создана для исторических данных")
    
    # 6. Обучение на исторических данных
    print("\n🎓 ЭТАП 6: ОБУЧЕНИЕ НА ИСТОРИЧЕСКИХ ДАННЫХ")
    print("-" * 50)
    model.learn(total_timesteps=HistoricalConfig.TOTAL_TIMESTEPS)
    print("✅ Обучение на исторических данных завершено")
    
    # 7. Тестирование на исторических данных
    print("\n🧪 ЭТАП 7: BACKTESTING НА ИСТОРИЧЕСКИХ ДАННЫХ")
    print("-" * 50)
    
    obs, _ = env.reset()
    results = []
    trades_log = []
    
    print("💼 Начинаем торговлю...")
    
    for step in range(min(3000, len(combined_df) - env.window_size - 1)):
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
            'sentiment_signal': info['sentiment_signal']
        })
        
        # Логируем сделки
        if info['trade_result']['executed']:
            trades_log.append(info['trade_result'])
            trade_type = info['trade_result']['type']
            price = info['trade_result']['price']
            datetime = info['trade_result']['datetime']
            print(f"🔄 {trade_type} по цене ${price:.2f} в {datetime}")
        
        if done:
            break
    
    # 8. Анализ результатов backtesting
    print("\n📊 ЭТАП 8: АНАЛИЗ РЕЗУЛЬТАТОВ BACKTESTING")
    print("-" * 50)
    
    if results:
        final_result = results[-1]
        start_date = results[0]['datetime']
        end_date = final_result['datetime']
        
        initial_value = HistoricalConfig.INITIAL_BALANCE
        final_value = final_result['portfolio_value']
        total_return = (final_value - initial_value) / initial_value * 100
        
        total_trades = final_result['total_trades']
        profitable_trades = final_result['profitable_trades']
        win_rate = (profitable_trades / max(total_trades, 1)) * 100
        max_drawdown = final_result['max_drawdown'] * 100
        
        # Расчет дополнительных метрик
        portfolio_values = [r['portfolio_value'] for r in results]
        returns = [portfolio_values[i] / portfolio_values[i-1] - 1 for i in range(1, len(portfolio_values))]
        volatility = np.std(returns) * np.sqrt(len(returns)) if returns else 0
        
        sharpe_ratio = (np.mean(returns) / volatility) if volatility > 0 else 0
        
        # Анализ Bitcoin Buy & Hold
        start_price = results[0]['current_price']
        end_price = final_result['current_price']
        bnh_return = (end_price - start_price) / start_price * 100
        
        print("📊 РЕЗУЛЬТАТЫ BACKTESTING НА ИСТОРИЧЕСКИХ ДАННЫХ")
        print("=" * 70)
        print(f"📅 Период тестирования: {start_date} - {end_date}")
        print(f"💰 Начальный баланс: ${initial_value:,.2f}")
        print(f"💰 Финальная стоимость: ${final_value:,.2f}")
        print(f"📈 Доходность стратегии: {total_return:+.2f}%")
        print(f"📈 Buy & Hold Bitcoin: {bnh_return:+.2f}%")
        print(f"🎯 Превосходство над B&H: {total_return - bnh_return:+.2f}%")
        print(f"🔄 Всего сделок: {total_trades}")
        print(f"✅ Прибыльных сделок: {profitable_trades} ({win_rate:.1f}%)")
        print(f"📉 Максимальная просадка: {max_drawdown:.2f}%")
        print(f"📊 Коэффициент Шарпа: {sharpe_ratio:.3f}")
        print(f"💎 Волатильность: {volatility*100:.2f}%")
        
        # Анализ сделок
        if trades_log:
            buy_trades = [t for t in trades_log if t['type'] == 'BUY']
            sell_trades = [t for t in trades_log if 'profit' in t]
            
            if sell_trades:
                profits = [t['profit'] for t in sell_trades]
                avg_profit = np.mean(profits)
                print(f"💰 Средняя прибыль с сделки: ${avg_profit:.2f}")
                print(f"🏆 Лучшая сделка: ${max(profits):.2f}")
                print(f"😞 Худшая сделка: ${min(profits):.2f}")
        
        print("\n🎯 ЗАКЛЮЧЕНИЕ ИСТОРИЧЕСКОГО BACKTESTING")
        print("=" * 60)
        
        if total_return > bnh_return and win_rate > 50 and max_drawdown < 30:
            print("🟢 ОТЛИЧНЫЙ РЕЗУЛЬТАТ: Стратегия превосходит Buy & Hold!")
        elif total_return > 0 and win_rate > 45:
            print("🟡 ХОРОШИЙ РЕЗУЛЬТАТ: Прибыльная стратегия с потенциалом")
        elif total_return > bnh_return:
            print("🔶 ПРИЕМЛЕМЫЙ РЕЗУЛЬТАТ: Превосходит пассивную стратегию")
        else:
            print("🔴 ТРЕБУЕТ УЛУЧШЕНИЯ: Не превосходит простое держание")
        
        print(f"\n💡 Стратегия показала доходность {total_return:+.2f}% за период")
        print(f"🚀 На исторических данных с {len(combined_df)} точками")
        print("🎉 BACKTESTING НА ИСТОРИЧЕСКИХ ДАННЫХ ЗАВЕРШЕН!")
    
    else:
        print("❌ Недостаточно данных для анализа")


if __name__ == "__main__":
    main() 