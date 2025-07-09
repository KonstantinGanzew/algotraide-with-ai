"""
🚀 AGGRESSIVE SENTIMENT ТОРГОВАЯ СИСТЕМА V3.8
Агрессивная стратегия с гарантированной торговлей для 20% годовых
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


class AggressiveConfig:
    """Агрессивная конфигурация для достижения 20% годовых"""
    
    # Агрессивное управление рисками
    INITIAL_BALANCE = 10000
    RISK_PER_TRADE = 0.15  # Очень агрессивно
    MAX_POSITION_SIZE = 0.8  # Почти весь капитал
    STOP_LOSS = 0.08  # Более широкий стоп
    TAKE_PROFIT = 0.25  # Большой профит
    
    # Упрощенные веса
    TECHNICAL_WEIGHT = 0.8
    SENTIMENT_WEIGHT = 0.2
    
    # Настройки модели
    WINDOW_SIZE = 48
    TOTAL_TIMESTEPS = 15000
    LEARNING_RATE = 1e-4
    
    # Агрессивные параметры
    MIN_SIGNAL_STRENGTH = 0.2  # Снижен порог
    MAX_DAILY_TRADES = 8  # Больше сделок
    FORCE_TRADE_PROBABILITY = 0.1  # Принудительные сделки


class SimpleDataLoader:
    """Простой загрузчик с основными индикаторами"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Загрузка и подготовка данных с основными индикаторами"""
        print(f"📊 Загрузка данных из {self.data_path}...")
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['datetime'] = df['timestamp']
        
        print(f"📅 Период: {df['datetime'].min()} - {df['datetime'].max()}")
        print(f"📈 Цены: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"📊 Записей: {len(df)}")
        
        # Основные индикаторы
        print("🔧 Расчет основных индикаторов...")
        
        # Базовые
        df['returns'] = df['close'].pct_change()
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
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Объемные индикаторы
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Моментум
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Простые сигналы настроений
        np.random.seed(42)
        n = len(df)
        
        # Базовое настроение связано с моментумом и RSI
        momentum_signal = df['momentum_10'].fillna(0)
        rsi_signal = (df['rsi'].fillna(50) - 50) / 50  # Нормализация RSI
        
        base_sentiment = (momentum_signal * 0.6 + rsi_signal * 0.4) + np.random.normal(0, 0.1, n)
        base_sentiment = np.clip(base_sentiment, -1, 1)
        
        df['sentiment_twitter'] = base_sentiment + np.random.normal(0, 0.05, n)
        df['sentiment_reddit'] = base_sentiment + np.random.normal(0, 0.08, n)
        df['sentiment_news'] = base_sentiment + np.random.normal(0, 0.06, n)
        
        df['overall_sentiment'] = (
            df['sentiment_twitter'] * 0.5 +
            df['sentiment_reddit'] * 0.3 +
            df['sentiment_news'] * 0.2
        )
        
        # Финальная очистка
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"✅ Подготовлено {len([col for col in df.columns if col not in ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']])} признаков")
        return df


class AggressiveFeatureExtractor(BaseFeaturesExtractor):
    """Простой Feature Extractor"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.input_features = observation_space.shape[1] if observation_space.shape else 20
        
        # Простая сеть
        self.net = nn.Sequential(
            nn.Linear(self.input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.Tanh()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Используем только последнее наблюдение для простоты
        last_obs = observations[:, -1, :]
        return self.net(last_obs)


class AggressiveTradingEnv(gym.Env):
    """Агрессивное торговое окружение"""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = AggressiveConfig.WINDOW_SIZE
        
        # Отбираем только важные признаки
        important_features = [
            'returns', 'volatility', 'rsi', 'macd', 'macd_histogram', 
            'bb_position', 'volume_ratio', 'momentum_5', 'momentum_10', 'overall_sentiment'
        ]
        
        self.feature_columns = [col for col in important_features if col in df.columns]
        
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
        """Простая подготовка данных"""
        print("🔧 Подготовка данных для агрессивной торговли...")
        
        feature_data = self.df[self.feature_columns].fillna(0)
        feature_data = feature_data.replace([np.inf, -np.inf], 0)
        
        # Простая нормализация
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(feature_data)
        normalized_data = np.clip(normalized_data, -3, 3)
        
        self.normalized_df = pd.DataFrame(normalized_data, columns=self.feature_columns, index=feature_data.index)
        
        print(f"✅ Подготовлено {len(self.feature_columns)} признаков")
    
    def _reset_state(self):
        """Сброс состояния"""
        self.current_step = self.window_size
        self.balance = AggressiveConfig.INITIAL_BALANCE
        self.btc_amount = 0.0
        self.entry_price = 0.0
        
        self.total_trades = 0
        self.profitable_trades = 0
        self.portfolio_history = [float(AggressiveConfig.INITIAL_BALANCE)]
        self.trades_history = []
        
        self.max_drawdown = 0.0
        self.peak_value = AggressiveConfig.INITIAL_BALANCE
        self.last_action = 0
        self.steps_since_last_trade = 0
    
    def reset(self, seed=None, options=None):
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Получение наблюдения"""
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
    
    def _calculate_simple_signal(self) -> float:
        """Простой расчет торгового сигнала"""
        if self.current_step >= len(self.df):
            return 0.0
        
        current_data = self.df.iloc[self.current_step]
        
        # RSI сигнал
        rsi = current_data.get('rsi', 50)
        if rsi < 30:
            rsi_signal = 0.5  # Oversold
        elif rsi > 70:
            rsi_signal = -0.5  # Overbought
        else:
            rsi_signal = (50 - rsi) / 50 * 0.3
        
        # MACD сигнал
        macd_hist = current_data.get('macd_histogram', 0)
        macd_signal = np.tanh(macd_hist * 20) * 0.3
        
        # Bollinger сигнал
        bb_pos = current_data.get('bb_position', 0.5)
        if bb_pos < 0.2:
            bb_signal = 0.4
        elif bb_pos > 0.8:
            bb_signal = -0.4
        else:
            bb_signal = 0
        
        # Настроение
        sentiment = current_data.get('overall_sentiment', 0)
        sentiment_signal = sentiment * 0.3
        
        # Моментум
        momentum = current_data.get('momentum_10', 0)
        momentum_signal = np.tanh(momentum * 5) * 0.2
        
        total_signal = rsi_signal + macd_signal + bb_signal + sentiment_signal + momentum_signal
        return np.clip(total_signal, -1, 1)
    
    def _execute_trade(self, action: int, current_price: float) -> Dict[str, Any]:
        """Агрессивная логика торговли"""
        signal_strength = self._calculate_simple_signal()
        trade_result = {'executed': False, 'type': None, 'signal_strength': signal_strength}
        
        self.steps_since_last_trade += 1
        
        # Принудительная торговля если долго нет сделок
        force_trade = (self.steps_since_last_trade > 100 and 
                      np.random.random() < AggressiveConfig.FORCE_TRADE_PROBABILITY)
        
        # Проверка стоп-лосса и тейк-профита
        if self.btc_amount > 0 and self.entry_price > 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            
            if price_change <= -AggressiveConfig.STOP_LOSS or price_change >= AggressiveConfig.TAKE_PROFIT:
                # Принудительная продажа
                revenue = self.btc_amount * current_price
                commission = revenue * 0.001
                profit = revenue - self.btc_amount * self.entry_price - commission
                
                if profit > 0:
                    self.profitable_trades += 1
                
                self.balance += revenue - commission
                self.btc_amount = 0.0
                self.entry_price = 0.0
                self.steps_since_last_trade = 0
                
                exit_type = 'TAKE_PROFIT' if price_change >= AggressiveConfig.TAKE_PROFIT else 'STOP_LOSS'
                trade_result.update({
                    'executed': True, 'type': f'SELL_{exit_type}',
                    'profit': profit, 'price': current_price,
                    'datetime': self._get_current_datetime()
                })
                self.total_trades += 1
                self.trades_history.append(trade_result)
                
                return trade_result
        
        # Обычная торговля или принудительная
        if (action == 1 or force_trade) and self.balance > 100:  # Buy
            # Проверяем сигнал или принудительная торговля
            if signal_strength > AggressiveConfig.MIN_SIGNAL_STRENGTH or force_trade:
                position_size = AggressiveConfig.RISK_PER_TRADE
                if signal_strength > 0.5:
                    position_size = min(position_size * 1.5, AggressiveConfig.MAX_POSITION_SIZE)
                
                investment = self.balance * position_size
                amount = investment / current_price
                commission = investment * 0.001
                
                if investment + commission <= self.balance:
                    self.btc_amount += amount
                    self.balance -= investment + commission
                    self.entry_price = current_price
                    self.steps_since_last_trade = 0
                    
                    trade_result.update({
                        'executed': True, 'type': 'BUY',
                        'amount': amount, 'price': current_price,
                        'investment': investment, 'force_trade': force_trade,
                        'datetime': self._get_current_datetime()
                    })
                    
        elif action == 2 and self.btc_amount > 0:  # Sell
            # Продаем если есть позиция и сигнал на продажу
            if signal_strength < -AggressiveConfig.MIN_SIGNAL_STRENGTH or force_trade:
                revenue = self.btc_amount * current_price
                commission = revenue * 0.001
                profit = revenue - self.btc_amount * self.entry_price - commission
                
                if profit > 0:
                    self.profitable_trades += 1
                
                self.balance += revenue - commission
                self.btc_amount = 0.0
                self.entry_price = 0.0
                self.steps_since_last_trade = 0
                
                trade_result.update({
                    'executed': True, 'type': 'SELL_MANUAL',
                    'profit': profit, 'price': current_price,
                    'force_trade': force_trade,
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
        """Агрессивная функция награды"""
        current_portfolio = self._calculate_portfolio_value()
        prev_portfolio = self.portfolio_history[-1]
        
        # Базовая награда
        portfolio_change = (current_portfolio - prev_portfolio) / prev_portfolio
        base_reward = portfolio_change * 500  # Высокий масштаб для агрессивности
        
        # Обновление максимальной просадки
        if current_portfolio > self.peak_value:
            self.peak_value = current_portfolio
        current_drawdown = (self.peak_value - current_portfolio) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Бонусы за активность
        if self.total_trades > 0:
            activity_bonus = min(self.total_trades * 0.1, 10)
            base_reward += activity_bonus
        
        # Штраф за отсутствие активности
        if self.steps_since_last_trade > 200:
            base_reward -= 5
        
        # Бонус за прибыльность
        if portfolio_change > 0.005:
            base_reward += 20
        
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
            portfolio_value <= AggressiveConfig.INITIAL_BALANCE * 0.2 or
            self.max_drawdown > 0.7
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
            'steps_since_last_trade': self.steps_since_last_trade
        }
        
        return self._get_observation(), reward, done, False, info


def main():
    """Главная функция агрессивной торговой системы"""
    print("🚀 AGGRESSIVE SENTIMENT ТОРГОВАЯ СИСТЕМА V3.8")
    print("🎯 ЦЕЛЬ: 20% ГОДОВЫХ С ГАРАНТИРОВАННОЙ ТОРГОВЛЕЙ")
    print("=" * 75)
    
    # 1. Загрузка и подготовка данных
    print("\n📊 ЭТАП 1: БЫСТРАЯ ПОДГОТОВКА ДАННЫХ")
    print("-" * 50)
    
    data_loader = SimpleDataLoader("data/BTC_5_2w.csv")
    combined_df = data_loader.load_and_prepare_data()
    
    # 2. Создание агрессивного окружения
    print("\n🎮 ЭТАП 2: СОЗДАНИЕ АГРЕССИВНОГО ОКРУЖЕНИЯ")
    print("-" * 50)
    env = AggressiveTradingEnv(combined_df)
    vec_env = DummyVecEnv([lambda: env])
    print(f"✅ Агрессивное окружение готово")
    
    # 3. Создание простой модели
    print("\n🧠 ЭТАП 3: СОЗДАНИЕ ПРОСТОЙ МОДЕЛИ")
    print("-" * 50)
    
    policy_kwargs = dict(
        features_extractor_class=AggressiveFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 128],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=AggressiveConfig.LEARNING_RATE,
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
    
    print("✅ Простая модель создана")
    
    # 4. Интенсивное обучение
    print("\n🎓 ЭТАП 4: ИНТЕНСИВНОЕ ОБУЧЕНИЕ")
    print("-" * 50)
    model.learn(total_timesteps=AggressiveConfig.TOTAL_TIMESTEPS)
    print("✅ Интенсивное обучение завершено")
    
    # 5. Агрессивное тестирование
    print("\n💰 ЭТАП 5: АГРЕССИВНОЕ ТЕСТИРОВАНИЕ")
    print("-" * 50)
    
    obs, _ = env.reset()
    results = []
    trades_log = []
    
    print("💼 Начинаем агрессивную торговлю...")
    
    for step in range(min(4000, len(combined_df) - env.window_size - 1)):
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
            'steps_since_last_trade': info['steps_since_last_trade']
        })
        
        # Логируем сделки
        if info['trade_result']['executed']:
            trades_log.append(info['trade_result'])
            trade_type = info['trade_result']['type']
            price = info['trade_result']['price']
            datetime = info['trade_result']['datetime']
            signal = info['signal_strength']
            force = info['trade_result'].get('force_trade', False)
            force_str = " [ПРИНУД]" if force else ""
            print(f"⚡ {trade_type} ${price:.2f} сигнал:{signal:.2f}{force_str} {datetime}")
        
        if done:
            break
    
    # 6. Анализ агрессивных результатов
    print("\n📊 ЭТАП 6: АНАЛИЗ АГРЕССИВНЫХ РЕЗУЛЬТАТОВ")
    print("-" * 50)
    
    if results:
        final_result = results[-1]
        
        initial_value = AggressiveConfig.INITIAL_BALANCE
        final_value = final_result['portfolio_value']
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Экстраполяция на 2 года
        days_tested = len(results) / 12  # 5-минутные интервалы
        years_tested = days_tested / 365
        annualized_return = (final_value / initial_value) ** (1 / years_tested) - 1 if years_tested > 0 else 0
        two_year_projection = initial_value * ((1 + annualized_return) ** 2)
        two_year_profit = two_year_projection - initial_value
        
        total_trades = final_result['total_trades']
        profitable_trades = final_result['profitable_trades']
        win_rate = (profitable_trades / max(total_trades, 1)) * 100
        max_drawdown = final_result['max_drawdown'] * 100
        
        # Buy & Hold сравнение
        start_price = results[0]['current_price']
        end_price = final_result['current_price']
        bnh_return = (end_price - start_price) / start_price * 100
        
        print("🎯 РЕЗУЛЬТАТЫ АГРЕССИВНОЙ СТРАТЕГИИ")
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
        
        # Детальный анализ сделок
        if trades_log:
            buy_trades = [t for t in trades_log if t['type'] == 'BUY']
            sell_trades = [t for t in trades_log if 'profit' in t]
            force_trades = [t for t in trades_log if t.get('force_trade', False)]
            
            print(f"📊 Покупок: {len(buy_trades)}")
            print(f"📊 Продаж: {len(sell_trades)}")
            print(f"⚡ Принудительных: {len(force_trades)}")
            
            if sell_trades:
                profits = [t['profit'] for t in sell_trades]
                avg_profit = np.mean(profits)
                print(f"💰 Средняя прибыль: ${avg_profit:.2f}")
                print(f"🏆 Лучшая сделка: ${max(profits):.2f}")
                print(f"😞 Худшая сделка: ${min(profits):.2f}")
        
        print("\n🎉 ЗАКЛЮЧЕНИЕ АГРЕССИВНОЙ СТРАТЕГИИ")
        print("=" * 60)
        
        if two_year_profit >= 14400 and total_trades > 10:
            print("🟢 ОТЛИЧНО! Агрессивная стратегия ДОСТИГЛА цели 20% годовых!")
            print(f"💰 Ожидаемая прибыль за 2 года: ${two_year_profit:,.2f}")
        elif total_return > 10 and total_trades > 5:
            print("🟡 ХОРОШО! Активная торговля показала результат")
            print(f"🚀 Требует оптимизации для достижения 20% годовых")
        elif total_trades > 0:
            print("🔶 ПРОГРЕСС! Есть торговая активность")
            print(f"🔧 Доходность: {total_return:+.2f}%, сделок: {total_trades}")
        else:
            print("🔴 НЕТ АКТИВНОСТИ! Требует кардинальной доработки")
        
        print(f"\n🚀 Агрессивная стратегия: {annualized_return*100:+.2f}% годовых")
        print(f"⚡ Активность: {total_trades} сделок за период")
        print("✨ Агрессивная оптимизация завершена!")
    
    else:
        print("❌ Недостаточно данных для анализа")


if __name__ == "__main__":
    main() 