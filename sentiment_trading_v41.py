"""
🚀 ОПТИМИЗИРОВАННАЯ ТОРГОВАЯ СИСТЕМА V4.1 - ПРАВИЛЬНОЕ ОБУЧЕНИЕ
Исправлена проблема "бездействия" агента.
✅ Награда дается ТОЛЬКО за закрытие прибыльной сделки. Награда за "Hold" = 0.
✅ Убраны все жесткие правила (signal_strength). Агент учится сам.
✅ Только одна позиция (в рынке / вне рынка) для простоты.
✅ Данные разделены на обучающую и тестовую выборки (Train/Test split).
✅ Добавлена комиссия для реализма.
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
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class OptimalConfig:
    """Конфигурация, сфокусированная на стабильности и реальной прибыли."""
    INITIAL_BALANCE = 10000
    ORDER_SIZE_RATIO = 0.50  # Используем 50% капитала, т.к. позиция всего одна
    STOP_LOSS = 0.02
    TAKE_PROFIT = 0.04
    TRANSACTION_FEE = 0.001 # Реалистичная комиссия 0.1%

    WINDOW_SIZE = 50
    TOTAL_TIMESTEPS = 150000
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01

class SimpleDataLoader:
    """Загрузчик данных с фокусом на качественных, относительных признаках."""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(f"📊 Загрузка и подготовка данных из {self.data_path}...")
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Индикаторы
        df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Создаем ОТНОСИТЕЛЬНЫЕ признаки, которые лучше для обучения
        features = pd.DataFrame(index=df.index)
        features['price_vs_ema_slow'] = (df['close'] - df['ema_slow']) / df['ema_slow']
        features['ema_spread'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow']
        features['rsi_norm'] = (df['rsi'] - 50) / 50
        features['macd_hist_norm'] = (df['macd'] - df['macd_signal']) / df['close']
        features['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        features['volatility'] = df['close'].pct_change().rolling(20).std()
        
        features.dropna(inplace=True)
        prices_df = df.loc[features.index].reset_index(drop=True)
        features.reset_index(drop=True, inplace=True)
        
        print(f"✅ Подготовлено данных: {len(features)} записей, {len(features.columns)} признаков.")
        return features, prices_df[['timestamp', 'open', 'high', 'low', 'close']]

class EfficientFeatureExtractor(BaseFeaturesExtractor):
    """Использует всю историю из окна наблюдения."""
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_features = observation_space.shape[0] * observation_space.shape[1]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input_features, 256), nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU()
        )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

class TradingEnv(gym.Env):
    """Среда, где награда дается только за закрытие сделки."""
    def __init__(self, features_df: pd.DataFrame, prices_df: pd.DataFrame):
        super().__init__()
        self.features_df = features_df.reset_index(drop=True)
        self.prices_df = prices_df.reset_index(drop=True)
        self.cfg = OptimalConfig()
        
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.WINDOW_SIZE, self.features_df.shape[1]),
            dtype=np.float32
        )
        self._reset_state()
    
    def _reset_state(self):
        self.balance = self.cfg.INITIAL_BALANCE
        self.equity = self.cfg.INITIAL_BALANCE
        self.current_step = self.cfg.WINDOW_SIZE
        self.position_amount = 0.0  # 0: нет позиции, > 0: long
        self.entry_price = 0.0
        self.trades = []
    
    def reset(self, seed=None, options=None):
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        start = self.current_step - self.cfg.WINDOW_SIZE
        end = self.current_step
        return self.features_df.iloc[start:end].values.astype(np.float32)

    def _get_current_price(self) -> float:
        return self.prices_df.iloc[self.current_step]['close']

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()
        reward = 0.0
        done = False

        # === ЛОГИКА ДЕЙСТВИЙ ===
        if action == 1 and self.position_amount == 0:  # Покупка, если нет открытой позиции
            order_size_usd = self.balance * self.cfg.ORDER_SIZE_RATIO
            fee = order_size_usd * self.cfg.TRANSACTION_FEE
            self.balance -= (order_size_usd + fee)
            self.position_amount = order_size_usd / current_price
            self.entry_price = current_price

        elif action == 2 and self.position_amount > 0:  # Продажа, если есть открытая позиция
            reward = self._close_position(current_price)
        
        # === ПРОВЕРКА SL/TP ===
        if self.position_amount > 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            if price_change <= -self.cfg.STOP_LOSS or price_change >= self.cfg.TAKE_PROFIT:
                reward = self._close_position(current_price)

        # === ОБНОВЛЕНИЕ СОСТОЯНИЯ И ЗАВЕРШЕНИЕ ===
        unrealized_pnl = (current_price - self.entry_price) * self.position_amount if self.position_amount > 0 else 0
        self.equity = self.balance + unrealized_pnl

        self.current_step += 1
        if self.current_step >= len(self.features_df) - 1 or self.equity <= self.cfg.INITIAL_BALANCE * 0.5:
            if self.position_amount > 0:
                self._close_position(current_price) # Закрываем последнюю позицию
            done = True

        info = {'equity': self.equity, 'trades': len(self.trades), 'position': self.position_amount > 0}
        return self._get_observation(), reward, done, False, info

    def _close_position(self, price: float) -> float:
        """Закрывает текущую позицию и возвращает награду (реализованный PnL)."""
        close_value = self.position_amount * price
        fee = close_value * self.cfg.TRANSACTION_FEE
        self.balance += (close_value - fee)
        
        realized_pnl = (price - self.entry_price) * self.position_amount - (close_value + self.entry_price * self.position_amount) * self.cfg.TRANSACTION_FEE
        self.trades.append(realized_pnl)
        
        self.position_amount = 0.0
        self.entry_price = 0.0
        
        # Награда - это и есть прибыль/убыток
        return realized_pnl

def main():
    print("🚀 СИСТЕМА V4.1 - ЗАПУСК")
    
    # 1. Загрузка данных
    data_loader = SimpleDataLoader("data/BTC_5_96w.csv")
    features_df, prices_df = data_loader.load_and_prepare_data()

    # 2. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
    train_split_idx = int(len(features_df) * 0.8)
    train_features = features_df.iloc[:train_split_idx]
    train_prices = prices_df.iloc[:train_split_idx]
    test_features = features_df.iloc[train_split_idx:]
    test_prices = prices_df.iloc[train_split_idx:]
    print(f"\nДанные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
    # 3. Создание окружения и модели
    train_env = TradingEnv(train_features, train_prices)
    vec_env = DummyVecEnv([lambda: train_env])
    
    policy_kwargs = dict(features_extractor_class=EfficientFeatureExtractor)
    model = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
                learning_rate=OptimalConfig.LEARNING_RATE, ent_coef=OptimalConfig.ENTROPY_COEF,
                verbose=1, device="cpu")
    
    # 4. Обучение
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ МОДЕЛИ (на обучающей выборке)...")
    model.learn(total_timesteps=OptimalConfig.TOTAL_TIMESTEPS)
    
    # 5. Тестирование
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    test_env = TradingEnv(test_features, test_prices)
    obs, _ = test_env.reset()
    
    equity_history = [test_env.equity]
    price_history = [test_env._get_current_price()]
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = test_env.step(int(action))
        equity_history.append(info['equity'])
        price_history.append(test_prices.iloc[test_env.current_step]['close'])
        if done: break
            
    # 6. Анализ результатов
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ")
    initial_equity = equity_history[0]
    final_equity = equity_history[-1]
    total_return = (final_equity - initial_equity) / initial_equity * 100
    
    start_price = price_history[0]
    end_price = price_history[-1]
    bnh_return = (end_price - start_price) / start_price * 100

    trade_log = test_env.trades
    total_trades = len(trade_log)
    win_rate = 0
    if total_trades > 0:
        profitable_trades = len([t for t in trade_log if t > 0])
        win_rate = (profitable_trades / total_trades) * 100

    print("=" * 60)
    print(f"💰 Финальный баланс: ${final_equity:,.2f} (Начальный: ${initial_equity:,.2f})")
    print(f"📈 Доходность стратегии: {total_return:+.2f}%")
    print(f"📊 Доходность Buy & Hold: {bnh_return:+.2f}%")
    print("-" * 30)
    print(f"🔄 Всего сделок: {total_trades}")
    print(f"✅ Процент прибыльных сделок: {win_rate:.1f}%")
    
    # Визуализация
    plt.figure(figsize=(15, 7))
    plt.title('Результаты на тестовой выборке')
    ax1 = plt.gca()
    ax1.plot(equity_history, label='Equity', color='blue', linewidth=2)
    ax1.set_xlabel('Шаги')
    ax1.set_ylabel('Equity ($)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(price_history, label='Цена BTC', color='orange', alpha=0.6)
    ax2.set_ylabel('Цена ($)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()