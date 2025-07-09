"""
🚀 ТОРГОВАЯ СИСТЕМА V5.0 - ТРАНСПЛАНТАЦИЯ МОЗГА (CNN)
✅ Заменен примитивный мозг (MlpPolicy) на продвинутый (CnnPolicy), способный видеть паттерны.
✅ Убраны все "костыли" (штрафы, бонусы). Среда максимально чистая.
✅ Увеличено время обучения, т.к. новый мозг сложнее.
✅ Цель: Увидеть первую осмысленную стратегию.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Важно! Для CNN нужен другой формат данных. Добавляем обертку.
# from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStack


class InvestorConfig:
    """Конфигурация для обучения "зрячего" инвестора с CNN."""
    INITIAL_BALANCE = 10000
    ORDER_SIZE_RATIO = 0.25
    STOP_LOSS = 0.05
    TAKE_PROFIT = 0.10
    TRANSACTION_FEE = 0.001

    WINDOW_SIZE = 64 # CNN любят размеры, кратные 2, например, 64
    # ИЗМЕНЕНО: Более сложный мозг требует больше времени на обучение
    TOTAL_TIMESTEPS = 250000
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01
    GAMMA = 0.99 # Возвращаем стандартный gamma

class SimpleDataLoader:
    """Загрузчик с чистым набором признаков для CNN."""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(f"📊 Загрузка и подготовка данных из {self.data_path}...")
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['sma_long'] = df['close'].rolling(window=200).mean()
        
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        features = pd.DataFrame(index=df.index)
        features['price_norm'] = df['close'] / df['sma_long'] # Цена относительно долгосрочного тренда
        features['ema_spread'] = (df['ema_fast'] - df['ema_slow']) / df['close']
        features['rsi_norm'] = (df['rsi'] - 50) / 50
        features['macd_hist_norm'] = (df['macd'] - df['macd_signal']) / df['close']
        
        features.replace([np.inf, -np.inf], 0, inplace=True)
        features.dropna(inplace=True)
        prices_df = df.loc[features.index].reset_index(drop=True)
        features.reset_index(drop=True, inplace=True)
        
        print(f"✅ Подготовлено данных: {len(features)} записей, {len(features.columns)} признаков.")
        return features, prices_df[['timestamp', 'open', 'high', 'low', 'close']]

class TradingEnv(gym.Env):
    # ИЗМЕНЕНО: Добавляем `render_mode` для совместимости
    metadata = {'render_modes': ['human']}

    def __init__(self, features_df: pd.DataFrame, prices_df: pd.DataFrame, render_mode=None):
        super().__init__()
        self.features_df = features_df.reset_index(drop=True)
        self.prices_df = prices_df.reset_index(drop=True)
        self.cfg = InvestorConfig()
        
        self.action_space = spaces.Discrete(3)
        # ИЗМЕНЕНО: Формат observation space для CNN (высота, ширина)
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
        self.position_amount = 0.0
        self.entry_price = 0.0
        self.trades = []
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Важно для совместимости с новыми версиями gym
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        start = self.current_step - self.cfg.WINDOW_SIZE
        return self.features_df.iloc[start:self.current_step].values.astype(np.float32)

    def _get_current_price(self) -> float:
        return self.prices_df.iloc[self.current_step]['close']

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()
        # ИЗМЕНЕНО: Среда предельно чистая. Награда = 0 по умолчанию.
        reward = 0.0
        done = False

        if action == 1 and self.position_amount == 0:
            self._open_position(current_price)
        elif action == 2 and self.position_amount > 0:
            reward = self._close_position(current_price)
        
        if self.position_amount > 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            if price_change <= -self.cfg.STOP_LOSS or price_change >= self.cfg.TAKE_PROFIT:
                reward = self._close_position(current_price)

        unrealized_pnl = (current_price - self.entry_price) * self.position_amount if self.position_amount > 0 else 0
        self.equity = self.balance + unrealized_pnl
        self.current_step += 1
        
        if self.current_step >= len(self.features_df) - 1 or self.equity <= self.cfg.INITIAL_BALANCE * 0.2:
            if self.position_amount > 0:
                self._close_position(current_price)
            done = True
        
        # truncated = False для совместимости с API
        return self._get_observation(), reward, done, False, {'equity': self.equity}

    def _open_position(self, price: float):
        order_size_usd = self.balance * self.cfg.ORDER_SIZE_RATIO
        fee = order_size_usd * self.cfg.TRANSACTION_FEE
        self.balance -= (order_size_usd + fee)
        self.position_amount = order_size_usd / price
        self.entry_price = price

    def _close_position(self, price: float) -> float:
        close_value = self.position_amount * price
        fee = close_value * self.cfg.TRANSACTION_FEE
        self.balance += (close_value - fee)
        realized_pnl = (price - self.entry_price) * self.position_amount - (self.entry_price * self.position_amount * self.cfg.TRANSACTION_FEE) - fee
        self.trades.append(realized_pnl)
        self.position_amount = 0.0
        self.entry_price = 0.0
        return realized_pnl / self.cfg.INITIAL_BALANCE

def main():
    print("🚀 СИСТЕМА V5.0 - ЗАПУСК")
    data_loader = SimpleDataLoader("data/BTC_5_96w.csv")
    features_df, prices_df = data_loader.load_and_prepare_data()

    train_split_idx = int(len(features_df) * 0.8)
    train_features, train_prices = features_df.iloc[:train_split_idx], prices_df.iloc[:train_split_idx]
    test_features, test_prices = features_df.iloc[train_split_idx:], prices_df.iloc[train_split_idx:]
    print(f"\nДанные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
    # Создаем lambda-функцию для окружения
    env_fn = lambda: TradingEnv(train_features, train_prices)
    # ИЗМЕНЕНО: Оборачиваем среду в DummyVecEnv
    vec_env = DummyVecEnv([env_fn])

    # ИСПРАВЛЕНО: Используем 'MlpPolicy' для числовых данных
    # CnnPolicy подходит только для изображений
    policy_kwargs = dict(net_arch=dict(pi=[64], vf=[64])) # Простая архитектура

    model = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
                learning_rate=InvestorConfig.LEARNING_RATE, ent_coef=InvestorConfig.ENTROPY_COEF,
                n_steps=2048, batch_size=64, gamma=InvestorConfig.GAMMA,
                verbose=1, device="cpu")
    
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ МОДЕЛИ С CNN...")
    model.learn(total_timesteps=InvestorConfig.TOTAL_TIMESTEPS)
    
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    test_env = TradingEnv(test_features, test_prices)
    obs, _ = test_env.reset()
    
    equity_history, price_history = [test_env.equity], [test_env._get_current_price()]
    
    while True:
        # Для MlpPolicy используем данные как есть
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = test_env.step(int(action))
        equity_history.append(info['equity'])
        price_history.append(test_env._get_current_price())
        if done: break
            
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ")
    initial_equity, final_equity = equity_history[0], equity_history[-1]
    total_return = (final_equity - initial_equity) / initial_equity * 100
    start_price, end_price = price_history[0], price_history[-1]
    bnh_return = (end_price - start_price) / start_price * 100
    total_trades = len(test_env.trades)
    win_rate = (len([t for t in test_env.trades if t > 0]) / total_trades) * 100 if total_trades > 0 else 0

    print("=" * 60)
    print(f"💰 Финальный баланс: ${final_equity:,.2f} (Начальный: ${initial_equity:,.2f})")
    print(f"📈 Доходность стратегии: {total_return:+.2f}%")
    print(f"📊 Доходность Buy & Hold: {bnh_return:+.2f}%")
    print("-" * 30)
    print(f"🔄 Всего сделок: {total_trades}")
    print(f"✅ Процент прибыльных сделок: {win_rate:.1f}%")
    
    plt.figure(figsize=(15, 7))
    plt.title('Результаты на тестовой выборке (V5.0 - CNN)')
    ax1 = plt.gca(); ax1.plot(equity_history, label='Equity', color='blue', linewidth=2)
    ax1.set_xlabel('Шаги'); ax1.set_ylabel('Equity ($)', color='blue'); ax1.grid(True)
    ax2 = ax1.twinx(); ax2.plot(price_history, label='Цена BTC', color='orange', alpha=0.6)
    ax2.set_ylabel('Цена ($)', color='orange'); ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()