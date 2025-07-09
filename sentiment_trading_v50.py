import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

"""
🚀 ТОРГОВАЯ СИСТЕМА V5.6 - ТРЕЙДЕР-СНАЙПЕР  
✅ ИЗМЕНЕНО: Добавлена "память" о портфеле в observation (состояние позиции, PnL, длительность).
✅ ИЗМЕНЕНО: Скорректированы награды и штрафы, чтобы "разбудить" агента и поощрять качественные сделки.
✅ ИСПРАВЛЕНО: Переход на MlpPolicy для правильной работы с временными рядами + портфельной памятью.
✅ Цель: Превратить безубыточного "Монаха" в прибыльного "Снайпера", повысив винрейт и сохранив низкую частоту сделок.
"""

class TrendTraderConfig:
    INITIAL_BALANCE = 10000
    ORDER_SIZE_RATIO = 0.15
    STOP_LOSS = 0.04
    TAKE_PROFIT = 0.12
    TRANSACTION_FEE = 0.001
    WINDOW_SIZE = 64
    TOTAL_TIMESTEPS = 500000 # Оставим 500k, этого должно хватить
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01
    GAMMA = 0.99
    # ИЗМЕНЕНО: Корректируем баланс "кнута и пряника"
    TREND_PROFIT_BONUS = 0.2
    TRADE_PENALTY = 0.005

class SimpleDataLoader:
    # ... (Этот класс без изменений)
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

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.ewm(span=14, adjust=False).mean()
        
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        features = pd.DataFrame(index=df.index)
        features['price_norm'] = df['close'] / df['sma_long']
        features['ema_spread'] = (df['ema_fast'] - df['ema_slow']) / df['close']
        features['rsi_norm'] = (df['rsi'] - 50) / 50
        features['macd_hist_norm'] = (df['macd'] - df['macd_signal']) / df['close']
        features['trend_signal'] = np.sign(df['close'] - df['sma_long'])
        features['atr_norm'] = df['atr'] / df['close']
        
        features.replace([np.inf, -np.inf], 0, inplace=True)
        features.dropna(inplace=True)
        prices_df = df.loc[features.index].reset_index(drop=True)
        features.reset_index(drop=True, inplace=True)
        
        print(f"✅ Подготовлено данных: {len(features)} записей, {len(features.columns)} признаков.")
        return features, prices_df[['timestamp', 'open', 'high', 'low', 'close']]


class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, features_df: pd.DataFrame, prices_df: pd.DataFrame):
        super().__init__()
        self.features_df = features_df.reset_index(drop=True)
        self.prices_df = prices_df.reset_index(drop=True)
        self.cfg = TrendTraderConfig()
        
        self.action_space = spaces.Discrete(3)
        
        # ИЗМЕНЕНО: Добавляем 3 новых признака для "памяти" агента
        self.market_features_shape = self.features_df.shape[1]
        self.portfolio_features_shape = 3 
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.WINDOW_SIZE, self.market_features_shape + self.portfolio_features_shape),
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
        self.entry_step = 0
        self.steps_in_position = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}
    
    # ИЗМЕНЕНО: _get_observation теперь добавляет состояние портфеля
    def _get_observation(self):
        # 1. Берем рыночные данные
        market_obs = self.features_df.iloc[self.current_step - self.cfg.WINDOW_SIZE : self.current_step].values

        # 2. Создаем данные о портфеле
        unrealized_pnl = 0.0
        position_state = 0.0
        if self.position_amount > 0:
            position_state = 1.0
            unrealized_pnl = (self._get_current_price() - self.entry_price) / self.entry_price
        
        # Нормализуем длительность сделки, чтобы она не росла до бесконечности
        normalized_steps = self.steps_in_position / self.cfg.WINDOW_SIZE 
        
        portfolio_obs = np.array([[position_state, unrealized_pnl, normalized_steps]] * self.cfg.WINDOW_SIZE)

        # 3. Объединяем их
        return np.concatenate([market_obs, portfolio_obs], axis=1).astype(np.float32)

    def _get_current_price(self) -> float:
        return self.prices_df.iloc[self.current_step]['close']

    # ИЗМЕНЕНО: Логика step теперь обновляет self.steps_in_position
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()
        reward = 0.0
        done = False

        if self.position_amount > 0:
            self.steps_in_position += 1
            unrealized_pnl = (current_price - self.entry_price) * self.position_amount
            reward += unrealized_pnl / (self.cfg.INITIAL_BALANCE * 10) 
        else:
            self.steps_in_position = 0 # Сбрасываем счетчик, если не в позиции
        
        if action == 1 and self.position_amount == 0:
            self._open_position(current_price)
            reward -= self.cfg.TRADE_PENALTY
        elif action == 2 and self.position_amount > 0:
            close_reward = self._close_position(current_price)
            reward += close_reward

        if self.position_amount > 0:
            pnl_ratio = (current_price - self.entry_price) / self.entry_price
            if pnl_ratio <= -self.cfg.STOP_LOSS or pnl_ratio >= self.cfg.TAKE_PROFIT:
                close_reward = self._close_position(current_price)
                reward += close_reward

        self.current_step += 1
        current_unrealized_pnl = (current_price - self.entry_price) * self.position_amount if self.position_amount > 0 else 0
        self.equity = self.balance + current_unrealized_pnl
        
        if self.current_step >= len(self.features_df) - 1 or self.equity <= self.cfg.INITIAL_BALANCE * 0.2:
            if self.position_amount > 0: self._close_position(current_price)
            done = True
        
        return self._get_observation(), reward, done, False, {'equity': self.equity}

    def _open_position(self, price: float):
        self.entry_step = self.current_step
        self.steps_in_position = 1
        order_size_usd = self.balance * self.cfg.ORDER_SIZE_RATIO
        fee = order_size_usd * self.cfg.TRANSACTION_FEE
        self.balance -= (order_size_usd + fee)
        self.position_amount = order_size_usd / price
        self.entry_price = price

    def _close_position(self, price: float) -> float:
        self.steps_in_position = 0 # Сбрасываем счетчик при закрытии
        close_value = self.position_amount * price
        fee = close_value * self.cfg.TRANSACTION_FEE
        self.balance += (close_value - fee)
        realized_pnl = (price - self.entry_price) * self.position_amount - (self.entry_price * self.position_amount * self.cfg.TRANSACTION_FEE) - fee
        self.trades.append(realized_pnl)
        reward = realized_pnl / self.cfg.INITIAL_BALANCE
        if realized_pnl > 0:
            trend_at_entry = self.features_df.iloc[self.entry_step]['trend_signal']
            if trend_at_entry > 0: reward += self.cfg.TREND_PROFIT_BONUS
        self.position_amount = 0.0
        self.entry_price = 0.0
        return reward

def main():
    print("🚀 СИСТЕМА V5.6 (Трейдер-Снайпер с MLP) - ЗАПУСК")
    data_loader = SimpleDataLoader("data/BTC_5_96w.csv")
    features_df, prices_df = data_loader.load_and_prepare_data()

    train_split_idx = int(len(features_df) * 0.8)
    train_features, train_prices = features_df.iloc[:train_split_idx], prices_df.iloc[:train_split_idx]
    test_features, test_prices = features_df.iloc[train_split_idx:], prices_df.iloc[train_split_idx:]
    print(f"\nДанные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
    # Используем MLP: создаем векторизованное окружение с портфельной памятью
    env_fn = lambda: TradingEnv(train_features, train_prices)
    vec_env = DummyVecEnv([lambda: TradingEnv(train_features, train_prices)])

    # ИЗМЕНЕНО: Параметры для MlpPolicy с усиленной архитектурой для портфельной памяти
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])  # Более глубокая сеть для обработки портфельной памяти
    )

    model = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
                learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF,
                n_steps=2048, batch_size=64, gamma=TrendTraderConfig.GAMMA,
                verbose=1, device="cpu")
    
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ МОДЕЛИ-СНАЙПЕРА С ПОРТФЕЛЬНОЙ ПАМЯТЬЮ...")
    model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS)
    
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    # Создаем тестовое окружение с портфельной памятью
    test_env_raw = TradingEnv(test_features, test_prices)
    obs, _ = test_env_raw.reset()
    
    equity_history = [test_env_raw.equity]
    price_history = [test_env_raw._get_current_price()]
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = test_env_raw.step(int(action))
        equity_history.append(test_env_raw.equity) 
        price_history.append(test_env_raw._get_current_price())
        if done: break
            
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ")
    initial_equity, final_equity = equity_history[0], equity_history[-1]
    total_return = (final_equity - initial_equity) / initial_equity * 100
    start_price, end_price = price_history[0], price_history[-1]
    bnh_return = (end_price - start_price) / start_price * 100
    total_trades = len(test_env_raw.trades)
    win_rate = (len([t for t in test_env_raw.trades if t > 0]) / total_trades) * 100 if total_trades > 0 else 0

    print("=" * 60)
    print(f"💰 Финальный баланс: ${final_equity:,.2f} (Начальный: ${initial_equity:,.2f})")
    print(f"📈 Доходность стратегии: {total_return:+.2f}%")
    print(f"📊 Доходность Buy & Hold: {bnh_return:+.2f}%")
    print("-" * 30)
    print(f"🔄 Всего сделок: {total_trades}")
    print(f"✅ Процент прибыльных сделок: {win_rate:.1f}%")
    
    plt.figure(figsize=(15, 7))
    plt.title('Результаты на тестовой выборке (V5.6 - Трейдер-Снайпер с портфельной памятью)')
    ax1 = plt.gca(); ax1.plot(equity_history, label='Equity', color='blue', linewidth=2)
    ax1.set_xlabel('Шаги'); ax1.set_ylabel('Equity ($)', color='blue'); ax1.grid(True)
    ax2 = ax1.twinx(); ax2.plot(price_history, label='Цена BTC', color='orange', alpha=0.6)
    ax2.set_ylabel('Цена ($)', color='orange'); ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()