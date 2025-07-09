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
🚀 ТОРГОВАЯ СИСТЕМА V5.10 - ПРАГМАТИК (MLP)
✅ ГЛАВНОЕ ИЗМЕНЕНИЕ: Пространство действий упрощено до `Discrete(2)`.
    - Агент принимает только релевантные решения (Hold/Buy или Hold/Sell), что резко повышает эффективность обучения.
✅ ВОЗВРАТ к чистому и простому reward shaping'у:
    - Есть награда за удержание прибыльной позиции (`unrealized_pnl`).
    - НЕТ конкурирующих наград (бонуса за ожидание).
✅ Сохранены все лучшие находки: Динамический SL/TP, качественные признаки, сильная MLP архитектура.
✅ Цель: Создать финальную, робастную и логичную базу, которая не обманывает симулятор и принимает осмысленные решения.
"""

class TrendTraderConfig:
    INITIAL_BALANCE = 10000
    ORDER_SIZE_RATIO = 0.15
    ATR_SL_MULTIPLIER = 2.0
    ATR_TP_MULTIPLIER = 6.0
    TRANSACTION_FEE = 0.001
    WINDOW_SIZE = 64
    TOTAL_TIMESTEPS = 750000
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01
    GAMMA = 0.99 # Возвращаем стандартное значение
    TREND_PROFIT_BONUS = 0.1
    TRADE_PENALTY = 0.01

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
        df['atr_value'] = tr.ewm(span=14, adjust=False).mean()
        df['rsi_delta'] = df['rsi'].diff(5)
        
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        features = pd.DataFrame(index=df.index)
        features['price_norm'] = df['close'] / df['sma_long']
        features['ema_spread'] = (df['ema_fast'] - df['ema_slow']) / df['close']
        features['rsi_norm'] = (df['rsi'] - 50) / 50
        features['macd_hist_norm'] = (df['macd'] - df['macd_signal']) / df['close']
        features['trend_signal'] = np.sign(df['close'] - df['sma_long'])
        features['atr_norm'] = df['atr_value'] / df['close']
        features['rsi_delta_norm'] = features['rsi_norm'].diff(5)
        
        features.replace([np.inf, -np.inf], 0, inplace=True)
        features.dropna(inplace=True)
        prices_df = df.loc[features.index].reset_index(drop=True)
        features.reset_index(drop=True, inplace=True)
        
        print(f"✅ Подготовлено данных: {len(features)} записей, {len(features.columns)} признаков.")
        return features, prices_df[['timestamp', 'open', 'high', 'low', 'close', 'atr_value']]

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, features_df: pd.DataFrame, prices_df: pd.DataFrame):
        super().__init__()
        self.features_df = features_df.reset_index(drop=True)
        self.prices_df = prices_df.reset_index(drop=True)
        self.cfg = TrendTraderConfig()
        
        # ИЗМЕНЕНО: Пространство действий теперь всегда Discrete(2)
        self.action_space = spaces.Discrete(2) # 0: Hold, 1: Act (Buy or Sell)
        # MLP ожидает плоские данные
        flattened_size = self.cfg.WINDOW_SIZE * self.features_df.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(flattened_size,), dtype=np.float32)
        self._reset_state()
    
    def _reset_state(self):
        self.balance = self.cfg.INITIAL_BALANCE
        self.equity = self.cfg.INITIAL_BALANCE
        self.current_step = self.cfg.WINDOW_SIZE
        self.position_amount = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.entry_step = 0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        start = self.current_step - self.cfg.WINDOW_SIZE
        # Возвращаем плоские данные для MLP
        return self.features_df.iloc[start:self.current_step].values.flatten().astype(np.float32)

    def _get_current_price(self) -> float:
        return self.prices_df.iloc[self.current_step]['close']

    def _get_current_atr(self) -> float:
        return self.prices_df.iloc[self.current_step]['atr_value']

    # ИЗМЕНЕНО: Логика step теперь обрабатывает бинарное действие
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()
        reward = 0.0
        done = False

        # Возвращаем награду за удержание
        if self.position_amount > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position_amount
            reward += unrealized_pnl / (self.cfg.INITIAL_BALANCE * 10) 
        
        # Action 1 теперь означает "Действовать"
        if action == 1:
            if self.position_amount == 0: # Если вне рынка, действие 1 - это "Купить"
                self._open_position(current_price)
                reward -= self.cfg.TRADE_PENALTY
            elif self.position_amount > 0: # Если в рынке, действие 1 - это "Продать"
                close_reward = self._close_position(current_price)
                reward += close_reward

        # Проверка SL/TP
        if self.position_amount > 0:
            if current_price <= self.stop_loss_price or current_price >= self.take_profit_price:
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
        current_atr = self._get_current_atr()
        
        self.stop_loss_price = price - (current_atr * self.cfg.ATR_SL_MULTIPLIER)
        self.take_profit_price = price + (current_atr * self.cfg.ATR_TP_MULTIPLIER)

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
        
        # Награда - это просто нормализованный PnL
        reward = realized_pnl / self.cfg.INITIAL_BALANCE
        
        if realized_pnl > 0:
            trend_at_entry = self.features_df.iloc[self.entry_step]['trend_signal']
            if trend_at_entry > 0: reward += self.cfg.TREND_PROFIT_BONUS
        
        self.position_amount = 0.0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        return reward

def main():
    print("🚀 СИСТЕМА V5.10 (Прагматик-MLP) - ЗАПУСК")
    data_loader = SimpleDataLoader("data/BTC_5_96w.csv")
    features_df, prices_df = data_loader.load_and_prepare_data()

    train_split_idx = int(len(features_df) * 0.8)
    train_features, train_prices = features_df.iloc[:train_split_idx], prices_df.iloc[:train_split_idx]
    test_features, test_prices = features_df.iloc[train_split_idx:], prices_df.iloc[train_split_idx:]
    print(f"\nДанные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
    env_fn = lambda: TradingEnv(train_features, train_prices)
    vec_env = DummyVecEnv([env_fn])

    # MLP архитектура для "Прагматика"
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
    )

    model = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
                learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF,
                n_steps=2048, batch_size=64, gamma=TrendTraderConfig.GAMMA,
                verbose=1, device="cpu")
    
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ ПРАГМАТИКА...")
    model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS)
    
    # ... (код тестирования и анализа без изменений) ...
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
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
    plt.title('Результаты на тестовой выборке (V5.10 - Прагматик-MLP)')
    ax1 = plt.gca(); ax1.plot(equity_history, label='Equity', color='blue', linewidth=2)
    ax1.set_xlabel('Шаги'); ax1.set_ylabel('Equity ($)', color='blue'); ax1.grid(True)
    ax2 = ax1.twinx(); ax2.plot(price_history, label='Цена BTC', color='orange', alpha=0.6)
    ax2.set_ylabel('Цена ($)', color='orange'); ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()