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
🚀 ТОРГОВАЯ СИСТЕМА V5.3 - УЛУЧШЕННАЯ ВЕРСИЯ
✅ Использует MLP политику для анализа временных паттернов.
✅ Кардинально переработана функция вознаграждения (reward shaping):
    - Добавлена награда/штраф за удержание позиции на каждом шаге.
    - Введен небольшой штраф за открытие сделки для борьбы с переторговлей.
✅ Увеличено время обучения для более сложной модели.
✅ Цель: Получить осмысленную стратегию, которая не "сливает" депозит на комиссиях.
"""


class TrendTraderConfig:
    INITIAL_BALANCE = 10000
    ORDER_SIZE_RATIO = 0.25
    STOP_LOSS = 0.04
    TAKE_PROFIT = 0.12
    TRANSACTION_FEE = 0.001
    WINDOW_SIZE = 64
    # ИЗМЕНЕНО: Увеличиваем время обучения для более сложной модели
    TOTAL_TIMESTEPS = 500000
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01
    GAMMA = 0.99
    # ИЗМЕНЕНО: Снизим бонус, чтобы он не был доминирующим фактором
    TREND_PROFIT_BONUS = 0.1
    TRADE_PENALTY = 0.01 # Штраф за открытие сделки

class SimpleDataLoader:
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
        features['price_norm'] = df['close'] / df['sma_long']
        features['ema_spread'] = (df['ema_fast'] - df['ema_slow']) / df['close']
        features['rsi_norm'] = (df['rsi'] - 50) / 50
        features['macd_hist_norm'] = (df['macd'] - df['macd_signal']) / df['close']
        features['trend_signal'] = np.sign(df['close'] - df['sma_long'])
        
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
        
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
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
        self.entry_step = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        start = self.current_step - self.cfg.WINDOW_SIZE
        return self.features_df.iloc[start:self.current_step].values.astype(np.float32)

    def _get_current_price(self) -> float:
        return self.prices_df.iloc[self.current_step]['close']

    # УЛУЧШЕНО: Полностью переработанный метод step с reward shaping
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()
        reward = 0.0
        done = False

        # 1. Награда/штраф за удержание позиции (unrealized PnL) на каждом шаге
        if self.position_amount > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position_amount
            # Нормализуем, чтобы вознаграждение не было слишком большим и не дестабилизировало обучение
            reward += unrealized_pnl / (self.cfg.INITIAL_BALANCE * 10) 
        
        # 2. ЛОГИКА ДЕЙСТВИЙ
        # Action 1: Купить (только если мы вне рынка)
        if action == 1 and self.position_amount == 0:
            self._open_position(current_price)
            # Штраф за открытие сделки, чтобы бороться с переторговлей
            reward -= self.cfg.TRADE_PENALTY

        # Action 2: Продать (только если мы в рынке)
        elif action == 2 and self.position_amount > 0:
            close_reward = self._close_position(current_price)
            reward += close_reward

        # 3. Проверка Stop Loss / Take Profit (если позиция все еще открыта после действий)
        if self.position_amount > 0:
            pnl_ratio = (current_price - self.entry_price) / self.entry_price
            if pnl_ratio <= -self.cfg.STOP_LOSS or pnl_ratio >= self.cfg.TAKE_PROFIT:
                close_reward = self._close_position(current_price)
                reward += close_reward

        # 4. ОБНОВЛЕНИЕ СОСТОЯНИЯ И ПРОВЕРКА НА ЗАВЕРШЕНИЕ
        self.current_step += 1
        
        # Обновляем equity для проверки на "слив депозита" и для info
        current_unrealized_pnl = (current_price - self.entry_price) * self.position_amount if self.position_amount > 0 else 0
        self.equity = self.balance + current_unrealized_pnl
        
        if self.current_step >= len(self.features_df) - 1 or self.equity <= self.cfg.INITIAL_BALANCE * 0.2:
            if self.position_amount > 0:
                self._close_position(current_price)
            done = True
        
        return self._get_observation(), reward, done, False, {'equity': self.equity}


    def _open_position(self, price: float):
        self.entry_step = self.current_step
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
        
        # Вознаграждение за закрытие сделки, нормализованное к начальному балансу
        reward = realized_pnl / self.cfg.INITIAL_BALANCE
        
        # Бонус за торговлю по тренду
        if realized_pnl > 0:
            trend_at_entry = self.features_df.iloc[self.entry_step]['trend_signal']
            if trend_at_entry > 0:
                reward += self.cfg.TREND_PROFIT_BONUS

        self.position_amount = 0.0
        self.entry_price = 0.0
        return reward

def main():
    print("🚀 СИСТЕМА V5.3 (улучшенная) - ЗАПУСК")
    data_loader = SimpleDataLoader("data/BTC_5_96w.csv")
    features_df, prices_df = data_loader.load_and_prepare_data()

    train_split_idx = int(len(features_df) * 0.8)
    train_features, train_prices = features_df.iloc[:train_split_idx], prices_df.iloc[:train_split_idx]
    test_features, test_prices = features_df.iloc[train_split_idx:], prices_df.iloc[train_split_idx:]
    print(f"\nДанные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
    # ИСПОЛЬЗУЕМ MLP: Создаем векторизованное окружение
    env_fn = lambda: TradingEnv(train_features, train_prices)
    vec_env = DummyVecEnv([lambda: TradingEnv(train_features, train_prices)])

    # ИЗМЕНЕНО: Параметры политики для MlpPolicy
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )

    # ИЗМЕНЕНО: Используем 'MlpPolicy' для временных рядов
    model = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
                learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF,
                n_steps=2048, batch_size=64, gamma=TrendTraderConfig.GAMMA,
                verbose=1, device="cpu")
    
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ МОДЕЛИ С MLP и улучшенным reward...")
    model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS)
    
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    # Создаем тестовое окружение
    test_env_raw = TradingEnv(test_features, test_prices)
    
    obs, _ = test_env_raw.reset()
    
    # Собираем историю из "сырого" окружения
    equity_history = [test_env_raw.equity]
    price_history = [test_env_raw._get_current_price()]
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = test_env_raw.step(int(action))
        
        # Собираем данные из тестового окружения
        equity_history.append(test_env_raw.equity) 
        price_history.append(test_env_raw._get_current_price())
        if done: break
            
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ")
    initial_equity, final_equity = equity_history[0], equity_history[-1]
    total_return = (final_equity - initial_equity) / initial_equity * 100
    start_price, end_price = price_history[0], price_history[-1]
    bnh_return = (end_price - start_price) / start_price * 100
    
    # Анализируем результаты торговли
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
    plt.title('Результаты на тестовой выборке (V5.3 - Улучшенная MLP версия)')
    ax1 = plt.gca()
    ax1.plot(equity_history, label='Equity', color='blue', linewidth=2)
    ax1.set_xlabel('Шаги')
    ax1.set_ylabel('Equity ($)', color='blue')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(price_history, label='Цена BTC', color='orange', alpha=0.6)
    ax2.set_ylabel('Цена ($)', color='orange')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()