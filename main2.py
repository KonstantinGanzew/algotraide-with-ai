import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, threshold=-0.001, patience=5, verbose=1):
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience
        self.counter = 0

    def _on_step(self) -> bool:
        # Попробуем получить explained_variance напрямую из модели
        if 'explained_variance' in self.locals.get('infos', [{}])[0]:
            ev = self.locals['infos'][0]['explained_variance']
        else:
            # Логгер не содержит нужной инфы, просто возвращаем True
            return True

        if ev < self.threshold:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] explained_variance={ev:.4f}, counter={self.counter}")
            if self.counter >= self.patience:
                print("[EarlyStopping] Stopping training!")
                return False
        else:
            self.counter = 0

        return True



class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, window_size=50, initial_balance=1000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, df.shape[1]), dtype=np.float32
        )

        # === Параметры позиции ===
        self.balance = self.initial_balance
        self.entry_price = 0
        self.position = 0
        self.position_size = 0
        self.current_step = self.window_size
        self.trades = []
        self.last_action = None
        self.order_size_usd = self.initial_balance / 3  # по 333 на ордер
        self.wait_counter = 0

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.entry_price = 0
        self.position = 0
        self.position_size = 0
        self.current_step = self.window_size
        self.trades = []
        self.last_action = None
        self.order_size_usd = self.initial_balance / 3  # по 333 на ордер
        self.wait_counter = 0

        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        reward = 0
        done = False

        current_price = self.df.iloc[self.current_step]['close']

        if action == 2 and self.position_size == 0:
            reward -= 0.5
            return self._get_observation(), reward, False, False, {}

        # === Купить ===
        if action == 1:
            if self.position_size < 3:
                self.entry_price = (
                        (self.entry_price * self.position_size + current_price)
                        / (self.position_size + 1)
                ) if self.position_size > 0 else current_price

                self.position_size += 1
                self.position = 1
                reward += 0.1  # 📌 усилили поощрение за вход
                self.wait_counter = 0
            else:
                reward -= 0.05  # ⚠️ уменьшили штраф

        # === Продать ===
        elif action == 2:
            if self.position_size > 0:
                profit_per_coin = current_price - self.entry_price
                profit_total = (
                        profit_per_coin
                        * self.order_size_usd
                        * self.position_size
                        / self.entry_price
                )

                self.balance += profit_total
                reward += profit_total

                if profit_total > 0:
                    reward += 2  # 💰 бонус за прибыль

                self.trades.append(profit_total)
                self.position_size = 0
                self.position = 0
                self.entry_price = 0
                self.wait_counter = 0
            else:
                reward -= 0.05  # ⚠️ мягкий штраф

        # === Нереализованная прибыль ===
        if self.position_size > 0 and self.entry_price > 0:
            unrealized = (
                    (current_price - self.entry_price)
                    * self.order_size_usd
                    * self.position_size
                    / self.entry_price
            )
            reward += 0.05 * unrealized

        # === Повторное действие подряд ===
        if self.last_action == action:
            reward -= 0.01

        # === Штраф за полную пассивность ===
        self.wait_counter += 1
        if self.wait_counter > 100:
            reward -= 0.1

        self.last_action = action
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            done = True

            # Закрываем открытую позицию, если есть
            if self.position_size > 0:
                final_profit = (
                        (current_price - self.entry_price)
                        * self.order_size_usd
                        * self.position_size
                        / self.entry_price
                )
                self.balance += final_profit
                reward += final_profit
                self.trades.append(final_profit)

                self.position_size = 0
                self.position = 0
                self.entry_price = 0

            # === 🧠 Новый блок: награда за финальный результат ===
            total_profit = self.balance - self.initial_balance

            # 💰 поощрение за итоговую прибыль
            reward += total_profit * 0.01  # например, 1% от результата

            # 🚨 штраф за убыток (опционально — или просто обнуляй)
            if self.balance < self.initial_balance:
                reward -= 50  # или -total_profit * 0.01

        return self._get_observation(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}")

# === 1. Загрузка данных ===
file_name = "BTC_5_96w.csv"
folder = "data/"
path = folder + file_name

df = pd.read_csv(path)
df = df[['open', 'high', 'low', 'close', 'volume']]

# EMA
df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Чистка NaN
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Нормализация только нужных колонок
cols_to_normalize = ['open', 'high', 'low', 'close', 'volume', 'ema_fast', 'ema_slow']
df[cols_to_normalize] = (df[cols_to_normalize] - df[cols_to_normalize].mean()) / df[cols_to_normalize].std()

# === 2. Создание среды и обучение ===
env = TradingEnv(df, initial_balance=10000)
from stable_baselines3.common.vec_env import DummyVecEnv
vec_env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1, device="cpu", ent_coef=0.05)  # или 0.02
model.learn(total_timesteps=50_000, callback=EarlyStoppingCallback())

# === 3. Тестирование ===
test_env = TradingEnv(df, initial_balance=10000)
obs, _ = test_env.reset()


actual_balance = []
prices = []
actions = []

while True:
    action = model.predict(obs, deterministic=True)[0]  # action уже int
    obs, reward, done, truncated, _ = test_env.step(action)

    step = test_env.current_step
    if step >= len(df):  # защита от выхода за границы
        break

    current_price = df.iloc[step]['close']

    if test_env.position_size > 0 and test_env.entry_price > 0:
        unrealized = (
                (current_price - test_env.entry_price)
                * test_env.order_size_usd
                * test_env.position_size
                / test_env.entry_price
        )
        total_balance = test_env.balance + unrealized
    else:
        total_balance = test_env.balance

    actual_balance.append(total_balance)
    prices.append(current_price)
    actions.append(action)  # <-- исправлено

    if done:
        break



print(f"Initial balance: 10000")
print(f"Final balance: {actual_balance[-1]:.2f}")

# === 5. Визуализация ===
plt.figure(figsize=(14, 6))

# Капитал
plt.subplot(2, 1, 1)
plt.plot(actual_balance, label='Agent Balance')
plt.title("Баланс агента")
plt.xlabel("Шаг")
plt.ylabel("Баланс (USDT)")
plt.legend()

# Цена и действия
plt.subplot(2, 1, 2)
plt.plot(prices, label='BTC Price', alpha=0.7)
buy_signals = [i for i, a in enumerate(actions) if a == 1]
sell_signals = [i for i, a in enumerate(actions) if a == 2]
plt.scatter(buy_signals, [prices[i] for i in buy_signals], marker='^', color='green', label='Buy')
plt.scatter(sell_signals, [prices[i] for i in sell_signals], marker='v', color='red', label='Sell')

plt.title("Цена BTC и действия агента")
plt.xlabel("Шаг")
plt.ylabel("Цена BTC")
plt.legend()
plt.tight_layout()
plt.show()