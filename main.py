import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from typing import Tuple, Dict, Any


# === КОНСТАНТЫ ===
class Config:
    # Файлы и пути
    DATA_FOLDER = "data/"
    DATA_FILE = "BTC_5_96w.csv"
    
    # Параметры окружения
    WINDOW_SIZE = 50
    INITIAL_BALANCE = 10000
    POSITIONS_LIMIT = 3
    PASSIVITY_THRESHOLD = 100
    
    # Параметры вознаграждений
    BUY_REWARD = 0.1
    SELL_PENALTY = 0.05
    INVALID_ACTION_PENALTY = 0.5
    PROFIT_BONUS = 2.0
    UNREALIZED_PROFIT_MULTIPLIER = 0.05
    REPETITIVE_ACTION_PENALTY = 0.01
    PASSIVITY_PENALTY = 0.1
    FINAL_PROFIT_MULTIPLIER = 0.01
    LOSS_PENALTY = 50
    
    # Технические индикаторы
    EMA_FAST_SPAN = 12
    EMA_SLOW_SPAN = 26
    RSI_WINDOW = 14
    
    # Параметры обучения
    TOTAL_TIMESTEPS = 50000
    PPO_ENT_COEF = 0.05
    EARLY_STOPPING_THRESHOLD = -0.001
    EARLY_STOPPING_PATIENCE = 5
    
    # Визуализация
    FIGURE_SIZE = (14, 6)


class EarlyStoppingCallback(BaseCallback):
    """Callback для раннего прекращения обучения"""
    
    def __init__(self, threshold: float = Config.EARLY_STOPPING_THRESHOLD, 
                 patience: int = Config.EARLY_STOPPING_PATIENCE, verbose: int = 1):
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience
        self.counter = 0

    def _on_step(self) -> bool:
        # Получаем explained_variance из модели
        if 'explained_variance' in self.locals.get('infos', [{}])[0]:
            ev = self.locals['infos'][0]['explained_variance']
        else:
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
    """Торговое окружение для обучения с подкреплением"""
    
    def __init__(self, df: pd.DataFrame, window_size: int = Config.WINDOW_SIZE, 
                 initial_balance: float = Config.INITIAL_BALANCE):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Пространства действий и наблюдений
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, df.shape[1]), dtype=np.float32
        )

        # Инициализация состояния
        self._reset_state()

    def _reset_state(self) -> None:
        """Сброс состояния окружения"""
        self.balance = self.initial_balance
        self.entry_price = 0.0
        self.position = 0
        self.position_size = 0
        self.current_step = self.window_size
        self.trades = []
        self.last_action = None
        self.order_size_usd = self.initial_balance / Config.POSITIONS_LIMIT
        self.wait_counter = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Сброс окружения"""
        self._reset_state()
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Получение текущего наблюдения"""
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return obs.astype(np.float32)

    def _calculate_profit(self, current_price: float) -> float:
        """Расчет прибыли от позиции"""
        if self.position_size == 0 or self.entry_price == 0:
            return 0.0
        
        profit_per_coin = current_price - self.entry_price
        return (profit_per_coin * self.order_size_usd * self.position_size) / self.entry_price

    def _execute_buy_action(self, current_price: float) -> float:
        """Выполнение покупки"""
        reward = 0.0
        
        if self.position_size < Config.POSITIONS_LIMIT:
            # Обновляем среднюю цену входа
            if self.position_size > 0:
                self.entry_price = ((self.entry_price * self.position_size + current_price) 
                                  / (self.position_size + 1))
            else:
                self.entry_price = current_price

            self.position_size += 1
            self.position = 1
            reward += Config.BUY_REWARD
            self.wait_counter = 0
        else:
            reward -= Config.SELL_PENALTY

        return reward

    def _execute_sell_action(self, current_price: float) -> float:
        """Выполнение продажи"""
        reward = 0.0
        
        if self.position_size > 0:
            profit_total = self._calculate_profit(current_price)
            
            self.balance += profit_total
            reward += profit_total

            if profit_total > 0:
                reward += Config.PROFIT_BONUS

            self.trades.append(profit_total)
            self._close_position()
        else:
            reward -= Config.SELL_PENALTY

        return reward

    def _close_position(self) -> None:
        """Закрытие позиции"""
        self.position_size = 0
        self.position = 0
        self.entry_price = 0.0
        self.wait_counter = 0

    def _calculate_final_reward(self, current_price: float) -> float:
        """Расчет финального вознаграждения в конце эпизода"""
        reward = 0.0
        
        # Закрываем открытую позицию, если есть
        if self.position_size > 0:
            final_profit = self._calculate_profit(current_price)
            self.balance += final_profit
            reward += final_profit
            self.trades.append(final_profit)
            self._close_position()

        # Награда за финальный результат
        total_profit = self.balance - self.initial_balance
        reward += total_profit * Config.FINAL_PROFIT_MULTIPLIER

        # Штраф за убыток
        if self.balance < self.initial_balance:
            reward -= Config.LOSS_PENALTY

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Выполнение шага в окружении"""
        reward = 0.0
        done = False

        current_price = self.df.iloc[self.current_step]['close']

        # Штраф за продажу без позиции
        if action == 2 and self.position_size == 0:
            reward -= Config.INVALID_ACTION_PENALTY
            return self._get_observation(), reward, False, False, {}

        # Выполнение действий
        if action == 1:  # Покупка
            reward += self._execute_buy_action(current_price)
        elif action == 2:  # Продажа
            reward += self._execute_sell_action(current_price)

        # Нереализованная прибыль
        if self.position_size > 0 and self.entry_price > 0:
            unrealized = self._calculate_profit(current_price)
            reward += Config.UNREALIZED_PROFIT_MULTIPLIER * unrealized

        # Штраф за повторные действия
        if self.last_action == action:
            reward -= Config.REPETITIVE_ACTION_PENALTY

        # Штраф за пассивность
        self.wait_counter += 1
        if self.wait_counter > Config.PASSIVITY_THRESHOLD:
            reward -= Config.PASSIVITY_PENALTY

        self.last_action = action
        self.current_step += 1

        # Проверка конца эпизода
        if self.current_step >= len(self.df) - 1:
            done = True
            reward += self._calculate_final_reward(current_price)

        return self._get_observation(), reward, done, False, {}

    def render(self) -> None:
        """Отображение текущего состояния"""
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}")


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Загрузка и подготовка данных"""
    df = pd.read_csv(file_path)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Технические индикаторы
    df['ema_fast'] = df['close'].ewm(span=Config.EMA_FAST_SPAN, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=Config.EMA_SLOW_SPAN, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=Config.RSI_WINDOW).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=Config.RSI_WINDOW).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Очистка NaN
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Нормализация
    cols_to_normalize = ['open', 'high', 'low', 'close', 'volume', 'ema_fast', 'ema_slow']
    df[cols_to_normalize] = ((df[cols_to_normalize] - df[cols_to_normalize].mean()) 
                            / df[cols_to_normalize].std())

    return df


def train_model(env: TradingEnv) -> PPO:
    """Обучение модели PPO"""
    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, verbose=1, device="cpu", ent_coef=Config.PPO_ENT_COEF)
    model.learn(total_timesteps=Config.TOTAL_TIMESTEPS, callback=EarlyStoppingCallback())
    return model


def test_model(model: PPO, test_env: TradingEnv, df: pd.DataFrame) -> Tuple[list, list, list]:
    """Тестирование обученной модели"""
    obs, _ = test_env.reset()
    
    actual_balance = []
    prices = []
    actions = []

    while True:
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, done, truncated, _ = test_env.step(action)

        step = test_env.current_step
        if step >= len(df):
            break

        current_price = df.iloc[step]['close']
        
        # Расчет баланса с учетом нереализованной прибыли
        if test_env.position_size > 0 and test_env.entry_price > 0:
            unrealized = test_env._calculate_profit(current_price)
            total_balance = test_env.balance + unrealized
        else:
            total_balance = test_env.balance

        actual_balance.append(total_balance)
        prices.append(current_price)
        actions.append(action)

        if done:
            break

    return actual_balance, prices, actions


def visualize_results(actual_balance: list, prices: list, actions: list) -> None:
    """Визуализация результатов"""
    plt.figure(figsize=Config.FIGURE_SIZE)

    # График баланса
    plt.subplot(2, 1, 1)
    plt.plot(actual_balance, label='Agent Balance', linewidth=2)
    plt.title("Баланс агента", fontsize=14)
    plt.xlabel("Шаг")
    plt.ylabel("Баланс (USDT)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График цены и действий
    plt.subplot(2, 1, 2)
    plt.plot(prices, label='BTC Price', alpha=0.7, linewidth=1)
    
    buy_signals = [i for i, a in enumerate(actions) if a == 1]
    sell_signals = [i for i, a in enumerate(actions) if a == 2]
    
    if buy_signals:
        plt.scatter(buy_signals, [prices[i] for i in buy_signals], 
                   marker='^', color='green', label='Buy', s=50, alpha=0.8)
    if sell_signals:
        plt.scatter(sell_signals, [prices[i] for i in sell_signals], 
                   marker='v', color='red', label='Sell', s=50, alpha=0.8)

    plt.title("Цена BTC и действия агента", fontsize=14)
    plt.xlabel("Шаг")
    plt.ylabel("Цена BTC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Основная функция"""
    # Загрузка и подготовка данных
    file_path = Config.DATA_FOLDER + Config.DATA_FILE
    df = load_and_prepare_data(file_path)
    
    print(f"Данные загружены: {len(df)} записей")
    print(f"Колонки: {list(df.columns)}")

    # Создание окружения и обучение
    print("\nНачинаем обучение...")
    env = TradingEnv(df, initial_balance=Config.INITIAL_BALANCE)
    model = train_model(env)
    print("Обучение завершено!")

    # Тестирование
    print("\nТестирование модели...")
    test_env = TradingEnv(df, initial_balance=Config.INITIAL_BALANCE)
    actual_balance, prices, actions = test_model(model, test_env, df)

    # Результаты
    initial_balance = Config.INITIAL_BALANCE
    final_balance = actual_balance[-1] if actual_balance else initial_balance
    profit_percentage = ((final_balance - initial_balance) / initial_balance) * 100
    
    print(f"\nРезультаты:")
    print(f"Начальный баланс: {initial_balance}")
    print(f"Финальный баланс: {final_balance:.2f}")
    print(f"Прибыль: {final_balance - initial_balance:.2f} ({profit_percentage:.2f}%)")
    print(f"Количество сделок: {len(test_env.trades)}")
    
    if test_env.trades:
        profitable_trades = sum(1 for trade in test_env.trades if trade > 0)
        win_rate = (profitable_trades / len(test_env.trades)) * 100
        print(f"Винрейт: {win_rate:.2f}%")

    # Визуализация
    visualize_results(actual_balance, prices, actions)


if __name__ == "__main__":
    main()