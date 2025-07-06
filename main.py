import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from typing import Tuple, Dict, Any
import torch


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
    
    # Параметры устройства
    AUTO_DEVICE = True        # Автоматическое определение устройства
    FORCE_CPU = False         # Принудительное использование CPU
    DEVICE = "cpu"            # Будет установлено автоматически
    
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


def setup_device():
    """Настройка устройства для обучения (CPU/GPU)"""
    if Config.FORCE_CPU:
        device = "cpu"
        print("🔧 Принудительно используется CPU")
    elif Config.AUTO_DEVICE:
        if torch.cuda.is_available():
            device = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"🚀 Используется GPU: {gpu_name}")
                print(f"💾 Память GPU: {gpu_memory:.1f} GB")
            except:
                print("🚀 Используется GPU")
        else:
            device = "cpu"
            print("⚠️  GPU недоступно, используется CPU")
    else:
        device = Config.DEVICE
        print(f"🎯 Используется указанное устройство: {device}")
    
    Config.DEVICE = device
    return device


def check_gpu_requirements():
    """Проверка требований для GPU"""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "gpu_memory": None
    }
    
    if torch.cuda.is_available():
        try:
            info["current_device"] = torch.cuda.current_device()
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except:
            pass
        
    return info


class EarlyStoppingCallback(BaseCallback):
    """Callback для раннего прекращения обучения"""
    
    def __init__(self, threshold=-0.001, patience=5, verbose=1):
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience
        self.counter = 0

    def _on_step(self) -> bool:
        return True  # Упрощенная версия


class TradingEnv(gym.Env):
    """Торговое окружение для обучения с подкреплением"""
    
    def __init__(self, df, window_size=50, initial_balance=10000):
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

    def _reset_state(self):
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

    def reset(self, seed=None, options=None):
        """Сброс окружения"""
        self._reset_state()
        return self._get_observation(), {}

    def _get_observation(self):
        """Получение текущего наблюдения"""
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return obs.astype(np.float32)

    def _calculate_profit(self, current_price):
        """Расчет прибыли от позиции"""
        if self.position_size == 0 or self.entry_price == 0:
            return 0.0
        
        profit_per_coin = current_price - self.entry_price
        return (profit_per_coin * self.order_size_usd * self.position_size) / self.entry_price

    def step(self, action):
        """Выполнение шага в окружении"""
        reward = 0.0
        done = False

        current_price = self.df.iloc[self.current_step]['close']

        # Обработка действий
        if action == 1:  # Покупка
            if self.position_size < Config.POSITIONS_LIMIT:
                if self.position_size > 0:
                    self.entry_price = ((self.entry_price * self.position_size + current_price) 
                                      / (self.position_size + 1))
                else:
                    self.entry_price = current_price
                self.position_size += 1
                self.position = 1
                reward += Config.BUY_REWARD
                
        elif action == 2:  # Продажа
            if self.position_size > 0:
                profit_total = self._calculate_profit(current_price)
                self.balance += profit_total
                reward += profit_total
                if profit_total > 0:
                    reward += Config.PROFIT_BONUS
                self.trades.append(profit_total)
                self.position_size = 0
                self.position = 0
                self.entry_price = 0.0

        self.last_action = action
        self.current_step += 1

        # Проверка конца эпизода
        if self.current_step >= len(self.df) - 1:
            done = True
            # Закрываем открытую позицию
            if self.position_size > 0:
                final_profit = self._calculate_profit(current_price)
                self.balance += final_profit
                reward += final_profit
                self.trades.append(final_profit)

        return self._get_observation(), reward, done, False, {}

    def render(self):
        """Отображение текущего состояния"""
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}")


def load_and_prepare_data(file_path):
    """Загрузка и подготовка данных"""
    df = pd.read_csv(file_path)
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

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
    for col in cols_to_normalize:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[col] = (df[col] - mean_val) / std_val

    return df


def train_model(env):
    """Обучение модели PPO"""
    device = setup_device()
    vec_env = DummyVecEnv([lambda: env])
    
    print(f"\n🎯 Создание модели PPO на устройстве: {device}")
    model = PPO("MlpPolicy", env, verbose=1, device=device, ent_coef=Config.PPO_ENT_COEF)
    
    print(f"🎮 Начинаем обучение на {Config.TOTAL_TIMESTEPS} шагах...")
    model.learn(total_timesteps=Config.TOTAL_TIMESTEPS, callback=EarlyStoppingCallback())
    return model


def test_model(model, test_env, df):
    """Тестирование обученной модели с защитой от зависания"""
    obs, _ = test_env.reset()
    
    actual_balance = []
    prices = []
    actions = []
    
    max_steps = len(df) - test_env.window_size - 10  # Защита от зависания
    step_count = 0

    print(f"Начинаем тестирование (максимум {max_steps} шагов)...")
    
    while step_count < max_steps:
        try:
            # Получение действия с защитой от ошибок типа
            action_result = model.predict(obs, deterministic=True)
            action = int(action_result[0]) if isinstance(action_result[0], (np.ndarray, list)) else int(action_result[0])
            
            obs, reward, done, truncated, _ = test_env.step(action)
            step_count += 1

            if test_env.current_step >= len(df):
                print("Достигнут конец данных")
                break

            current_price = df.iloc[test_env.current_step]['close']
            
            # Расчет баланса
            if test_env.position_size > 0 and test_env.entry_price > 0:
                unrealized = test_env._calculate_profit(current_price)
                total_balance = test_env.balance + unrealized
            else:
                total_balance = test_env.balance

            actual_balance.append(total_balance)
            prices.append(current_price)
            actions.append(action)

            if done:
                print("Эпизод завершен")
                break
                
            # Прогресс
            if step_count % 5000 == 0:
                print(f"Тестирование: {step_count}/{max_steps} шагов")
                
        except Exception as e:
            print(f"Ошибка в тестировании на шаге {step_count}: {e}")
            break

    print(f"Тестирование завершено за {step_count} шагов")
    return actual_balance, prices, actions


def visualize_results(actual_balance, prices, actions):
    """Визуализация результатов"""
    if not actual_balance:
        print("Нет данных для визуализации")
        return
        
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
    print("🚀 Торговый алгоритм на базе RL")
    print("=" * 50)
    
    try:
        # Проверка системных требований
        gpu_info = check_gpu_requirements()
        print(f"📊 PyTorch версия: {gpu_info['torch_version']}")
        print(f"🔧 Устройств GPU: {gpu_info['device_count']}")
        
        # Загрузка и подготовка данных
        file_path = Config.DATA_FOLDER + Config.DATA_FILE
        print(f"\n📁 Загружаем данные из {file_path}")
        df = load_and_prepare_data(file_path)
        
        print(f"📈 Данные загружены: {len(df)} записей")
        print(f"📊 Колонки: {list(df.columns)}")

        # Создание окружения и обучение
        print("\n🎓 Начинаем обучение...")
        env = TradingEnv(df, initial_balance=Config.INITIAL_BALANCE)
        model = train_model(env)
        print("✅ Обучение завершено!")

        # Тестирование
        print("\n🧪 Тестирование модели...")
        test_env = TradingEnv(df, initial_balance=Config.INITIAL_BALANCE)
        actual_balance, prices, actions = test_model(model, test_env, df)

        # Результаты
        if actual_balance:
            initial_balance = Config.INITIAL_BALANCE
            final_balance = actual_balance[-1]
            profit_percentage = ((final_balance - initial_balance) / initial_balance) * 100
            
            print(f"\n📊 Результаты:")
            print(f"Начальный баланс: {initial_balance}")
            print(f"Финальный баланс: {final_balance:.2f}")
            print(f"Прибыль: {final_balance - initial_balance:.2f} ({profit_percentage:.2f}%)")
            print(f"Количество сделок: {len(test_env.trades)}")
            
            if test_env.trades:
                profitable_trades = sum(1 for trade in test_env.trades if trade > 0)
                win_rate = (profitable_trades / len(test_env.trades)) * 100
                print(f"Винрейт: {win_rate:.2f}%")

            # Визуализация
            print("\n📈 Создание графиков...")
            visualize_results(actual_balance, prices, actions)
        else:
            print("⚠️ Нет данных для анализа результатов")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()