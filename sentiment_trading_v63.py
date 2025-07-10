import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List
import warnings
import platform

warnings.filterwarnings('ignore')

# Автоматическое определение и настройка GPU поддержки
def setup_gpu_support():
    """Настройка GPU поддержки для разных платформ"""
    system = platform.system().lower()
    
    if system == "windows":
        # Поддержка DirectML для AMD GPU на Windows
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
                print(f"✅ DirectML найден - AMD GPU поддержка включена: {device}")
                return device
        except ImportError:
            pass
    
    # Проверка NVIDIA CUDA (работает на всех платформах)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Используется NVIDIA CUDA: {gpu_name}")
        print(f"   Устройство: {device}")
        if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
            print(f"   CUDA версия: {torch.version.cuda}")
        return device
    
    # Проверка AMD ROCm (Linux)
    if system == "linux":
        try:
            if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
                device = torch.device("cuda")  # ROCm использует CUDA API
                print(f"🚀 Используется AMD ROCm/HIP")
                print(f"   Устройство: {device}")
                return device
        except:
            pass
    
    # Fallback на CPU
    device = torch.device("cpu")
    print(f"💻 Используется CPU: {device}")
    if system == "linux":
        print("💡 Для GPU ускорения установите:")
        print("   NVIDIA: драйверы + CUDA toolkit")  
        print("   AMD: ROCm (https://rocmdocs.amd.com/)")
    elif system == "windows":
        print("💡 Для GPU ускорения установите:")
        print("   NVIDIA: драйверы + CUDA toolkit")
        print("   AMD: pip install torch-directml")
    
    return device

def get_gpu_memory_info(device):
    """Получение информации о памяти GPU"""
    if device and device.type == "cuda":
        try:
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"📊 GPU память: {allocated:.1f}GB использовано / {total_memory:.1f}GB всего")
            return total_memory, allocated
        except Exception:
            pass
    return None, None

"""
🚀 ТОРГОВАЯ СИСТЕМА V6.4 - ОСТОРОЖНЫЙ ТРЕЙДЕР
✅ ИСПРАВЛЕНА "БОЯЗНЬ ТОРГОВЛИ": Штраф за открытие сделки (ACTION_COST) значительно снижен,
   чтобы агент не боялся входить в потенциально прибыльные позиции.
✅ УСИЛЕНО "ЧУТЬЕ" В ПОЗИЦИИ: Коэффициент награды за удержание (SHAPING_REWARD_COEFF) увеличен.
   Теперь агент более мотивирован удерживать прибыльные сделки.
✅ ОПТИМИЗИРОВАНА ПОДАЧА ДАННЫХ (ЛУЧШАЯ ПРАКТИКА): Пространство наблюдений разделено на "рынок" (картинка для CNN)
   и "состояние агента" (вектор). Это позволяет нейросети эффективнее обрабатывать разные типы информации.
✅ Цель: Обучить агента, который активно, но осмысленно торгует, а не бездействует.
"""

# ### ИЗМЕНЕНИЕ 3: Новый экстрактор для смешанных данных (картинка + вектор) ###
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Определяем размерности для каждого входа
        image_space = observation_space['image']
        state_space = observation_space['state']
        
        n_input_channels = image_space.shape[0]

        # CNN для обработки "картинки" рынка
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # Вычисляем размер выхода CNN
        with torch.no_grad():
            sample_image = torch.as_tensor(image_space.sample()[None]).float()
            n_flatten = self.cnn(sample_image).shape[1]

        # Вычисляем общий размер признаков (выход CNN + размер вектора состояния)
        combined_features_size = n_flatten + state_space.shape[0]

        # Линейный слой для объединения признаков
        self.linear = nn.Sequential(
            nn.Linear(combined_features_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Разделяем наблюдения на картинку и состояние
        image_obs = observations['image']
        state_obs = observations['state']
        
        # Прогоняем через соответствующие сети
        cnn_output = self.cnn(image_obs)
        
        # Объединяем выходы
        combined_features = torch.cat([cnn_output, state_obs], dim=1)
        
        return self.linear(combined_features)


class TrendTraderConfig:
    INITIAL_BALANCE = 10000
    ORDER_SIZE_RATIO = 0.15
    ATR_SL_MULTIPLIER = 3.5
    ATR_TP_MULTIPLIER = 6.0
    TRANSACTION_FEE = 0.001
    
    ### ИЗМЕНЕНИЕ 1: Снижаем барьер для входа ###
    # Штраф теперь очень мал, просто чтобы слегка наказать за беспорядочные действия.
    # Основной "штраф" - это реальная комиссия.
    ACTION_COST = 0.001 
    
    WINDOW_SIZE = 64
    # Уменьшено для быстрого тестирования на GPU. Увеличьте до 1000000+ для полного обучения
    TOTAL_TIMESTEPS = 500000  
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01
    GAMMA = 0.999
    TREND_PROFIT_BONUS = 0.1
    
    ### ИЗМЕНЕНИЕ 2: Усиливаем "чутье" ###
    # Коэффициент увеличен, чтобы изменения в PnL были более значимы для агента.
    SHAPING_REWARD_COEFF = 0.05

class SimpleDataLoader:
    # ... (Этот класс без изменений, я его сверну для краткости)
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
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        features = pd.DataFrame(index=df.index)
        features['price_norm'] = df['close'] / df['sma_long']
        features['ema_spread'] = (df['ema_fast'] - df['ema_slow']) / df['close']
        features['rsi_norm'] = (df['rsi'] - 50) / 50
        features['macd_hist_norm'] = (df['macd'] - df['macd_signal']) / df['close']
        features['trend_signal'] = np.sign(df['close'] - df['sma_long'])
        features['atr_norm'] = df['atr_value'] / df['close']
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
        
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
        
        # ### ИЗМЕНЕНИЕ 3 (продолжение): Используем Dict-пространство ###
        self.observation_space = spaces.Dict({
            # "Картинка" рыночных данных для CNN
            "image": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(1, self.cfg.WINDOW_SIZE, self.features_df.shape[1]), 
                dtype=np.float32
            ),
            # "Вектор" состояния агента
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })
        self._reset_state()
    
    def _reset_state(self):
        # ... (этот метод без изменений)
        self.balance = self.cfg.INITIAL_BALANCE
        self.equity = self.cfg.INITIAL_BALANCE
        self.current_step = self.cfg.WINDOW_SIZE
        self.position_amount = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.entry_step = 0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.last_unrealized_pnl = 0.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        # Получаем срез рыночных данных (картинка)
        market_obs = self.features_df.iloc[self.current_step - self.cfg.WINDOW_SIZE : self.current_step].values
        image_obs = np.expand_dims(market_obs, axis=0).astype(np.float32) # Добавляем channel dimension

        # Формируем вектор состояния
        position_active = 1.0 if self.position_amount > 0 else -1.0 # -1 для "нет позиции", 1 для "в позиции"
        if self.position_amount > 0:
            current_price = self._get_current_price()
            unrealized_pnl_norm = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl_norm = 0.0
            
        state_obs = np.array([position_active, unrealized_pnl_norm], dtype=np.float32)

        return {"image": image_obs, "state": state_obs}

    def _get_current_price(self) -> float:
        return self.prices_df.iloc[self.current_step]['close']

    def _get_current_atr(self) -> float:
        return self.prices_df.iloc[self.current_step]['atr_value']

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # ... (логика step почти не изменилась, только награда за действие)
        current_price = self._get_current_price()
        reward = 0.0 
        shaping_reward = 0.0
        done = False

        if action == 1 and self.position_amount == 0:
            self._open_position(current_price)
            reward -= self.cfg.ACTION_COST # Применяем наш новый, маленький штраф
        elif action == 2 and self.position_amount > 0:
            reward = self._close_position(current_price)
            reward -= self.cfg.ACTION_COST
        
        if self.position_amount > 0:
            low_price = self.prices_df.iloc[self.current_step]['low']
            high_price = self.prices_df.iloc[self.current_step]['high']
            
            if low_price <= self.stop_loss_price:
                reward = self._close_position(self.stop_loss_price)
            elif high_price >= self.take_profit_price:
                reward = self._close_position(self.take_profit_price)
            else:
                current_unrealized_pnl = (current_price - self.entry_price) * self.position_amount
                pnl_change = current_unrealized_pnl - self.last_unrealized_pnl
                shaping_reward = pnl_change * self.cfg.SHAPING_REWARD_COEFF
                self.last_unrealized_pnl = current_unrealized_pnl

        self.current_step += 1
        current_unrealized_pnl_for_equity = (self._get_current_price() - self.entry_price) * self.position_amount if self.position_amount > 0 else 0
        self.equity = self.balance + current_unrealized_pnl_for_equity
        
        if self.current_step >= len(self.features_df) - 1 or self.equity <= 0:
            if self.position_amount > 0:
                reward = self._close_position(self._get_current_price())
            done = True
        
        total_reward = reward + shaping_reward
        
        info = {'equity': self.equity}
        terminated = done
        truncated = False
        
        return self._get_observation(), total_reward, terminated, truncated, info

    def _open_position(self, price: float):
        # ... (этот метод без изменений)
        self.entry_step = self.current_step
        current_atr = self._get_current_atr()
        self.stop_loss_price = price - (current_atr * self.cfg.ATR_SL_MULTIPLIER)
        self.take_profit_price = price + (current_atr * self.cfg.ATR_TP_MULTIPLIER)
        order_size_usd = self.balance * self.cfg.ORDER_SIZE_RATIO
        if self.balance > 0 and order_size_usd > 0:
            fee = order_size_usd * self.cfg.TRANSACTION_FEE
            self.balance -= (order_size_usd + fee)
            self.position_amount = order_size_usd / price
            self.entry_price = price
            self.last_unrealized_pnl = 0.0

    def _close_position(self, price: float) -> float:
        # ... (этот метод без изменений)
        close_value = self.position_amount * price
        fee = close_value * self.cfg.TRANSACTION_FEE
        self.balance += (close_value - fee)
        entry_value = self.position_amount * self.entry_price
        entry_fee = entry_value * self.cfg.TRANSACTION_FEE
        realized_pnl = (close_value - fee) - (entry_value + entry_fee)
        self.trades.append(realized_pnl)
        reward = realized_pnl / self.cfg.INITIAL_BALANCE
        if realized_pnl > 0:
            trend_at_entry = self.features_df.iloc[self.entry_step]['trend_signal']
            if trend_at_entry > 0:
                reward += self.cfg.TREND_PROFIT_BONUS
        
        self.position_amount = 0.0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.last_unrealized_pnl = 0.0
        return reward


def main():
    print("🚀 СИСТЕМА V6.4 (Осторожный трейдер) - ЗАПУСК")
    
    # 🎯 ЭТАП 1: НАСТРОЙКА GPU ПОДДЕРЖКИ
    print("\n🎯 ЭТАП 1: ИНИЦИАЛИЗАЦИЯ УСТРОЙСТВА...")
    device = setup_gpu_support()
    
    # 📊 ЭТАП 2: ЗАГРУЗКА ДАННЫХ
    print("\n📊 ЭТАП 2: ЗАГРУЗКА ДАННЫХ...")
    data_loader = SimpleDataLoader("data/BTC_5_96w.csv")
    features_df, prices_df = data_loader.load_and_prepare_data()

    train_split_idx = int(len(features_df) * 0.8)
    train_features, train_prices = features_df.iloc[:train_split_idx], prices_df.iloc[:train_split_idx]
    test_features, test_prices = features_df.iloc[train_split_idx:], prices_df.iloc[train_split_idx:]
    print(f"✅ Данные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
    # 🏗️ ЭТАП 3: СОЗДАНИЕ ОКРУЖЕНИЯ И МОДЕЛИ
    print(f"\n🏗️ ЭТАП 3: СОЗДАНИЕ МОДЕЛИ НА УСТРОЙСТВЕ {device}...")
    vec_env = DummyVecEnv([lambda: TradingEnv(train_features, train_prices)])

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256), 
        net_arch=dict(pi=[256, 128], vf=[256, 128])
    )

    model = PPO('MultiInputPolicy', vec_env, policy_kwargs=policy_kwargs,
                learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF,
                n_steps=2048, batch_size=64, gamma=TrendTraderConfig.GAMMA,
                verbose=1, device=device)
    
    # Информация о памяти GPU (если доступно)
    if device.type == "cuda":
        get_gpu_memory_info(device)  

    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ ОСТОРОЖНОГО ТРЕЙДЕРА...")
    model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS)
    
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    test_env = TradingEnv(test_features, test_prices)
    obs, info = test_env.reset()
    
    equity_history = [test_env.equity]
    price_history = [test_env._get_current_price()]
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        equity_history.append(info['equity'])
        try:
            price_history.append(test_env._get_current_price())
        except IndexError:
            price_history.append(price_history[-1])
        done = terminated or truncated
            
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
    
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(15, 7))
    plt.title(f'Результаты на тестовой выборке (V6.4 - Осторожный трейдер)\nReturn: {total_return:.2f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}%')
    ax1 = plt.gca()
    ax1.plot(equity_history, label='Equity', color='royalblue', linewidth=2)
    ax1.set_xlabel('Шаги'); ax1.set_ylabel('Equity ($)', color='royalblue');
    ax2 = ax1.twinx()
    ax2.plot(price_history, label='Цена BTC', color='darkorange', alpha=0.6)
    ax2.set_ylabel('Цена ($)', color='darkorange');
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()