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

# Функция setup_gpu_support остается без изменений
def setup_gpu_support():
    """Настройка GPU поддержки для разных платформ"""
    system = platform.system().lower()
    
    if system == "windows":
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
                print(f"✅ DirectML найден - AMD GPU поддержка включена: {device}")
                return device
        except ImportError:
            pass
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Используется NVIDIA CUDA: {gpu_name}")
        return device
    
    device = torch.device("cpu")
    print(f"💻 Используется CPU: {device}")
    return device

# Функция get_gpu_memory_info остается без изменений
def get_gpu_memory_info(device):
    """Получение информации о памяти GPU"""
    if device and device.type == "cuda":
        try:
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            print(f"📊 GPU память: {allocated:.1f}GB использовано / {total_memory:.1f}GB всего")
        except Exception:
            pass

"""
🚀 ТОРГОВАЯ СИСТЕМА V7.0 - СТРАТЕГ
✅ ЦЕЛЬ: Качество сделок, а не количество. Учим агента терпению и точности.
✅ ПЕРЕРАБОТАНА СИСТЕМА НАГРАД: Убран искусственный штраф за действие. Введены четкие бонусы/штрафы
   за результат сделки и небольшой "штраф за бездействие", чтобы мотивировать поиск хороших входов.
✅ РАСШИРЕНЫ ВОЗМОЖНОСТИ: Добавлена возможность открывать короткие позиции (шорты),
   чтобы зарабатывать и на падении рынка.
✅ УЛУЧШЕНА ОСВЕДОМЛЕННОСТЬ АГЕНТА: В пространство наблюдений добавлена информация о
   длительности текущей сделки.
"""

# CustomCombinedExtractor остается без изменений
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        image_space = observation_space['image']
        state_space = observation_space['state']
        n_input_channels = image_space.shape[0]
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
        with torch.no_grad():
            sample_image = torch.as_tensor(image_space.sample()[None]).float()
            n_flatten = self.cnn(sample_image).shape[1]
        combined_features_size = n_flatten + state_space.shape[0]
        self.linear = nn.Sequential(
            nn.Linear(combined_features_size, features_dim),
            nn.ReLU()
        )
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_obs = observations['image']
        state_obs = observations['state']
        cnn_output = self.cnn(image_obs)
        combined_features = torch.cat([cnn_output, state_obs], dim=1)
        return self.linear(combined_features)


class TrendTraderConfig:
    INITIAL_BALANCE = 10000
    TRANSACTION_FEE = 0.001
    WINDOW_SIZE = 64
    
    # ### ИЗМЕНЕНИЕ V7.0: Настройка гиперпараметров "Стратега" ###
    ORDER_SIZE_RATIO = 0.10  # Снижаем агрессивность для уменьшения риска
    ATR_SL_MULTIPLIER = 3.0  # Немного более жесткий стоп-лосс
    ATR_TP_MULTIPLIER = 5.0  # Соответствующий тейк-профит (R:R ~ 1:1.67)
    
    # Обучение
    TOTAL_TIMESTEPS = 1000000  # Увеличиваем время обучения для лучшего исследования
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01
    GAMMA = 0.99
    
    # Новая система наград
    WIN_BONUS = 0.5         # Большой бонус за прибыльную сделку
    LOSS_PENALTY = -0.5     # Большой штраф за убыточную сделку
    HOLDING_PENALTY = 0.0001 # Небольшой штраф за каждый шаг без позиции (стимул искать сделку)
    PROFIT_HOLDING_REWARD = 0.01 # Коэффициент для поощрения удержания прибыльной позиции
    MAX_TRADE_DURATION = 288 # Макс. длительность сделки в шагах (для нормализации). 288*5мин = 1 сутки

# SimpleDataLoader остается без изменений
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
        
        # ### ИЗМЕНЕНИЕ V7.0: Новое пространство действий с шортами ###
        # 0: Hold, 1: Buy (Long), 2: Sell (Short), 3: Close Position
        self.action_space = spaces.Discrete(4)
        
        # ### ИЗМЕНЕНИЕ V7.0: Новое пространство наблюдений ###
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(1, self.cfg.WINDOW_SIZE, self.features_df.shape[1]), 
                dtype=np.float32
            ),
            # Вектор состояния: [тип позиции, нереализ. PnL, длительность позиции]
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        self._reset_state()
    
    def _reset_state(self):
        self.balance = self.cfg.INITIAL_BALANCE
        self.equity = self.cfg.INITIAL_BALANCE
        self.current_step = self.cfg.WINDOW_SIZE
        self.position_amount = 0.0 # >0 для Long, <0 для Short
        self.entry_price = 0.0
        self.entry_step = 0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.trades = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        market_obs = self.features_df.iloc[self.current_step - self.cfg.WINDOW_SIZE : self.current_step].values
        image_obs = np.expand_dims(market_obs, axis=0).astype(np.float32)

        position_type = np.sign(self.position_amount) # 1.0 для Long, -1.0 для Short, 0.0 для Flat
        
        if self.position_amount != 0:
            current_price = self._get_current_price()
            pnl = (current_price - self.entry_price) * self.position_amount
            unrealized_pnl_norm = pnl / (self.entry_price * abs(self.position_amount))
            duration_norm = (self.current_step - self.entry_step) / self.cfg.MAX_TRADE_DURATION
        else:
            unrealized_pnl_norm = 0.0
            duration_norm = 0.0
            
        state_obs = np.array([position_type, unrealized_pnl_norm, duration_norm], dtype=np.float32)

        return {"image": image_obs, "state": state_obs}

    def _get_current_price(self) -> float:
        return self.prices_df.iloc[self.current_step]['close']

    def _get_current_atr(self) -> float:
        return self.prices_df.iloc[self.current_step]['atr_value']

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()
        reward = 0.0
        done = False

        # ### ИЗМЕНЕНИЕ V7.0: Новая логика действий и наград ###
        # --- Логика действий ---
        if self.position_amount == 0: # Если мы вне позиции
            if action == 1: # Открыть Long
                self._open_position(current_price, is_long=True)
            elif action == 2: # Открыть Short
                self._open_position(current_price, is_long=False)
            else: # Держать (Hold)
                reward -= self.cfg.HOLDING_PENALTY # Наказываем за бездействие
        
        elif self.position_amount != 0: # Если мы в позиции
            if action == 3: # Закрыть текущую позицию
                reward = self._close_position(current_price)
            # Игнорируем действия "Buy" или "Sell", если уже в позиции
        
        # --- Логика SL/TP и наград за удержание ---
        if self.position_amount != 0:
            low_price = self.prices_df.iloc[self.current_step]['low']
            high_price = self.prices_df.iloc[self.current_step]['high']
            
            is_long = self.position_amount > 0
            
            # Проверка Stop Loss / Take Profit
            if (is_long and low_price <= self.stop_loss_price) or (not is_long and high_price >= self.stop_loss_price):
                reward = self._close_position(self.stop_loss_price)
            elif (is_long and high_price >= self.take_profit_price) or (not is_long and low_price <= self.take_profit_price):
                reward = self._close_position(self.take_profit_price)
            else:
                # Награда за удержание прибыльной позиции
                pnl = (current_price - self.entry_price) * self.position_amount
                if pnl > 0:
                    reward += self.cfg.PROFIT_HOLDING_REWARD

        # --- Обновление шага и проверка завершения ---
        self.current_step += 1
        current_unrealized_pnl = (self._get_current_price() - self.entry_price) * self.position_amount
        self.equity = self.balance + current_unrealized_pnl
        
        if self.current_step >= len(self.features_df) - 1 or self.equity <= 0:
            if self.position_amount != 0:
                reward = self._close_position(self._get_current_price())
            done = True
        
        info = {'equity': self.equity}
        return self._get_observation(), reward, done, False, info

    def _open_position(self, price: float, is_long: bool):
        self.entry_step = self.current_step
        current_atr = self._get_current_atr()
        
        if is_long:
            self.stop_loss_price = price - (current_atr * self.cfg.ATR_SL_MULTIPLIER)
            self.take_profit_price = price + (current_atr * self.cfg.ATR_TP_MULTIPLIER)
        else: # Short
            self.stop_loss_price = price + (current_atr * self.cfg.ATR_SL_MULTIPLIER)
            self.take_profit_price = price - (current_atr * self.cfg.ATR_TP_MULTIPLIER)

        order_size_usd = self.balance * self.cfg.ORDER_SIZE_RATIO
        if self.balance > 0 and order_size_usd > 0:
            fee = order_size_usd * self.cfg.TRANSACTION_FEE
            self.balance -= (order_size_usd + fee)
            position_size = order_size_usd / price
            self.position_amount = position_size if is_long else -position_size
            self.entry_price = price

    def _close_position(self, price: float) -> float:
        is_long = self.position_amount > 0
        position_size = abs(self.position_amount)
        close_value = position_size * price
        fee = close_value * self.cfg.TRANSACTION_FEE
        
        entry_value = position_size * self.entry_price
        entry_fee = entry_value * self.cfg.TRANSACTION_FEE
        
        if is_long:
            realized_pnl = (close_value - fee) - (entry_value + entry_fee)
        else: # Short
            realized_pnl = (entry_value - entry_fee) - (close_value + fee)

        self.balance += entry_value + realized_pnl # Возвращаем тело сделки и PnL
        self.trades.append(realized_pnl)
        
        # ### ИЗМЕНЕНИЕ V7.0: Награда с бонусом/штрафом ###
        reward = 0
        if realized_pnl > 0:
            reward += self.cfg.WIN_BONUS
        else:
            reward += self.cfg.LOSS_PENALTY
            
        self.position_amount = 0.0
        self.entry_price = 0.0
        return reward


def main():
    print("🚀 СИСТЕМА V7.0 (Стратег) - ЗАПУСК")
    device = setup_gpu_support()
    get_gpu_memory_info(device)
    
    data_loader = SimpleDataLoader("data/BTC_5_96w.csv")
    features_df, prices_df = data_loader.load_and_prepare_data()

    train_split_idx = int(len(features_df) * 0.8)
    train_features, train_prices = features_df.iloc[:train_split_idx], prices_df.iloc[:train_split_idx]
    test_features, test_prices = features_df.iloc[train_split_idx:], prices_df.iloc[train_split_idx:]
    print(f"✅ Данные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
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

    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ 'СТРАТЕГА' (может занять больше времени)...")
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
            
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ 'СТРАТЕГА'")
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
    plt.title(f'Результаты на тестовой выборке (V7.0 - Стратег)\nReturn: {total_return:.2f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}%')
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
