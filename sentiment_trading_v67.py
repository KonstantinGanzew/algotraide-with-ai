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

# Функции setup_gpu_support и get_gpu_memory_info без изменений
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
        except ImportError: pass
    if torch.cuda.is_available():
        device = torch.device("cuda"); gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Используется NVIDIA CUDA: {gpu_name}"); return device
    device = torch.device("cpu"); print(f"💻 Используется CPU: {device}"); return device

def get_gpu_memory_info(device):
    """Получение информации о памяти GPU"""
    if device and device.type == "cuda":
        try:
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            print(f"📊 GPU память: {allocated:.1f}GB использовано / {total_memory:.1f}GB всего")
        except Exception: pass

"""
🚀 ТОРГОВАЯ СИСТЕМА V10.0 - ТРЕНДОВЫЙ СПЕЦИАЛИСТ
✅ ЦЕЛЬ: Повысить качество сделок за счет торговли строго по тренду.
✅ ФИЛЬТР ТРЕНДА: Введена проверка по SMA 200. Лонги разрешены только выше SMA,
   шорты - только ниже. Это должно сократить количество сделок и повысить Win Rate.
✅ УЛУЧШЕННЫЙ STATE: Нереализованная прибыль в векторе состояния теперь нормализуется
   на ATR, что дает агенту лучшее понимание значимости PnL в контексте текущей волатильности.
"""

# CustomCombinedExtractor остается без изменений
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        image_space = observation_space['image']
        state_space = observation_space['state']
        n_input_channels = image_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(image_space.sample()[None]).float()).shape[1]
        combined_features_size = n_flatten + state_space.shape[0]
        self.linear = nn.Sequential(nn.Linear(combined_features_size, features_dim), nn.ReLU())

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_output = self.cnn(observations['image'])
        combined_features = torch.cat([cnn_output, observations['state']], dim=1)
        return self.linear(combined_features)


class TrendTraderConfig:
    INITIAL_BALANCE = 10000
    TRANSACTION_FEE = 0.001
    WINDOW_SIZE = 64
    
    ORDER_SIZE_RATIO = 0.10
    ATR_SL_MULTIPLIER = 2.0
    ATR_TP_MULTIPLIER = 4.0
    
    # Обучение
    TOTAL_TIMESTEPS = 1000000
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01
    GAMMA = 0.99
    MAX_TRADE_DURATION = 288 

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
        
        ### ИЗМЕНЕНИЕ V10.0: Добавляем sma_long в prices_df для фильтра тренда ###
        prices_columns = ['timestamp', 'open', 'high', 'low', 'close', 'atr_value', 'sma_long']
        
        print(f"✅ Подготовлено данных: {len(features)} записей, {len(features.columns)} признаков.")
        return features, prices_df[prices_columns]

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, features_df: pd.DataFrame, prices_df: pd.DataFrame):
        super().__init__()
        self.features_df = features_df.reset_index(drop=True)
        self.prices_df = prices_df.reset_index(drop=True)
        self.cfg = TrendTraderConfig()
        
        # 0: Short, 1: Flat, 2: Long
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.cfg.WINDOW_SIZE, self.features_df.shape[1]), dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        self._reset_state()
    
    def _reset_state(self):
        self.balance = self.cfg.INITIAL_BALANCE
        self.equity = self.cfg.INITIAL_BALANCE
        self.current_step = self.cfg.WINDOW_SIZE
        self.position_amount = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.trades = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._reset_state(); return self._get_observation(), {}
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        market_obs = self.features_df.iloc[self.current_step - self.cfg.WINDOW_SIZE : self.current_step].values
        image_obs = np.expand_dims(market_obs, axis=0).astype(np.float32)
        position_type = np.sign(self.position_amount)
        
        if self.position_amount != 0:
            current_price = self._get_current_price()
            pnl = (current_price - self.entry_price) * self.position_amount
            
            ### ИЗМЕНЕНИЕ V10.0: Нормализация PnL на волатильность (ATR) ###
            current_atr = self._get_current_atr()
            if current_atr > 0:
                unrealized_pnl_norm = pnl / (abs(self.position_amount) * current_atr)
            else:
                unrealized_pnl_norm = 0.0
                
            duration_norm = (self.current_step - self.entry_step) / self.cfg.MAX_TRADE_DURATION
        else:
            unrealized_pnl_norm = 0.0; duration_norm = 0.0
            
        state_obs = np.array([position_type, unrealized_pnl_norm, duration_norm], dtype=np.float32)
        return {"image": image_obs, "state": state_obs}

    def _get_current_price(self) -> float: return self.prices_df.iloc[self.current_step]['close']
    def _get_current_atr(self) -> float: return self.prices_df.iloc[self.current_step]['atr_value']
    
    ### ИЗМЕНЕНИЕ V10.0: Метод для проверки фильтра тренда ###
    def _check_trend_filter(self, price: float, is_long: bool) -> bool:
        """Проверяет, соответствует ли желаемое действие глобальному тренду."""
        sma_long = self.prices_df.iloc[self.current_step]['sma_long']
        if is_long:
            return price > sma_long  # Лонг только если цена выше SMA 200
        else: # is_short
            return price < sma_long  # Шорт только если цена ниже SMA 200

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()
        reward = 0.0
        done = False

        if self.position_amount != 0:
            low_price, high_price = self.prices_df.iloc[self.current_step]['low'], self.prices_df.iloc[self.current_step]['high']
            is_long = self.position_amount > 0
            if (is_long and low_price <= self.stop_loss_price) or (not is_long and high_price >= self.stop_loss_price):
                reward = self._close_position(self.stop_loss_price)
            elif (is_long and high_price >= self.take_profit_price) or (not is_long and low_price <= self.take_profit_price):
                reward = self._close_position(self.take_profit_price)
        
        current_position_type = np.sign(self.position_amount)
        desired_position_type = action - 1 

        if current_position_type != desired_position_type:
            if current_position_type != 0:
                reward += self._close_position(current_price)
            if desired_position_type != 0:
                is_long = (desired_position_type == 1)
                ### ИЗМЕНЕНИЕ V10.0: Применяем фильтр тренда перед открытием позиции ###
                if self._check_trend_filter(current_price, is_long):
                    self._open_position(current_price, is_long=is_long)

        self.current_step += 1
        current_unrealized_pnl = (self._get_current_price() - self.entry_price) * self.position_amount if self.position_amount != 0 else 0
        self.equity = self.balance + current_unrealized_pnl
        if self.current_step >= len(self.features_df) - 1 or self.equity <= 0:
            if self.position_amount != 0:
                reward += self._close_position(self._get_current_price())
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
        position_size = abs(self.position_amount); is_long = self.position_amount > 0
        close_value = position_size * price; fee = close_value * self.cfg.TRANSACTION_FEE
        entry_value = position_size * self.entry_price; entry_fee = entry_value * self.cfg.TRANSACTION_FEE
        if is_long:
            realized_pnl = (close_value - fee) - (entry_value + entry_fee)
        else: # Short
            realized_pnl = (entry_value - entry_fee) - (close_value + fee)
        self.balance += entry_value + realized_pnl; self.trades.append(realized_pnl)
        
        reward = realized_pnl / self.cfg.INITIAL_BALANCE
        self.position_amount = 0.0; self.entry_price = 0.0
        return reward

def main():
    print("🚀 СИСТЕМА V10.0 (Трендовый Специалист) - ЗАПУСК")
    device = setup_gpu_support(); get_gpu_memory_info(device)
    
    data_loader = SimpleDataLoader("data/BTC_5_96w.csv")
    features_df, prices_df = data_loader.load_and_prepare_data()
    
    train_split_idx = int(len(features_df) * 0.8)
    train_features, train_prices = features_df.iloc[:train_split_idx], prices_df.iloc[:train_split_idx]
    test_features, test_prices = features_df.iloc[train_split_idx:], prices_df.iloc[train_split_idx:]
    print(f"✅ Данные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
    vec_env = DummyVecEnv([lambda: TradingEnv(train_features, train_prices)])
    
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor, features_extractor_kwargs=dict(features_dim=256), net_arch=dict(pi=[256, 128], vf=[256, 128]))
    
    model = PPO('MultiInputPolicy', vec_env, policy_kwargs=policy_kwargs,
                learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF,
                n_steps=2048, batch_size=64, gamma=TrendTraderConfig.GAMMA,
                verbose=1, device=device)
                
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ 'ТРЕНДОВОГО СПЕЦИАЛИСТА'...")
    model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS)
    
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    test_env = TradingEnv(test_features, test_prices)
    obs, info = test_env.reset()
    equity_history = [test_env.equity]; price_history = [test_env._get_current_price()]
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        equity_history.append(info['equity'])
        try: price_history.append(test_env._get_current_price())
        except IndexError: price_history.append(price_history[-1])
        done = terminated or truncated
        
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ 'ТРЕНДОВОГО СПЕЦИАЛИСТА'")
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
    
    plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(15, 7))
    plt.title(f'Результаты на тестовой выборке (V10.0 - Трендовый Специалист)\nReturn: {total_return:.2f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}%')
    ax1 = plt.gca(); ax1.plot(equity_history, label='Equity', color='royalblue', linewidth=2)
    ax1.set_xlabel('Шаги'); ax1.set_ylabel('Equity ($)', color='royalblue');
    ax2 = ax1.twinx(); ax2.plot(price_history, label='Цена BTC', color='darkorange', alpha=0.6)
    ax2.set_ylabel('Цена ($)', color='darkorange');
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()