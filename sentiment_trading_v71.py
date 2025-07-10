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

# Функция setup_gpu_support и get_gpu_memory_info без изменений
def setup_gpu_support():
    """Настройка GPU поддержки. Этот код отработает и на вашем ПК, и на сервере."""
    system = platform.system().lower()
    if system == "windows":
        try:
            import torch_directml
            if torch_directml.is_available(): 
                device = torch_directml.device()
                print(f"✅ DirectML найден: {device}")
                return device
        except ImportError: 
            pass
            
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 NVIDIA CUDA: {gpu_name}")
        return device
        
    device = torch.device("cpu")
    print(f"💻 CPU: {device}")
    return device

def get_gpu_memory_info(device):
    """Получение информации о памяти GPU."""
    if device and device.type == "cuda":
        try:
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            print(f"📊 GPU память: {allocated:.1f}GB / {total:.1f}GB")
        except Exception: 
            pass
        
"""
🚀 ТОРГОВАЯ СИСТЕМА V12.1 - МНОГОМЕРНЫЙ СТРАТЕГ
✅ ЦЕЛЬ: Обеспечить агента полной картиной рынка за счет многотаймфреймового анализа.
✅ MTF DATA LOADER: Создан новый загрузчик данных, который безопасно объединяет
   индикаторы с 5m, 1H, 4H и 1D таймфреймов в единый набор признаков.
✅ ГЛУБОКИЙ КОНТЕКСТ: Агент теперь видит не только локальные 5-минутные паттерны,
   но и глобальный тренд, моментум и волатильность со старших ТФ.
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
    TOTAL_TIMESTEPS = 1500000 
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01
    GAMMA = 0.99
    MAX_TRADE_DURATION = 288


class MTFDataLoader:
    def __init__(self, data_paths: Dict[str, str]):
        self.paths = data_paths

    def _calculate_features(self, df: pd.DataFrame, timeframe_suffix: str) -> pd.DataFrame:
        df[f'ema_slow_{timeframe_suffix}'] = df['close'].ewm(span=50, adjust=False).mean()
        df[f'trend_{timeframe_suffix}'] = np.sign(df['close'] - df[f'ema_slow_{timeframe_suffix}'])
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df[f'rsi_{timeframe_suffix}'] = 100 - (100 / (1 + rs))
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{timeframe_suffix}'] = tr.ewm(span=14, adjust=False).mean()
        return df

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("📊 Загрузка и объединение данных с разных таймфреймов...")
        dfs = {}
        for tf, path in self.paths.items():
            print(f"  - Загрузка {path} ({tf})")
            df = pd.read_csv(path)
            # Убеждаемся, что колонка timestamp имеет правильный тип
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            dfs[tf] = df

        for tf, df in dfs.items():
            dfs[tf] = self._calculate_features(df, tf)

        merged_df = dfs['5m']
        for tf in ['1h', '4h', '1d']:
            cols_to_merge = ['timestamp', f'trend_{tf}', f'rsi_{tf}', f'atr_{tf}']
            merged_df = pd.merge_asof(
                merged_df.sort_values('timestamp'),
                dfs[tf][cols_to_merge].sort_values('timestamp'),
                on='timestamp',
                direction='backward'
            )
        
        features = pd.DataFrame(index=merged_df.index)
        ema_fast = merged_df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = merged_df['close'].ewm(span=26, adjust=False).mean()
        features['rsi_norm_5m'] = (merged_df['rsi_5m'] - 50) / 50
        features['macd_hist_norm_5m'] = (ema_fast - ema_slow) / merged_df['close']
        features['atr_norm_5m'] = merged_df['atr_5m'] / merged_df['close']
        features['trend_1h'] = merged_df['trend_1h']
        features['rsi_norm_1h'] = (merged_df['rsi_1h'] - 50) / 50
        features['trend_4h'] = merged_df['trend_4h']
        features['rsi_norm_4h'] = (merged_df['rsi_4h'] - 50) / 50
        features['trend_1d'] = merged_df['trend_1d']
        features['atr_norm_1d'] = merged_df['atr_1d'] / merged_df['close']
        
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.dropna(inplace=True)

        prices_df = merged_df.loc[features.index].reset_index(drop=True)
        features.reset_index(drop=True, inplace=True)
        
        prices_columns_to_select = ['timestamp', 'open', 'high', 'low', 'close', 'atr_5m']
        final_prices_df = prices_df[prices_columns_to_select].copy()
        final_prices_df.rename(columns={'atr_5m': 'atr_value'}, inplace=True)
        
        print(f"✅ Подготовлено данных: {len(features)} записей, {len(features.columns)} признаков.")
        return features, final_prices_df


class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, features_df: pd.DataFrame, prices_df: pd.DataFrame):
        super().__init__()
        self.features_df = features_df.reset_index(drop=True)
        self.prices_df = prices_df.reset_index(drop=True)
        self.cfg = TrendTraderConfig()
        
        self.action_space = spaces.Discrete(3)
        state_shape = (3,)
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.cfg.WINDOW_SIZE, self.features_df.shape[1]), dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32)
        })
        self._reset_state()
    
    def _reset_state(self):
        self.balance = self.cfg.INITIAL_BALANCE; self.equity = self.cfg.INITIAL_BALANCE
        self.current_step = self.cfg.WINDOW_SIZE
        self.position_amount = 0.0; self.entry_price = 0.0; self.entry_step = 0
        self.stop_loss_price = 0.0; self.take_profit_price = 0.0; self.trades = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._reset_state(); return self._get_observation(), {}
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        market_obs = self.features_df.iloc[self.current_step - self.cfg.WINDOW_SIZE : self.current_step].values
        image_obs = np.expand_dims(market_obs, axis=0).astype(np.float32)
        position_type = np.sign(self.position_amount)

        if self.position_amount != 0:
            current_price = self._get_current_price()
            pnl = (current_price - self.entry_price) * self.position_amount
            current_atr = self._get_current_atr()
            unrealized_pnl_norm = pnl / (abs(self.position_amount) * current_atr) if current_atr > 0 else 0.0
            duration_norm = (self.current_step - self.entry_step) / self.cfg.MAX_TRADE_DURATION
        else:
            unrealized_pnl_norm = 0.0; duration_norm = 0.0
        
        state_obs = np.array([position_type, unrealized_pnl_norm, duration_norm], dtype=np.float32)
        return {"image": image_obs, "state": state_obs}

    def _get_current_price(self) -> float: return self.prices_df.iloc[self.current_step]['close']
    def _get_current_atr(self) -> float: return self.prices_df.iloc[self.current_step]['atr_value']

    def step(self, action: int):
        current_price = self._get_current_price()
        reward = 0.0
        
        if self.position_amount != 0:
            low, high = self.prices_df.iloc[self.current_step][['low', 'high']]
            is_long = self.position_amount > 0
            if (is_long and low <= self.stop_loss_price) or (not is_long and high >= self.stop_loss_price):
                reward = self._close_position(self.stop_loss_price)
            elif (is_long and high >= self.take_profit_price) or (not is_long and low <= self.take_profit_price):
                reward = self._close_position(self.take_profit_price)
        
        current_pos_type = np.sign(self.position_amount); desired_pos_type = action - 1
        if current_pos_type != desired_pos_type:
            if current_pos_type != 0:
                reward += self._close_position(current_price)
            if desired_pos_type != 0:
                self._open_position(current_price, is_long=(desired_pos_type == 1))

        self.current_step += 1
        unrealized_pnl = (self._get_current_price() - self.entry_price) * self.position_amount if self.position_amount != 0 else 0
        self.equity = self.balance + unrealized_pnl
        done = self.current_step >= len(self.features_df) - 1 or self.equity <= 0
        if done and self.position_amount != 0:
            reward += self._close_position(self._get_current_price())
        
        return self._get_observation(), reward, done, False, {'equity': self.equity}

    def _open_position(self, price, is_long):
        self.entry_step = self.current_step
        atr = self._get_current_atr()
        sl, tp = self.cfg.ATR_SL_MULTIPLIER, self.cfg.ATR_TP_MULTIPLIER
        if is_long:
            self.stop_loss_price = price - (atr * sl); self.take_profit_price = price + (atr * tp)
        else:
            self.stop_loss_price = price + (atr * sl); self.take_profit_price = price - (atr * tp)
        order_size_usd = self.balance * self.cfg.ORDER_SIZE_RATIO
        if self.balance > 0 and order_size_usd > 0:
            fee = order_size_usd * self.cfg.TRANSACTION_FEE; self.balance -= (order_size_usd + fee)
            self.position_amount = (order_size_usd / price) * (1 if is_long else -1)
            self.entry_price = price

    def _close_position(self, price):
        size, is_long = abs(self.position_amount), self.position_amount > 0
        close_value = size * price; fee = close_value * self.cfg.TRANSACTION_FEE
        entry_value = size * self.entry_price; entry_fee = entry_value * self.cfg.TRANSACTION_FEE
        pnl = (close_value-fee) - (entry_value+entry_fee) if is_long else (entry_value-entry_fee) - (close_value+fee)
        self.balance += entry_value + pnl; self.trades.append(pnl)
        reward = pnl / self.cfg.INITIAL_BALANCE
        self.position_amount = 0.0; self.entry_price = 0.0
        return reward


def main():
    print("🚀 СИСТЕМА V12.1 (Многомерный Стратег) - ЗАПУСК")
    device = setup_gpu_support(); get_gpu_memory_info(device)
    
    data_paths = {
        '5m': 'data/BTCUSDT_5m_2y.csv',
        '1h': 'data/BTCUSDT_1h_2y.csv',
        '4h': 'data/BTCUSDT_4h_2y.csv',
        '1d': 'data/BTCUSDT_1d_2y.csv'
    }
    
    data_loader = MTFDataLoader(data_paths)
    features_df, prices_df = data_loader.load_and_prepare_data()
    
    split_idx = int(len(features_df) * 0.8)
    train_features, train_prices = features_df.iloc[:split_idx], prices_df.iloc[:split_idx]
    test_features, test_prices = features_df.iloc[split_idx:], prices_df.iloc[split_idx:]
    print(f"✅ Данные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
    env = DummyVecEnv([lambda: TradingEnv(train_features, train_prices)])
    
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor, features_extractor_kwargs=dict(features_dim=256), net_arch=dict(pi=[256, 128], vf=[256, 128]))
    
    model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs,
                learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF,
                n_steps=2048, batch_size=64, gamma=TrendTraderConfig.GAMMA, verbose=1, device=device)
                
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ 'МНОГОМЕРНОГО СТРАТЕГА'...")
    model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS)
    
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    test_env = TradingEnv(test_features, test_prices)
    obs, _ = test_env.reset()
    equity_history, price_history = [test_env.equity], [test_env._get_current_price()]
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = test_env.step(int(action))
        equity_history.append(info['equity'])
        try: 
            price_history.append(test_env._get_current_price())
        except IndexError: 
            price_history.append(price_history[-1])
        done = terminated or truncated
        
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ 'МНОГОМЕРНОГО СТРАТЕГА'")
    initial, final = equity_history[0], equity_history[-1]
    total_return = (final - initial) / initial * 100
    start_p, end_p = price_history[0], price_history[-1]
    bnh_return = (end_p - start_p) / start_p * 100
    trades = len(test_env.trades)
    win_rate = (len([t for t in test_env.trades if t > 0]) / trades) * 100 if trades > 0 else 0
    
    print("=" * 60)
    print(f"💰 Финальный баланс: ${final:,.2f} (Начальный: ${initial:,.2f})")
    print(f"📈 Доходность стратегии: {total_return:+.2f}%")
    print(f"📊 Доходность Buy & Hold: {bnh_return:+.2f}%")
    print("-" * 30)
    print(f"🔄 Всего сделок: {trades}")
    print(f"✅ Процент прибыльных сделок: {win_rate:.1f}%")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(15, 7))
    plt.title(f'Результаты на тестовой выборке (V12.1 - Многомерный Стратег)\nReturn: {total_return:.2f}% | Trades: {trades} | Win Rate: {win_rate:.1f}%')
    ax1 = plt.gca(); ax1.plot(equity_history, label='Equity', color='royalblue', linewidth=2)
    ax1.set_xlabel('Шаги'); ax1.set_ylabel('Equity ($)', color='royalblue')
    ax2 = ax1.twinx(); ax2.plot(price_history, label='Цена BTC', color='darkorange', alpha=0.6)
    ax2.set_ylabel('Цена ($)', color='darkorange'); ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    # Сохранение графика в файл, т.к. на сервере нет GUI для plt.show()
    plt.savefig('results_v12.png')
    plt.close() # Закрываем фигуру, чтобы скрипт мог завершиться
    print("✅ График с результатами сохранен в 'results_v12.png'")


if __name__ == "__main__":
    main()