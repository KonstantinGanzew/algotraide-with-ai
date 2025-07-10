"""
🚀 СИСТЕМА V12.0 (Многомерный Стратег) - УДАЛЕННОЕ GPU ОБУЧЕНИЕ
Оптимизированная версия для обучения на удаленном сервере с NVIDIA GPU
Сервер: 192.168.88.218 (CUDA 12.8)
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple
import matplotlib
matplotlib.use('Agg')  # Для headless сервера
import matplotlib.pyplot as plt
import time
import os
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_gpu_support():
    """Продвинутая настройка GPU с автоматическим выбором лучшего устройства"""
    logger.info("🎮 ИНИЦИАЛИЗАЦИЯ GPU...")
    
    if not torch.cuda.is_available():
        logger.warning("❌ CUDA недоступна, используется CPU")
        return torch.device('cpu')
    
    # Информация о доступных GPU
    gpu_count = torch.cuda.device_count()
    logger.info(f"🔍 Найдено GPU: {gpu_count}")
    
    best_device = 0
    best_memory = 0
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3  # GB
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
        free_memory = total_memory - allocated
        
        logger.info(f"📊 GPU {i}: {props.name}")
        logger.info(f"   💾 Память: {total_memory:.1f}GB (свободно: {free_memory:.1f}GB)")
        logger.info(f"   🔢 Выч. единиц: {props.multi_processor_count}")
        
        if free_memory > best_memory:
            best_memory = free_memory
            best_device = i
    
    device = torch.device(f'cuda:{best_device}')
    torch.cuda.set_device(best_device)
    
    logger.info(f"✅ Выбран GPU {best_device} с {best_memory:.1f}GB свободной памяти")
    
    # Очистка кэша GPU
    torch.cuda.empty_cache()
    
    return device


def get_gpu_memory_info(device):
    """Подробная информация о памяти GPU"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        cached = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        logger.info(f"🎮 GPU Память: {allocated:.2f}GB / {total:.2f}GB (кэш: {cached:.2f}GB)")


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Оптимизированный экстрактор признаков для GPU"""
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        image_space = observation_space.spaces["image"]
        state_space = observation_space.spaces["state"]
        
        # CNN для обработки временных рядов (оптимизирован для GPU)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, image_space.shape[2]), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # MLP для дополнительного состояния
        self.mlp = nn.Sequential(
            nn.Linear(state_space.shape[0], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Финальная комбинация
        self.final = nn.Sequential(
            nn.Linear(64 + 32, features_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_features = self.cnn(observations["image"])
        state_features = self.mlp(observations["state"])
        return self.final(torch.cat([image_features, state_features], dim=1))


class TrendTraderConfig:
    """Конфигурация оптимизированная для GPU обучения"""
    INITIAL_BALANCE = 10000
    TRANSACTION_FEE = 0.001
    WINDOW_SIZE = 64
    ORDER_SIZE_RATIO = 0.10
    ATR_SL_MULTIPLIER = 2.0
    ATR_TP_MULTIPLIER = 4.0
    TOTAL_TIMESTEPS = 2000000  # Увеличено для GPU
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01
    GAMMA = 0.99
    MAX_TRADE_DURATION = 288
    
    # GPU оптимизации
    BATCH_SIZE = 128  # Увеличен для GPU
    N_STEPS = 4096    # Увеличен для GPU


class MTFDataLoader:
    """Загрузчик мультитаймфрейм данных"""
    
    def __init__(self, data_paths: Dict[str, str]):
        self.paths = data_paths

    def _calculate_features(self, df: pd.DataFrame, timeframe_suffix: str) -> pd.DataFrame:
        """Расчет базовых признаков для одного таймфрейма."""
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
        logger.info("📊 Загрузка и объединение данных с разных таймфреймов...")
        
        dfs = {}
        for tf, path in self.paths.items():
            logger.info(f"  - Загрузка {path} ({tf})")
            df = pd.read_csv(path)
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
        
        # Исправлена ошибка с извлечением колонок
        prices_columns = ['timestamp', 'open', 'high', 'low', 'close', 'atr_5m']
        prices_df_final = prices_df[prices_columns].copy()
        prices_df_final.rename(columns={'atr_5m': 'atr_value'}, inplace=True)
        
        logger.info(f"✅ Подготовлено данных: {len(features)} записей, {len(features.columns)} признаков.")
        return features, prices_df_final


class TradingEnv(gym.Env):
    """Оптимизированное торговое окружение"""
    metadata = {'render_modes': ['human']}
    
    def __init__(self, features_df: pd.DataFrame, prices_df: pd.DataFrame):
        super().__init__()
        self.features_df = features_df.reset_index(drop=True)
        self.prices_df = prices_df.reset_index(drop=True)
        self.cfg = TrendTraderConfig()
        
        self.action_space = spaces.Discrete(3)
        state_shape = (3,)
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=-np.inf, high=np.inf, 
                              shape=(1, self.cfg.WINDOW_SIZE, self.features_df.shape[1]), 
                              dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32)
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
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}
    
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
            unrealized_pnl_norm = 0.0
            duration_norm = 0.0
        
        state_obs = np.array([position_type, unrealized_pnl_norm, duration_norm], dtype=np.float32)
        return {"image": image_obs, "state": state_obs}

    def _get_current_price(self) -> float:
        return self.prices_df.iloc[self.current_step]['close']
    
    def _get_current_atr(self) -> float:
        return self.prices_df.iloc[self.current_step]['atr_value']

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
        
        current_pos_type = np.sign(self.position_amount)
        desired_pos_type = action - 1
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
            self.stop_loss_price = price - (atr * sl)
            self.take_profit_price = price + (atr * tp)
        else:
            self.stop_loss_price = price + (atr * sl)
            self.take_profit_price = price - (atr * tp)
        order_size_usd = self.balance * self.cfg.ORDER_SIZE_RATIO
        if self.balance > 0 and order_size_usd > 0:
            fee = order_size_usd * self.cfg.TRANSACTION_FEE
            self.balance -= (order_size_usd + fee)
            self.position_amount = (order_size_usd / price) * (1 if is_long else -1)
            self.entry_price = price

    def _close_position(self, price):
        size, is_long = abs(self.position_amount), self.position_amount > 0
        close_value = size * price
        fee = close_value * self.cfg.TRANSACTION_FEE
        entry_value = size * self.entry_price
        entry_fee = entry_value * self.cfg.TRANSACTION_FEE
        pnl = (close_value-fee) - (entry_value+entry_fee) if is_long else (entry_value-entry_fee) - (close_value+fee)
        self.balance += entry_value + pnl
        self.trades.append(pnl)
        reward = pnl / self.cfg.INITIAL_BALANCE
        self.position_amount = 0.0
        self.entry_price = 0.0
        return reward


def save_results(equity_history, price_history, test_env, total_return, bnh_return):
    """Сохранение результатов на headless сервере"""
    logger.info("📊 Сохранение результатов...")
    
    # Создание графика
    plt.style.use('default')
    plt.figure(figsize=(15, 7))
    plt.title(f'Результаты на тестовой выборке (V12.0 - GPU)\nReturn: {total_return:.2f}% | Trades: {len(test_env.trades)} | Win Rate: {(len([t for t in test_env.trades if t > 0]) / len(test_env.trades)) * 100 if test_env.trades else 0:.1f}%')
    
    ax1 = plt.gca()
    ax1.plot(equity_history, label='Equity', color='royalblue', linewidth=2)
    ax1.set_xlabel('Шаги')
    ax1.set_ylabel('Equity ($)', color='royalblue')
    
    ax2 = ax1.twinx()
    ax2.plot(price_history, label='Цена BTC', color='darkorange', alpha=0.6)
    ax2.set_ylabel('Цена ($)', color='darkorange')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    
    # Сохранение в файл
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"📈 График сохранен: {filename}")
    
    # Сохранение статистики
    stats = {
        'timestamp': timestamp,
        'total_return': total_return,
        'bnh_return': bnh_return,
        'trades': len(test_env.trades),
        'win_rate': (len([t for t in test_env.trades if t > 0]) / len(test_env.trades)) * 100 if test_env.trades else 0,
        'final_equity': equity_history[-1],
        'initial_equity': equity_history[0]
    }
    
    with open(f'stats_{timestamp}.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"📋 Статистика сохранена: stats_{timestamp}.txt")


def main():
    """Главная функция для удаленного GPU обучения"""
    logger.info("🚀 СИСТЕМА V12.0 (GPU Многомерный Стратег) - ЗАПУСК")
    logger.info("=" * 60)
    
    # GPU инициализация
    device = setup_gpu_support()
    get_gpu_memory_info(device)
    
    # Пути к данным (проверяем существование)
    data_paths = {
        '5m': 'data/BTCUSDT_5m_2y.csv',
        '1h': 'data/BTCUSDT_1h_2y.csv',
        '4h': 'data/BTCUSDT_4h_2y.csv',
        '1d': 'data/BTCUSDT_1d_2y.csv'
    }
    
    # Проверка файлов
    missing_files = []
    for tf, path in data_paths.items():
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        logger.error(f"❌ Отсутствуют файлы данных: {missing_files}")
        return False
    
    # Загрузка и подготовка данных
    data_loader = MTFDataLoader(data_paths)
    features_df, prices_df = data_loader.load_and_prepare_data()
    
    split_idx = int(len(features_df) * 0.8)
    train_features, train_prices = features_df.iloc[:split_idx], prices_df.iloc[:split_idx]
    test_features, test_prices = features_df.iloc[split_idx:], prices_df.iloc[split_idx:]
    logger.info(f"✅ Данные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
    # Создание окружения и модели
    env = DummyVecEnv([lambda: TradingEnv(train_features, train_prices)])
    
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128])
    )
    
    config = TrendTraderConfig()
    model = PPO(
        'MultiInputPolicy', 
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=config.LEARNING_RATE,
        ent_coef=config.ENTROPY_COEF,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        gamma=config.GAMMA,
        verbose=1,
        device=device
    )
    
    logger.info(f"🧠 Модель создана на устройстве: {device}")
    get_gpu_memory_info(device)
    
    # Обучение
    logger.info("🎓 НАЧАЛО ОБУЧЕНИЯ...")
    start_time = time.time()
    
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS)
    
    training_time = time.time() - start_time
    logger.info(f"⏱️ Время обучения: {training_time:.2f} секунд")
    
    # Сохранение модели
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'model_gpu_{timestamp}.zip'
    model.save(model_path)
    logger.info(f"💾 Модель сохранена: {model_path}")
    
    # Тестирование
    logger.info("💰 ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
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
    
    # Анализ результатов
    initial, final = equity_history[0], equity_history[-1]
    total_return = (final - initial) / initial * 100
    start_p, end_p = price_history[0], price_history[-1]
    bnh_return = (end_p - start_p) / start_p * 100
    trades = len(test_env.trades)
    win_rate = (len([t for t in test_env.trades if t > 0]) / trades) * 100 if trades > 0 else 0
    
    logger.info("=" * 60)
    logger.info("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    logger.info("=" * 60)
    logger.info(f"💰 Финальный баланс: ${final:,.2f} (Начальный: ${initial:,.2f})")
    logger.info(f"📈 Доходность стратегии: {total_return:+.2f}%")
    logger.info(f"📊 Доходность Buy & Hold: {bnh_return:+.2f}%")
    logger.info(f"🔄 Всего сделок: {trades}")
    logger.info(f"✅ Процент прибыльных сделок: {win_rate:.1f}%")
    logger.info(f"⏱️ Время обучения: {training_time:.2f} сек")
    
    # Сохранение результатов
    save_results(equity_history, price_history, test_env, total_return, bnh_return)
    
    logger.info("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("✅ Программа завершена успешно")
        else:
            logger.error("❌ Программа завершена с ошибкой")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        raise 