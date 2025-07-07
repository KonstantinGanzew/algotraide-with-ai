import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn


# === КОНСТАНТЫ ===
class Config:
    # Файлы и пути
    DATA_FOLDER = "data/"
    DATA_FILE = "BTC_5_96w.csv"
    
    # Параметры окружения
    WINDOW_SIZE = 50
    INITIAL_BALANCE = 10000
    POSITIONS_LIMIT = 5  # Увеличено для частичного закрытия
    PASSIVITY_THRESHOLD = 100
    
    # Параметры устройства
    AUTO_DEVICE = True
    FORCE_CPU = False
    DEVICE = "cpu"
    
    # Риск-менеджмент
    RISK_PER_TRADE = 0.02      # 2% от баланса на сделку
    STOP_LOSS_PERCENTAGE = 0.02  # Стоп-лосс 2%
    TAKE_PROFIT_PERCENTAGE = 0.06  # Тейк-профит 6%
    MAX_DRAWDOWN_LIMIT = 0.15   # Максимальная просадка 15%
    
    # Продвинутые параметры вознаграждений
    BALANCE_CHANGE_MULTIPLIER = 10
    VOLATILITY_WINDOW = 20
    RISK_ADJUSTMENT_FACTOR = 0.5
    DRAWDOWN_PENALTY_MULTIPLIER = 20
    SHARPE_BONUS_MULTIPLIER = 5
    
    # Размеры позиций
    PARTIAL_CLOSE_PERCENTAGE = 0.33  # Закрывать 33% позиции
    MIN_POSITION_SIZE = 0.1
    
    # Технические индикаторы
    EMA_FAST_SPAN = 12
    EMA_SLOW_SPAN = 26
    RSI_WINDOW = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    BOLLINGER_WINDOW = 20
    
    # Параметры обучения - МАКСИМАЛЬНЫЕ НАСТРОЙКИ
    TOTAL_TIMESTEPS = 500000  # Увеличено в 5 раз для длительного обучения!
    PPO_ENT_COEF = 0.005      # Снижено для более фокусированного обучения
    LEARNING_RATE = 2e-4      # Снижено для стабильности
    
    # Отключение раннего завершения
    ENABLE_EARLY_STOPPING = False  # Отключаем early stopping
    EARLY_STOPPING_PATIENCE = 999999  # Очень большое значение
    
    # LSTM параметры
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    
    # Визуализация
    FIGURE_SIZE = (16, 10)


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


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """LSTM Feature Extractor для обработки временных рядов"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Получаем размеры входа
        n_input_features = observation_space.shape[-1]
        sequence_length = observation_space.shape[0]
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=n_input_features,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=0.2 if Config.LSTM_NUM_LAYERS > 1 else 0
        )
        
        # Дополнительные слои
        self.feature_net = nn.Sequential(
            nn.Linear(Config.LSTM_HIDDEN_SIZE, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Обработка входных данных через LSTM
        lstm_out, _ = self.lstm(observations)
        # Берем выход последнего временного шага
        features = lstm_out[:, -1, :]
        return self.feature_net(features)


class MaximalTrainingCallback(BaseCallback):
    """Callback для максимального обучения без раннего завершения"""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.step_count = 0
        self.best_reward = float('-inf')
        self.progress_interval = 10000  # Показывать прогресс каждые 10k шагов

    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Показываем прогресс обучения
        if self.step_count % self.progress_interval == 0:
            if len(self.locals.get('infos', [])) > 0:
                episode_rewards = [info.get('episode', {}).get('r', 0) for info in self.locals['infos']]
                if episode_rewards:
                    current_reward = np.mean(episode_rewards)
                    if current_reward > self.best_reward:
                        self.best_reward = current_reward
                        print(f"🚀 [Шаг {self.step_count}] Новый рекорд награды: {current_reward:.3f}")
                    else:
                        print(f"📊 [Шаг {self.step_count}] Текущая награда: {current_reward:.3f} (лучшая: {self.best_reward:.3f})")
        
        # НИКОГДА не останавливаем обучение досрочно!
        return True


class AdvancedTradingEnv(gym.Env):
    """Продвинутое торговое окружение с риск-менеджментом"""
    
    def __init__(self, df, window_size=50, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Пространства действий: 0-Hold, 1-Buy25%, 2-Buy50%, 3-Buy100%, 4-Sell25%, 5-Sell50%, 6-Sell100%
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, df.shape[1]), dtype=np.float32
        )

        self._reset_state()

    def _reset_state(self):
        """Сброс состояния окружения"""
        self.balance = self.initial_balance
        self.entry_price = 0.0
        self.position_size = 0.0  # Теперь float для частичных позиций
        self.current_step = self.window_size
        self.trades = []
        self.balance_history = [self.initial_balance]
        self.max_balance = self.initial_balance
        self.returns_history = []
        
        # Риск-менеджмент
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.max_drawdown = 0.0
        
        # Для расчета волатильности
        self.price_history = []

    def reset(self, seed=None, options=None):
        """Сброс окружения"""
        self._reset_state()
        return self._get_observation(), {}

    def _get_observation(self):
        """Получение текущего наблюдения"""
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return obs.astype(np.float32)

    def _calculate_dynamic_order_size(self):
        """Динамический расчет размера позиции"""
        return self.balance * Config.RISK_PER_TRADE

    def _calculate_profit(self, current_price):
        """Расчет прибыли от позиции"""
        if self.position_size <= 0 or self.entry_price <= 0:
            return 0.0
        
        order_size = self._calculate_dynamic_order_size()
        profit_per_coin = current_price - self.entry_price
        return (profit_per_coin * order_size * self.position_size) / self.entry_price

    def _calculate_risk_adjusted_reward(self, current_price):
        """Расчет награды с учетом риска"""
        # Обновляем историю цен
        self.price_history.append(current_price)
        if len(self.price_history) > Config.VOLATILITY_WINDOW:
            self.price_history.pop(0)
        
        # Базовая награда - изменение баланса
        prev_total_balance = self.balance_history[-1] if self.balance_history else self.initial_balance
        
        # Текущий общий баланс
        unrealized_profit = self._calculate_profit(current_price) if self.position_size > 0 else 0
        current_total_balance = self.balance + unrealized_profit
        
        # Базовая награда
        balance_change = current_total_balance - prev_total_balance
        base_reward = (balance_change / self.initial_balance) * Config.BALANCE_CHANGE_MULTIPLIER
        
        # Компонент с учетом риска
        risk_adjusted_return = 0.0
        if len(self.price_history) >= Config.VOLATILITY_WINDOW:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0.01
            
            if volatility > 0:
                sharpe_ratio = np.mean(returns) / volatility
                risk_adjusted_return = sharpe_ratio * Config.RISK_ADJUSTMENT_FACTOR
        
        # Штраф за просадку
        self.max_balance = max(self.max_balance, current_total_balance)
        drawdown = (self.max_balance - current_total_balance) / self.max_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        drawdown_penalty = 0.0
        if drawdown > Config.MAX_DRAWDOWN_LIMIT:
            drawdown_penalty = drawdown * Config.DRAWDOWN_PENALTY_MULTIPLIER
        
        # Итоговая награда
        total_reward = base_reward + risk_adjusted_return - drawdown_penalty
        
        # Обновляем историю
        self.balance_history.append(current_total_balance)
        if len(self.balance_history) > 100:  # Ограничиваем размер истории
            self.balance_history.pop(0)
            
        return total_reward, {
            'base_reward': base_reward,
            'risk_adjusted': risk_adjusted_return,
            'drawdown_penalty': drawdown_penalty,
            'current_drawdown': drawdown
        }

    def _execute_trade(self, action, current_price):
        """Выполнение торговых операций"""
        reward = 0.0
        trade_info = {}
        
        if action == 0:  # Hold
            return reward, trade_info
            
        # Покупка (1-25%, 2-50%, 3-100%)
        elif action in [1, 2, 3]:
            buy_percentages = {1: 0.25, 2: 0.5, 3: 1.0}
            buy_strength = buy_percentages[action]
            
            if self.position_size < Config.POSITIONS_LIMIT:
                # Рассчитываем новый размер позиции
                position_increase = min(buy_strength, Config.POSITIONS_LIMIT - self.position_size)
                
                if position_increase > 0:
                    # Обновляем среднюю цену входа
                    if self.position_size > 0:
                        total_cost = (self.entry_price * self.position_size) + (current_price * position_increase)
                        total_size = self.position_size + position_increase
                        self.entry_price = total_cost / total_size
                    else:
                        self.entry_price = current_price
                        # Устанавливаем стоп-лосс и тейк-профит
                        self.stop_loss_price = current_price * (1 - Config.STOP_LOSS_PERCENTAGE)
                        self.take_profit_price = current_price * (1 + Config.TAKE_PROFIT_PERCENTAGE)
                    
                    self.position_size += position_increase
                    reward += position_increase * 0.1  # Небольшая награда за открытие позиции
                    
                    trade_info = {
                        'action': f'BUY_{int(buy_strength*100)}%',
                        'size': position_increase,
                        'price': current_price,
                        'new_position': self.position_size
                    }
        
        # Продажа (4-25%, 5-50%, 6-100%)
        elif action in [4, 5, 6]:
            sell_percentages = {4: 0.25, 5: 0.5, 6: 1.0}
            sell_strength = sell_percentages[action]
            
            if self.position_size > 0:
                # Рассчитываем размер продажи
                sell_size = min(self.position_size * sell_strength, self.position_size)
                
                if sell_size >= Config.MIN_POSITION_SIZE:
                    # Рассчитываем прибыль
                    profit = self._calculate_profit(current_price) * (sell_size / self.position_size)
                    self.balance += profit
                    self.trades.append(profit)
                    
                    # Обновляем размер позиции
                    self.position_size = max(0, self.position_size - sell_size)
                    
                    # Сбрасываем стоп-лосс/тейк-профит если позиция закрыта полностью
                    if self.position_size <= Config.MIN_POSITION_SIZE:
                        self.position_size = 0
                        self.entry_price = 0
                        self.stop_loss_price = 0
                        self.take_profit_price = 0
                    
                    reward += profit / self.initial_balance  # Награда пропорциональна прибыли
                    
                    trade_info = {
                        'action': f'SELL_{int(sell_strength*100)}%',
                        'size': sell_size,
                        'price': current_price,
                        'profit': profit,
                        'remaining_position': self.position_size
                    }
        
        return reward, trade_info

    def _check_stop_loss_take_profit(self, current_price):
        """Проверка стоп-лосса и тейк-профита"""
        if self.position_size <= 0:
            return 0.0, {}
        
        trade_info = {}
        reward = 0.0
        
        # Проверка стоп-лосса
        if current_price <= self.stop_loss_price and self.stop_loss_price > 0:
            profit = self._calculate_profit(current_price)
            self.balance += profit
            self.trades.append(profit)
            
            trade_info = {
                'action': 'STOP_LOSS',
                'size': self.position_size,
                'price': current_price,
                'profit': profit
            }
            
            # Сбрасываем позицию
            self.position_size = 0
            self.entry_price = 0
            self.stop_loss_price = 0
            self.take_profit_price = 0
            
            reward = profit / self.initial_balance - 0.5  # Штраф за стоп-лосс
        
        # Проверка тейк-профита
        elif current_price >= self.take_profit_price and self.take_profit_price > 0:
            profit = self._calculate_profit(current_price)
            self.balance += profit
            self.trades.append(profit)
            
            trade_info = {
                'action': 'TAKE_PROFIT',
                'size': self.position_size,
                'price': current_price,
                'profit': profit
            }
            
            # Сбрасываем позицию
            self.position_size = 0
            self.entry_price = 0
            self.stop_loss_price = 0
            self.take_profit_price = 0
            
            reward = profit / self.initial_balance + 0.5  # Бонус за тейк-профит
        
        return reward, trade_info

    def step(self, action):
        """Выполнение шага в окружении"""
        current_price = self.df.iloc[self.current_step]['close']
        
        # Проверяем стоп-лосс/тейк-профит
        sl_tp_reward, sl_tp_info = self._check_stop_loss_take_profit(current_price)
        
        # Выполняем действие пользователя
        trade_reward, trade_info = self._execute_trade(action, current_price)
        
        # Рассчитываем основную награду с учетом риска
        risk_reward, risk_info = self._calculate_risk_adjusted_reward(current_price)
        
        # Суммарная награда
        total_reward = sl_tp_reward + trade_reward + risk_reward
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Финальное закрытие позиции
        if done and self.position_size > 0:
            final_profit = self._calculate_profit(current_price)
            self.balance += final_profit
            self.trades.append(final_profit)
            total_reward += final_profit / self.initial_balance

        # Информация для отладки
        info = {
            'balance': self.balance,
            'position_size': self.position_size,
            'max_drawdown': self.max_drawdown,
            'trade_info': trade_info,
            'sl_tp_info': sl_tp_info,
            'risk_info': risk_info,
            'price': current_price
        }

        return self._get_observation(), total_reward, done, False, info

    def render(self):
        """Отображение текущего состояния"""
        unrealized = self._calculate_profit(self.df.iloc[self.current_step-1]['close']) if self.position_size > 0 else 0
        total_balance = self.balance + unrealized
        print(f"Step: {self.current_step}, Balance: {total_balance:.2f}, Position: {self.position_size:.2f}, Drawdown: {self.max_drawdown:.2%}")


def load_and_prepare_data(file_path):
    """Загрузка и подготовка данных с расширенными техническими индикаторами"""
    df = pd.read_csv(file_path)
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Базовые технические индикаторы
    df['ema_fast'] = df['close'].ewm(span=Config.EMA_FAST_SPAN, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=Config.EMA_SLOW_SPAN, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=Config.RSI_WINDOW).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=Config.RSI_WINDOW).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=Config.MACD_FAST).mean()
    ema_26 = df['close'].ewm(span=Config.MACD_SLOW).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # OBV (On-Balance Volume)
    df['price_change'] = df['close'].diff()
    df['obv_raw'] = np.where(df['price_change'] > 0, df['volume'], 
                        np.where(df['price_change'] < 0, -df['volume'], 0))
    df['obv'] = df['obv_raw'].cumsum()

    # VWAP (Volume Weighted Average Price)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap_numerator'] = (df['typical_price'] * df['volume']).cumsum()
    df['vwap_denominator'] = df['volume'].cumsum()
    df['vwap'] = df['vwap_numerator'] / df['vwap_denominator']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=Config.BOLLINGER_WINDOW).mean()
    bb_std = df['close'].rolling(window=Config.BOLLINGER_WINDOW).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Дополнительные индикаторы
    df['price_change_pct'] = df['close'].pct_change()
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # ATR (Average True Range)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()

    # Очистка временных колонок
    df.drop(['price_change', 'obv_raw', 'vwap_numerator', 'vwap_denominator', 
             'typical_price', 'tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)

    # Очистка NaN
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Нормализация (исключая некоторые индикаторы)
    cols_to_normalize = ['open', 'high', 'low', 'close', 'volume', 'ema_fast', 'ema_slow', 
                        'macd', 'macd_signal', 'macd_histogram', 'obv', 'vwap', 
                        'bb_middle', 'bb_upper', 'bb_lower', 'volume_sma', 'atr']
    
    for col in cols_to_normalize:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val

    print(f"📊 Подготовлено {len(df.columns)} признаков: {list(df.columns)}")
    return df


def train_model(env):
    """Обучение модели PPO с LSTM архитектурой - МАКСИМАЛЬНОЕ ОБУЧЕНИЕ"""
    device = setup_device()
    
    print(f"\n🎯 Создание модели PPO с LSTM на устройстве: {device}")
    print(f"🔥 РЕЖИМ МАКСИМАЛЬНОГО ОБУЧЕНИЯ: {Config.TOTAL_TIMESTEPS:,} шагов БЕЗ раннего завершения!")
    
    # Настройки политики с LSTM
    policy_kwargs = {
        "features_extractor_class": LSTMFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": Config.LSTM_HIDDEN_SIZE},
        "net_arch": [dict(pi=[256, 128, 64], vf=[256, 128, 64])]  # Увеличенная сеть
    }
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=Config.LEARNING_RATE,
        n_steps=4096,        # Увеличено для более стабильного обучения
        batch_size=128,      # Увеличен batch size
        n_epochs=15,         # Больше эпох для лучшего обучения
        gamma=0.995,         # Увеличен discount factor
        gae_lambda=0.98,     # Увеличен GAE lambda
        clip_range=0.15,     # Чуть меньший clip range для стабильности
        clip_range_vf=None,
        ent_coef=Config.PPO_ENT_COEF,
        vf_coef=0.6,         # Увеличен value function coefficient
        max_grad_norm=0.3,   # Меньший gradient clipping
        device=device,
        verbose=1
    )
    
    print(f"🚀 МАКСИМАЛЬНОЕ ОБУЧЕНИЕ: {Config.TOTAL_TIMESTEPS:,} шагов...")
    print("⚠️  Обучение будет продолжаться до полного завершения!")
    print("💡 Прогресс будет показываться каждые 10,000 шагов")
    
    # Используем callback без раннего завершения
    callback = None if not Config.ENABLE_EARLY_STOPPING else MaximalTrainingCallback()
    
    model.learn(
        total_timesteps=Config.TOTAL_TIMESTEPS, 
        callback=MaximalTrainingCallback()  # Всегда используем максимальный callback
    )
    
    print("🎉 МАКСИМАЛЬНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    return model


def test_model(model, test_env, df):
    """Тестирование обученной модели с детальной аналитикой"""
    obs, _ = test_env.reset()
    
    results = {
        'balance_history': [],
        'prices': [],
        'actions': [],
        'trades': [],
        'drawdowns': [],
        'positions': [],
        'trade_details': []
    }
    
    max_steps = len(df) - test_env.window_size - 10
    step_count = 0

    print(f"Начинаем тестирование (максимум {max_steps} шагов)...")
    
    while step_count < max_steps:
        try:
            action_result = model.predict(obs, deterministic=True)
            action = int(action_result[0]) if isinstance(action_result[0], (np.ndarray, list)) else int(action_result[0])
            
            obs, reward, done, truncated, info = test_env.step(action)
            step_count += 1

            if test_env.current_step >= len(df):
                break

            current_price = df.iloc[test_env.current_step]['close']
            
            # Собираем статистику
            unrealized = test_env._calculate_profit(current_price) if test_env.position_size > 0 else 0
            total_balance = test_env.balance + unrealized
            
            results['balance_history'].append(total_balance)
            results['prices'].append(current_price)
            results['actions'].append(action)
            results['drawdowns'].append(info.get('max_drawdown', 0))
            results['positions'].append(test_env.position_size)
            
            if info.get('trade_info'):
                results['trade_details'].append({
                    'step': step_count,
                    'info': info['trade_info']
                })

            if done:
                print("Эпизод завершен")
                break
                
            if step_count % 5000 == 0:
                print(f"Тестирование: {step_count}/{max_steps} шагов, баланс: {total_balance:.2f}")
                
        except Exception as e:
            print(f"Ошибка в тестировании на шаге {step_count}: {e}")
            break

    results['trades'] = test_env.trades
    print(f"Тестирование завершено за {step_count} шагов")
    return results


def analyze_results(results, initial_balance):
    """Детальный анализ результатов торговли"""
    if not results['balance_history']:
        return {}
    
    balance_history = np.array(results['balance_history'])
    final_balance = balance_history[-1]
    
    # Основные метрики
    total_return = (final_balance - initial_balance) / initial_balance
    max_balance = np.max(balance_history)
    max_drawdown = np.max(results['drawdowns'])
    
    # Расчет доходности по дням
    returns = np.diff(balance_history) / balance_history[:-1]
    
    # Sharpe Ratio (предполагаем 252 торговых дня в году)
    if len(returns) > 1:
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Анализ сделок
    trades = results['trades']
    profitable_trades = [t for t in trades if t > 0]
    losing_trades = [t for t in trades if t < 0]
    
    win_rate = len(profitable_trades) / len(trades) * 100 if trades else 0
    avg_profit = np.mean(profitable_trades) if profitable_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    profit_factor = abs(sum(profitable_trades) / sum(losing_trades)) if losing_trades else float('inf')
    
    # Максимальная серия побед/поражений
    win_streak = 0
    loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    
    for trade in trades:
        if trade > 0:
            current_win_streak += 1
            current_loss_streak = 0
            win_streak = max(win_streak, current_win_streak)
        else:
            current_loss_streak += 1
            current_win_streak = 0
            loss_streak = max(loss_streak, current_loss_streak)
    
    analysis = {
        'total_return': total_return,
        'final_balance': final_balance,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'win_streak': win_streak,
        'loss_streak': loss_streak,
        'volatility': np.std(returns) if len(returns) > 1 else 0
    }
    
    return analysis


def visualize_results(results, analysis):
    """Расширенная визуализация результатов"""
    if not results['balance_history']:
        print("Нет данных для визуализации")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=Config.FIGURE_SIZE)
    
    # График баланса
    axes[0, 0].plot(results['balance_history'], label='Balance', linewidth=2, color='blue')
    axes[0, 0].set_title("Баланс агента", fontsize=14)
    axes[0, 0].set_ylabel("Баланс (USDT)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # График цены и позиций
    axes[0, 1].plot(results['prices'], label='BTC Price', alpha=0.7, linewidth=1, color='orange')
    ax2 = axes[0, 1].twinx()
    ax2.plot(results['positions'], label='Position Size', alpha=0.7, linewidth=1, color='green')
    axes[0, 1].set_title("Цена BTC и размер позиции", fontsize=14)
    axes[0, 1].set_ylabel("Цена BTC")
    ax2.set_ylabel("Размер позиции")
    axes[0, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # График просадки
    axes[1, 0].fill_between(range(len(results['drawdowns'])), results['drawdowns'], 
                           alpha=0.7, color='red', label='Drawdown')
    axes[1, 0].set_title("Просадка", fontsize=14)
    axes[1, 0].set_ylabel("Просадка (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Распределение сделок
    if results['trades']:
        axes[1, 1].hist(results['trades'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_title("Распределение прибыли по сделкам", fontsize=14)
        axes[1, 1].set_xlabel("Прибыль/Убыток")
        axes[1, 1].set_ylabel("Количество сделок")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Основная функция с улучшенной аналитикой"""
    print("🚀 Продвинутый торговый алгоритм на базе RL")
    print("=" * 60)
    
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
        print(f"🎯 Размер окна: {Config.WINDOW_SIZE}")
        print(f"💰 Начальный баланс: {Config.INITIAL_BALANCE}")

        # Создание окружения и обучение
        print("\n🎓 МАКСИМАЛЬНОЕ обучение с LSTM архитектурой...")
        print(f"🔥 Режим: БЕЗ раннего завершения, {Config.TOTAL_TIMESTEPS:,} шагов")
        env = AdvancedTradingEnv(df, 
                               window_size=Config.WINDOW_SIZE,
                               initial_balance=Config.INITIAL_BALANCE)
        
        model = train_model(env)
        print("✅ МАКСИМАЛЬНОЕ обучение завершено!")

        # Тестирование
        print("\n🧪 Тестирование модели...")
        test_env = AdvancedTradingEnv(df, 
                                    window_size=Config.WINDOW_SIZE,
                                    initial_balance=Config.INITIAL_BALANCE)
        
        results = test_model(model, test_env, df)

        # Анализ результатов
        if results['balance_history']:
            analysis = analyze_results(results, Config.INITIAL_BALANCE)
            
            print(f"\n📊 Детальные результаты:")
            print(f"{'='*50}")
            print(f"Финальный баланс: {analysis['final_balance']:.2f} USDT")
            print(f"Общая доходность: {analysis['total_return']:.2%}")
            print(f"Максимальная просадка: {analysis['max_drawdown']:.2%}")
            print(f"Коэффициент Шарпа: {analysis['sharpe_ratio']:.3f}")
            print(f"Волатильность: {analysis['volatility']:.4f}")
            print(f"\n📈 Торговая статистика:")
            print(f"{'='*50}")
            print(f"Всего сделок: {analysis['total_trades']}")
            print(f"Винрейт: {analysis['win_rate']:.2f}%")
            print(f"Средняя прибыль: {analysis['avg_profit']:.2f}")
            print(f"Средний убыток: {analysis['avg_loss']:.2f}")
            print(f"Фактор прибыли: {analysis['profit_factor']:.2f}")
            print(f"Макс. серия побед: {analysis['win_streak']}")
            print(f"Макс. серия поражений: {analysis['loss_streak']}")

            # Визуализация
            print("\n📈 Создание расширенных графиков...")
            visualize_results(results, analysis)
        else:
            print("⚠️ Нет данных для анализа результатов")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()