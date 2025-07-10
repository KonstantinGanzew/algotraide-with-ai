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
from typing import Dict, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

"""
🚀 ТОРГОВАЯ СИСТЕМА V6.2 - ЛОГИКА ИСПРАВЛЕНА
✅ ИСПРАВЛЕНА КЛЮЧЕВАЯ ОШИБКА ЛОГИКИ: Пространство действий расширено до 3 (Hold, Buy, Sell).
   - Агент теперь может принимать осмысленные решения: открыть позицию, закрыть ее или удерживать.
   - Устранен "цикл смерти" (открытие-закрытие на соседних шагах), который сжигал баланс.
✅ Улучшена логика штрафа за действие (ACTION_COST), теперь он применяется и на открытие, и на закрытие.
✅ Сохранена кастомная CNN и философия "Терпеливого Охотника".
✅ Цель: Получить первую логически корректную версию и оценить ее способность к обучению.
"""

# Враппер, необходимый для CnnPolicy
class ChannelFirstWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.expand_dims(observation, axis=0)
        
# ----------------- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: НАША СОБСТВЕННАЯ CNN -----------------
class CustomCNN(BaseFeaturesExtractor):
    """
    Кастомная сверточная сеть для наших данных (64, 7).
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # observation_space.shape = (1, 64, 7) (каналы, высота, ширина)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            # Ядра (kernels) меньше, чем ширина нашего "изображения" (7)
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

        # Вычисляем размерность после сверток, чтобы создать правильный линейный слой
        with torch.no_grad():
            # Добавляем .float() для совместимости типов
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# --------------------------------------------------------------------------

class TrendTraderConfig:
    INITIAL_BALANCE = 10000
    ORDER_SIZE_RATIO = 0.15
    ATR_SL_MULTIPLIER = 2.0
    ATR_TP_MULTIPLIER = 6.0
    TRANSACTION_FEE = 0.001
    ACTION_COST = 0.1  # Снизил стоимость, так как она теперь более значима
    WINDOW_SIZE = 64
    TOTAL_TIMESTEPS = 1000000
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01
    GAMMA = 0.999
    TREND_PROFIT_BONUS = 0.1

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
        df['rsi_delta'] = df['rsi'].diff(5)
        
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        features = pd.DataFrame(index=df.index)
        features['price_norm'] = df['close'] / df['sma_long']
        features['ema_spread'] = (df['ema_fast'] - df['ema_slow']) / df['close']
        features['rsi_norm'] = (df['rsi'] - 50) / 50
        features['macd_hist_norm'] = (df['macd'] - df['macd_signal']) / df['close']
        features['trend_signal'] = np.sign(df['close'] - df['sma_long'])
        features['atr_norm'] = df['atr_value'] / df['close']
        features['rsi_delta_norm'] = features['rsi_norm'].diff(5)
        
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
        
        ### ИЗМЕНЕНИЕ 1: Пространство действий ###
        # 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)
        
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
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        start = self.current_step - self.cfg.WINDOW_SIZE
        return self.features_df.iloc[start:self.current_step].values.astype(np.float32)

    def _get_current_price(self) -> float:
        return self.prices_df.iloc[self.current_step]['close']

    def _get_current_atr(self) -> float:
        return self.prices_df.iloc[self.current_step]['atr_value']

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()
        reward = 0.0
        done = False

        ### ИЗМЕНЕНИЕ 2: Новая логика обработки действий ###
        # Действие 1: Купить (только если нет открытой позиции)
        if action == 1 and self.position_amount == 0:
            self._open_position(current_price)
            self.balance -= self.cfg.ACTION_COST # Штраф за действие
        # Действие 2: Продать (только если есть открытая позиция)
        elif action == 2 and self.position_amount > 0:
            reward = self._close_position(current_price)
            self.balance -= self.cfg.ACTION_COST # Штраф за действие
        # Действие 0 (Hold) или нелогичные действия (купить при наличии, продать при отсутствии)
        # не приводят к изменению позиции, но могут привести к закрытию по SL/TP.

        # Проверка SL/TP происходит на каждом шаге, если позиция открыта
        if self.position_amount > 0:
            # Проверяем, не сработал ли SL или TP на текущей свече
            low_price = self.prices_df.iloc[self.current_step]['low']
            high_price = self.prices_df.iloc[self.current_step]['high']
            
            # Сначала проверяем стоп-лосс
            if low_price <= self.stop_loss_price:
                reward = self._close_position(self.stop_loss_price) # Закрываем по цене SL
            # Затем тейк-профит
            elif high_price >= self.take_profit_price:
                reward = self._close_position(self.take_profit_price) # Закрываем по цене TP

        # Обновляем шаг и состояние счета
        self.current_step += 1
        current_unrealized_pnl = (current_price - self.entry_price) * self.position_amount if self.position_amount > 0 else 0
        self.equity = self.balance + current_unrealized_pnl
        
        # Условие завершения эпизода
        if self.current_step >= len(self.features_df) - 1 or self.equity <= 0:
            if self.position_amount > 0:
                # Закрываем оставшуюся позицию по текущей цене, если данные закончились
                reward = self._close_position(current_price)
            done = True
        
        # info_dict должен быть последним возвращаемым значением в gymnasium
        info = {'equity': self.equity}
        # gymnasium возвращает 5 значений: obs, reward, terminated, truncated, info
        terminated = done 
        truncated = False # Мы не используем усечение по времени, done обрабатывает все
        
        return self._get_observation(), reward, terminated, truncated, info

    def _open_position(self, price: float):
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

    def _close_position(self, price: float) -> float:
        close_value = self.position_amount * price
        fee = close_value * self.cfg.TRANSACTION_FEE
        self.balance += (close_value - fee)
        
        # Расчет PnL теперь проще, т.к. баланс уже учитывает все комиссии
        entry_value = self.position_amount * self.entry_price
        realized_pnl = (close_value - fee) - (entry_value + entry_value * self.cfg.TRANSACTION_FEE)
        
        self.trades.append(realized_pnl)
        
        # Нормализуем награду относительно начального капитала для стабильности обучения
        reward = realized_pnl / self.cfg.INITIAL_BALANCE
        
        # Бонус за торговлю по тренду
        if realized_pnl > 0:
            trend_at_entry = self.features_df.iloc[self.entry_step]['trend_signal']
            if trend_at_entry > 0: # Если входили в лонг по бычьему тренду
                reward += self.cfg.TREND_PROFIT_BONUS
        
        # Сброс состояния позиции
        self.position_amount = 0.0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        return reward


def main():
    print("🚀 СИСТЕМА V6.2 (Логика исправлена) - ЗАПУСК")
    data_loader = SimpleDataLoader("data/BTC_5_96w.csv") # Убедись, что путь к файлу верный
    features_df, prices_df = data_loader.load_and_prepare_data()

    train_split_idx = int(len(features_df) * 0.8)
    train_features, train_prices = features_df.iloc[:train_split_idx], prices_df.iloc[:train_split_idx]
    test_features, test_prices = features_df.iloc[train_split_idx:], prices_df.iloc[train_split_idx:]
    print(f"\nДанные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
    env_fn = lambda: TradingEnv(train_features, train_prices)
    vec_env = DummyVecEnv([lambda: ChannelFirstWrapper(env_fn())])

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )

    model = PPO('CnnPolicy', vec_env, policy_kwargs=policy_kwargs,
                learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF,
                n_steps=2048, batch_size=64, gamma=TrendTraderConfig.GAMMA,
                verbose=1, device="cpu") # Используй "cuda", если есть GPU
    
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ МОДЕЛИ...")
    model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS)
    
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    test_env_raw = TradingEnv(test_features, test_prices)
    test_env_wrapped = ChannelFirstWrapper(test_env_raw)
    
    # ### ИЗМЕНЕНИЕ 3: Корректный цикл тестирования с новым API Gymnasium ###
    obs, info = test_env_wrapped.reset()
    
    equity_history = [test_env_raw.equity]
    price_history = [test_env_raw._get_current_price()]
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_wrapped.step(int(action))
        
        equity_history.append(info['equity']) # Берем equity из info dict
        price_history.append(test_env_raw._get_current_price())
        
        done = terminated or truncated
            
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ")
    initial_equity, final_equity = equity_history[0], equity_history[-1]
    total_return = (final_equity - initial_equity) / initial_equity * 100
    start_price, end_price = price_history[0], price_history[-1]
    bnh_return = (end_price - start_price) / start_price * 100
    
    total_trades = len(test_env_raw.trades)
    win_rate = (len([t for t in test_env_raw.trades if t > 0]) / total_trades) * 100 if total_trades > 0 else 0

    print("=" * 60)
    print(f"💰 Финальный баланс: ${final_equity:,.2f} (Начальный: ${initial_equity:,.2f})")
    print(f"📈 Доходность стратегии: {total_return:+.2f}%")
    print(f"📊 Доходность Buy & Hold: {bnh_return:+.2f}%")
    print("-" * 30)
    print(f"🔄 Всего сделок: {total_trades}")
    print(f"✅ Процент прибыльных сделок: {win_rate:.1f}%")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(15, 7))
    plt.title(f'Результаты на тестовой выборке (V6.2 - Исправленная логика)\nReturn: {total_return:.2f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}%')
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
    plt.show()

if __name__ == "__main__":
    main() 