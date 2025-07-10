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
🚀 ТОРГОВАЯ СИСТЕМА V6.3 - СТРАТЕГ
✅ ИСПРАВЛЕНА ПРОБЛЕМА СВЕРХЧУВСТВИТЕЛЬНОГО SL: Множитель ATR для стоп-лосса увеличен, чтобы дать сделкам "дышать".
✅ ВНЕДРЕНО ФОРМИРОВАНИЕ НАГРАД (REWARD SHAPING): Агент получает небольшую награду/штраф на каждом шаге удержания позиции,
   что решает проблему разреженных наград и ускоряет обучение.
✅ РАСШИРЕНО ПРОСТРАНСТВО НАБЛЮДЕНИЙ: Агент теперь "знает" о своем состоянии (открыта ли позиция, какой PnL),
   что делает его решения более контекстно-зависимыми.
✅ УВЕЛИЧЕНА ЕМКОСТЬ СЕТИ: Нейронная сеть стала немного больше для обработки более сложных паттернов.
✅ Цель: Обучить агента, который способен удерживать позиции и принимать стратегические решения.
"""

class ChannelFirstWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.expand_dims(observation, axis=0)
        
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256): # ИЗМЕНЕНИЕ: Увеличили features_dim
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
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
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class TrendTraderConfig:
    INITIAL_BALANCE = 10000
    ORDER_SIZE_RATIO = 0.15
    ### ИЗМЕНЕНИЕ 1: Даем сделке "дышать" ###
    ATR_SL_MULTIPLIER = 3.5  # Увеличено с 2.0
    ATR_TP_MULTIPLIER = 6.0
    TRANSACTION_FEE = 0.001
    ACTION_COST = 0.1
    WINDOW_SIZE = 64
    TOTAL_TIMESTEPS = 1000000 # Можно оставить 1М, но для теста можно и 500к
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.01
    GAMMA = 0.999
    TREND_PROFIT_BONUS = 0.1
    ### ИЗМЕНЕНИЕ 2: Коэффициент для формирования наград ###
    SHAPING_REWARD_COEFF = 0.005

class SimpleDataLoader:
    # ... (Этот класс без изменений)
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
        # Убрал rsi_delta_norm, так как CNN сама может находить такие зависимости
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
        
        ### ИЗМЕНЕНИЕ 3: Добавляем 2 новых признака в наблюдение ###
        # Исходные рыночные признаки + 2 признака состояния (в позиции ли мы, какой PnL)
        self.num_state_features = 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.cfg.WINDOW_SIZE, self.features_df.shape[1] + self.num_state_features), 
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
        self.last_unrealized_pnl = 0.0 # Для Reward Shaping
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Получаем срез рыночных данных
        market_obs = self.features_df.iloc[self.current_step - self.cfg.WINDOW_SIZE : self.current_step].values

        ### ИЗМЕНЕНИЕ 3 (продолжение): Динамически добавляем признаки состояния ###
        position_active = 1.0 if self.position_amount > 0 else 0.0
        
        if self.position_amount > 0:
            current_price = self._get_current_price()
            # Нормализуем PnL на цену входа, чтобы он не был слишком большим/маленьким
            unrealized_pnl_norm = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl_norm = 0.0

        # Создаем признаки состояния и расширяем их до формы (WINDOW_SIZE, 2)
        state_features = np.array([[position_active, unrealized_pnl_norm]] * self.cfg.WINDOW_SIZE)
        
        # Объединяем рыночные данные и признаки состояния
        full_obs = np.concatenate([market_obs, state_features], axis=1).astype(np.float32)
        return full_obs

    def _get_current_price(self) -> float:
        return self.prices_df.iloc[self.current_step]['close']

    def _get_current_atr(self) -> float:
        return self.prices_df.iloc[self.current_step]['atr_value']

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()
        reward = 0.0 # Награда за закрытие сделки (большая и редкая)
        shaping_reward = 0.0 # Награда за удержание (маленькая и частая)
        done = False

        # Логика действий
        if action == 1 and self.position_amount == 0:
            self._open_position(current_price)
            reward -= self.cfg.ACTION_COST
        elif action == 2 and self.position_amount > 0:
            reward = self._close_position(current_price)
            reward -= self.cfg.ACTION_COST
        
        # Логика SL/TP и Reward Shaping
        if self.position_amount > 0:
            # Проверка SL/TP
            low_price = self.prices_df.iloc[self.current_step]['low']
            high_price = self.prices_df.iloc[self.current_step]['high']
            
            if low_price <= self.stop_loss_price:
                reward = self._close_position(self.stop_loss_price)
            elif high_price >= self.take_profit_price:
                reward = self._close_position(self.take_profit_price)
            
            # Если позиция не была закрыта, считаем shaping reward
            else:
                current_unrealized_pnl = (current_price - self.entry_price) * self.position_amount
                pnl_change = current_unrealized_pnl - self.last_unrealized_pnl
                shaping_reward = pnl_change * self.cfg.SHAPING_REWARD_COEFF
                self.last_unrealized_pnl = current_unrealized_pnl

        # Обновление шага и эквити
        self.current_step += 1
        current_unrealized_pnl_for_equity = (self._get_current_price() - self.entry_price) * self.position_amount if self.position_amount > 0 else 0
        self.equity = self.balance + current_unrealized_pnl_for_equity
        
        # Условие завершения эпизода
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
            self.last_unrealized_pnl = 0.0 # Сброс при открытии

    def _close_position(self, price: float) -> float:
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
        self.last_unrealized_pnl = 0.0 # Сброс при закрытии
        return reward


def main():
    print("🚀 СИСТЕМА V6.3 (Стратег) - ЗАПУСК")
    # Убедитесь, что у вас есть этот файл данных или укажите свой
    data_loader = SimpleDataLoader("data/BTC_5_96w.csv")
    features_df, prices_df = data_loader.load_and_prepare_data()

    train_split_idx = int(len(features_df) * 0.8)
    train_features, train_prices = features_df.iloc[:train_split_idx], prices_df.iloc[:train_split_idx]
    test_features, test_prices = features_df.iloc[train_split_idx:], prices_df.iloc[train_split_idx:]
    print(f"\nДанные разделены: {len(train_features)} для обучения, {len(test_features)} для теста.")
    
    env_fn = lambda: TradingEnv(train_features, train_prices)
    vec_env = DummyVecEnv([lambda: ChannelFirstWrapper(env_fn())])

    ### ИЗМЕНЕНИЕ 4: Увеличиваем "мозг" агента ###
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256), # Больше признаков на выходе из CNN
        net_arch=dict(pi=[256, 128], vf=[256, 128]) # Более глубокие сети для политики и оценки
    )

    model = PPO('CnnPolicy', vec_env, policy_kwargs=policy_kwargs,
                learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF,
                n_steps=2048, batch_size=64, gamma=TrendTraderConfig.GAMMA,
                verbose=1, device="cpu") # Используй "cuda", если есть GPU
    
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ СТРАТЕГА...")
    model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS)
    
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    test_env_raw = TradingEnv(test_features, test_prices)
    test_env_wrapped = ChannelFirstWrapper(test_env_raw)
    
    obs, info = test_env_wrapped.reset()
    
    equity_history = [test_env_raw.equity]
    price_history = [test_env_raw._get_current_price()]
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_wrapped.step(int(action))
        equity_history.append(info['equity'])
        try:
            price_history.append(test_env_raw._get_current_price())
        except IndexError: # Если дошли до самого конца данных
            price_history.append(price_history[-1])

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
    plt.title(f'Результаты на тестовой выборке (V6.3 - Стратег)\nReturn: {total_return:.2f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}%')
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