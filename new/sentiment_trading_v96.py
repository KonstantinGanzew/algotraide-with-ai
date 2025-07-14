import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List
from collections import Counter
import warnings
from tqdm import tqdm
import logging
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_v13_20.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# --- Вспомогательные функции и классы (без изменений) ---
def setup_gpu_support():
    if torch.cuda.is_available(): device = torch.device("cuda"); gpu_name = torch.cuda.get_device_name(0); msg=f"🚀 NVIDIA CUDA: {gpu_name}"; logger.info(msg); print(msg); return device
    else: device = torch.device("cpu"); msg=f"💻 CPU: {device}"; logger.info(msg); print(msg); return device
def get_gpu_memory_info(device):
    if device and device.type == "cuda":
        try: total = torch.cuda.get_device_properties(device).total_memory/1e9; allocated = torch.cuda.memory_allocated(device)/1e9; msg=f"📊 GPU память: {allocated:.1f}GB / {total:.1f}GB"; logger.info(msg); print(msg)
        except Exception as e: logger.error(f"Ошибка получения инфо о GPU: {e}")
class TqdmCallback(BaseCallback):
    def __init__(self, pbar): super().__init__(verbose=0); self.pbar = pbar
    def _on_step(self) -> bool: self.pbar.update(1); return True
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        image_space = observation_space.spaces['image']
        state_space = observation_space.spaces['state']
        n_input_channels = image_space.shape[0]
        self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, 32, (3,3), 1, 1), nn.ReLU(), nn.Conv2d(32, 64, (3,3), 1, 1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Conv2d(64, 128, (3,3), 1, 1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Flatten())
        with torch.no_grad(): n_flatten = self.cnn(torch.as_tensor(image_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten + state_space.shape[0], features_dim), nn.ReLU())
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor: cnn_out = self.cnn(obs['image']); return self.linear(torch.cat([cnn_out, obs['state']], dim=1))

class MTFDataLoader:
    def __init__(self, data_paths: Dict[str, str]):
        self.paths = data_paths

    def _calc_indicators(self, df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        try:
            df[f'trend_{suffix}'] = np.sign(df['close'] - df['close'].ewm(span=50, adjust=False).mean())
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df[f'rsi_{suffix}'] = 100 - (100 / (1 + rs))
            tr = pd.concat([df['high'] - df['low'], np.abs(df['high'] - df['close'].shift()), np.abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
            df[f'atr_{suffix}'] = tr.ewm(span=14, adjust=False).mean()
            return df
        except Exception as e:
            logger.error(f"Ошибка при расчете индикаторов для {suffix}: {e}")
            raise

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Начало иерархической загрузки данных...")
        try:
            dfs = {tf: self._calc_indicators(pd.read_csv(p).assign(timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='ms')), tf) for tf, p in self.paths.items()}
            merged_df = dfs['5m']
            for tf in ['1h', '4h', '1d']:
                merged_df = pd.merge_asof(merged_df.sort_values(by='timestamp'), dfs[tf][['timestamp', f'trend_{tf}', f'rsi_{tf}', f'atr_{tf}']].sort_values(by='timestamp'), on='timestamp', direction='backward')
            
            initial_rows = len(merged_df)
            merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # --- НОВОЕ: РАСЧЕТ ПРИЗНАКА ВОЛАТИЛЬНОСТИ ---
            merged_df['log_returns'] = np.log(merged_df['close'] / merged_df['close'].shift(1))
            merged_df['volatility_24'] = merged_df['log_returns'].rolling(window=24).std() * np.sqrt(24)
            # -----------------------------------------------
            
            merged_df.dropna(inplace=True)
            logger.info(f"Удалено {initial_rows - len(merged_df)} строк с NaN/inf значениями.")
            
            image_features = merged_df[['open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
            
            state_features = pd.DataFrame(index=merged_df.index)
            state_features['rsi_5m_norm'] = (merged_df['rsi_5m'] - 50) / 50
            state_features['atr_5m_norm'] = merged_df['atr_5m'] / merged_df['close']
            
            # --- НОВОЕ: ДОБАВЛЕНИЕ ВОЛАТИЛЬНОСТИ В STATE ---
            state_features['volatility_norm'] = (merged_df['volatility_24'] - merged_df['volatility_24'].mean()) / merged_df['volatility_24'].std()
            # ---------------------------------------------

            for tf in ['1h', '4h', '1d']:
                state_features[f'trend_{tf}'] = merged_df[f'trend_{tf}']
                state_features[f'rsi_{tf}_norm'] = (merged_df[f'rsi_{tf}'] - 50) / 50
            
            state_features = state_features.reset_index(drop=True)
            prices_df = merged_df[['timestamp', 'open', 'high', 'low', 'close', 'atr_5m']].reset_index(drop=True)
            prices_df = prices_df.rename(columns={'atr_5m': 'atr_value'})
            
            min_len = min(len(prices_df), len(image_features), len(state_features))
            prices_df = prices_df.iloc[-min_len:].reset_index(drop=True)
            image_features = image_features.iloc[-min_len:].reset_index(drop=True)
            state_features = state_features.iloc[-min_len:].reset_index(drop=True)

            msg = f"Данные подготовлены. Image: {image_features.shape}, State: {state_features.shape}"
            logger.info(msg)
            print(f"✅ {msg}")
            return prices_df, image_features, state_features
        except Exception as e:
            logger.critical(f"Критическая ошибка при загрузке данных: {e}", exc_info=True)
            raise

class TrendTraderConfig:
    """
    🚀 КОНФИГУРАЦИЯ V13.20 - ОБОГАЩЕНИЕ ДАННЫХ И СБАЛАНСИРОВАННЫЙ ПОДХОД
    ✅ КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:
       1. FEATURE ENGINEERING: В state добавлен новый признак - нормализованная волатильность.
       2. REWARD BALANCE: Штрафы и бонусы возвращены на сбалансированный уровень,
          чтобы поощрять осмысленные действия, а не парализовывать агента.
       3. N_STEPS: Увеличен до 4096, чтобы агент собирал больше данных перед
          обновлением политики, что улучшает стабильность обучения.
       4. ATR_TP_MULTIPLIER: Установлен на 5.5 - амбициозная, но достижимая цель.
    """
    INITIAL_BALANCE = 10000; TRANSACTION_FEE = 0.001; WINDOW_SIZE = 64
    ORDER_SIZE_RATIO = 0.02
    ATR_SL_MULTIPLIER = 3.0
    ATR_TP_MULTIPLIER = 5.5
    TOTAL_TIMESTEPS = 1000000
    
    LEARNING_RATE = 1e-4
    ENTROPY_COEF = 0.005
    N_STEPS = 4096  # <-- ИЗМЕНЕНИЕ
    GAMMA = 0.99
    MAX_TRADE_DURATION = 288 

    # Гибридные параметры вознаграждения (сбалансированные)
    TRANSACTION_COST_PENALTY = 0.0025
    HOLD_REWARD_BONUS = 0.00002


class TradingEnv(gym.Env):
    def __init__(self, prices_df: pd.DataFrame, image_features: pd.DataFrame, state_features: pd.DataFrame):
        super().__init__()
        self.prices_df = prices_df; self.image_features = image_features; self.state_features = state_features; self.cfg = TrendTraderConfig()
        self.action_space = spaces.Discrete(3); self.image_shape=(1, self.cfg.WINDOW_SIZE, self.image_features.shape[1]); self.state_shape=(3 + self.state_features.shape[1],)
        self.observation_space=spaces.Dict({"image": spaces.Box(low=-1, high=2, shape=self.image_shape, dtype=np.float32), "state": spaces.Box(low=-5, high=5, shape=self.state_shape, dtype=np.float32)})
        self._reset_state()
    def _reset_state(self):
        self.balance, self.equity = self.cfg.INITIAL_BALANCE, self.cfg.INITIAL_BALANCE; self.current_step = self.cfg.WINDOW_SIZE; self.position_amount = 0.0; self.entry_price = 0.0; self.entry_step = 0; self.stop_loss_price = 0.0; self.take_profit_price = 0.0; self.trades = []; self.total_fees = 0.0; self.gross_profit = 0.0; self.gross_loss = 0.0; self.trade_durations = []; self.close_reasons = Counter()
        self.last_unrealized_pnl = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._reset_state(); return self._get_observation(), {}
    def _get_observation(self) -> Dict[str, np.ndarray]:
        # Проверка, что self.current_step не выходит за пределы
        if self.current_step >= len(self.image_features):
            self.current_step = len(self.image_features) - 1
            
        image_window=self.image_features.iloc[self.current_step-self.cfg.WINDOW_SIZE:self.current_step].copy()
        if image_window.empty: # Обработка случая, когда окно пустое
            return {"image": np.zeros(self.image_shape, dtype=np.float32), "state": np.zeros(self.state_shape, dtype=np.float32)}

        current_price=image_window.iloc[-1]['close'] if image_window.iloc[-1]['close']>0 else 1
        for col in ['open','high','low','close']: image_window[col]=(image_window[col]/current_price)-1.0
        mean_volume=image_window['volume'].mean(); image_window['volume']=(image_window['volume']/mean_volume)-1.0 if mean_volume>0 else 0
        image_obs=np.expand_dims(image_window.values,axis=0).astype(np.float32)
        pos_type=np.sign(self.position_amount); pnl_norm,duration_norm=0,0
        if self.position_amount!=0:
            pnl=(self._get_current_price()-self.entry_price)*self.position_amount; entry_value=self.entry_price*abs(self.position_amount); pnl_norm=pnl/entry_value if entry_value>0 else 0; pnl_norm=np.clip(pnl_norm,-5,5)
            duration_norm=(self.current_step-self.entry_step)/self.cfg.MAX_TRADE_DURATION
        operational_state=np.array([pos_type,pnl_norm,duration_norm]); analytical_state=self.state_features.iloc[self.current_step].values; state_obs=np.concatenate([operational_state,analytical_state]).astype(np.float32)
        return {"image":image_obs,"state":state_obs}
    def _get_current_price(self) -> float: return self.prices_df.iloc[self.current_step]['close']
    def _get_current_atr(self) -> float: return self.prices_df.iloc[self.current_step]['atr_value']

    def step(self, action: int):
        try:
            realized_pnl = 0.0
            transaction_penalty = 0.0
            hold_reward = 0.0

            if self.position_amount != 0:
                low, high = self.prices_df.iloc[self.current_step][['low', 'high']]
                is_long = self.position_amount > 0
                if (is_long and low <= self.stop_loss_price) or (not is_long and high >= self.stop_loss_price):
                    realized_pnl = self._close_position(self.stop_loss_price, "SL")
                elif (is_long and high >= self.take_profit_price) or (not is_long and low <= self.take_profit_price):
                    realized_pnl = self._close_position(self.take_profit_price, "TP")
                elif (self.current_step - self.entry_step) >= self.cfg.MAX_TRADE_DURATION:
                    realized_pnl = self._close_position(self._get_current_price(), "Max Duration")

            current_pos_sign = np.sign(self.position_amount)
            price = self._get_current_price()
            if action == 0:
                if current_pos_sign != 0:
                    hold_reward = self.cfg.HOLD_REWARD_BONUS
            elif action == 1 and current_pos_sign != 1:
                transaction_penalty = self.cfg.TRANSACTION_COST_PENALTY
                if current_pos_sign == -1:
                    realized_pnl += self._close_position(price, "Flip to Long")
                self._open_position(price, is_long=True)
            elif action == 2 and current_pos_sign != -1:
                transaction_penalty = self.cfg.TRANSACTION_COST_PENALTY
                if current_pos_sign == 1:
                    realized_pnl += self._close_position(price, "Flip to Short")
                self._open_position(price, is_long=False)

            self.current_step += 1

            current_unrealized_pnl = (self._get_current_price() - self.entry_price) * self.position_amount if self.position_amount != 0 else 0
            pnl_change = current_unrealized_pnl - self.last_unrealized_pnl
            self.last_unrealized_pnl = current_unrealized_pnl

            pnl_based_reward = (realized_pnl + pnl_change) / self.cfg.INITIAL_BALANCE
            reward = pnl_based_reward + hold_reward - transaction_penalty
            
            self.equity = self.balance + current_unrealized_pnl
            done = self.current_step >= len(self.image_features) - 1 or self.equity <= 0

            if done and self.position_amount != 0:
                final_pnl = self._close_position(self._get_current_price(), "End of Episode")
                reward += final_pnl / self.cfg.INITIAL_BALANCE
                self.equity = self.balance

            return self._get_observation(), reward, done, False, {'equity': self.equity}
        except Exception as e:
            logger.critical(f"Критическая ошибка на шаге {self.current_step}: {e}", exc_info=True)
            obs = self._get_observation()
            return obs, 0, True, False, {'equity': self.equity}

    def _open_position(self, price: float, is_long: bool):
        self.entry_step = self.current_step; atr = self._get_current_atr(); sl, tp = self.cfg.ATR_SL_MULTIPLIER, self.cfg.ATR_TP_MULTIPLIER
        self.stop_loss_price = price - (atr * sl) if is_long else price + (atr * sl); self.take_profit_price = price + (atr * tp) if is_long else price - (atr * tp)
        order_size = self.equity * self.cfg.ORDER_SIZE_RATIO
        if self.balance > order_size and order_size > 0:
            fee = order_size * self.cfg.TRANSACTION_FEE; self.total_fees += fee; self.balance -= order_size; self.position_amount = (order_size - fee) / price * (1 if is_long else -1); self.entry_price = price
            self.last_unrealized_pnl = 0.0
    
    def _close_position(self, price: float, reason: str) -> float:
        self.close_reasons[reason] += 1
        duration = self.current_step - self.entry_step; self.trade_durations.append(duration); size, is_long = abs(self.position_amount), self.position_amount > 0
        entry_value = size * self.entry_price; gross_close_value = size * price; fee = gross_close_value * self.cfg.TRANSACTION_FEE; self.total_fees += fee
        net_close_value = gross_close_value - fee; pnl = net_close_value - entry_value if is_long else entry_value - net_close_value
        if pnl > 0: self.gross_profit += pnl
        else: self.gross_loss += pnl
        self.balance += entry_value + pnl; self.trades.append(pnl)
        self.position_amount, self.entry_price = 0.0, 0.0
        self.last_unrealized_pnl = 0.0
        return pnl

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, eval_freq: int, patience: int, model_save_path: str, verbose: int = 1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.model_save_path = model_save_path
        self.best_metric = -np.inf
        self.patience_counter = 0
        self.best_model_path = ""
        os.makedirs(self.model_save_path, exist_ok=True)
    def _on_step(self) -> bool:
        # Увеличиваем eval_freq, так как n_steps стал больше
        effective_eval_freq = self.eval_freq * (self.model.n_steps / 2048)
        if self.n_calls > 0 and self.n_calls % effective_eval_freq == 0:
            metric = self._evaluate_performance()
            logger.info(f"Callback Step {self.n_calls}: Eval Metric (Final Equity): {metric:.2f}, Best: {self.best_metric:.2f}")
            if self.verbose > 0: print(f"\n---"); print(f"🧠 EarlyStopping Check @ Step {self.n_calls}:"); print(f"   - Validation Equity: ${metric:,.2f}"); print(f"   - Best Equity So Far: ${self.best_metric:,.2f}")
            if metric > self.best_metric:
                self.best_metric = metric; self.patience_counter = 0; self.best_model_path = os.path.join(self.model_save_path, f"best_model_step_{self.n_calls}.zip"); self.model.save(self.best_model_path)
                logger.info(f"New best model found with metric {metric:.2f}. Saved to {self.best_model_path}")
                if self.verbose > 0: print(f"   - ✨ New best model found! Saved to {self.best_model_path}")
            else:
                self.patience_counter += 1
                if self.verbose > 0: print(f"   - ⏳ No improvement. Patience: {self.patience_counter}/{self.patience}")
            if self.patience_counter >= self.patience:
                if self.verbose > 0: print(f"\nStopping training early: metric hasn't improved for {self.patience} checks.")
                logger.warning(f"Stopping training early at step {self.n_calls}.")
                return False
            if self.verbose > 0: print(f"---")
        return True
    def _evaluate_performance(self) -> float:
        eval_vec_env = DummyVecEnv([lambda: self.eval_env]); obs = eval_vec_env.reset(); done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True); obs, _, terminated, truncated = eval_vec_env.step(action); done = terminated[0] or truncated[0]
        final_equity = eval_vec_env.get_attr('equity')[0]
        return final_equity

def main():
    try:
        logger.info("="*20 + " ЗАПУСК СИСТЕМЫ V13.20 (Обогащение данных) " + "="*20)
        device=setup_gpu_support(); get_gpu_memory_info(device)
        
        data_paths={'5m':'data/BTCUSDT_5m_2y.csv','1h':'data/BTCUSDT_1h_2y.csv','4h':'data/BTCUSDT_4h_2y.csv','1d':'data/BTCUSDT_1d_2y.csv'}
        data_loader = MTFDataLoader(data_paths)
        prices_df,image_features,state_features=data_loader.load_and_prepare_data()
        
        train_end_idx = int(len(prices_df) * 0.7); val_end_idx = int(len(prices_df) * 0.8)
        train_prices, val_prices, test_prices = prices_df.iloc[:train_end_idx], prices_df.iloc[train_end_idx:val_end_idx], prices_df.iloc[val_end_idx:]
        train_image, val_image, test_image = image_features.iloc[:train_end_idx], image_features.iloc[train_end_idx:val_end_idx], image_features.iloc[val_end_idx:]
        train_state, val_state, test_state = state_features.iloc[:train_end_idx], state_features.iloc[train_end_idx:val_end_idx], state_features.iloc[val_end_idx:]
        msg = f"Данные разделены: {len(train_prices)} обуч., {len(val_prices)} валид., {len(test_prices)} тест."; logger.info(msg); print(f"✅ {msg}")
        
        train_env=DummyVecEnv([lambda:TradingEnv(train_prices, train_image, train_state)])
        val_env = TradingEnv(val_prices, val_image, val_state)

        policy_kwargs=dict(features_extractor_class=CustomCombinedExtractor,features_extractor_kwargs=dict(features_dim=512),net_arch=dict(pi=[256,128],vf=[256,128]))
        model=PPO('MultiInputPolicy', train_env, policy_kwargs=policy_kwargs,learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF, n_steps=TrendTraderConfig.N_STEPS, batch_size=128, gamma=TrendTraderConfig.GAMMA, verbose=0, device=device)
        
        # Передаем eval_freq как есть, колбэк сам скорректирует его
        early_stopping_callback = EarlyStoppingCallback(eval_env=val_env, eval_freq=5000, patience=5, model_save_path='models/')
        logger.info("Модель PPO создана. Начало обучения с Early Stopping...")
        print("\n🎓 ЭТАП 4: ОПТИМАЛЬНОЕ ОБУЧЕНИЕ..."); 
        with tqdm(total=TrendTraderConfig.TOTAL_TIMESTEPS, desc="Обучение", unit=" шагов") as pbar:
            tqdm_callback = TqdmCallback(pbar); model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS, callback=[tqdm_callback, early_stopping_callback], reset_num_timesteps=False)
        
        if not early_stopping_callback.best_model_path:
            logger.warning("Обучение завершено без улучшений. Тестирование последней модели.")
            print("\n⚠️ Обучение завершено без улучшений. Тестирование последней модели.")
            model.save("models/last_model.zip")
            best_model_path = "models/last_model.zip"
        else:
            best_model_path = early_stopping_callback.best_model_path
            logger.info(f"Обучение завершено. Лучшая модель сохранена в: {best_model_path}")
            print(f"\n✅ Обучение остановлено. Лучшая модель: {best_model_path}")


        print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ ЛУЧШЕЙ МОДЕЛИ...")
        model = PPO.load(best_model_path, env=train_env)
        
        test_env = TradingEnv(test_prices, test_image, test_state)
        obs, _ = test_env.reset(); equity_history, price_history = [test_env.equity], [test_env._get_current_price()]; done=False
        
        test_steps = len(test_prices) - test_env.cfg.WINDOW_SIZE - 1
        for _ in tqdm(range(test_steps), desc="Тестирование"):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = test_env.step(int(action))
            equity_history.append(info['equity'])
            price_history.append(test_env._get_current_price())
            done = terminated or truncated
            if done: break
        
        logger.info("Тестирование завершено. Расчет финальных метрик.")
        print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ"); 
        equity_series = pd.Series(equity_history, index=test_prices.index[test_env.cfg.WINDOW_SIZE : test_env.cfg.WINDOW_SIZE + len(equity_history)])
        peak = equity_series.expanding(min_periods=1).max(); drawdown = (equity_series - peak) / peak; max_drawdown = drawdown.min()
        returns = equity_series.pct_change().dropna(); annualization_factor = np.sqrt(252 * 288)
        profit_factor = abs(test_env.gross_profit / test_env.gross_loss) if test_env.gross_loss != 0 else float('inf')
        sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor if returns.std() > 0 else 0
        avg_duration_steps = np.mean(test_env.trade_durations) if test_env.trade_durations else 0; avg_duration_hours = (avg_duration_steps * 5) / 60
        wins = [t for t in test_env.trades if t > 0]; losses = [t for t in test_env.trades if t < 0]
        avg_win = np.mean(wins) if wins else 0; avg_loss = np.mean(losses) if losses else 0
        reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        initial, final = equity_history[0], equity_history[-1]; total_return=(final-initial)/initial*100
        bnh_return=(price_history[-1]-price_history[0])/price_history[0]*100
        trades=len(test_env.trades); win_rate=(len(wins)/trades)*100 if trades > 0 else 0
        
        print("="*60); print(f"💰 Финальный баланс: ${final:,.2f} (Начальный: ${initial:,.2f})"); print(f"📈 Доходность стратегии: {total_return:+.2f}%"); print(f"📊 Доходность Buy & Hold: {bnh_return:+.2f}%"); print("-"*60)
        print("📈 Детальная статистика PnL:"); print(f"   🟢 Валовая прибыль: ${test_env.gross_profit:,.2f}"); print(f"   🔴 Валовый убыток:  ${test_env.gross_loss:,.2f}"); print(f"   💸 Всего комиссий:  ${test_env.total_fees:,.2f}"); print(f"   ⚖️ Профит-фактор:    {profit_factor:.2f}"); print("-"*60)
        print("🛡️ Статистика Риска и Дисциплины:"); print(f"   📉 Макс. просадка:   {max_drawdown:.2%}"); print(f"   ⏳ Средняя длительность сделки: {avg_duration_hours:.2f} часов"); print(f"   🏆 Соотношение Прибыль/Риск:  {reward_risk_ratio:.2f}"); print(f"   ⚡ Коэффициент Шарпа (год.): {sharpe_ratio:.2f}"); print("-"*60)
        print("🤔 Причины закрытия сделок:")
        for reason, count in test_env.close_reasons.items(): print(f"   - {reason}: {count} раз")
        print("-"*60)
        print(f"🔄 Всего сделок: {trades}"); print(f"✅ Процент прибыльных сделок: {win_rate:.1f}%")
        
        plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(15,7)); plt.title(f'V13.20 - Data Enrichment\nReturn: {total_return:.2f}% | Trades: {trades} | Win Rate: {win_rate:.1f}% | R/R: {reward_risk_ratio:.2f} | P/F: {profit_factor:.2f}'); ax1=plt.gca(); ax1.plot(equity_series, label='Equity',c='royalblue'); ax1.set_xlabel('Дата'); ax1.set_ylabel('Equity ($)',color='royalblue'); ax2=ax1.twinx(); ax2.plot(equity_series.index, price_history, label='Цена BTC',c='darkorange',alpha=0.6); ax2.set_ylabel('Цена ($)',color='darkorange'); ax1.legend(loc='upper left'); ax2.legend(loc='upper right'); plt.savefig('results_v13.20.png'); plt.close(); print("✅ График сохранен в 'results_v13.20.png'")
        logger.info("Скрипт успешно завершил работу.")
    
    except Exception as e:
        logger.critical(f"Неперехваченная ошибка в main: {e}", exc_info=True)
        print(f"❌ Критическая ошибка! Подробности в файле 'trading_v13_20.log'.")

if __name__ == "__main__":
    main()