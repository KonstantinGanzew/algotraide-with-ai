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
    if torch.cuda.is_available(): device = torch.device("cuda"); gpu_name = torch.cuda.get_device_name(0); print(f"🚀 NVIDIA CUDA: {gpu_name}"); return device
    else: device = torch.device("cpu"); print(f"💻 CPU: {device}"); return device
def get_gpu_memory_info(device):
    if device and device.type == "cuda":
        try: total = torch.cuda.get_device_properties(device).total_memory/1e9; allocated = torch.cuda.memory_allocated(device)/1e9; print(f"📊 GPU память: {allocated:.1f}GB / {total:.1f}GB")
        except: pass

"""
🚀 ТОРГОВАЯ СИСТЕМА V13.1 - ИЕРАРХИЧЕСКИЙ АНАЛИТИК С ЭКОНОМИЧЕСКОЙ ЛОГИКОЙ
✅ ЦЕЛЬ: Устранить хаотичную торговлю путем введения корректной экономической модели.
✅ КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:
   1. ПОШАГОВАЯ НАГРАДА (REWARD SHAPING): Агент теперь получает награду на каждом шаге, равную изменению equity. Это мгновенно штрафует его за комиссию при открытии сделки и поощряет удержание прибыльных позиций.
   2. КОРРЕКТНАЯ НОРМАЛИЗАЦИЯ PNL: PnL в состоянии (state) теперь нормализуется относительно размера вложенного капитала, что дает сети осмысленный процентный показатель.
   3. УЛУЧШЕННАЯ НОРМАЛИЗАЦИЯ ДАННЫХ: Данные для CNN ('image') нормализуются более логично: цены относительно текущей цены, объем - относительно своего среднего.
✅ ОЖИДАЕМЫЙ РЕЗУЛЬТАТ: Значительное снижение количества сделок, обучение осмысленному удержанию позиций и более стабильные результаты.
"""

# CustomCombinedExtractor остается без изменений
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        image_space, state_space = observation_space['image'], observation_space['state']
        n_input_channels = image_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, (3, 3), 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3, 3), 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(image_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten + state_space.shape[0], features_dim), nn.ReLU())
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_out = self.cnn(obs['image']); return self.linear(torch.cat([cnn_out, obs['state']], dim=1))

class TrendTraderConfig:
    INITIAL_BALANCE = 10000; TRANSACTION_FEE = 0.001; WINDOW_SIZE = 64
    ORDER_SIZE_RATIO = 0.10; ATR_SL_MULTIPLIER = 2.0; ATR_TP_MULTIPLIER = 4.0
    TOTAL_TIMESTEPS = 2000000; LEARNING_RATE = 1e-4
    ENTROPY_COEF = 0.01; GAMMA = 0.99; MAX_TRADE_DURATION = 288 # 24 часа в 5-минутках

# MTFDataLoader остается без изменений
class MTFDataLoader:
    def __init__(self, data_paths: Dict[str, str]): self.paths = data_paths
    def _calc_indicators(self, df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        df[f'trend_{suffix}']=np.sign(df['close']-df['close'].ewm(span=50,adjust=False).mean())
        delta=df['close'].diff();gain=(delta.where(delta>0,0)).rolling(14).mean();loss=(-delta.where(delta<0,0)).rolling(14).mean();df[f'rsi_{suffix}']=100-(100/(1+gain/loss))
        tr=pd.concat([df['high']-df['low'],np.abs(df['high']-df['close'].shift()),np.abs(df['low']-df['close'].shift())],axis=1).max(axis=1);df[f'atr_{suffix}']=tr.ewm(span=14,adjust=False).mean()
        return df
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("📊 Иерархическая загрузка данных..."); dfs={tf:self._calc_indicators(pd.read_csv(p).assign(timestamp=lambda x:pd.to_datetime(x['timestamp'],unit='ms')),tf)for tf,p in self.paths.items()}
        merged_df=dfs['5m']; [merged_df:=pd.merge_asof(merged_df.sort_values('timestamp'),dfs[tf][['timestamp',f'trend_{tf}',f'rsi_{tf}',f'atr_{tf}']].sort_values('timestamp'),on='timestamp',direction='backward') for tf in ['1h','4h','1d']]
        merged_df.replace([np.inf,-np.inf],np.nan,inplace=True); merged_df.dropna(inplace=True)
        image_features=merged_df[['open','high','low','close','volume']].reset_index(drop=True)
        state_features=pd.DataFrame(index=merged_df.index); state_features['rsi_5m_norm']=(merged_df['rsi_5m']-50)/50; state_features['atr_5m_norm']=merged_df['atr_5m']/merged_df['close']
        for tf in ['1h','4h','1d']: state_features[f'trend_{tf}']=merged_df[f'trend_{tf}']; state_features[f'rsi_{tf}_norm']=(merged_df[f'rsi_{tf}']-50)/50
        state_features=state_features.reset_index(drop=True)
        prices_df=merged_df[['timestamp','open','high','low','close','atr_5m']].reset_index(drop=True); prices_df.rename(columns={'atr_5m':'atr_value'},inplace=True)
        print(f"✅ Данные подготовлены. Image: {image_features.shape}, State: {state_features.shape}"); return prices_df,image_features,state_features

class TradingEnv(gym.Env):
    def __init__(self, prices_df: pd.DataFrame, image_features: pd.DataFrame, state_features: pd.DataFrame):
        super().__init__()
        self.prices_df, self.image_features, self.state_features = prices_df, image_features, state_features
        self.cfg = TrendTraderConfig()
        self.action_space = spaces.Discrete(3) # 0: Sell, 1: Hold, 2: Buy
        self.image_shape = (1, self.cfg.WINDOW_SIZE, self.image_features.shape[1])
        self.state_shape = (3 + self.state_features.shape[1],)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=-1, high=2, shape=self.image_shape, dtype=np.float32),
            "state": spaces.Box(low=-2, high=2, shape=self.state_shape, dtype=np.float32)
        })
        self._reset_state()
    
    def _reset_state(self):
        self.balance, self.equity = self.cfg.INITIAL_BALANCE, self.cfg.INITIAL_BALANCE
        self.current_step = self.cfg.WINDOW_SIZE; self.position_amount = 0.0; self.entry_price = 0.0
        self.entry_step = 0; self.stop_loss_price = 0.0; self.take_profit_price = 0.0; self.trades = []

    def reset(self, seed=None, options=None): super().reset(seed=seed); self._reset_state(); return self._get_observation(), {}
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        # --- Image Observation (История цен) ---
        # ИЗМЕНЕНИЕ 1: Улучшенная, более стабильная нормализация "картинки"
        image_window = self.image_features.iloc[self.current_step - self.cfg.WINDOW_SIZE : self.current_step].copy()
        current_price = image_window.iloc[-1]['close'] if image_window.iloc[-1]['close'] > 0 else 1
        
        # Нормализуем OHLC относительно текущей цены
        for col in ['open', 'high', 'low', 'close']:
            image_window[col] = (image_window[col] / current_price) - 1.0
        # Нормализуем объем отдельно, чтобы избежать искажения масштаба
        mean_volume = image_window['volume'].mean()
        image_window['volume'] = (image_window['volume'] / mean_volume) -1.0 if mean_volume > 0 else 0
        
        image_obs = np.expand_dims(image_window.values, axis=0).astype(np.float32)

        # --- State Observation (Текущий Контекст) ---
        pos_type = np.sign(self.position_amount); pnl_norm, duration_norm = 0, 0
        if self.position_amount != 0:
            pnl = (self._get_current_price() - self.entry_price) * self.position_amount
            # ИЗМЕНЕНИЕ 2: Корректная нормализация PnL относительно стоимости входа
            entry_value = self.entry_price * abs(self.position_amount)
            pnl_norm = pnl / entry_value if entry_value > 0 else 0
            duration_norm = (self.current_step - self.entry_step) / self.cfg.MAX_TRADE_DURATION
        operational_state = np.array([pos_type, pnl_norm, duration_norm])
        analytical_state = self.state_features.iloc[self.current_step].values
        state_obs = np.concatenate([operational_state, analytical_state]).astype(np.float32)
        
        return {"image": image_obs, "state": state_obs}

    def _get_current_price(self)->float: return self.prices_df.iloc[self.current_step]['close']
    def _get_current_atr(self)->float: return self.prices_df.iloc[self.current_step]['atr_value']
    
    def step(self, action:int):
        # ИЗМЕНЕНИЕ 3: Внедрение пошаговой награды (Reward Shaping)
        previous_equity = self.equity # Запоминаем equity до совершения действий
        
        # 1. Проверяем SL/TP
        if self.position_amount != 0:
            low, high = self.prices_df.iloc[self.current_step][['low', 'high']]
            is_long = self.position_amount > 0
            if (is_long and low <= self.stop_loss_price) or (not is_long and high >= self.stop_loss_price):
                self._close_position(self.stop_loss_price)
            elif (is_long and high >= self.take_profit_price) or (not is_long and low <= self.take_profit_price):
                self._close_position(self.take_profit_price)

        # 2. Обрабатываем действие агента (0:Sell, 1:Hold, 2:Buy)
        current_pos = np.sign(self.position_amount)
        # В прошлой версии было action-1, что давало (-1, 0, 1). Поменяем на более явную логику.
        # 0 -> Sell (-1), 1 -> Hold (0), 2 -> Buy (1)
        desired_pos = action - 1 

        if current_pos != desired_pos:
            price = self._get_current_price()
            if current_pos != 0: self._close_position(price)
            if desired_pos != 0: self._open_position(price, is_long=(desired_pos == 1))
        
        # 3. Обновляем состояние на конец шага
        self.current_step += 1
        unrealized_pnl = (self._get_current_price() - self.entry_price) * self.position_amount if self.position_amount != 0 else 0
        self.equity = self.balance + unrealized_pnl
        
        # 4. Расчет награды и завершение эпизода
        # Награда - это изменение equity, нормализованное на начальный баланс
        reward = (self.equity - previous_equity) / self.cfg.INITIAL_BALANCE
        
        done = self.current_step >= len(self.image_features) - 1 or self.equity <= 0
        if done and self.position_amount != 0:
            self._close_position(self._get_current_price()) # Закрываем последнюю сделку
            # Финальное обновление equity для корректного расчета последней награды
            self.equity = self.balance 
            reward = (self.equity - previous_equity) / self.cfg.INITIAL_BALANCE

        return self._get_observation(), reward, done, False, {'equity': self.equity}

    def _open_position(self, price, is_long):
        self.entry_step=self.current_step; atr=self._get_current_atr(); sl,tp=self.cfg.ATR_SL_MULTIPLIER,self.cfg.ATR_TP_MULTIPLIER
        self.stop_loss_price=price-(atr*sl) if is_long else price+(atr*sl)
        self.take_profit_price=price+(atr*tp) if is_long else price-(atr*tp)
        order_size=self.balance*self.cfg.ORDER_SIZE_RATIO
        if self.balance > 0 and order_size > 0:
            self.balance -= (order_size * (1 + self.cfg.TRANSACTION_FEE)) # Комиссия при покупке
            self.position_amount = (order_size / price) * (1 if is_long else -1)
            self.entry_price = price

    def _close_position(self, price):
        # Эта функция теперь не возвращает награду, а только обновляет состояние баланса
        size, is_long = abs(self.position_amount), self.position_amount > 0
        close_value = size * price * (1 - self.cfg.TRANSACTION_FEE) # Комиссия при продаже
        entry_value = size * self.entry_price
        
        pnl = (close_value - entry_value) if is_long else (entry_value - (size * price * (1 + self.cfg.TRANSACTION_FEE)))

        self.balance += entry_value + pnl
        self.trades.append(pnl)
        self.position_amount, self.entry_price = 0.0, 0.0

def main():
    print("🚀 СИСТЕМА V13.1 (Экономическая Логика) - ЗАПУСК")
    device = setup_gpu_support(); get_gpu_memory_info(device)
    
    data_paths = {
        '5m': 'data/BTCUSDT_5m_2y.csv', '1h': 'data/BTCUSDT_1h_2y.csv',
        '4h': 'data/BTCUSDT_4h_2y.csv', '1d': 'data/BTCUSDT_1d_2y.csv'
    }
    
    data_loader = MTFDataLoader(data_paths)
    prices_df, image_features, state_features = data_loader.load_and_prepare_data()
    
    split_idx = int(len(prices_df) * 0.8)
    train_prices, test_prices = prices_df.iloc[:split_idx], prices_df.iloc[split_idx:]
    train_image_feats, test_image_feats = image_features.iloc[:split_idx], image_features.iloc[split_idx:]
    train_state_feats, test_state_feats = state_features.iloc[:split_idx], state_features.iloc[split_idx:]
    print(f"✅ Данные разделены: {len(train_prices)} для обучения, {len(test_prices)} для теста.")
    
    env = DummyVecEnv([lambda: TradingEnv(train_prices, train_image_feats, train_state_feats)])
    
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor, features_extractor_kwargs=dict(features_dim=512), net_arch=dict(pi=[256, 128], vf=[256, 128]))
    
    model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs,
                learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF,
                n_steps=4096, batch_size=128, gamma=TrendTraderConfig.GAMMA, verbose=1, device=device)
                
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ 'АНАЛИТИКА С ЭКОНОМИЧЕСКОЙ ЛОГИКОЙ'..."); model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS)
    
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    test_env = TradingEnv(test_prices, test_image_feats, test_state_feats)
    obs, _ = test_env.reset(); equity_history, price_history = [test_env.equity], [test_env._get_current_price()]; done=False
    while not done:
        action, _ = model.predict(obs, deterministic=True); obs, _, terminated, truncated, info = test_env.step(int(action))
        equity_history.append(info['equity'])
        try: price_history.append(test_env._get_current_price())
        except IndexError: price_history.append(price_history[-1])
        done = terminated or truncated

    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ"); initial, final = equity_history[0], equity_history[-1]
    total_return=(final-initial)/initial*100; start_p, end_p=price_history[0],price_history[-1]; bnh_return=(end_p-start_p)/start_p*100
    trades=len(test_env.trades); win_rate=(len([t for t in test_env.trades if t > 0])/trades)*100 if trades > 0 else 0
    print("="*60);print(f"💰 Финальный баланс: ${final:,.2f} (Начальный: ${initial:,.2f})");print(f"📈 Доходность стратегии: {total_return:+.2f}%")
    print(f"📊 Доходность Buy & Hold: {bnh_return:+.2f}%"); print("-"*30); print(f"🔄 Всего сделок: {trades}"); print(f"✅ Процент прибыльных сделок: {win_rate:.1f}%")
    plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(15,7)); plt.title(f'V13.1 - Экономическая Логика\nReturn: {total_return:.2f}%|Trades:{trades}|Win Rate:{win_rate:.1f}%')
    ax1=plt.gca(); ax1.plot(equity_history, label='Equity',c='royalblue'); ax1.set_xlabel('Шаги'); ax1.set_ylabel('Equity ($)',color='royalblue')
    ax2=ax1.twinx(); ax2.plot(price_history, label='Цена BTC',c='darkorange',alpha=0.6); ax2.set_ylabel('Цена ($)',color='darkorange')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right'); plt.savefig('results_v13.1.png'); plt.close(); print("✅ График сохранен в 'results_v13.1.png'")

if __name__=="__main__": main()