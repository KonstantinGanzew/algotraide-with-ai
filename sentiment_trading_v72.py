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
🚀 ТОРГОВАЯ СИСТЕМА V13.0 - ИЕРАРХИЧЕСКИЙ АНАЛИТИК
✅ ЦЕЛЬ: Построить архитектуру, имитирующую мышление трейдера, разделяя контекст и историю.
✅ ИЕРАРХИЯ ДАННЫХ: Данные теперь разделены на два потока:
   1. 'image' (для CNN): Только история ценового движения (OHLCV) на 5м. Показывает "КАК" цена двигалась.
   2. 'state' (для MLP): Весь аналитический контекст на СЕЙЧАС (MTF индикаторы, PnL, тип позиции). Показывает "ЧТО" происходит на рынке.
✅ ОСМЫСЛЕННОЕ ОБУЧЕНИЕ: Эта архитектура заставляет CNN искать свечные паттерны, а MLP - сопоставлять их с глобальным рыночным контекстом. Это должно устранить хаотичное поведение.
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
    TOTAL_TIMESTEPS = 2000000; LEARNING_RATE = 1e-4 # Снижаем LR для более сложной задачи
    ENTROPY_COEF = 0.01; GAMMA = 0.99; MAX_TRADE_DURATION = 288


class MTFDataLoader:
    def __init__(self, data_paths: Dict[str, str]):
        self.paths = data_paths
    def _calc_indicators(self, df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        # EMA Trend
        df[f'trend_{suffix}'] = np.sign(df['close'] - df['close'].ewm(span=50, adjust=False).mean())
        # RSI
        delta=df['close'].diff(); gain=(delta.where(delta>0,0)).rolling(14).mean(); loss=(-delta.where(delta<0,0)).rolling(14).mean()
        df[f'rsi_{suffix}'] = 100-(100/(1+gain/loss))
        # ATR
        tr=pd.concat([df['high']-df['low'],np.abs(df['high']-df['close'].shift()),np.abs(df['low']-df['close'].shift())],axis=1).max(axis=1)
        df[f'atr_{suffix}'] = tr.ewm(span=14, adjust=False).mean()
        return df

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("📊 Иерархическая загрузка данных...")
        dfs = {tf: self._calc_indicators(pd.read_csv(p).assign(timestamp=lambda x: pd.to_datetime(x['timestamp'],unit='ms')), tf) for tf, p in self.paths.items()}
        
        merged_df = dfs['5m']
        for tf in ['1h', '4h', '1d']:
            cols_to_merge = ['timestamp', f'trend_{tf}', f'rsi_{tf}', f'atr_{tf}']
            merged_df=pd.merge_asof(merged_df.sort_values('timestamp'), dfs[tf][cols_to_merge].sort_values('timestamp'), on='timestamp', direction='backward')

        merged_df.replace([np.inf, -np.inf], np.nan, inplace=True); merged_df.dropna(inplace=True)

        # 1. Данные для 'image' (история цен)
        image_features = merged_df[['open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)

        # 2. Данные для 'state' (текущий контекст)
        state_features = pd.DataFrame(index=merged_df.index)
        state_features['rsi_5m_norm'] = (merged_df['rsi_5m'] - 50) / 50
        state_features['atr_5m_norm'] = merged_df['atr_5m'] / merged_df['close']
        for tf in ['1h', '4h', '1d']:
            state_features[f'trend_{tf}'] = merged_df[f'trend_{tf}']
            state_features[f'rsi_{tf}_norm'] = (merged_df[f'rsi_{tf}'] - 50) / 50
        state_features = state_features.reset_index(drop=True)

        # 3. Данные для самой среды (цены для PnL и ATR для SL/TP)
        prices_df = merged_df[['timestamp', 'open', 'high', 'low', 'close', 'atr_5m']].reset_index(drop=True)
        prices_df.rename(columns={'atr_5m': 'atr_value'}, inplace=True)
        
        print(f"✅ Данные подготовлены. Image: {image_features.shape}, State: {state_features.shape}")
        return prices_df, image_features, state_features

class TradingEnv(gym.Env):
    def __init__(self, prices_df: pd.DataFrame, image_features: pd.DataFrame, state_features: pd.DataFrame):
        super().__init__()
        self.prices_df = prices_df; self.image_features = image_features; self.state_features = state_features
        self.cfg = TrendTraderConfig()
        
        self.action_space = spaces.Discrete(3)
        self.image_shape = (1, self.cfg.WINDOW_SIZE, self.image_features.shape[1])
        # [pos_type, pnl, duration] + индикаторы из state_features
        self.state_shape = (3 + self.state_features.shape[1],)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=-1, high=2, shape=self.image_shape, dtype=np.float32),
            "state": spaces.Box(low=-2, high=2, shape=self.state_shape, dtype=np.float32)
        })
        self._reset_state()
    
    def _reset_state(self):
        # ... без изменений ...
        self.balance, self.equity = self.cfg.INITIAL_BALANCE, self.cfg.INITIAL_BALANCE
        self.current_step = self.cfg.WINDOW_SIZE; self.position_amount = 0.0; self.entry_price = 0.0
        self.entry_step = 0; self.stop_loss_price = 0.0; self.take_profit_price = 0.0; self.trades = []

    def reset(self, seed=None, options=None): super().reset(seed=seed); self._reset_state(); return self._get_observation(), {}
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        # --- Image Observation (История цен) ---
        image_window = self.image_features.iloc[self.current_step - self.cfg.WINDOW_SIZE : self.current_step]
        # Нормализация окна относительно цены открытия этого окна
        start_price = image_window.iloc[0]['close']
        normalized_image_window = (image_window / start_price) - 1.0
        image_obs = np.expand_dims(normalized_image_window.values, axis=0).astype(np.float32)

        # --- State Observation (Текущий Контекст) ---
        # 1. Оперативный state (внутри позиции)
        pos_type = np.sign(self.position_amount)
        if self.position_amount != 0:
            pnl = (self._get_current_price() - self.entry_price) * self.position_amount
            atr = self._get_current_atr(); pnl_norm = pnl/(abs(self.position_amount)*atr) if atr > 0 else 0
            duration_norm = (self.current_step - self.entry_step) / self.cfg.MAX_TRADE_DURATION
        else: pnl_norm, duration_norm = 0, 0
        operational_state = np.array([pos_type, pnl_norm, duration_norm])

        # 2. Аналитический state (индикаторы)
        analytical_state = self.state_features.iloc[self.current_step].values

        # 3. Объединение
        state_obs = np.concatenate([operational_state, analytical_state]).astype(np.float32)
        
        return {"image": image_obs, "state": state_obs}

    # Методы _get_current_price, _get_current_atr, step, _open_position, _close_position без изменений
    def _get_current_price(self)->float: return self.prices_df.iloc[self.current_step]['close']
    def _get_current_atr(self)->float: return self.prices_df.iloc[self.current_step]['atr_value']
    def step(self, action:int):
        price=self._get_current_price(); reward=0.0
        if self.position_amount!=0:
            low,high=self.prices_df.iloc[self.current_step][['low','high']]; is_long=self.position_amount>0
            if (is_long and low<=self.stop_loss_price) or (not is_long and high>=self.stop_loss_price): reward=self._close_position(self.stop_loss_price)
            elif (is_long and high>=self.take_profit_price) or (not is_long and low<=self.take_profit_price): reward=self._close_position(self.take_profit_price)
        current_pos,desired_pos=np.sign(self.position_amount),action-1
        if current_pos!=desired_pos:
            if current_pos!=0: reward+=self._close_position(price)
            if desired_pos!=0: self._open_position(price,is_long=(desired_pos==1))
        self.current_step+=1
        pnl_u=(self._get_current_price()-self.entry_price)*self.position_amount if self.position_amount!=0 else 0
        self.equity=self.balance+pnl_u; done=self.current_step>=len(self.image_features)-1 or self.equity<=0
        if done and self.position_amount!=0: reward+=self._close_position(self._get_current_price())
        return self._get_observation(),reward,done,False,{'equity':self.equity}
    def _open_position(self, price, is_long):
        self.entry_step=self.current_step; atr=self._get_current_atr(); sl,tp=self.cfg.ATR_SL_MULTIPLIER,self.cfg.ATR_TP_MULTIPLIER
        self.stop_loss_price=price-(atr*sl) if is_long else price+(atr*sl); self.take_profit_price=price+(atr*tp) if is_long else price-(atr*tp)
        order_size=self.balance*self.cfg.ORDER_SIZE_RATIO
        if self.balance>0 and order_size>0: self.balance-=(order_size*(1+self.cfg.TRANSACTION_FEE)); self.position_amount=(order_size/price)*(1 if is_long else-1); self.entry_price=price
    def _close_position(self,price):
        size,is_long=abs(self.position_amount),self.position_amount>0; close_value=size*price
        entry_value=size*self.entry_price; pnl=close_value-entry_value if is_long else entry_value-close_value
        pnl-=(entry_value+close_value)*self.cfg.TRANSACTION_FEE # Упрощенный расчет комиссии
        self.balance+=entry_value+pnl; self.trades.append(pnl); reward=pnl/self.cfg.INITIAL_BALANCE
        self.position_amount,self.entry_price=0.0,0.0; return reward

def main():
    print("🚀 СИСТЕМА V13.0 (Иерархический Аналитик) - ЗАПУСК")
    device = setup_gpu_support(); get_gpu_memory_info(device)
    
    data_paths = {
        '5m': 'data/BTCUSDT_5m_2y.csv', '1h': 'data/BTCUSDT_1h_2y.csv',
        '4h': 'data/BTCUSDT_4h_2y.csv', '1d': 'data/BTCUSDT_1d_2y.csv'
    }
    
    # 1. Загружаем все три набора данных
    data_loader = MTFDataLoader(data_paths)
    prices_df, image_features, state_features = data_loader.load_and_prepare_data()
    
    # 2. Делим каждый набор данных на train/test
    split_idx = int(len(prices_df) * 0.8)
    train_prices, test_prices = prices_df.iloc[:split_idx], prices_df.iloc[split_idx:]
    train_image_feats, test_image_feats = image_features.iloc[:split_idx], image_features.iloc[split_idx:]
    train_state_feats, test_state_feats = state_features.iloc[:split_idx], state_features.iloc[split_idx:]
    print(f"✅ Данные разделены: {len(train_prices)} для обучения, {len(test_prices)} для теста.")
    
    # 3. Передаем все три набора в среду
    env = DummyVecEnv([lambda: TradingEnv(train_prices, train_image_feats, train_state_feats)])
    
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor, features_extractor_kwargs=dict(features_dim=512), net_arch=dict(pi=[256, 128], vf=[256, 128]))
    
    model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs,
                learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF,
                n_steps=4096, batch_size=128, gamma=TrendTraderConfig.GAMMA, verbose=1, device=device) # Увеличены n_steps и batch_size
                
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ 'ИЕРАРХИЧЕСКОГО АНАЛИТИКА'..."); model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS)
    
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    test_env = TradingEnv(test_prices, test_image_feats, test_state_feats)
    # ... остальной код для тестирования и отрисовки графика без изменений ...
    obs, _ = test_env.reset(); equity_history, price_history = [test_env.equity], [test_env._get_current_price()]; done=False
    while not done:
        action, _ = model.predict(obs, deterministic=True); obs, _, terminated, truncated, info = test_env.step(int(action))
        equity_history.append(info['equity'])
        try: price_history.append(test_env._get_current_price())
        except IndexError: price_history.append(price_history[-1])
        done=terminated or truncated
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ"); initial, final = equity_history[0], equity_history[-1]
    total_return=(final-initial)/initial*100; start_p, end_p=price_history[0],price_history[-1]; bnh_return=(end_p-start_p)/start_p*100
    trades=len(test_env.trades); win_rate=(len([t for t in test_env.trades if t > 0])/trades)*100 if trades > 0 else 0
    print("="*60);print(f"💰 Финальный баланс: ${final:,.2f} (Начальный: ${initial:,.2f})");print(f"📈 Доходность стратегии: {total_return:+.2f}%")
    print(f"📊 Доходность Buy & Hold: {bnh_return:+.2f}%"); print("-"*30); print(f"🔄 Всего сделок: {trades}"); print(f"✅ Процент прибыльных сделок: {win_rate:.1f}%")
    plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(15,7)); plt.title(f'V13.0 - Иерархический Аналитик\nReturn: {total_return:.2f}%|Trades:{trades}|Win Rate:{win_rate:.1f}%')
    ax1=plt.gca(); ax1.plot(equity_history, label='Equity',c='royalblue'); ax1.set_xlabel('Шаги'); ax1.set_ylabel('Equity ($)',color='royalblue')
    ax2=ax1.twinx(); ax2.plot(price_history, label='Цена BTC',c='darkorange',alpha=0.6); ax2.set_ylabel('Цена ($)',color='darkorange')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right'); plt.savefig('results_v13.png'); plt.close(); print("✅ График сохранен в 'results_v13.png'")

if __name__=="__main__": main()


📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ
============================================================
💰 Финальный баланс: $368.49 (Начальный: $10,000.00)
📈 Доходность стратегии: -96.32%
📊 Доходность Buy & Hold: +16.73%
------------------------------
🔄 Всего сделок: 10886
✅ Процент прибыльных сделок: 10.8%
✅ График сохранен в 'results_v13.png'