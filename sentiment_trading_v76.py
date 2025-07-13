import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback # ✅ ИМПОРТ для правильного коллбэка
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List
import warnings
from tqdm import tqdm

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
🚀 ТОРГОВАЯ СИСТЕМА V13.4 - АТАКА НА ГИПЕРАКТИВНОСТЬ (С ИСПРАВЛЕНИЯМИ)
✅ ЦЕЛЬ: Подавить иррациональную торговлю путем тонкой настройки экономических стимулов и гиперпараметров обучения.
✅ КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:
   1. ПРЯМОЙ ШТРАФ ЗА ТРАНЗАКЦИЮ: Введен явный отрицательный reward за каждое торговое действие.
   2. СНИЖЕНИЕ ЭНТРОПИИ и УВЕЛИЧЕНИЕ ГОРИЗОНТА ПЛАНИРОВАНИЯ: Настройка `ent_coef` и `n_steps`.
   3. УЛУЧШЕННАЯ ДИАГНОСТИКА: Добавлен прогресс-бар и детальный отчет по PnL и комиссиям.
   4. ИСПРАВЛЕНИЕ ОШИБКИ: Коллбэк для прогресс-бара реализован через правильный класс, а не словарь.
"""

# ✅ НОВИНКА: Правильный класс для коллбэка
class TqdmCallback(BaseCallback):
    def __init__(self, pbar):
        super().__init__(verbose=0)
        self.pbar = pbar

    def _on_step(self) -> bool:
        # Обновляем прогресс-бар на 1 шаг
        self.pbar.update(1)
        return True

# CustomCombinedExtractor, MTFDataLoader, TrendTraderConfig, TradingEnv без изменений
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim); image_space, state_space = observation_space['image'], observation_space['state']; n_input_channels = image_space.shape[0]
        self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, 32, (3,3), 1, 1), nn.ReLU(), nn.Conv2d(32, 64, (3,3), 1, 1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Conv2d(64, 128, (3,3), 1, 1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Flatten())
        with torch.no_grad(): n_flatten = self.cnn(torch.as_tensor(image_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten + state_space.shape[0], features_dim), nn.ReLU())
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor: cnn_out = self.cnn(obs['image']); return self.linear(torch.cat([cnn_out, obs['state']], dim=1))
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
class TrendTraderConfig:
    INITIAL_BALANCE = 10000; TRANSACTION_FEE = 0.001; WINDOW_SIZE = 64
    ORDER_SIZE_RATIO = 0.10; ATR_SL_MULTIPLIER = 2.0; ATR_TP_MULTIPLIER = 4.0
    TOTAL_TIMESTEPS = 2000000; LEARNING_RATE = 1e-4
    ENTROPY_COEF = 0.001; N_STEPS = 8192; GAMMA = 0.99; MAX_TRADE_DURATION = 288
    TRANSACTION_PENALTY = 0.0005
class TradingEnv(gym.Env):
    def __init__(self, prices_df: pd.DataFrame, image_features: pd.DataFrame, state_features: pd.DataFrame):
        super().__init__(); self.prices_df=prices_df; self.image_features=image_features; self.state_features=state_features; self.cfg=TrendTraderConfig()
        self.action_space=spaces.Discrete(4); self.image_shape=(1,self.cfg.WINDOW_SIZE,self.image_features.shape[1]); self.state_shape=(3+self.state_features.shape[1],)
        self.observation_space=spaces.Dict({"image":spaces.Box(low=-1,high=2,shape=self.image_shape,dtype=np.float32),"state":spaces.Box(low=-5,high=5,shape=self.state_shape,dtype=np.float32)})
        self._reset_state()
    def _reset_state(self):
        self.balance,self.equity=self.cfg.INITIAL_BALANCE,self.cfg.INITIAL_BALANCE; self.current_step=self.cfg.WINDOW_SIZE; self.position_amount=0.0; self.entry_price=0.0; self.entry_step=0; self.stop_loss_price=0.0; self.take_profit_price=0.0; self.trades=[]
        self.total_fees = 0.0; self.gross_profit = 0.0; self.gross_loss = 0.0
    def reset(self,seed=None,options=None): super().reset(seed=seed); self._reset_state(); return self._get_observation(),{}
    def _get_observation(self)->Dict[str,np.ndarray]:
        image_window=self.image_features.iloc[self.current_step-self.cfg.WINDOW_SIZE:self.current_step].copy(); current_price=image_window.iloc[-1]['close'] if image_window.iloc[-1]['close']>0 else 1
        for col in ['open','high','low','close']: image_window[col]=(image_window[col]/current_price)-1.0
        mean_volume=image_window['volume'].mean(); image_window['volume']=(image_window['volume']/mean_volume)-1.0 if mean_volume>0 else 0
        image_obs=np.expand_dims(image_window.values,axis=0).astype(np.float32)
        pos_type=np.sign(self.position_amount); pnl_norm,duration_norm=0,0
        if self.position_amount!=0:
            pnl=(self._get_current_price()-self.entry_price)*self.position_amount; entry_value=self.entry_price*abs(self.position_amount)
            pnl_norm=pnl/entry_value if entry_value>0 else 0; pnl_norm=np.clip(pnl_norm,-5,5)
            duration_norm=(self.current_step-self.entry_step)/self.cfg.MAX_TRADE_DURATION
        operational_state=np.array([pos_type,pnl_norm,duration_norm])
        analytical_state=self.state_features.iloc[self.current_step].values
        state_obs=np.concatenate([operational_state,analytical_state]).astype(np.float32)
        return {"image":image_obs,"state":state_obs}
    def _get_current_price(self)->float: return self.prices_df.iloc[self.current_step]['close']
    def _get_current_atr(self)->float: return self.prices_df.iloc[self.current_step]['atr_value']
    def step(self, action: int):
        realized_reward = 0.0; transaction_penalty = 0.0
        if self.position_amount != 0:
            low, high = self.prices_df.iloc[self.current_step][['low', 'high']]; is_long = self.position_amount > 0
            if (is_long and low <= self.stop_loss_price) or (not is_long and high >= self.stop_loss_price): realized_reward = self._close_position(self.stop_loss_price)
            elif (is_long and high >= self.take_profit_price) or (not is_long and low <= self.take_profit_price): realized_reward = self._close_position(self.take_profit_price)
        current_pos = np.sign(self.position_amount); price = self._get_current_price()
        if action == 1:
            if current_pos != 1: transaction_penalty = self.cfg.TRANSACTION_PENALTY
            if current_pos == -1: realized_reward += self._close_position(price)
            if current_pos != 1: self._open_position(price, is_long=True)
        elif action == 2:
            if current_pos != -1: transaction_penalty = self.cfg.TRANSACTION_PENALTY
            if current_pos == 1: realized_reward += self._close_position(price)
            if current_pos != -1: self._open_position(price, is_long=False)
        elif action == 3:
            if current_pos != 0: transaction_penalty = self.cfg.TRANSACTION_PENALTY; realized_reward += self._close_position(price)
        self.current_step += 1
        unrealized_pnl = (self._get_current_price() - self.entry_price) * self.position_amount if self.position_amount != 0 else 0; self.equity = self.balance + unrealized_pnl
        holding_penalty = 0.0
        if self.position_amount != 0 and unrealized_pnl < 0: holding_penalty = unrealized_pnl / self.cfg.INITIAL_BALANCE
        reward = (realized_reward / self.cfg.INITIAL_BALANCE) + holding_penalty - transaction_penalty
        done = self.current_step >= len(self.image_features) - 1 or self.equity <= 0
        if done and self.position_amount != 0:
            final_pnl = self._close_position(self._get_current_price()); reward += final_pnl / self.cfg.INITIAL_BALANCE
            self.equity = self.balance
        return self._get_observation(), reward, done, False, {'equity': self.equity}
    def _open_position(self,price,is_long):
        self.entry_step=self.current_step; atr=self._get_current_atr(); sl,tp=self.cfg.ATR_SL_MULTIPLIER,self.cfg.ATR_TP_MULTIPLIER
        self.stop_loss_price=price-(atr*sl) if is_long else price+(atr*sl); self.take_profit_price=price+(atr*tp) if is_long else price-(atr*tp)
        order_size = self.equity * self.cfg.ORDER_SIZE_RATIO
        if self.balance > order_size and order_size > 0:
            fee = order_size * self.cfg.TRANSACTION_FEE; self.total_fees += fee; self.balance -= order_size
            self.position_amount = (order_size - fee) / price * (1 if is_long else -1); self.entry_price = price
    def _close_position(self, price):
        size, is_long = abs(self.position_amount), self.position_amount > 0
        entry_value = size * self.entry_price; gross_close_value = size * price
        fee = gross_close_value * self.cfg.TRANSACTION_FEE; self.total_fees += fee
        net_close_value = gross_close_value - fee; pnl = net_close_value - entry_value if is_long else entry_value - gross_close_value
        if pnl > 0: self.gross_profit += pnl
        else: self.gross_loss += pnl
        self.balance += entry_value + pnl; self.trades.append(pnl)
        self.position_amount, self.entry_price = 0.0, 0.0
        return pnl

def main():
    print("🚀 СИСТЕМА V13.4 (Атака на гиперактивность) - ЗАПУСК")
    device=setup_gpu_support(); get_gpu_memory_info(device)
    data_paths={'5m':'data/BTCUSDT_5m_2y.csv','1h':'data/BTCUSDT_1h_2y.csv','4h':'data/BTCUSDT_4h_2y.csv','1d':'data/BTCUSDT_1d_2y.csv'}
    data_loader=MTFDataLoader(data_paths); prices_df,image_features,state_features=data_loader.load_and_prepare_data()
    split_idx=int(len(prices_df)*0.8); train_prices,test_prices=prices_df.iloc[:split_idx],prices_df.iloc[split_idx:]; train_image_feats,test_image_feats=image_features.iloc[:split_idx],image_features.iloc[split_idx:]; train_state_feats,test_state_feats=state_features.iloc[:split_idx],state_features.iloc[split_idx:]
    print(f"✅ Данные разделены: {len(train_prices)} для обучения, {len(test_prices)} для теста.")
    env=DummyVecEnv([lambda:TradingEnv(train_prices,train_image_feats,train_state_feats)])
    policy_kwargs=dict(features_extractor_class=CustomCombinedExtractor,features_extractor_kwargs=dict(features_dim=512),net_arch=dict(pi=[256,128],vf=[256,128]))
    
    model=PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs,
              learning_rate=TrendTraderConfig.LEARNING_RATE, ent_coef=TrendTraderConfig.ENTROPY_COEF,
              n_steps=TrendTraderConfig.N_STEPS, batch_size=128,
              gamma=TrendTraderConfig.GAMMA, verbose=0, device=device)
              
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ РАЦИОНАЛЬНОГО АНАЛИТИКА..."); 
    
    # ✅ ИСПРАВЛЕНИЕ: Использование правильного коллбэка
    with tqdm(total=TrendTraderConfig.TOTAL_TIMESTEPS, desc="Обучение", unit=" шагов") as pbar:
        # Создаем экземпляр нашего коллбэка
        tqdm_callback = TqdmCallback(pbar)
        # Передаем его в model.learn()
        model.learn(total_timesteps=TrendTraderConfig.TOTAL_TIMESTEPS, callback=tqdm_callback)

    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ НА НЕВИДИМЫХ ДАННЫХ...")
    test_env = TradingEnv(test_prices, test_image_feats, test_state_feats)
    obs, _ = test_env.reset()
    equity_history, price_history = [test_env.equity], [test_env._get_current_price()]
    done=False
    # Прогресс-бар для тестирования остается без изменений, он был реализован правильно
    for _ in tqdm(range(len(test_prices) - test_env.cfg.WINDOW_SIZE -1), desc="Тестирование"):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = test_env.step(int(action))
        equity_history.append(info['equity'])
        try: price_history.append(test_env._get_current_price())
        except IndexError: price_history.append(price_history[-1])
        done = terminated or truncated
        if done: break
        
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ"); initial, final = equity_history[0], equity_history[-1]
    total_return=(final-initial)/initial*100; start_p,end_p=price_history[0],price_history[-1]; bnh_return=(end_p-start_p)/start_p*100
    trades=len(test_env.trades); win_rate=(len([t for t in test_env.trades if t > 0])/trades)*100 if trades > 0 else 0
    print("="*60); print(f"💰 Финальный баланс: ${final:,.2f} (Начальный: ${initial:,.2f})"); print(f"📈 Доходность стратегии: {total_return:+.2f}%")
    print(f"📊 Доходность Buy & Hold: {bnh_return:+.2f}%"); print("-"*30); 
    
    print("📈 Детальная статистика PnL:")
    print(f"   🟢 Валовая прибыль: ${test_env.gross_profit:,.2f}")
    print(f"   🔴 Валовый убыток:  ${test_env.gross_loss:,.2f}")
    print(f"   💸 Всего комиссий:  ${test_env.total_fees:,.2f}")
    print(f"   ⚖️ Профит-фактор:    {abs(test_env.gross_profit / test_env.gross_loss):.2f}" if test_env.gross_loss != 0 else "∞")
    print("-"*30)
    
    print(f"🔄 Всего сделок: {trades}"); print(f"✅ Процент прибыльных сделок: {win_rate:.1f}%")
    plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(15,7)); plt.title(f'V13.4 - Атака на гиперактивность\nReturn: {total_return:.2f}%|Trades:{trades}|Win Rate:{win_rate:.1f}%')
    ax1=plt.gca(); ax1.plot(equity_history,label='Equity',c='royalblue'); ax1.set_xlabel('Шаги'); ax1.set_ylabel('Equity ($)',color='royalblue')
    ax2=ax1.twinx(); ax2.plot(price_history,label='Цена BTC',c='darkorange',alpha=0.6); ax2.set_ylabel('Цена ($)',color='darkorange')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right'); plt.savefig('results_v13.4.png'); plt.close(); print("✅ График сохранен в 'results_v13.4.png'")

if __name__=="__main__": main()