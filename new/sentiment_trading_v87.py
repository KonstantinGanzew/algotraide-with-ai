import logging
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_v13_13.log', # ✅ ИЗМЕНЕНИЕ: Новый файл лога
    filemode='w'
)
logger = logging.getLogger(__name__)

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from utils.data_utils import load_data, preprocess_data
from utils.env_utils import TradingEnv
from utils.rl_utils import setup_gpu_support, get_gpu_memory_info

class TrendTraderConfig:
    """
    🚀 КОНФИГУРАЦИЯ V13.13 - УЧИМ АГЕНТА ТЕРПЕНИЮ
    ✅ КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:
       1. TRANSACTION_PENALTY: Значительно увеличен (x5), чтобы наказать за панические флипы.
       2. HOLD_REWARD: Введена небольшая награда за удержание позиции, чтобы мотивировать
          агента придерживаться своего решения и ждать TP/SL.
    """
    INITIAL_BALANCE = 10000; TRANSACTION_FEE = 0.001; WINDOW_SIZE = 64; ORDER_SIZE_RATIO = 0.05
    ATR_SL_MULTIPLIER = 2.0; ATR_TP_MULTIPLIER = 4.0; TOTAL_TIMESTEPS = 1000000
    LEARNING_RATE = 1e-4; ENTROPY_COEF = 0.001; N_STEPS = 2048; GAMMA = 0.99
    MAX_TRADE_DURATION = 288
    TRANSACTION_PENALTY = 0.01     # ✅ ИЗМЕНЕНИЕ: x5, делаем действия "дорогими"
    PNL_SHAPING_COEF = 0.02
    HOLD_REWARD = 0.0001           # ✅ ИЗМЕНЕНИЕ: Награда за терпение

class TradingEnv(gym.Env):
    # ... (Содержимое класса TradingEnv без изменений) ...
    def __init__(self, prices_df: pd.DataFrame, image_features: pd.DataFrame, state_features: pd.DataFrame):
        super().__init__()
        self.cfg = TrendTraderConfig()
        self.prices_df = prices_df
        self.image_features = image_features
        self.state_features = state_features
        self.current_step = 0
        self.balance = self.cfg.INITIAL_BALANCE
        self.position_amount = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.entry_step = 0
        self.equity = self.balance
        self.close_reasons = {}

    def step(self, action: int):
        realized_reward, transaction_penalty, hold_reward = 0.0, 0.0, 0.0
        
        # 1. Проверка на срабатывание SL/TP/Max Duration
        if self.position_amount != 0:
            low, high = self.prices_df.iloc[self.current_step][['low', 'high']]; is_long = self.position_amount > 0
            if (is_long and low <= self.stop_loss_price): realized_reward = self._close_position(self.stop_loss_price, "SL")
            elif (is_long and high >= self.take_profit_price) or (not is_long and low <= self.take_profit_price): realized_reward = self._close_position(self.take_profit_price, "TP")
            elif (self.current_step - self.entry_step) >= self.cfg.MAX_TRADE_DURATION: realized_reward = self._close_position(self._get_current_price(), "Max Duration")
        
        # 2. Обработка действий агента
        current_pos, price = np.sign(self.position_amount), self._get_current_price()
        if action == 0 and self.position_amount != 0:
             # ✅ ИЗМЕНЕНИЕ: Награда за удержание позиции
            hold_reward = self.cfg.HOLD_REWARD
        elif self.position_amount != 0:
            if action == 1 and current_pos != 1: transaction_penalty = self.cfg.TRANSACTION_PENALTY; realized_reward += self._close_position(price, "Flip to Long"); self._open_position(price, is_long=True)
            elif action == 2 and current_pos != -1: transaction_penalty = self.cfg.TRANSACTION_PENALTY; realized_reward += self._close_position(price, "Flip to Short"); self._open_position(price, is_long=False)
        elif action in [1, 2] and current_pos == 0: transaction_penalty = self.cfg.TRANSACTION_PENALTY; self._open_position(price, is_long=(action==1))
        
        # 3. Обновление состояния и расчет вознаграждения
        self.current_step += 1
        unrealized_pnl = (self._get_current_price() - self.entry_price) * self.position_amount if self.position_amount != 0 else 0
        self.equity = self.balance + unrealized_pnl
        pnl_shaping_reward = (unrealized_pnl / self.cfg.INITIAL_BALANCE) * self.cfg.PNL_SHAPING_COEF if self.position_amount != 0 else 0
        
        reward = (realized_reward / self.cfg.INITIAL_BALANCE) + pnl_shaping_reward - transaction_penalty + hold_reward

        # 4. Проверка на конец эпизода
        done = self.current_step >= len(self.image_features) - 1 or self.equity <= 0
        if done and self.position_amount != 0: final_pnl = self._close_position(self._get_current_price(), "End of Episode"); reward += final_pnl / self.cfg.INITIAL_BALANCE; self.equity = self.balance
        return self._get_observation(), reward, done, False, {'equity': self.equity}
    def _open_position(self, price: float, is_long: bool):
        self.position_amount, self.entry_price = 0.0, 0.0
        return pnl

def main():
    try:
        logger.info("="*20 + " ЗАПУСК СИСТЕМЫ V13.13 (Учим терпению) " + "="*20)
        device=setup_gpu_support(); get_gpu_memory_info(device)
        
        data_paths={'5m':'data/BTCUSDT_5m_2y.csv','1h':'data/BTCUSDT_1h_2y.csv','4h':'data/BTCUSDT_4h_2y.csv','1d':'data/BTCUSDT_1d_2y.csv'}
        prices_df, image_features, state_features = load_data(data_paths)
        
        # Предобработка данных
        prices_df, image_features, state_features = preprocess_data(prices_df, image_features, state_features)
        
        # Создание окружения
        env = TradingEnv(prices_df, image_features, state_features)
        check_env(env)
        
        # Создание модели PPO
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")
        
        # Обучение модели
        model.learn(total_timesteps=env.cfg.TOTAL_TIMESTEPS, callback=EvalCallback(env, best_model_save_path="models/", log_path="logs/"))
        
        # Сохранение модели
        model.save("models/final_model")
        
        # Тестирование модели
        test_env = TradingEnv(prices_df, image_features, state_features)
        test_env.load_model("models/final_model")
        
        total_return, trades, win_rate, reward_risk_ratio, profit_factor = test_env.run_simulation()
        
        print("🤔 Причины закрытия сделок:")
        for reason, count in test_env.close_reasons.items(): print(f"   - {reason}: {count} раз")
        print("-"*60)
        print(f"🔄 Всего сделок: {trades}"); print(f"✅ Процент прибыльных сделок: {win_rate:.1f}%")
        
        plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(15,7)); plt.title(f'V13.13 - Patience Training\nReturn: {total_return:.2f}% | Trades: {trades} | Win Rate: {win_rate:.1f}% | R/R: {reward_risk_ratio:.2f} | P/F: {profit_factor:.2f}'); ax1=plt.gca(); ax1.plot(equity_series, label='Equity',c='royalblue'); ax1.set_xlabel('Дата'); ax1.set_ylabel('Equity ($)',color='royalblue'); ax2=ax1.twinx(); ax2.plot(equity_series.index, price_history, label='Цена BTC',c='darkorange',alpha=0.6); ax2.set_ylabel('Цена ($)',color='darkorange'); ax1.legend(loc='upper left'); ax2.legend(loc='upper right'); plt.savefig('results_v13.13.png'); plt.close(); print("✅ График сохранен в 'results_v13.13.png'")
        logger.info("Скрипт успешно завершил работу.")
    
    except Exception as e:
        logger.critical(f"Неперехваченная ошибка в main: {e}", exc_info=True)
        print(f"❌ Критическая ошибка! Подробности в файле 'trading_v13_13.log'.")


if __name__ == "__main__":
    main() 