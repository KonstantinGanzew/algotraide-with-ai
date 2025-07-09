"""
🚀 SENTIMENT ТОРГОВАЯ СИСТЕМА V3.6 - ULTRA ACTIVE EDITION
Ультра-агрессивная версия с принудительными покупками И продажами
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class UltraActiveConfig:
    """Ультра-агрессивная конфигурация"""
    
    # Максимально агрессивные параметры
    INITIAL_BALANCE = 10000
    RISK_PER_TRADE = 0.15  # 15% риск
    MAX_POSITION_SIZE = 0.9  # 90% максимальная позиция
    
    # Частая принудительная торговля
    FORCE_BUY_INTERVAL = 30   # Принудительная покупка каждые 30 шагов
    FORCE_SELL_INTERVAL = 40  # Принудительная продажа каждые 40 шагов при наличии позиции
    MAX_HOLD_TIME = 60       # Максимальное время удержания позиции
    
    # Очень низкие пороги
    MIN_SIGNAL_THRESHOLD = 0.01  # Минимальный сигнал
    
    # Настройки модели
    WINDOW_SIZE = 12  # Еще меньше окно
    TOTAL_TIMESTEPS = 5000  # Быстрое обучение
    LEARNING_RATE = 5e-4
    
    # Максимальные стимулы
    TRADING_BONUS = 10.0      # Большой бонус за торговлю
    INACTIVITY_PENALTY = -5.0  # Большой штраф за бездействие
    COMPLETION_BONUS = 15.0   # Бонус за завершение цикла покупка-продажа


class UltraSimpleAnalyzer:
    """Ультра-простой анализ"""
    
    def add_ultra_simple_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Минимальный набор индикаторов"""
        print("🔧 Добавление ультра-простых индикаторов...")
        
        # Базовые
        df['returns'] = df['close'].pct_change()
        df['price_change'] = df['close'].diff()
        
        # Простые сигналы
        df['sma_3'] = df['close'].rolling(3).mean()
        df['sma_7'] = df['close'].rolling(7).mean()
        
        # Сигнал тренда
        df['trend_signal'] = np.where(df['close'] > df['sma_7'], 1, -1)
        df['momentum_signal'] = np.where(df['returns'] > 0, 1, -1)
        
        # Простой скор
        df['simple_score'] = (df['trend_signal'] + df['momentum_signal']) / 2
        
        # Случайные настроения для активности
        np.random.seed(42)
        df['random_sentiment'] = np.random.uniform(-0.5, 0.5, len(df))
        df['overall_sentiment'] = df['simple_score'] * 0.7 + df['random_sentiment'] * 0.3
        
        print(f"✅ Добавлено 8 ультра-простых индикаторов")
        return df


class UltraActiveExtractor(BaseFeaturesExtractor):
    """Минимальный Feature Extractor"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        if observation_space.shape is not None:
            self.input_features = observation_space.shape[1]
        else:
            self.input_features = 10
        
        # Супер простая сеть
        self.net = nn.Sequential(
            nn.Linear(self.input_features, 32),
            nn.ReLU(),
            nn.Linear(32, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Только последнее наблюдение
        last_obs = observations[:, -1, :]
        return self.net(last_obs)


class UltraActiveTradingEnv(gym.Env):
    """Ультра-активное торговое окружение"""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = UltraActiveConfig.WINDOW_SIZE
        
        # Минимальный набор признаков
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numeric_columns if col not in 
                               ['open', 'high', 'low', 'close', 'volume']][:10]  # Максимум 10 признаков
        
        self.action_space = spaces.Discrete(3)
        
        n_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, n_features),
            dtype=np.float32
        )
        
        self._prepare_data()
        self._reset_state()
    
    def _prepare_data(self):
        """Простейшая подготовка данных"""
        print("🔧 Ультра-простая подготовка данных...")
        
        feature_data = self.df[self.feature_columns].fillna(0)
        feature_data = feature_data.replace([np.inf, -np.inf], 0)
        
        # Простая нормализация
        normalized_data = (feature_data - feature_data.mean()) / (feature_data.std() + 1e-8)
        normalized_data = np.clip(normalized_data, -2, 2)
        
        self.normalized_df = pd.DataFrame(normalized_data, columns=self.feature_columns)
        print(f"✅ Подготовлено {len(self.feature_columns)} признаков")
    
    def _reset_state(self):
        """Сброс состояния"""
        self.current_step = self.window_size
        self.balance = UltraActiveConfig.INITIAL_BALANCE
        self.btc_amount = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        
        self.total_trades = 0
        self.profitable_trades = 0
        self.portfolio_history = [float(UltraActiveConfig.INITIAL_BALANCE)]
        self.trades_history = []
        
        # Счетчики для принудительной торговли
        self.steps_since_last_buy = 0
        self.steps_since_last_sell = 0
        self.position_hold_time = 0
        
        self.max_drawdown = 0.0
        self.peak_value = UltraActiveConfig.INITIAL_BALANCE
    
    def reset(self, seed=None, options=None):
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Получение наблюдения"""
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        obs = self.normalized_df.iloc[start_idx:end_idx].values
        
        if len(obs) < self.window_size:
            padding = np.tile(obs[0], (self.window_size - len(obs), 1))
            obs = np.vstack([padding, obs])
        
        return obs.astype(np.float32)
    
    def _get_current_price(self) -> float:
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.iloc[self.current_step]['close']
    
    def _get_current_datetime(self) -> str:
        if self.current_step >= len(self.df):
            return str(self.df.iloc[-1]['datetime'])
        return str(self.df.iloc[self.current_step]['datetime'])
    
    def _should_force_buy(self) -> bool:
        """Проверка принудительной покупки"""
        return (self.steps_since_last_buy >= UltraActiveConfig.FORCE_BUY_INTERVAL and 
                self.btc_amount == 0 and self.balance > 100)
    
    def _should_force_sell(self) -> bool:
        """Проверка принудительной продажи"""
        return (self.btc_amount > 0 and (
            self.steps_since_last_sell >= UltraActiveConfig.FORCE_SELL_INTERVAL or
            self.position_hold_time >= UltraActiveConfig.MAX_HOLD_TIME
        ))
    
    def _execute_ultra_trade(self, action: int, current_price: float) -> Dict[str, Any]:
        """Ультра-агрессивная торговля"""
        
        force_buy = self._should_force_buy()
        force_sell = self._should_force_sell()
        
        trade_result = {'executed': False, 'type': None, 'forced': False}
        
        # ПРИНУДИТЕЛЬНАЯ ПРОДАЖА (приоритет)
        if force_sell:
            revenue = self.btc_amount * current_price
            commission = revenue * 0.001
            profit = revenue - self.btc_amount * self.entry_price
            
            if profit > 0:
                self.profitable_trades += 1
            
            self.balance += revenue - commission
            self.btc_amount = 0.0
            self.entry_price = 0.0
            self.steps_since_last_sell = 0
            self.position_hold_time = 0
            
            trade_result.update({
                'executed': True, 'type': 'SELL',
                'profit': profit, 'price': current_price,
                'datetime': self._get_current_datetime(),
                'forced': True, 'reason': 'FORCE_SELL'
            })
            
            print(f"🔄 ПРИНУДИТЕЛЬНАЯ ПРОДАЖА по ${current_price:.2f}")
        
        # ПРИНУДИТЕЛЬНАЯ ПОКУПКА
        elif force_buy:
            position_size = UltraActiveConfig.RISK_PER_TRADE
            investment = self.balance * position_size
            amount = investment / current_price
            commission = investment * 0.001
            
            self.btc_amount += amount
            self.balance -= investment + commission
            self.entry_price = current_price
            self.entry_step = self.current_step
            self.steps_since_last_buy = 0
            self.position_hold_time = 0
            
            trade_result.update({
                'executed': True, 'type': 'BUY',
                'amount': amount, 'price': current_price,
                'investment': investment,
                'datetime': self._get_current_datetime(),
                'forced': True, 'reason': 'FORCE_BUY'
            })
            
            print(f"🔄 ПРИНУДИТЕЛЬНАЯ ПОКУПКА по ${current_price:.2f}")
        
        # ОБЫЧНАЯ ТОРГОВЛЯ (очень либеральные условия)
        elif action == 1 and self.balance > 100 and self.btc_amount == 0:  # Buy
            # Покупаем почти всегда
            position_size = UltraActiveConfig.RISK_PER_TRADE
            investment = self.balance * position_size
            amount = investment / current_price
            commission = investment * 0.001
            
            self.btc_amount += amount
            self.balance -= investment + commission
            self.entry_price = current_price
            self.entry_step = self.current_step
            self.steps_since_last_buy = 0
            self.position_hold_time = 0
            
            trade_result.update({
                'executed': True, 'type': 'BUY',
                'amount': amount, 'price': current_price,
                'investment': investment,
                'datetime': self._get_current_datetime()
            })
            
        elif action == 2 and self.btc_amount > 0:  # Sell
            # Продаем очень легко
            revenue = self.btc_amount * current_price
            commission = revenue * 0.001
            profit = revenue - self.btc_amount * self.entry_price
            
            if profit > 0:
                self.profitable_trades += 1
            
            self.balance += revenue - commission
            self.btc_amount = 0.0
            self.entry_price = 0.0
            self.steps_since_last_sell = 0
            self.position_hold_time = 0
            
            trade_result.update({
                'executed': True, 'type': 'SELL',
                'profit': profit, 'price': current_price,
                'datetime': self._get_current_datetime()
            })
        
        # Обновляем счетчики
        self.steps_since_last_buy += 1
        self.steps_since_last_sell += 1
        if self.btc_amount > 0:
            self.position_hold_time += 1
        
        if trade_result['executed']:
            self.total_trades += 1
            self.trades_history.append(trade_result)
        
        return trade_result
    
    def _calculate_portfolio_value(self) -> float:
        current_price = self._get_current_price()
        return self.balance + self.btc_amount * current_price
    
    def _calculate_ultra_reward(self) -> float:
        """Ультра-агрессивная функция награды"""
        current_portfolio = self._calculate_portfolio_value()
        prev_portfolio = self.portfolio_history[-1]
        
        # Базовая награда
        portfolio_change = (current_portfolio - prev_portfolio) / prev_portfolio
        base_reward = portfolio_change * 100
        
        # ОГРОМНЫЙ бонус за торговлю
        if len(self.trades_history) > 0:
            last_trade = self.trades_history[-1]
            if last_trade.get('executed', False):
                base_reward += UltraActiveConfig.TRADING_BONUS
                
                # Дополнительный бонус за завершение цикла
                if last_trade['type'] == 'SELL':
                    base_reward += UltraActiveConfig.COMPLETION_BONUS
        
        # ОГРОМНЫЙ штраф за бездействие
        if self.steps_since_last_buy > UltraActiveConfig.FORCE_BUY_INTERVAL:
            base_reward += UltraActiveConfig.INACTIVITY_PENALTY
        
        if self.btc_amount > 0 and self.position_hold_time > UltraActiveConfig.MAX_HOLD_TIME:
            base_reward += UltraActiveConfig.INACTIVITY_PENALTY
        
        return base_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Шаг ультра-активной симуляции"""
        current_price = self._get_current_price()
        
        # Выполнение ультра-агрессивной торговли
        trade_result = self._execute_ultra_trade(action, current_price)
        
        # Обновление состояния
        self.current_step += 1
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_history.append(portfolio_value)
        
        # Ультра-агрессивная награда
        reward = self._calculate_ultra_reward()
        
        # Проверка завершения
        done = (
            self.current_step >= len(self.df) - 1 or
            portfolio_value <= UltraActiveConfig.INITIAL_BALANCE * 0.2
        )
        
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'btc_amount': self.btc_amount,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'current_price': current_price,
            'datetime': self._get_current_datetime(),
            'steps_since_last_buy': self.steps_since_last_buy,
            'steps_since_last_sell': self.steps_since_last_sell,
            'position_hold_time': self.position_hold_time,
            'trade_result': trade_result
        }
        
        return self._get_observation(), reward, done, False, info


def load_and_prepare_ultra_data() -> pd.DataFrame:
    """Ультра-быстрая загрузка данных"""
    print("📊 Ультра-быстрая загрузка данных...")
    
    df = pd.read_csv("data/BTC_5_2w.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['datetime'] = df['timestamp']
    
    print(f"📊 Загружено {len(df)} записей")
    
    # Ультра-простые индикаторы
    analyzer = UltraSimpleAnalyzer()
    df = analyzer.add_ultra_simple_indicators(df)
    
    # Очистка
    df = df.fillna(method='ffill').fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    print(f"✅ Готов ультра-датасет с {len(df.columns)} колонками")
    return df


def main():
    """Ультра-активная торговая система V3.6"""
    print("🚀 ЗАПУСК SENTIMENT ТОРГОВОЙ СИСТЕМЫ V3.6 - ULTRA ACTIVE EDITION")
    print("=" * 85)
    
    # 1. Ультра-быстрая подготовка
    print("\n📊 ЭТАП 1: УЛЬТРА-БЫСТРАЯ ПОДГОТОВКА ДАННЫХ")
    print("-" * 65)
    combined_df = load_and_prepare_ultra_data()
    
    # 2. Ультра-активное окружение
    print("\n🎮 ЭТАП 2: СОЗДАНИЕ УЛЬТРА-АКТИВНОГО ОКРУЖЕНИЯ")
    print("-" * 65)
    env = UltraActiveTradingEnv(combined_df)
    vec_env = DummyVecEnv([lambda: env])
    print(f"✅ Ультра-окружение: принудительная покупка каждые {UltraActiveConfig.FORCE_BUY_INTERVAL} шагов")
    print(f"✅ Принудительная продажа каждые {UltraActiveConfig.FORCE_SELL_INTERVAL} шагов")
    
    # 3. Ультра-простая модель
    print("\n🧠 ЭТАП 3: СОЗДАНИЕ УЛЬТРА-ПРОСТОЙ МОДЕЛИ")
    print("-" * 65)
    
    policy_kwargs = dict(
        features_extractor_class=UltraActiveExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=[64, 32],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=UltraActiveConfig.LEARNING_RATE,
        n_steps=256,  # Еще меньше шагов
        batch_size=32,
        n_epochs=2,   # Меньше эпох
        gamma=0.9,    # Меньше gamma для быстрых решений
        gae_lambda=0.8,
        clip_range=0.3,  # Больше clip для агрессивности
        ent_coef=0.05,   # Больше энтропии
        vf_coef=0.3,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    print("✅ Ультра-модель готова к агрессивной торговле")
    
    # 4. Быстрое обучение
    print("\n🎓 ЭТАП 4: БЫСТРОЕ АГРЕССИВНОЕ ОБУЧЕНИЕ")
    print("-" * 65)
    model.learn(total_timesteps=UltraActiveConfig.TOTAL_TIMESTEPS)
    print("✅ Быстрое обучение завершено")
    
    # 5. Ультра-активное тестирование
    print("\n🧪 ЭТАП 5: УЛЬТРА-АКТИВНОЕ BACKTESTING")
    print("-" * 65)
    
    obs, _ = env.reset()
    results = []
    trades_log = []
    
    print("💼 Начинаем УЛЬТРА-АКТИВНУЮ торговлю...")
    print(f"⚡ Принудительная покупка каждые {UltraActiveConfig.FORCE_BUY_INTERVAL} шагов")
    print(f"⚡ Принудительная продажа каждые {UltraActiveConfig.FORCE_SELL_INTERVAL} шагов")
    print(f"⚡ Максимальное удержание: {UltraActiveConfig.MAX_HOLD_TIME} шагов")
    
    for step in range(min(1000, len(combined_df) - env.window_size - 1)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        
        results.append({
            'step': step,
            'datetime': info['datetime'],
            'portfolio_value': info['portfolio_value'],
            'current_price': info['current_price'],
            'balance': info['balance'],
            'btc_amount': info['btc_amount'],
            'total_trades': info['total_trades'],
            'profitable_trades': info['profitable_trades'],
            'steps_since_last_buy': info['steps_since_last_buy'],
            'position_hold_time': info['position_hold_time']
        })
        
        # Логируем все сделки
        if info['trade_result']['executed']:
            trades_log.append(info['trade_result'])
            trade = info['trade_result']
            trade_type = trade['type']
            price = trade['price']
            datetime = trade['datetime']
            forced = " (ПРИНУДИТЕЛЬНО)" if trade.get('forced', False) else ""
            reason = f" - {trade.get('reason', '')}" if trade.get('forced', False) else ""
            print(f"🔄 {trade_type} по ${price:.2f} в {datetime}{forced}{reason}")
        
        if done:
            break
    
    # 6. Анализ УЛЬТРА-результатов
    print("\n📊 ЭТАП 6: АНАЛИЗ УЛЬТРА-АКТИВНОЙ ТОРГОВЛИ")
    print("-" * 65)
    
    if results:
        final_result = results[-1]
        start_date = results[0]['datetime']
        end_date = final_result['datetime']
        
        initial_value = UltraActiveConfig.INITIAL_BALANCE
        final_value = final_result['portfolio_value']
        total_return = (final_value - initial_value) / initial_value * 100
        
        total_trades = final_result['total_trades']
        profitable_trades = final_result['profitable_trades']
        win_rate = (profitable_trades / max(total_trades, 1)) * 100
        
        # Buy & Hold
        start_price = results[0]['current_price']
        end_price = final_result['current_price']
        bnh_return = (end_price - start_price) / start_price * 100
        
        print("📊 РЕЗУЛЬТАТЫ УЛЬТРА-АКТИВНОЙ ТОРГОВЛИ V3.6")
        print("=" * 70)
        print(f"📅 Период: {start_date} - {end_date}")
        print(f"💰 Начальный баланс: ${initial_value:,.2f}")
        print(f"💰 Финальная стоимость: ${final_value:,.2f}")
        print(f"📈 УЛЬТРА-доходность: {total_return:+.2f}%")
        print(f"📈 Buy & Hold Bitcoin: {bnh_return:+.2f}%")
        print(f"🎯 Превосходство: {total_return - bnh_return:+.2f}%")
        print(f"🔄 ВСЕГО СДЕЛОК: {total_trades}")
        print(f"✅ Прибыльных: {profitable_trades} ({win_rate:.1f}%)")
        print(f"⚡ УЛЬТРА-активность: {total_trades / len(results) * 100:.1f} сделок/100 шагов")
        
        # Детальный анализ сделок
        if trades_log:
            buy_trades = [t for t in trades_log if t['type'] == 'BUY']
            sell_trades = [t for t in trades_log if t['type'] == 'SELL']
            forced_trades = [t for t in trades_log if t.get('forced', False)]
            
            print(f"\n📊 ДЕТАЛИЗАЦИЯ СДЕЛОК:")
            print(f"🛒 Покупок: {len(buy_trades)}")
            print(f"💰 Продаж: {len(sell_trades)}")
            print(f"⚡ Принудительных: {len(forced_trades)} ({len(forced_trades)/total_trades*100:.1f}%)")
            
            if sell_trades:
                profits = [t.get('profit', 0) for t in sell_trades if 'profit' in t]
                if profits:
                    avg_profit = np.mean(profits)
                    print(f"💵 Средняя прибыль: ${avg_profit:.2f}")
                    print(f"🏆 Лучшая сделка: ${max(profits):.2f}")
                    print(f"😞 Худшая сделка: ${min(profits):.2f}")
        
        print("\n🎯 ЗАКЛЮЧЕНИЕ УЛЬТРА-АКТИВНОЙ ТОРГОВЛИ V3.6")
        print("=" * 70)
        
        if total_trades >= 10:
            if total_return > bnh_return and win_rate > 40:
                print("🟢 УЛЬТРА-УСПЕХ: Превосходная активная торговля!")
            elif total_trades >= 20:
                print("🟡 УЛЬТРА-ПРОГРЕСС: Очень активная система!")
            else:
                print("🔶 УЛЬТРА-РАЗВИТИЕ: Максимальная активность достигнута!")
        else:
            print("🔴 НУЖНА ДОРАБОТКА: Недостаточно даже для ультра-режима")
        
        print(f"\n⚡ УЛЬТРА-система совершила {total_trades} сделок")
        print(f"🚀 Активность: {total_trades / len(results) * 100:.1f} сделок на 100 шагов")
        print(f"📈 Доходность: {total_return:+.2f}% vs {bnh_return:+.2f}% B&H")
        print("🎉 УЛЬТРА-АКТИВНОЕ BACKTESTING V3.6 ЗАВЕРШЕНО!")
    
    else:
        print("❌ Нет данных для анализа")


if __name__ == "__main__":
    main() 