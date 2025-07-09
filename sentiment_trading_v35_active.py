"""
🚀 SENTIMENT ТОРГОВАЯ СИСТЕМА V3.5 - ACTIVE TRADING EDITION
Агрессивная версия с активной торговлей на исторических данных
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
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ActiveTradingConfig:
    """Конфигурация для активной торговли"""
    
    # Агрессивные параметры риска
    INITIAL_BALANCE = 10000
    RISK_PER_TRADE = 0.1  # Увеличен риск для активности
    MAX_POSITION_SIZE = 0.8  # Больше максимальная позиция
    STOP_LOSS = 0.05  # 5% стоп-лосс
    TAKE_PROFIT = 0.1  # 10% тейк-профит
    
    # Упрощенные условия входа
    MIN_SIGNAL_THRESHOLD = 0.05  # Снижен порог сигнала
    FORCE_TRADE_INTERVAL = 50   # Принудительная торговля каждые 50 шагов
    
    # Настройки модели
    WINDOW_SIZE = 24  # Уменьшено окно
    TOTAL_TIMESTEPS = 8000
    LEARNING_RATE = 3e-4  # Увеличена скорость обучения
    
    # Стимулы для торговли
    TRADING_BONUS = 5.0    # Бонус за совершение сделок
    INACTIVITY_PENALTY = -2.0  # Штраф за бездействие


class SimpleTechnicalAnalyzer:
    """Упрощенный технический анализ"""
    
    def add_simple_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление простых технических индикаторов"""
        print("🔧 Расчет упрощенных технических индикаторов...")
        
        # Основные индикаторы
        df['returns'] = df['close'].pct_change()
        df['price_change'] = df['close'].diff()
        df['volatility'] = df['returns'].rolling(10).std()
        
        # Простые скользящие средние
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        # Сигналы пересечения
        df['sma_signal'] = np.where(df['sma_5'] > df['sma_10'], 1, -1)
        df['trend_signal'] = np.where(df['close'] > df['sma_20'], 1, -1)
        
        # RSI упрощенный
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
        rs = gain / (loss + 1e-8)  # Избегаем деления на ноль
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_signal'] = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
        
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_signal'] = np.where(df['momentum'] > 0.02, 1, np.where(df['momentum'] < -0.02, -1, 0))
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(10).mean()
        df['volume_signal'] = np.where(df['volume'] > df['volume_ma'] * 1.5, 1, 0)
        
        # Комбинированный технический сигнал
        df['technical_score'] = (
            df['sma_signal'] * 0.3 +
            df['trend_signal'] * 0.3 +
            df['rsi_signal'] * 0.2 +
            df['momentum_signal'] * 0.2
        )
        
        print(f"✅ Добавлено 15 упрощенных технических индикаторов")
        return df


class ActiveSentimentGenerator:
    """Генератор активных сигналов настроений"""
    
    def generate_active_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Генерация более активных сигналов настроений"""
        print("📱 Генерация активных сигналов настроений...")
        
        n_points = len(df)
        np.random.seed(42)
        
        # Базовое настроение с большей амплитудой
        base_sentiment = np.random.normal(0, 0.4, n_points)
        
        # Более частые события
        events = np.random.choice([0, 1], size=n_points, p=[0.9, 0.1])
        event_impact = np.random.normal(0, 0.8, n_points) * events
        
        # Тренд на основе цены
        price_momentum = df['close'].pct_change(periods=5).fillna(0)
        sentiment_from_price = price_momentum * 5  # Усиливаем связь
        
        # Комбинируем
        total_sentiment = base_sentiment + event_impact + sentiment_from_price
        total_sentiment = np.clip(total_sentiment, -1, 1)
        
        # Создаем сигналы
        df['sentiment_twitter'] = total_sentiment + np.random.normal(0, 0.1, n_points)
        df['sentiment_reddit'] = total_sentiment + np.random.normal(0, 0.15, n_points)
        df['sentiment_news'] = total_sentiment + np.random.normal(0, 0.1, n_points)
        
        # Нормализация
        for col in ['sentiment_twitter', 'sentiment_reddit', 'sentiment_news']:
            df[col] = np.clip(df[col], -1, 1)
        
        # Агрегированное настроение
        df['overall_sentiment'] = (
            df['sentiment_twitter'] * 0.4 +
            df['sentiment_reddit'] * 0.3 +
            df['sentiment_news'] * 0.3
        )
        
        # Сильные сигналы настроений
        df['sentiment_signal'] = np.where(
            df['overall_sentiment'] > ActiveTradingConfig.MIN_SIGNAL_THRESHOLD, 1,
            np.where(df['overall_sentiment'] < -ActiveTradingConfig.MIN_SIGNAL_THRESHOLD, -1, 0)
        )
        
        print(f"✅ Сгенерированы активные сигналы настроений для {n_points} точек")
        return df


class ActiveFeatureExtractor(BaseFeaturesExtractor):
    """Упрощенный Feature Extractor для активной торговли"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        if observation_space.shape is not None:
            self.seq_len = observation_space.shape[0]
            self.input_features = observation_space.shape[1]
        else:
            self.seq_len = ActiveTradingConfig.WINDOW_SIZE
            self.input_features = 20
        
        # Простая архитектура
        self.net = nn.Sequential(
            nn.Linear(self.input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = observations.shape
        
        # Используем только последнее наблюдение для простоты
        last_obs = observations[:, -1, :]
        output = self.net(last_obs)
        
        return output


class ActiveTradingEnv(gym.Env):
    """Активное торговое окружение"""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = ActiveTradingConfig.WINDOW_SIZE
        
        # Выбираем только численные признаки
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numeric_columns if col not in 
                               ['open', 'high', 'low', 'close', 'volume']]
        
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        
        n_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, n_features),
            dtype=np.float32
        )
        
        self._prepare_data()
        self._reset_state()
    
    def _prepare_data(self):
        """Подготовка данных"""
        print("🔧 Подготовка данных для активной торговли...")
        
        # Простая нормализация
        feature_data = self.df[self.feature_columns].fillna(0)
        feature_data = feature_data.replace([np.inf, -np.inf], 0)
        
        # Z-score нормализация
        normalized_data = (feature_data - feature_data.mean()) / (feature_data.std() + 1e-8)
        normalized_data = np.clip(normalized_data, -3, 3)
        
        self.normalized_df = pd.DataFrame(normalized_data, columns=self.feature_columns)
        
        print(f"✅ Подготовлено {len(self.feature_columns)} признаков")
    
    def _reset_state(self):
        """Сброс состояния"""
        self.current_step = self.window_size
        self.balance = ActiveTradingConfig.INITIAL_BALANCE
        self.btc_amount = 0.0
        self.entry_price = 0.0
        self.last_trade_step = 0
        
        self.total_trades = 0
        self.profitable_trades = 0
        self.portfolio_history = [float(ActiveTradingConfig.INITIAL_BALANCE)]
        self.trades_history = []
        
        # Для стимулирования торговли
        self.steps_without_trade = 0
        self.max_drawdown = 0.0
        self.peak_value = ActiveTradingConfig.INITIAL_BALANCE
    
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
        """Получение текущей цены"""
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.iloc[self.current_step]['close']
    
    def _get_current_datetime(self) -> str:
        """Получение текущего времени"""
        if self.current_step >= len(self.df):
            return str(self.df.iloc[-1]['datetime'])
        return str(self.df.iloc[self.current_step]['datetime'])
    
    def _should_force_trade(self) -> bool:
        """Проверка принудительной торговли"""
        return self.steps_without_trade >= ActiveTradingConfig.FORCE_TRADE_INTERVAL
    
    def _get_trading_signals(self) -> Dict[str, float]:
        """Получение торговых сигналов"""
        if self.current_step >= len(self.df):
            return {'technical': 0, 'sentiment': 0, 'combined': 0}
        
        current_data = self.df.iloc[self.current_step]
        
        # Технический сигнал
        technical_signal = current_data.get('technical_score', 0)
        
        # Сигнал настроений
        sentiment_signal = current_data.get('overall_sentiment', 0)
        
        # Комбинированный сигнал
        combined_signal = technical_signal * 0.6 + sentiment_signal * 0.4
        
        return {
            'technical': technical_signal,
            'sentiment': sentiment_signal,
            'combined': combined_signal
        }
    
    def _execute_active_trade(self, action: int, current_price: float) -> Dict[str, Any]:
        """Выполнение активной торговой операции"""
        signals = self._get_trading_signals()
        force_trade = self._should_force_trade()
        
        trade_result = {'executed': False, 'type': None, 'signals': signals}
        
        # Принудительная торговля при длительном бездействии
        if force_trade and self.balance > 100 and self.btc_amount == 0:
            action = 1  # Принудительная покупка
            print(f"🔄 Принудительная торговля на шаге {self.current_step}")
        
        # Упрощенные условия торговли
        if action == 1 and self.balance > 100:  # Buy
            # Покупаем при слабых положительных сигналах или принудительно
            if signals['combined'] > -0.5 or force_trade:
                position_size = ActiveTradingConfig.RISK_PER_TRADE
                
                # Увеличиваем позицию при сильных сигналах
                if signals['combined'] > 0.2:
                    position_size *= 1.5
                
                position_size = min(position_size, ActiveTradingConfig.MAX_POSITION_SIZE)
                
                investment = self.balance * position_size
                amount = investment / current_price
                commission = investment * 0.001
                
                self.btc_amount += amount
                self.balance -= investment + commission
                self.entry_price = current_price
                self.last_trade_step = self.current_step
                self.steps_without_trade = 0
                
                trade_result.update({
                    'executed': True, 'type': 'BUY',
                    'amount': amount, 'price': current_price,
                    'investment': investment,
                    'datetime': self._get_current_datetime(),
                    'forced': force_trade
                })
                
        elif action == 2 and self.btc_amount > 0:  # Sell
            # Продаем при любом негативном сигнале или для фиксации прибыли
            current_profit = (current_price - self.entry_price) / self.entry_price
            
            if (signals['combined'] < 0.1 or 
                current_profit > 0.03 or  # 3% прибыль
                current_profit < -0.04 or  # 4% убыток
                self.current_step - self.last_trade_step > 100):  # Долгое держание
                
                revenue = self.btc_amount * current_price
                commission = revenue * 0.001
                profit = revenue - self.btc_amount * self.entry_price
                
                if profit > 0:
                    self.profitable_trades += 1
                
                self.balance += revenue - commission
                self.btc_amount = 0.0
                self.entry_price = 0.0
                self.last_trade_step = self.current_step
                self.steps_without_trade = 0
                
                trade_result.update({
                    'executed': True, 'type': 'SELL',
                    'profit': profit, 'price': current_price,
                    'datetime': self._get_current_datetime()
                })
        
        if trade_result['executed']:
            self.total_trades += 1
            self.trades_history.append(trade_result)
        else:
            self.steps_without_trade += 1
        
        return trade_result
    
    def _calculate_portfolio_value(self) -> float:
        """Расчет стоимости портфеля"""
        current_price = self._get_current_price()
        return self.balance + self.btc_amount * current_price
    
    def _calculate_active_reward(self) -> float:
        """Расчет награды с стимулами для торговли"""
        current_portfolio = self._calculate_portfolio_value()
        prev_portfolio = self.portfolio_history[-1]
        
        # Базовая награда от изменения портфеля
        portfolio_change = (current_portfolio - prev_portfolio) / prev_portfolio
        base_reward = portfolio_change * 100
        
        # Бонус за торговлю
        if len(self.trades_history) > 0:
            last_trade = self.trades_history[-1]
            if last_trade.get('executed', False):
                base_reward += ActiveTradingConfig.TRADING_BONUS
        
        # Штраф за длительное бездействие
        if self.steps_without_trade > ActiveTradingConfig.FORCE_TRADE_INTERVAL:
            base_reward += ActiveTradingConfig.INACTIVITY_PENALTY
        
        # Обновление просадки
        if current_portfolio > self.peak_value:
            self.peak_value = current_portfolio
        current_drawdown = (self.peak_value - current_portfolio) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        return base_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Шаг симуляции"""
        current_price = self._get_current_price()
        
        # Выполнение действия
        trade_result = self._execute_active_trade(action, current_price)
        
        # Обновление состояния
        self.current_step += 1
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_history.append(portfolio_value)
        
        # Расчет награды
        reward = self._calculate_active_reward()
        
        # Проверка завершения
        done = (
            self.current_step >= len(self.df) - 1 or
            portfolio_value <= ActiveTradingConfig.INITIAL_BALANCE * 0.3
        )
        
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'btc_amount': self.btc_amount,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'max_drawdown': self.max_drawdown,
            'current_price': current_price,
            'datetime': self._get_current_datetime(),
            'steps_without_trade': self.steps_without_trade,
            'trade_result': trade_result
        }
        
        return self._get_observation(), reward, done, False, info


def load_and_prepare_data() -> pd.DataFrame:
    """Загрузка и подготовка исторических данных"""
    print("📊 Загрузка исторических данных...")
    
    # Загружаем данные
    df = pd.read_csv("data/BTC_5_2w.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['datetime'] = df['timestamp']
    
    print(f"📅 Период: {df['datetime'].min()} - {df['datetime'].max()}")
    print(f"📈 Цены: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"📊 Записей: {len(df)}")
    
    # Добавляем технические индикаторы
    analyzer = SimpleTechnicalAnalyzer()
    df = analyzer.add_simple_indicators(df)
    
    # Добавляем сигналы настроений
    sentiment_gen = ActiveSentimentGenerator()
    df = sentiment_gen.generate_active_sentiment(df)
    
    # Очистка данных
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.replace([np.inf, -np.inf], 0)
    
    print(f"✅ Подготовлен датасет с {len(df.columns)} признаками")
    return df


def main():
    """Главная функция активной торговой системы V3.5"""
    print("🚀 ЗАПУСК SENTIMENT ТОРГОВОЙ СИСТЕМЫ V3.5 - ACTIVE TRADING EDITION")
    print("=" * 80)
    
    # 1. Загрузка и подготовка данных
    print("\n📊 ЭТАП 1: ПОДГОТОВКА ДАННЫХ ДЛЯ АКТИВНОЙ ТОРГОВЛИ")
    print("-" * 60)
    combined_df = load_and_prepare_data()
    
    # 2. Создание активного окружения
    print("\n🎮 ЭТАП 2: СОЗДАНИЕ АКТИВНОГО ТОРГОВОГО ОКРУЖЕНИЯ")
    print("-" * 60)
    env = ActiveTradingEnv(combined_df)
    vec_env = DummyVecEnv([lambda: env])
    print(f"✅ Активное окружение создано с {len(env.feature_columns)} признаками")
    
    # 3. Создание модели для активной торговли
    print("\n🧠 ЭТАП 3: СОЗДАНИЕ МОДЕЛИ ДЛЯ АКТИВНОЙ ТОРГОВЛИ")
    print("-" * 60)
    
    policy_kwargs = dict(
        features_extractor_class=ActiveFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 64],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=ActiveTradingConfig.LEARNING_RATE,
        n_steps=512,
        batch_size=64,
        n_epochs=3,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.2,
        ent_coef=0.02,  # Больше энтропии для разнообразия
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    print("✅ Модель для активной торговли создана")
    
    # 4. Обучение модели
    print("\n🎓 ЭТАП 4: ОБУЧЕНИЕ МОДЕЛИ НА АКТИВНУЮ ТОРГОВЛЮ")
    print("-" * 60)
    model.learn(total_timesteps=ActiveTradingConfig.TOTAL_TIMESTEPS)
    print("✅ Обучение завершено")
    
    # 5. Активное тестирование
    print("\n🧪 ЭТАП 5: АКТИВНОЕ BACKTESTING")
    print("-" * 60)
    
    obs, _ = env.reset()
    results = []
    trades_log = []
    
    print("💼 Начинаем активную торговлю...")
    
    for step in range(min(1500, len(combined_df) - env.window_size - 1)):
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
            'steps_without_trade': info['steps_without_trade']
        })
        
        # Логируем сделки
        if info['trade_result']['executed']:
            trades_log.append(info['trade_result'])
            trade_type = info['trade_result']['type']
            price = info['trade_result']['price']
            datetime = info['trade_result']['datetime']
            forced = info['trade_result'].get('forced', False)
            force_msg = " (ПРИНУДИТЕЛЬНО)" if forced else ""
            print(f"🔄 {trade_type} по цене ${price:.2f} в {datetime}{force_msg}")
        
        if done:
            break
    
    # 6. Анализ результатов активной торговли
    print("\n📊 ЭТАП 6: АНАЛИЗ АКТИВНОЙ ТОРГОВЛИ")
    print("-" * 60)
    
    if results:
        final_result = results[-1]
        start_date = results[0]['datetime']
        end_date = final_result['datetime']
        
        initial_value = ActiveTradingConfig.INITIAL_BALANCE
        final_value = final_result['portfolio_value']
        total_return = (final_value - initial_value) / initial_value * 100
        
        total_trades = final_result['total_trades']
        profitable_trades = final_result['profitable_trades']
        win_rate = (profitable_trades / max(total_trades, 1)) * 100
        
        # Buy & Hold сравнение
        start_price = results[0]['current_price']
        end_price = final_result['current_price']
        bnh_return = (end_price - start_price) / start_price * 100
        
        print("📊 РЕЗУЛЬТАТЫ АКТИВНОЙ ТОРГОВЛИ V3.5")
        print("=" * 65)
        print(f"📅 Период тестирования: {start_date} - {end_date}")
        print(f"💰 Начальный баланс: ${initial_value:,.2f}")
        print(f"💰 Финальная стоимость: ${final_value:,.2f}")
        print(f"📈 Доходность активной торговли: {total_return:+.2f}%")
        print(f"📈 Buy & Hold Bitcoin: {bnh_return:+.2f}%")
        print(f"🎯 Превосходство над B&H: {total_return - bnh_return:+.2f}%")
        print(f"🔄 Всего сделок: {total_trades}")
        print(f"✅ Прибыльных сделок: {profitable_trades} ({win_rate:.1f}%)")
        print(f"⚡ Активность: {total_trades / len(results) * 100:.1f} сделок на 100 шагов")
        
        # Анализ сделок
        if trades_log:
            buy_trades = [t for t in trades_log if t['type'] == 'BUY']
            sell_trades = [t for t in trades_log if t['type'] == 'SELL' and 'profit' in t]
            forced_trades = [t for t in trades_log if t.get('forced', False)]
            
            print(f"🔄 Покупок: {len(buy_trades)}")
            print(f"🔄 Продаж: {len(sell_trades)}")
            print(f"⚡ Принудительных сделок: {len(forced_trades)}")
            
            if sell_trades:
                profits = [t['profit'] for t in sell_trades]
                avg_profit = np.mean(profits)
                print(f"💰 Средняя прибыль с сделки: ${avg_profit:.2f}")
                print(f"🏆 Лучшая сделка: ${max(profits):.2f}")
                print(f"😞 Худшая сделка: ${min(profits):.2f}")
        
        print("\n🎯 ЗАКЛЮЧЕНИЕ АКТИВНОЙ ТОРГОВЛИ V3.5")
        print("=" * 65)
        
        if total_trades > 5:
            if total_return > bnh_return and win_rate > 45:
                print("🟢 УСПЕХ: Активная торговля превосходит пассивную стратегию!")
            elif total_trades > 10 and win_rate > 40:
                print("🟡 ПРОГРЕСС: Система активно торгует, есть потенциал!")
            else:
                print("🔶 РАЗВИТИЕ: Активность есть, нужна оптимизация прибыльности")
        else:
            print("🔴 ПРОБЛЕМА: Недостаточная торговая активность")
        
        print(f"\n💡 Система совершила {total_trades} сделок за период")
        print(f"📈 Показала доходность {total_return:+.2f}% против {bnh_return:+.2f}% B&H")
        print("🎉 АКТИВНОЕ BACKTESTING V3.5 ЗАВЕРШЕНО!")
    
    else:
        print("❌ Недостаточно данных для анализа")


if __name__ == "__main__":
    main() 