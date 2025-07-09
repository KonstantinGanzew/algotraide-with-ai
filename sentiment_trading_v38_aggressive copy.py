"""
🚀 ИСПРАВЛЕННАЯ ТОРГОВАЯ СИСТЕМА V3.8 - СТАБИЛЬНАЯ ВЕРСИЯ
Исправлены основные проблемы:
✅ Убраны фейковые sentiment данные
✅ Feature Extractor теперь видит всю историю (nn.Flatten)
✅ Упрощена система наград - основана на изменении баланса
✅ Увеличено время обучения до 150,000 шагов
✅ Снижены риски - 8% капитала на сделку
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


class SuperActiveConfig:
    """Улучшенная конфигурация для стабильной прибыли"""
    
    # КОНСЕРВАТИВНОЕ управление рисками
    INITIAL_BALANCE = 10000
    ORDER_SIZE_RATIO = 0.08  # 8% капитала на ордер - консервативно
    MAX_POSITIONS = 3
    STOP_LOSS = 0.02  # 2% стоп-лосс
    TAKE_PROFIT = 0.04  # 4% тейк-профит
    
    # Настройки модели
    WINDOW_SIZE = 50
    TOTAL_TIMESTEPS = 150000  # Увеличиваем время обучения
    LEARNING_RATE = 3e-4
    
    # УПРОЩЕННЫЕ параметры
    MIN_SIGNAL_STRENGTH = 0.15
    ENTROPY_COEF = 0.01


class SimpleDataLoader:
    """Простой загрузчик с проверенными индикаторами"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Загрузка и подготовка данных как в main2.py"""
        print(f"📊 Загрузка данных из {self.data_path}...")
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['datetime'] = df['timestamp']
        
        print(f"📅 Период: {df['datetime'].min()} - {df['datetime'].max()}")
        print(f"📈 Цены: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"📊 Записей: {len(df)}")
        
        # Основные колонки как в main2.py
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # EMA как в main2.py
        df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # RSI как в main2.py
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Дополнительные простые индикаторы
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # MACD
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Простые сигналы настроений (менее случайные)
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        
        # Убираем фейковые sentiment данные - используем только реальные технические индикаторы
        # rsi_normalized = (df['rsi'] - 50) / 50
        
        # Настроение связано с техническими показателями
        # base_sentiment = (df['momentum'].fillna(0) * 0.6 + rsi_normalized.fillna(0) * 0.4)
        # base_sentiment = np.clip(base_sentiment, -1, 1)
        
        # df['sentiment_twitter'] = base_sentiment + np.random.normal(0, 0.02, len(df))
        # df['sentiment_reddit'] = base_sentiment + np.random.normal(0, 0.03, len(df))
        # df['sentiment_news'] = base_sentiment + np.random.normal(0, 0.025, len(df))
        
        # df['overall_sentiment'] = (
        #     df['sentiment_twitter'] * 0.5 +
        #     df['sentiment_reddit'] * 0.3 +
        #     df['sentiment_news'] * 0.2
        # )
        
        # Очистка NaN как в main2.py
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Нормализация только нужных колонок как в main2.py
        cols_to_normalize = base_cols + ['ema_fast', 'ema_slow', 'rsi', 'macd', 'macd_signal', 
                                        'bb_position', 'volatility']
        
        # Простая стандартизация
        for col in cols_to_normalize:
            if col in df.columns:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        # Оставляем только численные колонки для торговли
        numeric_cols = [col for col in cols_to_normalize + ['returns', 'momentum'] if col in df.columns]
        
        # Создаем новый DataFrame только с численными данными
        df_trading = pd.DataFrame()
        for col in numeric_cols:
            df_trading[col] = df[col]
        
        print(f"✅ Подготовлено данных: {len(df_trading)} записей, {len(numeric_cols)} признаков")
        return df_trading


class ProfitableFeatureExtractor(BaseFeaturesExtractor):
    """Улучшенный Feature Extractor, использующий всю историю"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Вычисляем размер входа: window_size * n_features
        if hasattr(observation_space, 'shape') and observation_space.shape is not None:
            n_input_features = observation_space.shape[0] * observation_space.shape[1]
        else:
            # Резервное значение для случаев, когда shape недоступен
            n_input_features = 50 * 11  # window_size * приблизительное количество признаков
        
        # Архитектура, которая видит всю историю
        self.net = nn.Sequential(
            nn.Flatten(),  # Расплющиваем (window_size, n_features) -> (window_size * n_features)
            nn.Linear(n_input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Теперь используем всю историю, а не только последний шаг
        return self.net(observations)


class ProfitableTradingEnv(gym.Env):
    """Упрощенное торговое окружение с простой системой наград"""
    
    def __init__(self, df: pd.DataFrame, original_df: pd.DataFrame):
        super().__init__()
        
        self.df = df.reset_index(drop=True)  # Численные данные для наблюдений
        self.original_df = original_df.reset_index(drop=True)  # Исходные данные для цен
        self.window_size = SuperActiveConfig.WINDOW_SIZE
        self.initial_balance = SuperActiveConfig.INITIAL_BALANCE
        
        # Пространства как в main2.py
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, df.shape[1]),
            dtype=np.float32
        )
        
        self._reset_state()
    
    def _reset_state(self):
        """Сброс состояния как в main2.py"""
        self.balance = self.initial_balance
        self.entry_price = 0
        self.position = 0
        self.position_size = 0  # Количество позиций (0-3)
        self.current_step = self.window_size
        self.trades = []
        self.last_action = None
        self.order_size_usd = self.initial_balance * SuperActiveConfig.ORDER_SIZE_RATIO
        self.previous_balance = self.initial_balance  # Для расчета изменения баланса
    
    def reset(self, seed=None, options=None):
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Получение наблюдения как в main2.py"""
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return obs.astype(np.float32)
    
    def _get_current_price(self) -> float:
        """Получение текущей цены"""
        return self.original_df.iloc[self.current_step]['close']
    
    def _calculate_signal_strength(self) -> float:
        """Простой расчет силы сигнала без sentiment"""
        current_data = self.df.iloc[self.current_step]
        
        # RSI сигнал
        rsi = current_data['rsi']
        rsi_signal = 0.5 if rsi < -0.5 else (-0.5 if rsi > 0.5 else 0)  # RSI нормализован
        
        # EMA кроссовер
        ema_signal = 0.3 if current_data['ema_fast'] > current_data['ema_slow'] else -0.3
        
        # MACD
        macd_signal = 0.2 if current_data['macd'] > current_data['macd_signal'] else -0.2
        
        # Bollinger позиция
        bb_signal = 0.2 if current_data['bb_position'] < 0.2 else (-0.2 if current_data['bb_position'] > 0.8 else 0)
        
        # Momentum
        momentum_signal = np.clip(current_data['momentum'] * 0.3, -0.3, 0.3)
        
        total_signal = rsi_signal + ema_signal + macd_signal + bb_signal + momentum_signal
        return np.clip(total_signal, -1, 1)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Упрощенный шаг симуляции с простой системой наград"""
        reward = 0
        done = False
        
        current_price = self._get_current_price()
        signal_strength = self._calculate_signal_strength()
        
        # Сохраняем предыдущий баланс для расчета изменения
        prev_total_balance = self._get_total_balance()
        
        # === ПОКУПКА ===
        if action == 1:
            if self.position_size < SuperActiveConfig.MAX_POSITIONS:
                # Простая проверка сигнала
                if signal_strength > SuperActiveConfig.MIN_SIGNAL_STRENGTH:
                    # Усредняем цену входа
                    self.entry_price = (
                        (self.entry_price * self.position_size + current_price)
                        / (self.position_size + 1)
                    ) if self.position_size > 0 else current_price
                    
                    self.position_size += 1
                    self.position = 1
                else:
                    reward -= 0.01  # Небольшой штраф за плохой сигнал
            else:
                reward -= 0.01  # Штраф за превышение лимита позиций
        
        # === ПРОДАЖА ===
        elif action == 2:
            if self.position_size > 0:
                profit_per_coin = current_price - self.entry_price
                profit_total = (
                    profit_per_coin
                    * self.order_size_usd
                    * self.position_size
                    / self.entry_price
                )
                
                self.balance += profit_total
                self.trades.append(profit_total)
                self.position_size = 0
                self.position = 0
                self.entry_price = 0
            else:
                reward -= 0.01  # Штраф за продажу без позиции
        
        # === АВТОМАТИЧЕСКИЙ СТОП-ЛОСС И ТЕЙК-ПРОФИТ ===
        if self.position_size > 0 and self.entry_price > 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            if price_change <= -SuperActiveConfig.STOP_LOSS or price_change >= SuperActiveConfig.TAKE_PROFIT:
                # Принудительная продажа
                profit_total = (
                    (current_price - self.entry_price)
                    * self.order_size_usd
                    * self.position_size
                    / self.entry_price
                )
                self.balance += profit_total
                self.trades.append(profit_total)
                self.position_size = 0
                self.position = 0
                self.entry_price = 0
        
        # === ОСНОВНАЯ НАГРАДА: ИЗМЕНЕНИЕ БАЛАНСА ===
        current_total_balance = self._get_total_balance()
        balance_change = current_total_balance - prev_total_balance
        reward += balance_change * 0.01  # Масштабируем изменение баланса
        
        self.current_step += 1
        
        # === ЗАВЕРШЕНИЕ ЭПИЗОДА ===
        if self.current_step >= len(self.df) - 1:
            done = True
            
            # Закрытие позиций при завершении
            if self.position_size > 0:
                final_profit = (
                    (current_price - self.entry_price)
                    * self.order_size_usd
                    * self.position_size
                    / self.entry_price
                )
                self.balance += final_profit
                self.trades.append(final_profit)
                self.position_size = 0
                self.position = 0
                self.entry_price = 0
            
            # Финальная награда основана на общей прибыльности
            total_profit = self.balance - self.initial_balance
            reward += total_profit * 0.001  # Небольшой бонус за итоговую прибыль
        
        info = {
            'balance': self.balance,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'current_price': current_price,
            'signal_strength': signal_strength,
            'total_trades': len(self.trades),
            'unrealized_pnl': 0 if self.position_size == 0 else (current_price - self.entry_price) * self.order_size_usd * self.position_size / self.entry_price
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _get_total_balance(self) -> float:
        """Получение общего баланса с учетом нереализованной прибыли"""
        total_balance = self.balance
        if self.position_size > 0 and self.entry_price > 0:
            current_price = self._get_current_price()
            unrealized = (
                (current_price - self.entry_price)
                * self.order_size_usd
                * self.position_size
                / self.entry_price
            )
            total_balance += unrealized
        return total_balance


def main():
    """Главная функция исправленной торговой системы"""
    print("🚀 ИСПРАВЛЕННАЯ ТОРГОВАЯ СИСТЕМА V3.8 - СТАБИЛЬНАЯ ВЕРСИЯ")
    print("✅ ИСПРАВЛЕНЫ ВСЕ ОСНОВНЫЕ ПРОБЛЕМЫ")
    print("=" * 75)
    
    # 1. Загрузка данных
    print("\n📊 ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("-" * 50)
    
    data_loader = SimpleDataLoader("data/BTC_5_96w.csv")  # Используем тот же файл что и main2.py
    
    # Загружаем сначала исходные данные
    original_df = pd.read_csv("data/BTC_5_96w.csv")
    original_df['timestamp'] = pd.to_datetime(original_df['timestamp'], unit='ms')
    
    # Затем обработанные
    df = data_loader.load_and_prepare_data()
    
    # 2. Создание окружения
    print("\n🎮 ЭТАП 2: СОЗДАНИЕ СТАБИЛЬНОГО ОКРУЖЕНИЯ")
    print("-" * 50)
    env = ProfitableTradingEnv(df, original_df)
    vec_env = DummyVecEnv([lambda: env])
    print(f"✅ Окружение готово. Баланс: ${SuperActiveConfig.INITIAL_BALANCE}")
    print(f"✅ Риск на сделку: {SuperActiveConfig.ORDER_SIZE_RATIO*100:.1f}% капитала")
    
    # 3. Создание модели
    print("\n🧠 ЭТАП 3: СОЗДАНИЕ УЛУЧШЕННОЙ МОДЕЛИ")
    print("-" * 50)
    
    policy_kwargs = dict(
        features_extractor_class=ProfitableFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 64]
    )
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=SuperActiveConfig.LEARNING_RATE,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=SuperActiveConfig.ENTROPY_COEF,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu"
    )
    
    print("✅ Модель создана с исправленными параметрами")
    print("✅ Feature Extractor теперь видит всю историю за 50 шагов")
    
    # 4. Обучение
    print("\n🎓 ЭТАП 4: РАСШИРЕННОЕ ОБУЧЕНИЕ МОДЕЛИ")
    print("-" * 50)
    print(f"⏱️ Обучение на {SuperActiveConfig.TOTAL_TIMESTEPS:,} шагов...")
    model.learn(total_timesteps=SuperActiveConfig.TOTAL_TIMESTEPS)
    print("✅ Обучение завершено")
    
    # 5. Тестирование
    print("\n💰 ЭТАП 5: ТЕСТИРОВАНИЕ ИСПРАВЛЕННОЙ СТРАТЕГИИ")
    print("-" * 50)
    
    test_env = ProfitableTradingEnv(df, original_df)
    obs, _ = test_env.reset()
    
    balance_history = []
    price_history = []
    action_history = []
    trade_log = []
    
    print("💼 Начинаем торговлю с исправленной системой...")
    
    step_count = 0
    while step_count < 5000:  # Ограничим количество шагов
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(int(action))
        
        # Расчет общего баланса с нереализованной прибылью
        total_balance = test_env._get_total_balance()
        
        balance_history.append(total_balance)
        price_history.append(test_env._get_current_price())
        action_history.append(int(action))
        
        # Логирование сделок
        if len(test_env.trades) > len(trade_log):
            new_trades = test_env.trades[len(trade_log):]
            for trade in new_trades:
                trade_log.append(trade)
                profit_str = f"+${trade:.2f}" if trade > 0 else f"${trade:.2f}"
                signal_strength = test_env._calculate_signal_strength()
                print(f"💸 Сделка: {profit_str}, Позиций: {test_env.position_size}, Сигнал: {signal_strength:.2f}")
        
        step_count += 1
        if done:
            break
    
    # 6. Анализ результатов
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ ИСПРАВЛЕННОЙ СИСТЕМЫ")
    print("-" * 50)
    
    if balance_history:
        initial_balance = SuperActiveConfig.INITIAL_BALANCE
        final_balance = balance_history[-1]
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        total_trades = len(trade_log)
        profitable_trades = len([t for t in trade_log if t > 0])
        win_rate = (profitable_trades / max(total_trades, 1)) * 100
        
        # Расчет максимальной просадки
        peak = initial_balance
        max_drawdown = 0
        for balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Buy & Hold сравнение
        start_price = price_history[0]
        end_price = price_history[-1]
        bnh_return = (end_price - start_price) / start_price * 100
        
        print("🎯 РЕЗУЛЬТАТЫ ИСПРАВЛЕННОЙ СТРАТЕГИИ")
        print("=" * 60)
        print(f"💰 Начальный баланс: ${initial_balance:,.2f}")
        print(f"💰 Финальный баланс: ${final_balance:,.2f}")
        print(f"📈 Доходность: {total_return:+.2f}%")
        print(f"📊 Buy & Hold: {bnh_return:+.2f}%")
        print(f"🔄 Всего сделок: {total_trades}")
        print(f"✅ Прибыльных: {profitable_trades} ({win_rate:.1f}%)")
        print(f"📉 Макс. просадка: {max_drawdown*100:.2f}%")
        
        if trade_log:
            avg_profit = np.mean(trade_log)
            best_trade = max(trade_log)
            worst_trade = min(trade_log)
            print(f"💰 Средняя сделка: ${avg_profit:.2f}")
            print(f"🏆 Лучшая сделка: ${best_trade:.2f}")
            print(f"😞 Худшая сделка: ${worst_trade:.2f}")
        
        print("\n🎉 ЗАКЛЮЧЕНИЕ ПО ИСПРАВЛЕННОЙ СИСТЕМЕ")
        print("=" * 50)
        
        print("✅ ИСПРАВЛЕНИЯ ВНЕСЕНЫ:")
        print("  • Убраны фейковые sentiment данные")
        print("  • Feature Extractor видит всю историю")
        print("  • Упрощена система наград")
        print("  • Увеличено время обучения")
        print("  • Снижены риски на сделку")
        
        if total_return > 5 and total_trades > 0:
            print("🟢 ОТЛИЧНО! Исправленная стратегия показывает прибыль!")
            print(f"💰 Прибыль: ${final_balance - initial_balance:,.2f}")
        elif total_return > 0:
            print("🟡 ХОРОШО! Есть небольшая прибыль")
        elif total_return > -5:
            print("🔶 НЕЙТРАЛЬНО! Близко к безубыточности")
        else:
            print("🔴 ТРЕБУЕТ ДОПОЛНИТЕЛЬНОЙ НАСТРОЙКИ")
        
        print(f"📊 Итоговая доходность: {total_return:+.2f}%")
        print("✨ Анализ исправленной системы завершен!")
    
    else:
        print("❌ Недостаточно данных для анализа")


if __name__ == "__main__":
    main() 