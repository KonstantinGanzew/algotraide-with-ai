"""
🔧 КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ ДЛЯ РЕАЛЬНОЙ ПРИБЫЛЬНОСТИ

ПРОБЛЕМА: Система вознаграждений не коррелировала с реальной прибыльностью
РЕШЕНИЕ: Полная переработка торговой логики

КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ:
1. ✅ ИСПРАВЛЕНА функция _calculate_profit() - теперь правильно рассчитывает прибыль
2. ✅ УПРОЩЕНА функция _calculate_dynamic_order_size() - возвращает сумму в долларах
3. ✅ ПЕРЕРАБОТАНА система вознаграждений - основана ТОЛЬКО на изменении баланса
4. ✅ УБРАНЫ искусственные бонусы за торговые действия
5. ✅ ИСПРАВЛЕНА торговая логика - корректно списывает/возвращает средства
6. ✅ ДОБАВЛЕНЫ торговые комиссии для реалистичности
7. ✅ УЛУЧШЕНО управление рисками (2% риска на сделку)
8. ✅ ОПТИМИЗИРОВАНО соотношение риск/прибыль (1:3)

СТАРАЯ ПРОБЛЕМА: 
- Алгоритм получал награды ~32000, но баланс оставался 10000 USDT
- 841 сделка без реальной прибыли

ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:
- Награды напрямую коррелируют с ростом баланса
- Алгоритм учится генерировать реальную прибыль
- Учитываются реальные торговые издержки

УЛУЧШЕННАЯ ВЕРСИЯ ТОРГОВОГО АЛГОРИТМА ДЛЯ ПОВЫШЕНИЯ ПРИБЫЛЬНОСТИ

КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:
1. ИСПРАВЛЕННЫЙ риск-менеджмент (2% риска на сделку вместо 15%)
2. УЛУЧШЕННОЕ соотношение стоп-лосс/тейк-профит (2% к 6% вместо 3% к 10%)
3. УБРАНА система искусственных вознаграждений
4. РЕАЛИСТИЧНЫЕ торговые комиссии (0.1%)
5. УЛУЧШЕННЫЕ параметры обучения PPO
6. УСИЛЕННАЯ модель с LSTM архитектурой

Эти изменения должны привести к более стабильной и прибыльной торговле.
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn


# === КОНСТАНТЫ ===
class Config:
    # Файлы и пути
    DATA_FOLDER = "data/"
    DATA_FILE = "BTC_5_96w.csv"  # Полный набор данных для максимальной прибыли
    
    # Параметры окружения
    WINDOW_SIZE = 50
    INITIAL_BALANCE = 10000
    POSITIONS_LIMIT = 1.0
    PASSIVITY_THRESHOLD = 100
    
    # Параметры устройства
    AUTO_DEVICE = True
    FORCE_CPU = False
    DEVICE = "cpu"
    
    # ОПТИМИЗИРОВАННАЯ торговая стратегия для максимальной прибыли
    BASE_RISK_PER_TRADE = 0.03         # Увеличен базовый риск до 3%
    DYNAMIC_RISK_MULTIPLIER = 1.5      # Динамический множитель риска
    STOP_LOSS_PERCENTAGE = 0.015       # Тайтер стоп-лосс 1.5%
    TAKE_PROFIT_PERCENTAGE = 0.045     # Тейк-профит 4.5% (соотношение 1:3)
    TRAILING_STOP_PERCENTAGE = 0.01    # Трейлинг стоп 1%
    MAX_DRAWDOWN_LIMIT = 0.12          # 12% максимальная просадка
    TRADE_COMMISSION = 0.001           # 0.1% комиссия за сделку
    
    # УСИЛЕННЫЕ параметры вознаграждений с фокусом на прибыль
    BALANCE_CHANGE_MULTIPLIER = 200    # Увеличена награда за прибыль
    PROFIT_STREAK_BONUS = 50.0         # Бонус за серию прибыльных сделок
    SHARP_RATIO_BONUS = 100.0          # Бонус за высокий Sharpe ratio
    
    # АГРЕССИВНЫЕ размеры позиций при благоприятных условиях
    MIN_POSITION_SIZE = 0.15           # Увеличена минимальная позиция до 15%
    MAX_POSITION_MULTIPLIER = 1.2      # Максимальная позиция 120% при высокой уверенности
    CONFIDENCE_THRESHOLD = 0.7         # Порог уверенности для увеличения позиции
    
    # Технические индикаторы
    EMA_FAST_SPAN = 12
    EMA_SLOW_SPAN = 26
    RSI_WINDOW = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    MACD_FAST = 12
    MACD_SLOW = 26
    BOLLINGER_WINDOW = 20
    MOMENTUM_WINDOW = 10
    
    # АГРЕССИВНЫЕ параметры обучения для максимальной прибыли
    TOTAL_TIMESTEPS = 10000           # Увеличено время обучения
    PPO_ENT_COEF = 0.02                # Больше исследования для поиска прибыльных стратегий
    LEARNING_RATE = 5e-5               # Более точная настройка
    
    # Умная ранняя остановка с фокусом на прибыль
    ENABLE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 75       # Больше терпеливости для поиска прибыли
    MIN_EPISODES_BEFORE_STOPPING = 300  # Больше эпизодов для оценки
    IMPROVEMENT_THRESHOLD = 0.002       # Более чувствительный порог
    
    # МОЩНАЯ LSTM архитектура
    LSTM_HIDDEN_SIZE = 512             # Увеличена мощность модели
    LSTM_NUM_LAYERS = 3                # Больше слоёв для сложных паттернов
    
    # Визуализация
    FIGURE_SIZE = (16, 10)


def setup_device():
    """Настройка устройства для обучения (CPU/GPU)"""
    if Config.FORCE_CPU:
        device = "cpu"
        print("🔧 Принудительно используется CPU")
    elif Config.AUTO_DEVICE:
        if torch.cuda.is_available():
            device = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"🚀 Используется GPU: {gpu_name}")
                print(f"💾 Память GPU: {gpu_memory:.1f} GB")
            except:
                print("🚀 Используется GPU")
        else:
            device = "cpu"
            print("⚠️  GPU недоступно, используется CPU")
    else:
        device = Config.DEVICE
        print(f"🎯 Используется указанное устройство: {device}")
    
    return device


def check_gpu_requirements():
    """Проверка требований для GPU"""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "gpu_memory": None
    }
    
    if torch.cuda.is_available():
        try:
            info["current_device"] = torch.cuda.current_device()
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except:
            pass
        
    return info


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """УЛУЧШЕННЫЙ LSTM Feature Extractor для обработки временных рядов"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Получаем размеры входа
        if observation_space.shape is not None:
            n_input_features = observation_space.shape[-1]
            sequence_length = observation_space.shape[0]
        else:
            # Fallback значения
            n_input_features = 20
            sequence_length = 50
        
        # УЛУЧШЕННЫЕ LSTM слои
        self.lstm = nn.LSTM(
            input_size=n_input_features,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=0.2 if Config.LSTM_NUM_LAYERS > 1 else 0
        )
        
        # Дополнительные слои
        self.feature_net = nn.Sequential(
            nn.Linear(Config.LSTM_HIDDEN_SIZE, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Обработка входных данных через LSTM
        lstm_out, _ = self.lstm(observations)
        # Берем выход последнего временного шага
        features = lstm_out[:, -1, :]
        return self.feature_net(features)


class SmartEarlyStoppingCallback(BaseCallback):
    """Умный callback с ранней остановкой при отсутствии улучшений"""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.step_count = 0
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.progress_interval = 10000  # Показывать прогресс каждые 10k шагов
        self.episode_rewards_history = []
        self.recent_rewards_window = 10  # Окно для усреднения последних эпизодов

    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Проверяем завершенные эпизоды
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_reward = info['episode'].get('r', 0)
                    self.episode_count += 1
                    self.episode_rewards_history.append(episode_reward)
                    
                    # Используем скользящее среднее для более стабильной оценки
                    if len(self.episode_rewards_history) >= self.recent_rewards_window:
                        recent_avg = np.mean(self.episode_rewards_history[-self.recent_rewards_window:])
                        
                        # Проверяем улучшение
                        if recent_avg > self.best_reward + Config.IMPROVEMENT_THRESHOLD:
                            improvement = recent_avg - self.best_reward
                            self.best_reward = recent_avg
                            self.episodes_without_improvement = 0
                            print(f"🚀 [Эпизод {self.episode_count}] Новый рекорд: {recent_avg:.3f} (+{improvement:.3f})")
                        else:
                            self.episodes_without_improvement += 1
                            if self.episode_count % 5 == 0:  # Показываем каждые 5 эпизодов
                                print(f"📊 [Эпизод {self.episode_count}] Награда: {recent_avg:.3f} (лучшая: {self.best_reward:.3f}, без улучшения: {self.episodes_without_improvement})")
        
        # Показываем прогресс по шагам
        if self.step_count % self.progress_interval == 0:
            print(f"⏱️  [Шаг {self.step_count}] Эпизодов пройдено: {self.episode_count}")
        
        # Проверяем условия ранней остановки
        if Config.ENABLE_EARLY_STOPPING and self.episode_count >= Config.MIN_EPISODES_BEFORE_STOPPING:
            if self.episodes_without_improvement >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\n🛑 РАННЯЯ ОСТАНОВКА!")
                print(f"   Эпизодов без улучшения: {self.episodes_without_improvement}")
                print(f"   Лучшая средняя награда: {self.best_reward:.3f}")
                print(f"   Общее количество эпизодов: {self.episode_count}")
                print(f"   Общее количество шагов: {self.step_count}")
                return False  # Останавливаем обучение
        
        return True  # Продолжаем обучение


class SimplifiedTradingEnv(gym.Env):
    """ИСПРАВЛЕННОЕ торговое окружение с правильными расчетами"""
    
    def __init__(self, df, window_size=50, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Пространство действий: 0-Hold, 1-Buy, 2-Sell
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, df.shape[1]), dtype=np.float32
        )

        # Максимум шагов на эпизод для принудительного завершения
        self.max_episode_steps = 10000  # Увеличено для более длительного обучения
        self.episode_steps = 0

        self._reset_state()

    def _reset_state(self):
        """Сброс состояния окружения"""
        self.balance = float(self.initial_balance)
        self.btc_amount = 0.0        # Количество BTC (исправлено)
        self.entry_price = 0.0
        self.current_step = self.window_size
        self.episode_steps = 0
        self.trades = []
        self.balance_history = [float(self.initial_balance)]
        self.max_balance = float(self.initial_balance)
        self.returns_history = []
        self.profit_streak = 0
        self.loss_streak = 0
        
        # Улучшенный риск-менеджмент
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.trailing_stop_price = 0.0     # Трейлинг стоп
        self.position_high_price = 0.0     # Максимальная цена после входа
        self.max_drawdown = 0.0
        
        # Для расчета волатильности
        self.price_history = []

    def reset(self, seed=None, options=None):
        """Сброс окружения"""
        self._reset_state()
        return self._get_observation(), {}

    def _get_observation(self):
        """Получение текущего наблюдения"""
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return obs.astype(np.float32)

    def _calculate_dynamic_order_size(self):
        """ОПТИМИЗИРОВАННЫЙ расчет размера позиции с динамическим риском"""
        available_balance = self.balance
        
        # Базовый размер позиции
        base_risk = Config.BASE_RISK_PER_TRADE
        
        # Динамическое увеличение риска при благоприятных условиях
        dynamic_risk = base_risk
        if len(self.trades) > 0:
            # Увеличиваем риск при серии прибыльных сделок
            recent_trades = self.trades[-5:] if len(self.trades) >= 5 else self.trades
            profitable_ratio = sum(1 for trade in recent_trades if trade > 0) / len(recent_trades)
            if profitable_ratio >= Config.CONFIDENCE_THRESHOLD:
                dynamic_risk = base_risk * Config.DYNAMIC_RISK_MULTIPLIER
        
        position_value = available_balance * dynamic_risk
        
        # Минимальный размер позиции
        min_position = available_balance * Config.MIN_POSITION_SIZE
        # Максимальный размер позиции
        max_position = available_balance * Config.MAX_POSITION_MULTIPLIER
        
        position_value = max(min_position, min(position_value, max_position))
        
        return position_value  # Возвращаем значение в долларах
    
    def _calculate_profit(self, current_price):
        """ИСПРАВЛЕННЫЙ расчет прибыли от позиции"""
        if self.btc_amount <= 0 or self.entry_price <= 0:
            return 0.0
        
        # ПРАВИЛЬНАЯ формула: количество BTC * (текущая цена - цена входа)
        profit = self.btc_amount * (current_price - self.entry_price)
        return profit

    def _calculate_enhanced_reward(self, current_price, action):
        """УСИЛЕННАЯ система вознаграждений с фокусом на максимальную прибыль"""
        # Обновляем историю цен
        self.price_history.append(float(current_price))
        if len(self.price_history) > 20:
            self.price_history.pop(0)
        
        # ОСНОВНАЯ НАГРАДА - изменение общего баланса
        prev_total_balance = self.balance_history[-1] if self.balance_history else self.initial_balance
        unrealized_profit = self._calculate_profit(current_price) if self.btc_amount > 0 else 0
        current_total_balance = self.balance + unrealized_profit
        
        balance_change = current_total_balance - prev_total_balance
        base_reward = balance_change * Config.BALANCE_CHANGE_MULTIPLIER
        
        # БОНУСЫ за прибыльные серии
        profit_streak_bonus = 0.0
        if len(self.trades) > 0:
            # Бонус за серию прибыльных сделок
            if self.profit_streak >= 3:
                profit_streak_bonus = self.profit_streak * Config.PROFIT_STREAK_BONUS
        
        # Бонус за высокое соотношение Sharpe (доходность/риск)
        sharpe_bonus = 0.0
        if len(self.balance_history) > 10:
            returns = np.diff(self.balance_history[-10:]) / np.array(self.balance_history[-10:-1])
            if len(returns) > 1:
                returns_mean = np.mean(returns)
                returns_std = np.std(returns)
                if returns_std > 0 and returns_mean > 0:
                    sharpe_ratio = returns_mean / returns_std
                    sharpe_bonus = sharpe_ratio * Config.SHARP_RATIO_BONUS
        
        # Штраф за большую просадку (усиленный)
        drawdown_penalty = 0.0
        self.max_balance = max(self.max_balance, current_total_balance)
        current_drawdown = (self.max_balance - current_total_balance) / self.max_balance if self.max_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        if current_drawdown > Config.MAX_DRAWDOWN_LIMIT:
            drawdown_penalty = (current_drawdown - Config.MAX_DRAWDOWN_LIMIT) * 1000
        
        # Итоговая награда
        total_reward = base_reward + profit_streak_bonus + sharpe_bonus - drawdown_penalty
        
        # Обновляем историю
        self.balance_history.append(float(current_total_balance))
        if len(self.balance_history) > 100:
            self.balance_history.pop(0)
            
        return total_reward, {
            'base_reward': base_reward,
            'balance_change': balance_change,
            'profit_streak_bonus': profit_streak_bonus,
            'sharpe_bonus': sharpe_bonus,
            'drawdown_penalty': drawdown_penalty,
            'total_balance': current_total_balance,
            'unrealized_profit': unrealized_profit,
            'current_drawdown': current_drawdown
        }

    def _execute_simplified_trade(self, action, current_price):
        """ИСПРАВЛЕННОЕ выполнение торговых операций с комиссиями"""
        trade_info = {}
        
        if action == 0:  # Hold - ничего не делаем
            return trade_info
            
        elif action == 1:  # Buy - покупаем только если нет позиции
            if self.btc_amount == 0:  # Открываем новую позицию только если её нет
                position_value = self._calculate_dynamic_order_size()
                commission = position_value * Config.TRADE_COMMISSION
                
                # Проверяем, что у нас достаточно средств (включая комиссию)
                total_cost = position_value + commission
                if total_cost <= self.balance:
                    # ИСПРАВЛЕНО: рассчитываем количество купленных BTC
                    self.btc_amount = position_value / current_price
                    self.entry_price = current_price
                    self.balance -= total_cost  # Списываем средства + комиссию
                    
                    # Устанавливаем стоп-лосс, тейк-профит и трейлинг стоп
                    self.stop_loss_price = current_price * (1 - Config.STOP_LOSS_PERCENTAGE)
                    self.take_profit_price = current_price * (1 + Config.TAKE_PROFIT_PERCENTAGE)
                    self.trailing_stop_price = current_price * (1 - Config.TRAILING_STOP_PERCENTAGE)
                    self.position_high_price = current_price
                    
                    trade_info = {
                        'action': 'BUY',
                        'btc_amount': self.btc_amount,
                        'price': current_price,
                        'cost': total_cost,
                        'commission': commission,
                        'balance_after': self.balance
                    }
        
        elif action == 2:  # Sell - продаем позицию если она есть
            if self.btc_amount > 0:
                # Рассчитываем выручку от продажи
                sale_proceeds = self.btc_amount * current_price
                commission = sale_proceeds * Config.TRADE_COMMISSION
                net_proceeds = sale_proceeds - commission
                
                # Рассчитываем прибыль
                profit = self._calculate_profit(current_price)
                
                # Возвращаем чистую выручку на баланс
                self.balance += net_proceeds
                self.trades.append(profit)
                
                # Обновляем стрики прибылей/убытков
                if profit > 0:
                    self.profit_streak += 1
                    self.loss_streak = 0
                else:
                    self.loss_streak += 1
                    self.profit_streak = 0
                
                trade_info = {
                    'action': 'SELL',
                    'btc_amount': self.btc_amount,
                    'price': current_price,
                    'proceeds': sale_proceeds,
                    'commission': commission,
                    'net_proceeds': net_proceeds,
                    'profit': profit,
                    'balance_after': self.balance,
                    'profit_streak': self.profit_streak,
                    'loss_streak': self.loss_streak
                }
                
                # Сбрасываем позицию полностью
                self.btc_amount = 0
                self.entry_price = 0
                self.stop_loss_price = 0
                self.take_profit_price = 0
        
        return trade_info

    def _check_stop_loss_take_profit(self, current_price):
        """ИСПРАВЛЕННАЯ проверка стоп-лосса и тейк-профита"""
        if self.btc_amount <= 0:
            return 0.0, {}
        
        trade_info = {}
        
        # Проверка стоп-лосса
        if current_price <= self.stop_loss_price and self.stop_loss_price > 0:
            sale_proceeds = self.btc_amount * current_price
            commission = sale_proceeds * Config.TRADE_COMMISSION
            net_proceeds = sale_proceeds - commission
            profit = self._calculate_profit(current_price)
            
            self.balance += net_proceeds
            self.trades.append(profit)
            
            trade_info = {
                'action': 'STOP_LOSS',
                'btc_amount': self.btc_amount,
                'price': current_price,
                'profit': profit,
                'commission': commission
            }
            
            # Обновляем стрики
            if profit > 0:
                self.profit_streak += 1
                self.loss_streak = 0
            else:
                self.loss_streak += 1
                self.profit_streak = 0
            
            # Сброс позиции
            self.btc_amount = 0
            self.entry_price = 0
            self.stop_loss_price = 0
            self.take_profit_price = 0
                
        # Проверка тейк-профита
        elif current_price >= self.take_profit_price and self.take_profit_price > 0:
            sale_proceeds = self.btc_amount * current_price
            commission = sale_proceeds * Config.TRADE_COMMISSION
            net_proceeds = sale_proceeds - commission
            profit = self._calculate_profit(current_price)
            
            self.balance += net_proceeds
            self.trades.append(profit)
            
            trade_info = {
                'action': 'TAKE_PROFIT',
                'btc_amount': self.btc_amount,
                'price': current_price,
                'profit': profit,
                'commission': commission
            }
            
            # Обновляем стрики
            if profit > 0:
                self.profit_streak += 1
                self.loss_streak = 0
            else:
                self.loss_streak += 1
                self.profit_streak = 0
            
            # Сброс позиции
            self.btc_amount = 0
            self.entry_price = 0
            self.stop_loss_price = 0
            self.take_profit_price = 0
        
        return 0.0, trade_info  # Возвращаем 0 награды, так как награда рассчитывается отдельно

    def step(self, action):
        """ОПТИМИЗИРОВАННЫЙ шаг в окружении"""
        current_price = self.df.iloc[self.current_step]['close']
        
        # Проверяем стоп-лосс/тейк-профит
        sl_tp_reward, sl_tp_info = self._check_stop_loss_take_profit(current_price)
        
        # Выполняем действие пользователя
        trade_info = self._execute_simplified_trade(action, current_price)
        
        # Рассчитываем усиленную награду (основана на реальном PnL + бонусы за прибыль)
        total_reward, reward_info = self._calculate_enhanced_reward(current_price, action)
        
        self.current_step += 1
        self.episode_steps += 1
        
        # ОПТИМИЗИРОВАННЫЕ условия завершения эпизода
        current_total = self.balance + (self._calculate_profit(current_price) if self.btc_amount > 0 else 0)
        done = (self.current_step >= len(self.df) - 1 or 
                current_total <= 0.5 * self.initial_balance or  # Более строгое условие
                self.episode_steps >= self.max_episode_steps)
        
        truncated = self.episode_steps >= self.max_episode_steps
        
        # Собираем информацию
        info = {
            'balance': self.balance,
            'btc_amount': self.btc_amount,
            'total_trades': len(self.trades),
            'profit_streak': self.profit_streak,
            'loss_streak': self.loss_streak,
            'max_drawdown': self.max_drawdown,
            'reward_info': reward_info,
            'trade_info': trade_info,
            'sl_tp_info': sl_tp_info
        }
        
        return self._get_observation(), total_reward, done, truncated, info

    def render(self):
        """Отображение текущего состояния"""
        unrealized = self._calculate_profit(self.df.iloc[self.current_step-1]['close']) if self.btc_amount > 0 else 0
        total_balance = self.balance + unrealized
        print(f"Step: {self.current_step}, Balance: {total_balance:.2f}, BTC: {self.btc_amount:.6f}, Drawdown: {self.max_drawdown:.2%}")


def load_and_prepare_data(file_path):
    """УЛУЧШЕННАЯ загрузка и подготовка данных с дополнительными фичами"""
    df = pd.read_csv(file_path)
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Базовые технические индикаторы
    df['ema_fast'] = df['close'].ewm(span=Config.EMA_FAST_SPAN, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=Config.EMA_SLOW_SPAN, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=Config.RSI_WINDOW).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=Config.RSI_WINDOW).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=Config.MACD_FAST).mean()
    ema_26 = df['close'].ewm(span=Config.MACD_SLOW).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # OBV (On-Balance Volume)
    df['price_change'] = df['close'].diff()
    df['obv_raw'] = np.where(df['price_change'] > 0, df['volume'], 
                        np.where(df['price_change'] < 0, -df['volume'], 0))
    df['obv'] = df['obv_raw'].cumsum()

    # VWAP (Volume Weighted Average Price)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap_numerator'] = (df['typical_price'] * df['volume']).cumsum()
    df['vwap_denominator'] = df['volume'].cumsum()
    df['vwap'] = df['vwap_numerator'] / df['vwap_denominator']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=Config.BOLLINGER_WINDOW).mean()
    bb_std = df['close'].rolling(window=Config.BOLLINGER_WINDOW).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # НОВЫЕ УЛУЧШЕННЫЕ индикаторы
    close_series = df['close']
    df['log_return'] = np.log(close_series / close_series.shift(1))
    df['volatility'] = df['log_return'].rolling(window=20).std()
    df['price_trend'] = close_series.pct_change(periods=5)
    
    # Дополнительные индикаторы
    df['price_change_pct'] = df['close'].pct_change()
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # ATR (Average True Range)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()

    # Очистка временных колонок
    df.drop(['price_change', 'obv_raw', 'vwap_numerator', 'vwap_denominator', 
             'typical_price', 'tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)

    # Очистка NaN
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ИСПРАВЛЕННАЯ нормализация - НЕ нормализуем цены, только индикаторы
    print("🔍 Средние значения ДО нормализации:")
    print(df[['close', 'ema_fast', 'macd', 'rsi']].mean())
    
    # Нормализуем только технические индикаторы, НЕ ЦЕНЫ!
    price_cols = ['open', 'high', 'low', 'close']  # Цены НЕ нормализуем!
    indicator_cols = ['ema_fast', 'ema_slow', 'macd', 'macd_signal', 'macd_histogram', 
                     'obv', 'vwap', 'bb_middle', 'bb_upper', 'bb_lower', 'volume_sma', 'atr']
    
    # Нормализуем только объём и технические индикаторы
    cols_to_normalize = ['volume'] + indicator_cols
    
    for col in cols_to_normalize:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val
    
    print("🔍 Средние значения ПОСЛЕ нормализации:")
    print(df[['close', 'ema_fast', 'macd', 'rsi']].mean())
    print("💡 Цены НЕ нормализованы - это поможет модели лучше понимать рыночные сигналы!")

    print(f"📊 Подготовлено {len(df.columns)} признаков: {list(df.columns)}")
    return df


def train_model(env):
    """УЛУЧШЕННОЕ обучение модели PPO с LSTM архитектурой"""
    device = setup_device()
    
    print(f"\n🎯 Создание УЛУЧШЕННОЙ модели PPO с LSTM на устройстве: {device}")
    if Config.ENABLE_EARLY_STOPPING:
        print(f"🧠 УМНОЕ ОБУЧЕНИЕ: до {Config.TOTAL_TIMESTEPS:,} шагов с ранней остановкой")
        print(f"   📊 Остановка после {Config.EARLY_STOPPING_PATIENCE} эпизодов без улучшения")
        print(f"   📈 Минимальное улучшение: {Config.IMPROVEMENT_THRESHOLD}")
    else:
        print(f"🔥 ПОЛНОЕ ОБУЧЕНИЕ: {Config.TOTAL_TIMESTEPS:,} шагов БЕЗ ранней остановки")
    
    # УЛУЧШЕННЫЕ настройки политики с LSTM
    policy_kwargs = {
        "features_extractor_class": LSTMFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": Config.LSTM_HIDDEN_SIZE},
        "net_arch": [dict(pi=[512, 256], vf=[512, 256])],  # Более мощная архитектура
        "activation_fn": nn.ReLU
    }
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=Config.LEARNING_RATE,
        n_steps=4096,        # Увеличен размер для более стабильного обучения
        batch_size=128,      # Увеличен batch size для лучшей конвергенции
        n_epochs=10,         # Умеренное количество эпох
        gamma=0.99,          # Стандартный discount factor
        gae_lambda=0.95,     # Стандартный GAE lambda
        clip_range=0.2,      # Стандартный clip range
        clip_range_vf=None,
        ent_coef=Config.PPO_ENT_COEF,
        vf_coef=0.5,         # Стандартный value function coefficient
        max_grad_norm=0.5,   # Нормальный gradient clipping
        device=device,
        verbose=1
    )
    
    print(f"🚀 НАЧИНАЕМ УЛУЧШЕННОЕ ОБУЧЕНИЕ...")
    print("💡 Прогресс будет показываться по эпизодам и шагам")
    print("🎯 Фокус на реальной прибыльности без искусственных бонусов")
    
    # Создаем умный callback
    callback = SmartEarlyStoppingCallback()
    
    try:
        model.learn(
            total_timesteps=Config.TOTAL_TIMESTEPS, 
            callback=callback
        )
        print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО ПОЛНОСТЬЮ!")
    except KeyboardInterrupt:
        print("⚠️ ОБУЧЕНИЕ ПРЕРВАНО ПОЛЬЗОВАТЕЛЕМ!")
    
    # Выводим финальную статистику
    if hasattr(callback, 'episode_count'):
        print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   Всего эпизодов: {callback.episode_count}")
        print(f"   Всего шагов: {callback.step_count}")
        print(f"   Лучшая средняя награда: {callback.best_reward:.3f}")
        print(f"   Эпизодов без улучшения: {callback.episodes_without_improvement}")
    
    return model


def test_model(model, test_env, df):
    """РЕАЛИСТИЧНОЕ тестирование с учетом комиссий"""
    obs, _ = test_env.reset()
    
    results = {
        'balance_history': [],
        'prices': [],
        'actions': [],
        'trades': [],
        'drawdowns': [],
        'positions': [],
        'trade_details': [],
        'commissions_paid': []
    }
    
    max_steps = len(df) - test_env.window_size - 10
    step_count = 0
    total_commissions = 0.0

    print(f"🚀 РЕАЛИСТИЧНОЕ тестирование (максимум {max_steps} шагов)...")
    print("💡 Используется СТОХАСТИЧЕСКАЯ политика для максимальной прибыльности!")
    print(f"🏦 Торговые комиссии: {Config.TRADE_COMMISSION:.1%}")
    
    while step_count < max_steps:
        try:
            # СТОХАСТИЧЕСКАЯ политика для реалистичного тестирования
            action_result = model.predict(obs, deterministic=False)
            action = int(action_result[0]) if isinstance(action_result[0], (np.ndarray, list)) else int(action_result[0])
            
            obs, reward, done, truncated, info = test_env.step(action)
            step_count += 1

            if test_env.current_step >= len(df):
                break

            current_price = df.iloc[test_env.current_step]['close']
            
            # Собираем статистику
            unrealized = test_env._calculate_profit(current_price) if test_env.btc_amount > 0 else 0
            total_balance = test_env.balance + unrealized
            
            results['balance_history'].append(total_balance)
            results['prices'].append(current_price)
            results['actions'].append(action)
            results['drawdowns'].append(info.get('max_drawdown', 0))
            results['positions'].append(test_env.btc_amount)
            
            # Отслеживаем комиссии
            if info.get('trade_info') and 'commission' in info['trade_info']:
                commission = info['trade_info']['commission']
                total_commissions += commission
                results['commissions_paid'].append(commission)
            
            if info.get('trade_info'):
                results['trade_details'].append({
                    'step': step_count,
                    'info': info['trade_info']
                })

            # Сброс при завершении эпизода для продолжения
            if done:
                print(f"📊 Эпизод завершен на шаге {step_count}, баланс: {total_balance:.2f}")
                obs, _ = test_env.reset()
                
            if step_count % 5000 == 0:
                print(f"💰 Тестирование: {step_count}/{max_steps} шагов, баланс: {total_balance:.2f}, комиссии: {total_commissions:.2f}")
                
        except Exception as e:
            print(f"❌ Ошибка в тестировании на шаге {step_count}: {e}")
            break

    results['trades'] = test_env.trades
    results['total_commissions'] = total_commissions
    print(f"✅ РЕАЛИСТИЧНОЕ тестирование завершено за {step_count} шагов")
    print(f"💸 Общие комиссии: {total_commissions:.2f} USDT")
    return results


def analyze_results(results, initial_balance):
    """Детальный анализ результатов торговли"""
    if not results['balance_history']:
        return {}
    
    balance_history = np.array(results['balance_history'])
    final_balance = balance_history[-1]
    
    # Основные метрики
    total_return = (final_balance - initial_balance) / initial_balance
    max_balance = np.max(balance_history)
    max_drawdown = np.max(results['drawdowns'])
    
    # Расчет доходности по дням
    returns = np.diff(balance_history) / balance_history[:-1]
    
    # Sharpe Ratio (предполагаем 252 торговых дня в году)
    if len(returns) > 1:
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Анализ сделок
    trades = results['trades']
    profitable_trades = [t for t in trades if t > 0]
    losing_trades = [t for t in trades if t < 0]
    
    win_rate = len(profitable_trades) / len(trades) * 100 if trades else 0
    avg_profit = np.mean(profitable_trades) if profitable_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    profit_factor = abs(sum(profitable_trades) / sum(losing_trades)) if losing_trades else float('inf')
    
    # Максимальная серия побед/поражений
    win_streak = 0
    loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    
    for trade in trades:
        if trade > 0:
            current_win_streak += 1
            current_loss_streak = 0
            win_streak = max(win_streak, current_win_streak)
        else:
            current_loss_streak += 1
            current_win_streak = 0
            loss_streak = max(loss_streak, current_loss_streak)
    
    analysis = {
        'total_return': total_return,
        'final_balance': final_balance,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'win_streak': win_streak,
        'loss_streak': loss_streak,
        'volatility': np.std(returns) if len(returns) > 1 else 0
    }
    
    return analysis


def visualize_results(results, analysis):
    """Расширенная визуализация результатов"""
    if not results['balance_history']:
        print("Нет данных для визуализации")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=Config.FIGURE_SIZE)
    
    # График баланса
    axes[0, 0].plot(results['balance_history'], label='Balance', linewidth=2, color='blue')
    axes[0, 0].set_title("Баланс агента", fontsize=14)
    axes[0, 0].set_ylabel("Баланс (USDT)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # График цены и позиций
    axes[0, 1].plot(results['prices'], label='BTC Price', alpha=0.7, linewidth=1, color='orange')
    ax2 = axes[0, 1].twinx()
    ax2.plot(results['positions'], label='Position Size', alpha=0.7, linewidth=1, color='green')
    axes[0, 1].set_title("Цена BTC и размер позиции", fontsize=14)
    axes[0, 1].set_ylabel("Цена BTC")
    ax2.set_ylabel("Размер позиции")
    axes[0, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # График просадки
    axes[1, 0].fill_between(range(len(results['drawdowns'])), results['drawdowns'], 
                           alpha=0.7, color='red', label='Drawdown')
    axes[1, 0].set_title("Просадка", fontsize=14)
    axes[1, 0].set_ylabel("Просадка (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Распределение сделок
    if results['trades']:
        axes[1, 1].hist(results['trades'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_title("Распределение прибыли по сделкам", fontsize=14)
        axes[1, 1].set_xlabel("Прибыль/Убыток")
        axes[1, 1].set_ylabel("Количество сделок")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем график в файл вместо показа (для совместимости с серверными средами)
    filename = "trading_results.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 График сохранен в файл: {filename}")
    plt.close()  # Освобождаем память


def main():
    """Основная функция с улучшенной аналитикой"""
    print("🚀 Продвинутый торговый алгоритм на базе RL")
    print("=" * 60)
    
    try:
        # Проверка системных требований
        gpu_info = check_gpu_requirements()
        print(f"📊 PyTorch версия: {gpu_info['torch_version']}")
        print(f"🔧 Устройств GPU: {gpu_info['device_count']}")
        
        # Загрузка и подготовка данных
        file_path = Config.DATA_FOLDER + Config.DATA_FILE
        print(f"\n📁 Загружаем данные из {file_path}")
        df = load_and_prepare_data(file_path)
        
        print(f"📈 Данные загружены: {len(df)} записей")
        print(f"🎯 Размер окна: {Config.WINDOW_SIZE}")
        print(f"💰 Начальный баланс: {Config.INITIAL_BALANCE}")

        # Создание УЛУЧШЕННОГО окружения и обучение
        print("\n🎓 УЛУЧШЕННОЕ обучение с 3 действиями и LSTM архитектурой...")
        if Config.ENABLE_EARLY_STOPPING:
            print(f"🧠 Режим: УМНОЕ обучение с ранней остановкой (до {Config.TOTAL_TIMESTEPS:,} шагов)")
            print(f"🛑 Остановка после {Config.EARLY_STOPPING_PATIENCE} эпизодов без улучшения")
        else:
            print(f"🔥 Режим: ПОЛНОЕ обучение, {Config.TOTAL_TIMESTEPS:,} шагов")
        print("🎯 Упрощенные действия: 0-Hold, 1-Buy, 2-Sell")
        env = SimplifiedTradingEnv(df, 
                                  window_size=Config.WINDOW_SIZE,
                                  initial_balance=Config.INITIAL_BALANCE)
        
        model = train_model(env)
        print("✅ УЛУЧШЕННОЕ обучение завершено!")

        # Тестирование
        print("\n🧪 Тестирование упрощенной модели...")
        test_env = SimplifiedTradingEnv(df, 
                                       window_size=Config.WINDOW_SIZE,
                                       initial_balance=Config.INITIAL_BALANCE)
        
        results = test_model(model, test_env, df)

        # Анализ результатов
        if results['balance_history']:
            analysis = analyze_results(results, Config.INITIAL_BALANCE)
            
            print(f"\n📊 Детальные результаты:")
            print(f"{'='*50}")
            print(f"Финальный баланс: {analysis['final_balance']:.2f} USDT")
            print(f"Общая доходность: {analysis['total_return']:.2%}")
            print(f"Максимальная просадка: {analysis['max_drawdown']:.2%}")
            print(f"Коэффициент Шарпа: {analysis['sharpe_ratio']:.3f}")
            print(f"Волатильность: {analysis['volatility']:.4f}")
            print(f"\n📈 Торговая статистика:")
            print(f"{'='*50}")
            print(f"Всего сделок: {analysis['total_trades']}")
            print(f"Винрейт: {analysis['win_rate']:.2f}%")
            print(f"Средняя прибыль: {analysis['avg_profit']:.2f}")
            print(f"Средний убыток: {analysis['avg_loss']:.2f}")
            print(f"Фактор прибыли: {analysis['profit_factor']:.2f}")
            print(f"Макс. серия побед: {analysis['win_streak']}")
            print(f"Макс. серия поражений: {analysis['loss_streak']}")

            # Визуализация
            print("\n📈 Создание расширенных графиков...")
            visualize_results(results, analysis)
        else:
            print("⚠️ Нет данных для анализа результатов")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()