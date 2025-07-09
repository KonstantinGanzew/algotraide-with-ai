"""
Торговое окружение для обучения RL агента

КРИТИЧЕСКИЕ ПРОБЛЕМЫ В ОРИГИНАЛЬНОМ КОДЕ:
1. Слишком агрессивные параметры риска
2. Неправильная система вознаграждений  
3. Отсутствие proper валидации данных
4. Проблемы с переобучением

ИСПРАВЛЕНИЯ В ЭТОЙ ВЕРСИИ:
1. Консервативные параметры риска
2. Упрощенная система наград на основе P&L
3. Добавлена валидация и логирование
4. Улучшенная обработка ошибок
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import logging

from ..core.config import ActiveTradingConfig as TradingConfig, ActiveRewardConfig as RewardConfig, SystemConfig


class ImprovedTradingEnv(gym.Env):
    """
    УЛУЧШЕННОЕ торговое окружение с исправленными проблемами прибыльности
    
    Основные изменения:
    1. Консервативный риск-менеджмент
    2. Простая система наград (только P&L)
    3. Лучшая валидация данных
    4. Детальное логирование
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 window_size: int = 50, 
                 initial_balance: float = 10000,
                 validation_mode: bool = False):
        super().__init__()
        
        # Основные параметры
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.validation_mode = validation_mode
        
        # Валидация данных
        self._validate_data()
        
        # Пространство действий: 0-Hold, 1-Buy, 2-Sell
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, df.shape[1]), 
            dtype=np.float32
        )
        
        # Ограничения эпизода
        self.max_episode_steps = min(10000, len(self.df) - window_size - 10)
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
        # Инициализация состояния
        self._reset_state()
    
    def _validate_data(self):
        """Валидация входных данных"""
        if len(self.df) < self.window_size + 100:
            raise ValueError(f"Недостаточно данных: {len(self.df)} < {self.window_size + 100}")
        
        required_cols = ['close', 'open', 'high', 'low', 'volume']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
        
        # Проверка на NaN
        has_nan = self.df.isnull().sum().sum() > 0
        if has_nan:
            self.logger.warning("Обнаружены NaN значения в данных")
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        
        # Проверка на аномальные значения
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in self.df.columns:
                if (self.df[col] <= 0).any():
                    raise ValueError(f"Обнаружены нулевые или отрицательные цены в {col}")
    
    def _reset_state(self):
        """Сброс состояния окружения"""
        # Торговые переменные
        self.balance = float(self.initial_balance)
        self.btc_amount = 0.0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        
        # Статистика
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_commission_paid = 0.0
        self.max_balance = float(self.initial_balance)
        self.max_drawdown = 0.0
        self.consecutive_holds = 0
        self.steps_since_trade = 0
        
        # История для анализа
        self.balance_history = [float(self.initial_balance)]
        self.price_history = []  # НОВОЕ: для анализа трендов
        self.trades = []
        self.actions_history = []
        
        # НОВЫЕ переменные для высокоприбыльной торговли
        self.steps_since_last_trade = 0    # Отслеживание времени между сделками
        self.daily_trades_count = 0        # Счетчик сделок за день
        self.last_trade_step = -1000       # Шаг последней сделки
        self.signal_confirmations = []     # История подтверждений сигналов
        self.last_trade_profit = None      # Прибыль от последней сделки
        
        # Состояние шага
        self.current_step = self.window_size
        self.episode_steps = 0
        self.prev_portfolio_value = float(self.initial_balance)
    
    @property
    def data(self):
        """Совместимость с кодом, который ожидает атрибут data"""
        return self.df
    
    def reset(self, seed=None, options=None):
        """Сброс окружения"""
        if seed is not None:
            np.random.seed(seed)
        
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Получение текущего наблюдения"""
        if self.current_step >= len(self.df):
            # Возвращаем последнее валидное наблюдение
            obs = self.df.iloc[-self.window_size:].values
        else:
            obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        
        return obs.astype(np.float32)
    
    def _get_current_price(self) -> float:
        """Получение текущей цены с валидацией"""
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.iloc[self.current_step]['close']
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Расчет общей стоимости портфеля"""
        unrealized_pnl = 0.0
        if self.btc_amount > 0 and self.entry_price > 0:
            unrealized_pnl = self.btc_amount * (current_price - self.entry_price)
        
        return self.balance + unrealized_pnl
    
    def _calculate_position_size(self, current_price: float) -> float:
        """УМНАЯ система управления позициями с анализом тренда и уверенности"""
        available_balance = self.balance
        
        # 1. БАЗОВЫЙ РАЗМЕР ПОЗИЦИИ
        base_risk = TradingConfig.BASE_RISK_PER_TRADE
        
        # 2. ДИНАМИЧЕСКАЯ КОРРЕКТИРОВКА НА ОСНОВЕ ПРОИЗВОДИТЕЛЬНОСТИ
        performance_multiplier = 1.0
        if hasattr(self, 'profitable_trades') and hasattr(self, 'total_trades') and self.total_trades > 5:
            win_rate = self.profitable_trades / self.total_trades
            
            if win_rate > 0.65:  # Очень хорошая производительность
                performance_multiplier = 1.4
            elif win_rate > 0.55:  # Хорошая производительность
                performance_multiplier = 1.2
            elif win_rate < 0.35:  # Плохая производительность
                performance_multiplier = 0.6
            elif win_rate < 0.45:  # Ниже среднего
                performance_multiplier = 0.8
        
        # 3. АНАЛИЗ ТРЕНДА ДЛЯ КОРРЕКТИРОВКИ РАЗМЕРА
        trend_multiplier = 1.0
        if hasattr(self, 'price_history') and len(self.price_history) >= 20:
            short_trend = (self.price_history[-5:] if len(self.price_history) >= 5 else self.price_history)
            long_trend = self.price_history[-20:]
            
            short_change = (short_trend[-1] - short_trend[0]) / short_trend[0]
            long_change = (long_trend[-1] - long_trend[0]) / long_trend[0]
            
            # Сильный восходящий тренд
            if short_change > 0.01 and long_change > 0.02:
                trend_multiplier = 1.3
            # Умеренный восходящий тренд
            elif short_change > 0.005 and long_change > 0.01:
                trend_multiplier = 1.1
            # Нисходящий тренд - осторожность
            elif short_change < -0.01 or long_change < -0.02:
                trend_multiplier = 0.7
        
        # 4. УПРАВЛЕНИЕ РИСКАМИ НА ОСНОВЕ ПРОСАДКИ
        drawdown_multiplier = 1.0
        current_portfolio = self._calculate_portfolio_value(current_price)
        if hasattr(self, 'max_balance') and self.max_balance > 0:
            current_drawdown = (self.max_balance - current_portfolio) / self.max_balance
            
            if current_drawdown > 0.08:  # Значительная просадка
                drawdown_multiplier = 0.5
            elif current_drawdown > 0.05:  # Умеренная просадка
                drawdown_multiplier = 0.7
        
        # 5. КОРРЕКТИРОВКА НА ОСНОВЕ ПОСЛЕДНИХ СДЕЛОК
        recent_trades_multiplier = 1.0
        if hasattr(self, 'trades') and len(self.trades) >= 3:
            recent_trades = self.trades[-3:]
            if all(trade > 0 for trade in recent_trades):  # 3 прибыльные подряд
                recent_trades_multiplier = 1.2
            elif all(trade < 0 for trade in recent_trades):  # 3 убыточные подряд
                recent_trades_multiplier = 0.6
        
        # 6. РАСЧЕТ ИТОГОВОГО РАЗМЕРА
        adjusted_risk = base_risk * performance_multiplier * trend_multiplier * drawdown_multiplier * recent_trades_multiplier
        
        # Ограничения безопасности
        adjusted_risk = max(adjusted_risk, 0.005)  # Минимум 0.5%
        adjusted_risk = min(adjusted_risk, 0.05)   # Максимум 5%
        
        risk_amount = available_balance * adjusted_risk
        
        # Проверяем минимальные и максимальные границы
        min_size = available_balance * TradingConfig.MIN_POSITION_SIZE
        max_size = available_balance * TradingConfig.MAX_POSITION_MULTIPLIER
        
        position_value = max(min_size, min(risk_amount, max_size))
        
        # Учет комиссий (универсальный доступ)
        commission_rate = getattr(TradingConfig, 'COMMISSION_RATE', 
                                getattr(TradingConfig, 'TRADE_COMMISSION', 0.001))
        commission = position_value * commission_rate
        if position_value + commission > available_balance:
            position_value = available_balance - commission
        
        final_position = max(0, position_value)
        
        # Логирование для диагностики (реже)
        if hasattr(self, '_debug_step_count') and self._debug_step_count % 200 == 0:
            self.logger.info(f"📊 УМНЫЙ РАСЧЕТ РАЗМЕРА ПОЗИЦИИ:")
            self.logger.info(f"   Базовый риск: {base_risk:.1%}")
            self.logger.info(f"   Производительность: x{performance_multiplier:.2f}")
            self.logger.info(f"   Тренд: x{trend_multiplier:.2f}")
            self.logger.info(f"   Просадка: x{drawdown_multiplier:.2f}")
            self.logger.info(f"   Последние сделки: x{recent_trades_multiplier:.2f}")
            self.logger.info(f"   Итоговый риск: {adjusted_risk:.2%}")
            self.logger.info(f"   Размер позиции: ${final_position:.2f}")
        
        return final_position
    
    def _execute_trade(self, action: int, current_price: float) -> Dict[str, Any]:
        """ИСПРАВЛЕННОЕ выполнение торговых операций с детальной диагностикой"""
        trade_info = {'action': 'HOLD', 'executed': False}
        
        # Логирование для диагностики
        if hasattr(self, '_debug_step_count'):
            self._debug_step_count += 1
        else:
            self._debug_step_count = 1
        
        # Детальная диагностика каждые 100 шагов
        if self._debug_step_count % 100 == 0:
            self.logger.info(f"🔍 [Шаг {self._debug_step_count}] Диагностика торговли:")
            self.logger.info(f"   Действие: {action} ({'Hold' if action == 0 else 'Buy' if action == 1 else 'Sell'})")
            self.logger.info(f"   Цена: ${current_price:.2f}")
            self.logger.info(f"   Баланс: ${self.balance:.2f}")
            self.logger.info(f"   BTC: {self.btc_amount:.6f}")
            self.logger.info(f"   В позиции: {'Да' if self.btc_amount > 0 else 'Нет'}")
        
        if action == 0:  # Hold
            return trade_info
        
        elif action == 1:  # Buy
            if self.btc_amount == 0:  # Покупаем только если нет позиции
                position_value = self._calculate_position_size(current_price)
                commission_rate = getattr(TradingConfig, 'COMMISSION_RATE', 
                                        getattr(TradingConfig, 'TRADE_COMMISSION', 0.001))
                commission = position_value * commission_rate
                total_cost = position_value + commission
                
                # Детальная диагностика покупки
                if self._debug_step_count % 100 == 0 or position_value == 0:
                    self.logger.info(f"   🛒 ПОПЫТКА ПОКУПКИ:")
                    self.logger.info(f"     Размер позиции: ${position_value:.2f}")
                    self.logger.info(f"     Комиссия: ${commission:.2f}")
                    self.logger.info(f"     Общая стоимость: ${total_cost:.2f}")
                    self.logger.info(f"     Доступно средств: ${self.balance:.2f}")
                    self.logger.info(f"     Достаточно средств: {'Да' if total_cost <= self.balance else 'НЕТ'}")
                    self.logger.info(f"     Позиция > 0: {'Да' if position_value > 0 else 'НЕТ'}")
                
                if total_cost <= self.balance and position_value > 0:
                    # Выполняем покупку
                    self.btc_amount = position_value / current_price
                    self.entry_price = current_price
                    self.balance -= total_cost
                    self.total_commission_paid += commission
                    
                    # Устанавливаем стоп-лосс и тейк-профит
                    self.stop_loss_price = current_price * (1 - TradingConfig.STOP_LOSS_PERCENTAGE)
                    self.take_profit_price = current_price * (1 + TradingConfig.TAKE_PROFIT_PERCENTAGE)
                    
                    # Записываем сделку
                    self.total_trades += 1
                    self.daily_trades_count += 1  # НОВОЕ: счетчик дневных сделок
                    self.steps_since_last_trade = 0  # НОВОЕ: сброс счетчика времени
                    self.last_trade_step = self.episode_steps  # НОВОЕ: запоминаем шаг сделки
                    
                    # НОВОЕ: Записываем в историю для адаптивной системы
                    if not hasattr(self, 'trade_steps_history'):
                        self.trade_steps_history = []
                    self.trade_steps_history.append(self.episode_steps)
                    
                    trade_info = {
                        'action': 'BUY',
                        'executed': True,
                        'amount': self.btc_amount,
                        'price': current_price,
                        'cost': total_cost,
                        'commission': commission
                    }
                    
                    self.logger.info(f"✅ ПОКУПКА ВЫПОЛНЕНА: {self.btc_amount:.6f} BTC за ${total_cost:.2f}")
                else:
                    # Логируем причину отклонения
                    reasons = []
                    if total_cost > self.balance:
                        reasons.append(f"недостаточно средств ({total_cost:.2f} > {self.balance:.2f})")
                    if position_value <= 0:
                        reasons.append("размер позиции <= 0")
                    
                    if self._debug_step_count % 100 == 0:
                        self.logger.warning(f"❌ ПОКУПКА ОТКЛОНЕНА: {', '.join(reasons)}")
            else:
                if self._debug_step_count % 100 == 0:
                    self.logger.info(f"   ⚠️ Покупка пропущена: уже в позиции ({self.btc_amount:.6f} BTC)")
        
        elif action == 2:  # Sell
            if self.btc_amount > 0:  # Продаем только если есть позиция
                gross_proceeds = self.btc_amount * current_price
                commission_rate = getattr(TradingConfig, 'COMMISSION_RATE', 
                                        getattr(TradingConfig, 'TRADE_COMMISSION', 0.001))
                commission = gross_proceeds * commission_rate
                net_proceeds = gross_proceeds - commission
                
                # Рассчитываем прибыль/убыток
                profit = gross_proceeds - (self.btc_amount * self.entry_price)
                profit_percentage = (profit / (self.btc_amount * self.entry_price)) * 100 if self.entry_price > 0 else 0
                
                # Детальная диагностика продажи
                if self._debug_step_count % 100 == 0:
                    self.logger.info(f"   💰 ПОПЫТКА ПРОДАЖИ:")
                    self.logger.info(f"     Количество: {self.btc_amount:.6f} BTC")
                    self.logger.info(f"     Цена входа: ${self.entry_price:.2f}")
                    self.logger.info(f"     Текущая цена: ${current_price:.2f}")
                    self.logger.info(f"     Валовая выручка: ${gross_proceeds:.2f}")
                    self.logger.info(f"     Комиссия: ${commission:.2f}")
                    self.logger.info(f"     Чистая выручка: ${net_proceeds:.2f}")
                    self.logger.info(f"     Прибыль: ${profit:.2f} ({profit_percentage:+.1f}%)")
                
                # Выполняем продажу
                self.balance += net_proceeds
                self.total_commission_paid += commission
                self.trades.append(profit)
                self.last_trade_profit = profit
                self.daily_trades_count += 1  # НОВОЕ: счетчик дневных сделок
                self.steps_since_last_trade = 0  # НОВОЕ: сброс счетчика времени
                self.last_trade_step = self.episode_steps  # НОВОЕ: запоминаем шаг сделки
                
                # НОВОЕ: Записываем в историю для адаптивной системы
                if not hasattr(self, 'trade_steps_history'):
                    self.trade_steps_history = []
                if not hasattr(self, 'recent_trades_profit'):
                    self.recent_trades_profit = []
                    
                self.trade_steps_history.append(self.episode_steps)
                self.recent_trades_profit.append(profit)
                
                if profit > 0:
                    self.profitable_trades += 1
                
                trade_info = {
                    'action': 'SELL',
                    'executed': True,
                    'amount': self.btc_amount,
                    'price': current_price,
                    'proceeds': net_proceeds,
                    'profit': profit,
                    'commission': commission
                }
                
                self.logger.info(f"✅ ПРОДАЖА ВЫПОЛНЕНА: {self.btc_amount:.6f} BTC за ${net_proceeds:.2f} (P&L: {profit:+.2f})")
                
                # Сбрасываем позицию
                self.btc_amount = 0.0
                self.entry_price = 0.0
                self.stop_loss_price = 0.0
                self.take_profit_price = 0.0
            else:
                if self._debug_step_count % 100 == 0:
                    self.logger.info(f"   ⚠️ Продажа пропущена: нет позиции")
        
        return trade_info
    
    def _check_stop_loss_take_profit(self, current_price: float) -> Dict[str, Any]:
        """Проверка автоматических стоп-лоссов и тейк-профитов"""
        if self.btc_amount <= 0:
            return {'action': 'NONE', 'executed': False}
        
        # Проверка стоп-лосса
        if current_price <= self.stop_loss_price:
            return self._execute_trade(2, current_price)  # Принудительная продажа
        
        # Проверка тейк-профита
        if current_price >= self.take_profit_price:
            return self._execute_trade(2, current_price)  # Принудительная продажа
        
        return {'action': 'NONE', 'executed': False}
    
    def _calculate_reward(self, current_price: float) -> float:
        """УЛУЧШЕННАЯ система наград для повышения прибыльности"""
        current_portfolio = self._calculate_portfolio_value(current_price)
        
        # 1. БАЗОВАЯ НАГРАДА - изменение портфеля (основа)
        portfolio_change = (current_portfolio - self.prev_portfolio_value) / self.prev_portfolio_value
        base_reward = portfolio_change * RewardConfig.BALANCE_CHANGE_MULTIPLIER
        
        # 2. НАГРАДА ЗА КАЧЕСТВЕННЫЕ СДЕЛКИ
        trade_reward = 0.0
        if hasattr(self, 'last_trade_profit') and self.last_trade_profit is not None:
            profit_percent = self.last_trade_profit / self.initial_balance
            
            if self.last_trade_profit > 0:
                # Градация бонусов за прибыльные сделки
                if profit_percent > 0.03:  # >3% прибыль
                    trade_reward = getattr(RewardConfig, 'EXCELLENT_TRADE_BONUS', RewardConfig.HIGH_PROFIT_BONUS)
                elif profit_percent > 0.02:  # >2% прибыль
                    trade_reward = RewardConfig.HIGH_PROFIT_BONUS
                else:
                    trade_reward = RewardConfig.PROFITABLE_TRADE_BONUS
            else:
                # Штраф за убыточные сделки (с защитой от маленьких потерь)
                if abs(profit_percent) < 0.005:  # <0.5% потеря
                    trade_reward = RewardConfig.SMALL_LOSS_PROTECTION
                else:
                    trade_reward = RewardConfig.LOSING_TRADE_PENALTY
            
            self.last_trade_profit = None
        
        # 3. АНАЛИЗ ТРЕНДА И НАГРАДА ЗА УМНУЮ ТОРГОВЛЮ
        trend_reward = 0.0
        if len(self.price_history) >= 10:
            recent_prices = self.price_history[-10:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            last_action = self.actions_history[-1] if self.actions_history else 0
            
            # Награда за торговлю по тренду
            if last_action == 1 and price_trend > 0.001:  # Покупка на росте
                trend_reward = RewardConfig.TREND_FOLLOWING_BONUS
            elif last_action == 2 and price_trend < -0.001:  # Продажа на падении
                trend_reward = RewardConfig.TREND_FOLLOWING_BONUS
            elif last_action != 0:  # Торговля против тренда
                trend_reward = RewardConfig.COUNTER_TREND_PENALTY
        
        # 4. УМНАЯ НАГРАДА ЗА АКТИВНОСТЬ
        activity_reward = 0.0
        last_action = self.actions_history[-1] if self.actions_history else 0
        
        if last_action != 0:  # Не hold
            activity_reward = RewardConfig.ACTION_REWARD
            self.consecutive_holds = 0
            self.steps_since_trade = 0
            
            # Проверка на избыточную торговлю
            if len(self.actions_history) >= 20:
                recent_trades = sum(1 for a in self.actions_history[-20:] if a != 0)
                if recent_trades > 15:  # Слишком много торгов
                    activity_reward += RewardConfig.OVERTRADING_PENALTY
        else:
            self.consecutive_holds += 1
            self.steps_since_trade += 1
        
        # 5. НОВАЯ ЛОГИКА: Награды за селективность и терпение
        inactivity_penalty = 0.0
        selective_trading_bonus = 0.0
        patience_bonus = 0.0
        
        # В высокоприбыльном режиме НЕ штрафуем за бездействие
        if getattr(RewardConfig, 'INACTIVITY_PENALTY', 0) > 0:
            # Мягкий штраф за бездействие (только если включен)
            if len(self.actions_history) >= 10:
                recent_holds = sum(1 for a in self.actions_history[-10:] if a == 0)
                hold_ratio = recent_holds / 10
                inactivity_penalty = RewardConfig.INACTIVITY_PENALTY * hold_ratio
        
        # Бонус за селективную торговлю (редкие качественные сделки)
        if hasattr(self, 'steps_since_last_trade') and self.steps_since_last_trade > 30:
            selective_bonus = getattr(RewardConfig, 'SELECTIVE_TRADING_BONUS', 0)
            if selective_bonus > 0 and last_action != 0:  # Торгуем после долгого ожидания
                selective_trading_bonus = selective_bonus
        
        # Бонус за терпение (долгое ожидание качественного сигнала)
        patience_bonus_value = getattr(RewardConfig, 'PATIENCE_BONUS', 0)
        if patience_bonus_value > 0 and self.consecutive_holds > 20:
            patience_bonus = patience_bonus_value * min(self.consecutive_holds / 50, 1.0)
        
        # Принуждение к торговле (только если включено)
        force_trade_penalty = 0.0
        long_inactivity_penalty = getattr(RewardConfig, 'LONG_INACTIVITY_PENALTY', 0)
        if long_inactivity_penalty > 0 and self.steps_since_trade > TradingConfig.FORCE_TRADE_EVERY_N_STEPS:
            force_trade_penalty = long_inactivity_penalty
        
        # 6. УПРАВЛЕНИЕ РИСКАМИ
        # Отслеживание просадки
        self.max_balance = max(self.max_balance, current_portfolio)
        drawdown = (self.max_balance - current_portfolio) / self.max_balance if self.max_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Штраф за просадку (прогрессивный)
        drawdown_penalty = 0
        # Используем новое или старое название для лимита просадки
        max_drawdown_limit = getattr(TradingConfig, 'MAX_DRAWDOWN_STOP', 
                                   getattr(TradingConfig, 'MAX_DRAWDOWN_LIMIT', 0.15))
        if drawdown > max_drawdown_limit:
            excess_drawdown = drawdown - max_drawdown_limit
            drawdown_penalty = excess_drawdown * RewardConfig.DRAWDOWN_PENALTY_MULTIPLIER
        
        # 7. БОНУС ЗА ДИВЕРСИФИКАЦИЮ
        diversification_bonus = 0.0
        if current_portfolio > 0:
            portfolio_cash_ratio = self.balance / current_portfolio
            # Оптимальная диверсификация: 20-80% в кеше
            if 0.2 <= portfolio_cash_ratio <= 0.8:
                diversification_bonus = 0.2
        
        # 8. БОНУС ЗА СТАБИЛЬНОСТЬ
        stability_bonus = 0.0
        if len(self.balance_history) >= 50:
            recent_balances = self.balance_history[-50:]
            volatility = np.std(recent_balances) / np.mean(recent_balances)
            if volatility < 0.05:  # Низкая волатильность
                stability_bonus = 0.5
        
        # ИТОГОВАЯ НАГРАДА
        total_reward = (
            base_reward +           # Основа: изменение портфеля
            trade_reward +          # Качество сделок
            trend_reward +          # Умная торговля по тренду
            activity_reward +       # Активность
            diversification_bonus + # Диверсификация
            stability_bonus +       # Стабильность
            selective_trading_bonus + # НОВОЕ: За селективную торговлю
            patience_bonus +        # НОВОЕ: За терпение
            inactivity_penalty +    # Штрафы за бездействие (если включены)
            force_trade_penalty -   # Принуждение к торговле (если включено)
            drawdown_penalty        # Управление рисками
        )
        
        # Обновляем состояние
        self.prev_portfolio_value = current_portfolio
        if not hasattr(self, 'price_history'):
            self.price_history = []
        self.price_history.append(current_price)
        
        # Ограничиваем историю цен
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-50:]
        
        return total_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Выполнение одного шага в окружении с принуждением к торговле"""
        if self.current_step >= len(self.df):
            # Эпизод завершен
            obs = self._get_observation()
            return obs, 0.0, True, True, {'reason': 'data_end'}

        current_price = self._get_current_price()
        original_action = action
        
        # НОВОЕ: Интеллектуальное исправление действий
        action = self._smart_action_correction(action, current_price)
        
        # НОВОЕ: Проверка требований качественной торговли
        can_trade, quality_reason = self._check_trade_quality_requirements(action)
        
        # Проверяем автоматические стоп-лоссы/тейк-профиты
        sl_tp_info = self._check_stop_loss_take_profit(current_price)
        
        # Выполняем действие пользователя (если не сработал SL/TP и проходит проверки качества)
        if not sl_tp_info['executed']:
            if can_trade:
                trade_info = self._execute_trade(action, current_price)
            else:
                # Принудительно Hold если не прошли проверки качества
                trade_info = self._execute_trade(0, current_price)
                if hasattr(self, '_debug_step_count') and self._debug_step_count % 100 == 0:
                    self.logger.info(f"   🚫 Торговля заблокирована: {quality_reason}")
        else:
            trade_info = sl_tp_info
        
        # Рассчитываем награду
        reward = self._calculate_reward(current_price)
        
        # Обновляем историю с оригинальным действием
        self.actions_history.append(original_action)
        portfolio_value = self._calculate_portfolio_value(current_price)
        self.balance_history.append(portfolio_value)
        
        # Переходим к следующему шагу
        self.current_step += 1
        self.episode_steps += 1
        self.steps_since_last_trade += 1  # НОВОЕ: отслеживание времени с последней сделки
        
        # НОВОЕ: Мониторинг торговой активности для адаптации
        self._monitor_trading_activity()
        
        # Проверяем условия завершения
        done = False
        truncated = False
        
        # НОВОЕ: Завершение эпизода раньше для частых сбросов
        max_episode_length = min(5000, len(self.df) // 4)  # Короче эпизоды
        
        # Завершение по достижению конца данных
        if self.current_step >= len(self.df) - 1:
            done = True
            
        # Завершение по большим потерям
        if portfolio_value <= 0.5 * self.initial_balance:  # Потеря 50% вместо 70%
            done = True
            
        # Завершение по лимиту шагов (сокращено)
        if self.episode_steps >= max_episode_length:
            truncated = True
            
        # НОВОЕ: Завершение если недостаточно торгов за эпизод
        if (self.episode_steps > 1000 and 
            self.total_trades < TradingConfig.MIN_TRADES_PER_EPISODE):
            done = True  # Принуждаем к новому эпизоду
        
        # Информация для анализа
        info = {
            'balance': self.balance,
            'btc_amount': self.btc_amount,
            'portfolio_value': portfolio_value,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'max_drawdown': self.max_drawdown,
            'commission_paid': self.total_commission_paid,
            'trade_info': trade_info,
            'current_price': current_price,
            'win_rate': (self.profitable_trades / max(1, self.total_trades)) * 100,
            'consecutive_holds': self.consecutive_holds,
            'forced_action': action != original_action,
            'action_corrected': action != original_action
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _smart_action_correction(self, action: int, current_price: float) -> int:
        """НОВАЯ ФУНКЦИЯ: Интеллектуальное исправление действий модели"""
        
        # Отслеживаем попытки продать без позиции
        if not hasattr(self, '_sell_attempts_without_position'):
            self._sell_attempts_without_position = 0
            self._buy_attempts_with_position = 0
            self._consecutive_sell_attempts = 0
            self._action_correction_count = 0
        
        # Логика исправления
        corrected_action = action
        correction_reason = None
        
        # 1. Если модель пытается продать без позиции
        if action == 2 and self.btc_amount == 0:
            self._sell_attempts_without_position += 1
            self._consecutive_sell_attempts += 1
            
            # После 5 попыток продать без позиции - принуждаем к покупке
            if self._consecutive_sell_attempts >= 5:
                corrected_action = 1  # Принуждаем к покупке
                correction_reason = f"Принуждение к покупке после {self._consecutive_sell_attempts} попыток продать"
                self._consecutive_sell_attempts = 0
                self._action_correction_count += 1
        
        # 2. Если модель пытается купить уже имея позицию  
        elif action == 1 and self.btc_amount > 0:
            self._buy_attempts_with_position += 1
            # Можно оставить как есть или принудить к продаже
            corrected_action = 2  # Принуждаем к продаже
            correction_reason = "Принуждение к продаже - уже есть позиция"
        
        # 3. Сброс счетчиков при правильных действиях
        else:
            if action == 1 and self.btc_amount == 0:  # Правильная покупка
                self._consecutive_sell_attempts = 0
            elif action == 2 and self.btc_amount > 0:  # Правильная продажа
                self._consecutive_sell_attempts = 0
        
        # Логирование исправлений
        if corrected_action != action and hasattr(self, '_debug_step_count'):
            if self._debug_step_count % 100 == 0 or self._action_correction_count <= 10:
                self.logger.warning(f"🔧 ИСПРАВЛЕНИЕ ДЕЙСТВИЯ [Шаг {self._debug_step_count}]:")
                self.logger.warning(f"   Оригинальное: {action} ({'Hold' if action == 0 else 'Buy' if action == 1 else 'Sell'})")
                self.logger.warning(f"   Исправленное: {corrected_action} ({'Hold' if corrected_action == 0 else 'Buy' if corrected_action == 1 else 'Sell'})")
                self.logger.warning(f"   Причина: {correction_reason}")
                self.logger.warning(f"   Позиция: {self.btc_amount:.6f} BTC")
                self.logger.warning(f"   Всего исправлений: {self._action_correction_count}")
        
        return corrected_action
    
    def _check_trade_quality_requirements(self, action: int) -> Tuple[bool, str]:
        """ОБНОВЛЕННАЯ ФУНКЦИЯ: Проверка требований качественной торговли с адаптивными порогами"""
        
        # Пропускаем проверки для hold
        if action == 0:
            return True, "Hold действие разрешено"
        
        # НОВОЕ: Получаем адаптивные пороги
        adaptive_confidence, adaptive_signal_strength = self._get_adaptive_thresholds()
        
        # 1. Проверка минимального времени между сделками (динамические лимиты)
        if hasattr(self, 'dynamic_limits') and 'min_time_between_trades' in self.dynamic_limits:
            min_time = self.dynamic_limits['min_time_between_trades']
        else:
            # Используем базовые или статичные лимиты
            min_time = getattr(TradingConfig, 'BASE_MIN_TIME_BETWEEN_TRADES', 
                             getattr(TradingConfig, 'MIN_TIME_BETWEEN_TRADES', 25))
        
        if hasattr(self, 'steps_since_last_trade') and self.steps_since_last_trade < min_time:
            return False, f"Слишком рано для торговли (прошло {self.steps_since_last_trade}/{min_time} шагов)"
        
        # 2. Проверка лимита сделок в день (динамические лимиты)
        if hasattr(self, 'dynamic_limits') and 'max_daily_trades' in self.dynamic_limits:
            max_daily = self.dynamic_limits['max_daily_trades']
        else:
            # Используем базовые или статичные лимиты
            max_daily = getattr(TradingConfig, 'BASE_MAX_DAILY_TRADES', 
                              getattr(TradingConfig, 'MAX_DAILY_TRADES', 15))
        
        if hasattr(self, 'daily_trades_count'):
            # Сброс счетчика каждые 1440 шагов (условный "день")
            if self.episode_steps % 1440 == 0:
                self.daily_trades_count = 0
            
            if self.daily_trades_count >= max_daily:
                return False, f"Достигнут лимит сделок в день ({self.daily_trades_count}/{max_daily})"
        
        # 3. НОВОЕ: Адаптивная проверка силы сигнала
        signal_strength = self._calculate_signal_strength()
        if signal_strength < adaptive_signal_strength:
            return False, f"Слабый сигнал ({signal_strength:.2f}/{adaptive_signal_strength:.2f})"
        
        # 4. НОВОЕ: Проверка адаптивной уверенности модели
        model_confidence = getattr(self, 'last_model_confidence', None)
        if model_confidence is not None and model_confidence < adaptive_confidence:
            return False, f"Низкая уверенность модели ({model_confidence:.2f}/{adaptive_confidence:.2f})"
        
        return True, "Все проверки качества пройдены"

    def _get_adaptive_thresholds(self) -> Tuple[float, float]:
        """ОБНОВЛЕННАЯ ФУНКЦИЯ: Динамические пороги с самооптимизацией"""
        
        # Получаем базовые значения
        base_confidence = getattr(TradingConfig, 'BASE_CONFIDENCE_THRESHOLD', 
                                 getattr(TradingConfig, 'CONFIDENCE_THRESHOLD', 0.5))
        base_signal_strength = getattr(TradingConfig, 'BASE_MIN_SIGNAL_STRENGTH', 
                                     getattr(TradingConfig, 'MIN_SIGNAL_STRENGTH', 0.4))
        
        # НОВОЕ: Динамическая оптимизация (только если включена)
        if getattr(TradingConfig, 'USE_DYNAMIC_OPTIMIZATION', False):
            confidence, signal = self._calculate_dynamic_thresholds(base_confidence, base_signal_strength)
        else:
            # Оригинальная логика адаптивных порогов
            confidence, signal = self._calculate_adaptive_thresholds(base_confidence, base_signal_strength)
        
        return confidence, signal
    
    def _calculate_dynamic_thresholds(self, base_confidence: float, base_signal: float) -> Tuple[float, float]:
        """НОВОЕ: Динамическое вычисление порогов на основе производительности"""
        
        # Если недостаточно данных для адаптации
        min_trades = getattr(TradingConfig, 'MIN_TRADES_FOR_ADAPTATION', 5)
        if len(getattr(self, 'recent_trades_profit', [])) < min_trades:
            return base_confidence, base_signal
        
        # Анализируем производительность
        performance_score = self._calculate_performance_score()
        
        # Адаптируем на основе производительности
        adaptation_rate = getattr(TradingConfig, 'CONFIDENCE_ADAPTATION_RATE', 0.1)
        
        # Динамически корректируем пороги
        if performance_score > 0.7:  # Хорошая производительность - расслабляем фильтры
            confidence_adj = -adaptation_rate * 0.5
            signal_adj = -adaptation_rate * 0.3
        elif performance_score < 0.3:  # Плохая производительность - ужесточаем фильтры  
            confidence_adj = adaptation_rate * 0.8
            signal_adj = adaptation_rate * 0.5
        else:  # Средняя производительность - небольшие корректировки
            confidence_adj = (0.5 - performance_score) * adaptation_rate * 0.3
            signal_adj = (0.5 - performance_score) * adaptation_rate * 0.2
        
        # Применяем ограничения
        new_confidence = max(0.1, min(0.8, base_confidence + confidence_adj))
        new_signal = max(0.1, min(0.9, base_signal + signal_adj))
        
        # Сохраняем для логирования
        if not hasattr(self, 'adaptive_history'):
            self.adaptive_history = []
        self.adaptive_history.append({
            'step': self.episode_steps,
            'performance': performance_score,
            'confidence': new_confidence,
            'signal': new_signal
        })
        
        return new_confidence, new_signal
    
    def _calculate_adaptive_thresholds(self, base_confidence: float, base_signal: float) -> Tuple[float, float]:
        """Оригинальная логика адаптивных порогов"""
        
        # Подсчитываем активность в окне
        activity_window = getattr(TradingConfig, 'ACTIVITY_CHECK_WINDOW', 100)
        
        if hasattr(self, 'trade_steps_history') and len(self.trade_steps_history) > 0:
            recent_trades = [step for step in self.trade_steps_history 
                           if self.episode_steps - step <= activity_window]
            trades_in_window = len(recent_trades)
        else:
            trades_in_window = 0
        
        min_trades_in_window = getattr(TradingConfig, 'MIN_TRADES_IN_WINDOW', 3)
        
        # Адаптируем пороги на основе активности
        if trades_in_window < min_trades_in_window:
            # Низкая активность - снижаем пороги
            boost_factor = getattr(TradingConfig, 'ACTIVITY_BOOST_FACTOR', 0.9)
            adaptive_confidence = base_confidence * boost_factor
            adaptive_signal_strength = base_signal * boost_factor
        else:
            # Нормальная активность - используем базовые пороги
            adaptive_confidence = base_confidence
            adaptive_signal_strength = base_signal
        
        # Применяем ограничения
        min_conf = getattr(TradingConfig, 'ADAPTIVE_CONFIDENCE_MIN', 0.2)
        max_conf = getattr(TradingConfig, 'ADAPTIVE_CONFIDENCE_MAX', 0.8)
        
        adaptive_confidence = max(min_conf, min(max_conf, adaptive_confidence))
        adaptive_signal_strength = max(0.1, min(0.9, adaptive_signal_strength))
        
        return adaptive_confidence, adaptive_signal_strength
    
    def _calculate_performance_score(self) -> float:
        """НОВОЕ: Вычисление общего балла производительности (0-1)"""
        
        # Веса компонентов
        profitability_weight = getattr(TradingConfig, 'PROFITABILITY_WEIGHT', 0.4)
        activity_weight = getattr(TradingConfig, 'ACTIVITY_WEIGHT', 0.3)
        risk_weight = getattr(TradingConfig, 'RISK_WEIGHT', 0.3)
        
        # 1. Прибыльность (0-1)
        profitability_score = self._calculate_profitability_score()
        
        # 2. Активность (0-1)  
        activity_score = self._calculate_activity_score()
        
        # 3. Управление рисками (0-1)
        risk_score = self._calculate_risk_score()
        
        # Взвешенный общий балл
        total_score = (profitability_score * profitability_weight + 
                      activity_score * activity_weight + 
                      risk_score * risk_weight)
        
        return max(0.0, min(1.0, total_score))
    
    def _calculate_profitability_score(self) -> float:
        """Балл прибыльности"""
        if not hasattr(self, 'recent_trades_profit') or len(self.recent_trades_profit) == 0:
            return 0.5  # Нейтральный балл без данных
        
        # Анализируем недавние сделки
        window = getattr(TradingConfig, 'PERFORMANCE_WINDOW', 100)
        recent_profits = self.recent_trades_profit[-window:]
        
        if len(recent_profits) == 0:
            return 0.5
        
        # Винрейт
        profitable_trades = sum(1 for p in recent_profits if p > 0)
        win_rate = profitable_trades / len(recent_profits) if recent_profits else 0
        
        # Средняя прибыль
        avg_profit = sum(recent_profits) / len(recent_profits) if recent_profits else 0
        avg_profit_percent = avg_profit / self.initial_balance
        
        # Нормализуем к 0-1
        win_rate_score = win_rate  # Уже в диапазоне 0-1
        profit_score = max(0, min(1, 0.5 + avg_profit_percent * 10))  # ±10% = ±0.5 балла
        
        return (win_rate_score + profit_score) / 2
    
    def _calculate_activity_score(self) -> float:
        """Балл торговой активности"""
        
        # Подсчитываем сделки в окне производительности
        window = getattr(TradingConfig, 'PERFORMANCE_WINDOW', 100)
        recent_steps = max(1, min(window, self.episode_steps))
        
        if hasattr(self, 'trade_steps_history'):
            recent_trades = [step for step in self.trade_steps_history 
                           if self.episode_steps - step <= window]
            trades_count = len(recent_trades)
        else:
            trades_count = 0
        
        # Целевая активность (например, 1 сделка на 20 шагов)
        target_trades = recent_steps / 20
        activity_ratio = trades_count / max(1, target_trades)
        
        # Оптимальная активность около 1.0
        if activity_ratio < 0.5:
            return activity_ratio * 2 * 0.7  # Штраф за низкую активность
        elif activity_ratio > 2.0:
            return max(0.3, 1.0 - (activity_ratio - 2.0) * 0.2)  # Штраф за гиперактивность
        else:
            return 1.0  # Оптимальная активность
    
    def _calculate_risk_score(self) -> float:
        """Балл управления рисками"""
        
        # Текущая просадка
        current_portfolio = self._calculate_portfolio_value(self._get_current_price())
        drawdown = (self.max_balance - current_portfolio) / self.max_balance if self.max_balance > 0 else 0
        
        # Максимально допустимая просадка
        max_allowed_dd = getattr(TradingConfig, 'MAX_DRAWDOWN_STOP', 0.15)
        
        # Балл просадки (чем меньше просадка, тем выше балл)
        if drawdown <= max_allowed_dd * 0.5:
            drawdown_score = 1.0  # Отличное управление рисками
        elif drawdown <= max_allowed_dd:
            drawdown_score = 0.7  # Приемлемое управление рисками
        else:
            drawdown_score = max(0.0, 0.3 - (drawdown - max_allowed_dd) * 2)  # Плохое управление
        
        # Стабильность баланса
        if len(self.balance_history) >= 20:
            recent_balances = self.balance_history[-20:]
            balance_volatility = float(np.std(recent_balances)) / float(np.mean(recent_balances)) if np.mean(recent_balances) > 0 else 1.0
            volatility_score = max(0.0, 1.0 - balance_volatility * 10)  # Чем меньше волатильность, тем лучше
        else:
            volatility_score = 0.5  # Нейтрально
        
        return (drawdown_score + volatility_score) / 2

    def _monitor_trading_activity(self):
        """ОБНОВЛЕННАЯ ФУНКЦИЯ: Мониторинг торговой активности для динамической адаптации"""
        
        # Инициализация истории если нужно
        if not hasattr(self, 'trade_steps_history'):
            self.trade_steps_history = []
        if not hasattr(self, 'recent_trades_profit'):
            self.recent_trades_profit = []
        
        # Очищаем старую историю (оставляем только последние 500 шагов)
        cutoff_step = self.episode_steps - 500
        self.trade_steps_history = [step for step in self.trade_steps_history if step > cutoff_step]
        
        # Ограничиваем размер истории прибыли
        if len(self.recent_trades_profit) > 20:
            self.recent_trades_profit = self.recent_trades_profit[-20:]
        
        # НОВОЕ: Динамическая адаптация лимитов (только если включена)
        if getattr(TradingConfig, 'ADAPTIVE_LIMIT_ADJUSTMENT', False):
            self._adjust_dynamic_limits()
        
        # Логирование каждые 100 шагов для диагностики
        if self.episode_steps % 100 == 0 and hasattr(self, 'adaptive_history'):
            if self.adaptive_history:
                last_perf = self.adaptive_history[-1]
                self.logger.debug(f"📊 Динамическая адаптация [Шаг {self.episode_steps}]: "
                                f"Performance={last_perf['performance']:.2f}, "
                                f"Confidence={last_perf['confidence']:.2f}, "
                                f"Signal={last_perf['signal']:.2f}")
    
    def _adjust_dynamic_limits(self):
        """НОВОЕ: Динамическая корректировка торговых лимитов"""
        
        # Получаем текущий балл производительности
        if not hasattr(self, 'recent_trades_profit') or len(self.recent_trades_profit) < 3:
            return  # Недостаточно данных для адаптации
        
        performance_score = self._calculate_performance_score()
        sensitivity = getattr(TradingConfig, 'ADAPTATION_SENSITIVITY', 0.15)
        
        # Адаптируем лимиты времени между сделками
        base_time_limit = getattr(TradingConfig, 'BASE_MIN_TIME_BETWEEN_TRADES', 15)
        if performance_score > 0.7:
            # Хорошая производительность - ускоряем торговлю
            new_time_limit = max(5, int(base_time_limit * (1 - sensitivity)))
        elif performance_score < 0.3:
            # Плохая производительность - замедляем торговлю
            new_time_limit = min(50, int(base_time_limit * (1 + sensitivity * 2)))
        else:
            new_time_limit = base_time_limit
        
        # Адаптируем дневные лимиты сделок
        base_daily_limit = getattr(TradingConfig, 'BASE_MAX_DAILY_TRADES', 30)
        if performance_score > 0.7:
            # Хорошая производительность - увеличиваем лимит
            new_daily_limit = min(60, int(base_daily_limit * (1 + sensitivity)))
        elif performance_score < 0.3:
            # Плохая производительность - снижаем лимит
            new_daily_limit = max(10, int(base_daily_limit * (1 - sensitivity)))
        else:
            new_daily_limit = base_daily_limit
        
        # Сохраняем адаптированные лимиты
        if not hasattr(self, 'dynamic_limits'):
            self.dynamic_limits = {}
        
        self.dynamic_limits['min_time_between_trades'] = new_time_limit
        self.dynamic_limits['max_daily_trades'] = new_daily_limit
        
        # Логируем изменения
        if self.episode_steps % 200 == 0:  # Каждые 200 шагов
            self.logger.info(f"🔧 Динамические лимиты [Performance: {performance_score:.2f}]: "
                           f"Time={new_time_limit}, Daily={new_daily_limit}")

    def _calculate_signal_strength(self) -> int:
        """НОВАЯ ФУНКЦИЯ: Расчет силы торгового сигнала"""
        if not hasattr(self, 'price_history') or len(self.price_history) < 20:
            return 1  # Минимальная сила при недостатке данных
        
        strength = 0
        current_price = self.price_history[-1]
        
        # 1. Анализ краткосрочного тренда
        if len(self.price_history) >= 5:
            short_trend = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
            if abs(short_trend) > 0.01:  # Тренд >1%
                strength += 1
        
        # 2. Анализ среднесрочного тренда
        if len(self.price_history) >= 10:
            medium_trend = (self.price_history[-1] - self.price_history[-10]) / self.price_history[-10]
            if abs(medium_trend) > 0.02:  # Тренд >2%
                strength += 1
        
        # 3. Анализ волатильности (низкая волатильность = лучше)
        if len(self.price_history) >= 10:
            volatility = np.std(self.price_history[-10:]) / np.mean(self.price_history[-10:])
            if volatility < 0.05:  # Низкая волатильность
                strength += 1
        
        # 4. Согласованность трендов
        if len(self.price_history) >= 20:
            short_trend = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
            long_trend = (self.price_history[-1] - self.price_history[-20]) / self.price_history[-20]
            
            # Если краткосрочный и долгосрочный тренды в одном направлении
            if (short_trend > 0 and long_trend > 0) or (short_trend < 0 and long_trend < 0):
                strength += 1
        
        return min(strength, 5)  # Максимум 5 баллов
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """Получение торговой статистики"""
        if not self.trades:
            return {}
        
        profits = [t for t in self.trades if t > 0]
        losses = [t for t in self.trades if t <= 0]
        
        return {
            'total_trades': len(self.trades),
            'profitable_trades': len(profits),
            'losing_trades': len(losses),
            'win_rate': len(profits) / len(self.trades) * 100 if self.trades else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': abs(sum(profits) / sum(losses)) if losses else float('inf'),
            'total_pnl': sum(self.trades),
            'max_drawdown': self.max_drawdown,
            'commission_paid': self.total_commission_paid
        } 