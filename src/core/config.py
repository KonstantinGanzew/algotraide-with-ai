"""
Конфигурационные настройки для алготрейдинг системы

ВАЖНО: Этот файл содержит все настройки для торгового алгоритма.
При изменении параметров риска будьте осторожны!
"""

import os
from typing import Dict, Any


class TradingConfig:
    """Торговые параметры и риск-менеджмент"""
    
    # === КРИТИЧЕСКАЯ ПРОБЛЕМА: СЛИШКОМ АГРЕССИВНЫЕ ПАРАМЕТРЫ ===
    # Текущие настройки могут привести к большим потерям!
    
    # Риск-менеджмент (АГРЕССИВНЫЕ НАСТРОЙКИ ДЛЯ АКТИВНОЙ ТОРГОВЛИ!)
    BASE_RISK_PER_TRADE = 0.03        # УВЕЛИЧЕНО с 2% до 3%
    DYNAMIC_RISK_MULTIPLIER = 1.5      # УВЕЛИЧЕНО с 1.3 до 1.5
    MIN_POSITION_SIZE = 0.15           # УВЕЛИЧЕНО с 8% до 15%  
    MAX_POSITION_MULTIPLIER = 0.95     # УВЕЛИЧЕНО с 90% до 95%
    CONFIDENCE_THRESHOLD = 0.5         # КРИТИЧНО: УМЕНЬШЕНО с 0.7 до 0.5
    
    # Стоп-лоссы и тейк-профиты
    STOP_LOSS_PERCENTAGE = 0.02        # 2% стоп-лосс
    TAKE_PROFIT_PERCENTAGE = 0.04      # 4% тейк-профит (1:2 вместо 1:3)
    TRAILING_STOP_PERCENTAGE = 0.015   # 1.5% трейлинг стоп
    MAX_DRAWDOWN_LIMIT = 0.08          # УМЕНЬШЕНО до 8% макс. просадка
    
    # Торговые издержки
    TRADE_COMMISSION = 0.001           # 0.1% комиссия
    SLIPPAGE = 0.0005                  # 0.05% проскальзывание
    
    # НОВЫЕ ПАРАМЕТРЫ ПРИНУЖДЕНИЯ К ТОРГОВЛЕ
    FORCE_TRADE_EVERY_N_STEPS = 100    # Принудительная торговля каждые 100 шагов
    MIN_TRADES_PER_EPISODE = 5         # Минимум сделок за эпизод
    MAX_CONSECUTIVE_HOLDS = 20         # Макс. подряд "держать"
    
    
class TradingConfigFixed:
    """ЭКСТРЕМАЛЬНО АГРЕССИВНАЯ конфигурация для принуждения к торговле"""
    
    # Максимально агрессивные параметры для диагностики
    BASE_RISK_PER_TRADE = 0.20        # 20% на сделку!
    DYNAMIC_RISK_MULTIPLIER = 2.0      
    MIN_POSITION_SIZE = 0.30           # 30% минимум
    MAX_POSITION_MULTIPLIER = 0.90     
    CONFIDENCE_THRESHOLD = 0.1         # Очень низкий порог
    
    # Широкие стоп-лоссы для предотвращения ранних выходов
    STOP_LOSS_PERCENTAGE = 0.10        # 10% стоп-лосс
    TAKE_PROFIT_PERCENTAGE = 0.15      # 15% тейк-профит
    TRAILING_STOP_PERCENTAGE = 0.05    # 5% трейлинг стоп
    MAX_DRAWDOWN_LIMIT = 0.50          # 50% макс. просадка
    
    # Низкие издержки
    TRADE_COMMISSION = 0.0001          # 0.01% комиссия
    SLIPPAGE = 0.0001                  # 0.01% проскальзывание
    
    # Принуждение к торговле
    FORCE_TRADE_EVERY_N_STEPS = 20     # Каждые 20 шагов
    MIN_TRADES_PER_EPISODE = 10        # Минимум 10 сделок
    MAX_CONSECUTIVE_HOLDS = 5          # Максимум 5 hold подряд
    

class ConservativeTradingConfig:
    """Консервативные торговые параметры для стабильной прибыльности"""
    
    # Консервативный риск-менеджмент
    BASE_RISK_PER_TRADE = 0.01        # Снижаем до 1%
    DYNAMIC_RISK_MULTIPLIER = 1.2     # Более осторожно
    MIN_POSITION_SIZE = 0.05          # Минимум 5%
    MAX_POSITION_MULTIPLIER = 0.3     # Максимум 30% капитала
    CONFIDENCE_THRESHOLD = 0.65       # Повышаем порог уверенности
    
    # Строгие стоп-лоссы
    STOP_LOSS_PERCENTAGE = 0.015      # 1.5% стоп-лосс
    TAKE_PROFIT_PERCENTAGE = 0.025    # 2.5% тейк-профит
    ENABLE_STOP_LOSS = True
    ENABLE_TAKE_PROFIT = True
    
    # Комиссии
    TRADE_COMMISSION = 0.001          # 0.1%
    SLIPPAGE = 0.0001                 # 0.01%
    
    # Диверсификация
    MAX_CORRELATION_THRESHOLD = 0.7
    REBALANCING_FREQUENCY = 24        # каждые 24 часа
    
    # Психология торговли
    MAX_CONSECUTIVE_LOSSES = 3
    MAX_CONSECUTIVE_HOLDS = 10        # Максимум 10 последовательных удержаний
    COOLDOWN_AFTER_LOSSES = 5
    CONFIDENCE_DECAY_RATE = 0.95
    FORCE_TRADE_EVERY_N_STEPS = 200   # Принуждение к торговле каждые 200 шагов
    
    # Дополнительные лимиты безопасности
    MAX_DRAWDOWN_LIMIT = 0.15         # 15% максимальная просадка
    MAX_DAILY_LOSS_LIMIT = 0.05       # 5% дневные потери
    MIN_BALANCE_THRESHOLD = 0.8       # 80% от начального баланса
    MIN_TRADES_PER_EPISODE = 5        # Минимум 5 сделок за эпизод


class ProfitOptimizedConfig:
    """Конфигурация оптимизированная для прибыльности"""
    
    # Более умный риск-менеджмент
    BASE_RISK_PER_TRADE = 0.015       # 1.5% риск (больше чем консервативный)
    DYNAMIC_RISK_MULTIPLIER = 1.3     # Динамическое увеличение
    MIN_POSITION_SIZE = 0.08          # 8% минимум 
    MAX_POSITION_MULTIPLIER = 0.4     # 40% максимум (сбалансированно)
    CONFIDENCE_THRESHOLD = 0.6        # Немного выше консервативного
    
    # Умные стоп-лоссы и тейк-профиты
    STOP_LOSS_PERCENTAGE = 0.012      # 1.2% стоп-лосс
    TAKE_PROFIT_PERCENTAGE = 0.030    # 3% тейк-профит (соотношение 1:2.5)
    ENABLE_STOP_LOSS = True
    ENABLE_TAKE_PROFIT = True
    
    # Комиссии
    TRADE_COMMISSION = 0.001          # 0.1%
    SLIPPAGE = 0.0001                 # 0.01%
    
    # Психология торговли (сбалансированная)
    MAX_CONSECUTIVE_LOSSES = 2        # Более строго
    MAX_CONSECUTIVE_HOLDS = 15        # Больше терпения
    COOLDOWN_AFTER_LOSSES = 3         # Короче пауза
    CONFIDENCE_DECAY_RATE = 0.97      # Медленнее снижение уверенности
    FORCE_TRADE_EVERY_N_STEPS = 300   # Реже принуждение
    
    # Лимиты безопасности
    MAX_DRAWDOWN_LIMIT = 0.12         # 12% максимальная просадка
    MAX_DAILY_LOSS_LIMIT = 0.03       # 3% дневные потери
    MIN_BALANCE_THRESHOLD = 0.85      # 85% от начального баланса
    MIN_TRADES_PER_EPISODE = 8        # Больше минимум сделок


class ImprovedRewardConfig:
    """Улучшенная система наград для прибыльности"""
    
    # Базовые множители (сбалансированные)
    BALANCE_CHANGE_MULTIPLIER = 100.0     # Основная награда
    
    # Награды за действия (более умные)
    ACTION_REWARD = 0.1                   # Небольшая награда за активность
    
    # Награды за качественные сделки (сбалансированные)
    PROFITABLE_TRADE_BONUS = 2.0         # Снижено с 5.0
    LOSING_TRADE_PENALTY = -1.5          # Увеличено с -0.5
    
    # Бонусы за качество (новые)
    HIGH_PROFIT_BONUS = 5.0              # За сделки >2%
    SMALL_LOSS_PROTECTION = -0.5         # Меньший штраф за маленькие потери <0.5%
    
    # Штрафы за бездействие (смягченные)
    INACTIVITY_PENALTY = -0.1            # Снижено
    LONG_INACTIVITY_PENALTY = -2.0       # Вместо -10.0
    
    # Риск-штрафы
    DRAWDOWN_PENALTY_MULTIPLIER = 5.0    # Умеренный штраф
    OVERTRADING_PENALTY = -0.2           # Штраф за слишком частую торговлю
    
    # Бонусы за тренд (новые)
    TREND_FOLLOWING_BONUS = 1.0          # За торговлю по тренду
    COUNTER_TREND_PENALTY = -0.5         # За торговлю против тренда


class HighProfitConfig:
    """Конфигурация оптимизированная для МАКСИМАЛЬНОЙ прибыльности (Этап 1 - ИСПРАВЛЕННАЯ)"""
    
    # СМЯГЧЕННЫЕ УСЛОВИЯ ВХОДА для увеличения количества сделок
    BASE_RISK_PER_TRADE = 0.012       # 1.2% риск (умеренный)
    DYNAMIC_RISK_MULTIPLIER = 1.2     # Консервативное увеличение
    MIN_POSITION_SIZE = 0.06          # 6% минимум 
    MAX_POSITION_MULTIPLIER = 0.35    # 35% максимум (сбалансированно)
    CONFIDENCE_THRESHOLD = 0.55       # СНИЖЕНО: 55% уверенность (было 75%)
    
    # УЛУЧШЕННОЕ СООТНОШЕНИЕ РИСК/ПРИБЫЛЬ (1:3.1 вместо 1:2.5)
    STOP_LOSS_PERCENTAGE = 0.008      # УМЕНЬШЕНО: 0.8% стоп-лосс (было 1.2%)
    TAKE_PROFIT_PERCENTAGE = 0.025    # УМЕНЬШЕНО: 2.5% тейк-профит (было 3.0%)
    ENABLE_STOP_LOSS = True
    ENABLE_TAKE_PROFIT = True
    
    # Комиссии
    COMMISSION_RATE = 0.001          # 0.1% (реалистично для большинства бирж)
    
    # СМЯГЧЕННЫЕ ОГРАНИЧЕНИЯ ТОРГОВЛИ для увеличения активности
    MIN_TIME_BETWEEN_TRADES = 25     # СНИЖЕНО: 25 шагов между сделками (было 50)
    MAX_DAILY_TRADES = 15           # УВЕЛИЧЕНО: до 15 сделок в день (было 8)
    
    # СМЯГЧЕННЫЕ ТРЕБОВАНИЯ К КАЧЕСТВУ СИГНАЛОВ
    MIN_SIGNAL_STRENGTH = 0.4       # СНИЖЕНО: минимальная сила сигнала 40% (было 60%)
    REQUIRED_CONFIRMATIONS = 2      # СНИЖЕНО: 2 подтверждения (было 3)
    
    # Управление рисками
    MAX_CONSECUTIVE_HOLDS = 100     # УВЕЛИЧЕНО: больше терпения к hold (было 50)
    FORCE_TRADE_EVERY_N_STEPS = 200 # СНИЖЕНО: принуждение к торговле (было 300)
    MAX_DRAWDOWN_STOP = 0.15        # 15% макс. просадка для остановки
    MIN_TRADES_PER_EPISODE = 5      # Минимум сделок за эпизод
    
    # Адаптивность
    ENABLE_DYNAMIC_POSITION_SIZING = True
    ENABLE_TREND_FOLLOWING = True
    ENABLE_MARKET_REGIME_DETECTION = True


class HighProfitRewardConfig:
    """Система наград оптимизированная для прибыльности (без перетрейдинга)"""
    
    # Базовые награды (консервативные)
    BALANCE_CHANGE_MULTIPLIER = 150.0     # Немного выше базового
    
    # СТРОГИЕ награды за качество
    ACTION_REWARD = 0.05                  # УМЕНЬШЕНО: меньше поощрения за активность
    
    # СБАЛАНСИРОВАННЫЕ награды за сделки
    PROFITABLE_TRADE_BONUS = 3.0         # Увеличено за прибыльные
    LOSING_TRADE_PENALTY = -2.5          # Увеличен штраф за убыточные
    
    # НОВЫЕ бонусы за качество
    HIGH_PROFIT_BONUS = 8.0              # Большой бонус за сделки >2%
    SMALL_LOSS_PROTECTION = -0.3         # Меньший штраф за потери <0.5%
    EXCELLENT_TRADE_BONUS = 15.0         # За сделки >3%
    
    # АНТИПЕРЕТРЕЙДИНГ штрафы
    INACTIVITY_PENALTY = 0.0             # НЕТ штрафа за бездействие
    LONG_INACTIVITY_PENALTY = 0.0        # НЕТ принуждения к торговле
    OVERTRADING_PENALTY = -1.0           # УВЕЛИЧЕН штраф за частую торговлю
    
    # Риск-штрафы (строгие)
    DRAWDOWN_PENALTY_MULTIPLIER = 10.0   # Высокий штраф за просадку
    
    # НОВЫЕ бонусы за качественную торговлю
    TREND_FOLLOWING_BONUS = 2.0          # За торговлю по тренду
    COUNTER_TREND_PENALTY = -1.5         # За торговлю против тренда
    SELECTIVE_TRADING_BONUS = 1.0        # За редкие качественные сделки
    PATIENCE_BONUS = 0.5                 # За терпеливое ожидание сигналов


class SmartBalancedConfig:
    """Конфигурация с ИНТЕЛЛЕКТУАЛЬНОЙ БАЛАНСИРОВКОЙ активности и качества (Этап 2)"""
    
    # АДАПТИВНЫЕ УСЛОВИЯ ВХОДА
    BASE_RISK_PER_TRADE = 0.015       # 1.5% риск (умеренно-агрессивный)
    DYNAMIC_RISK_MULTIPLIER = 1.3     # Динамическое увеличение
    MIN_POSITION_SIZE = 0.05          # 5% минимум 
    MAX_POSITION_MULTIPLIER = 0.4     # 40% максимум
    CONFIDENCE_THRESHOLD = 0.45       # АДАПТИВНО: 45% базовый уровень
    
    # СООТНОШЕНИЕ РИСК/ПРИБЫЛЬ (более агрессивное)
    STOP_LOSS_PERCENTAGE = 0.012      # 1.2% стоп-лосс
    TAKE_PROFIT_PERCENTAGE = 0.03     # 3.0% тейк-профит (1:2.5)
    ENABLE_STOP_LOSS = True
    ENABLE_TAKE_PROFIT = True
    
    # Комиссии
    COMMISSION_RATE = 0.001           # 0.1% комиссия
    
    # ИНТЕЛЛЕКТУАЛЬНОЕ УПРАВЛЕНИЕ АКТИВНОСТЬЮ
    MIN_TIME_BETWEEN_TRADES = 15      # СНИЖЕНО: 15 шагов между сделками
    MAX_DAILY_TRADES = 25            # УВЕЛИЧЕНО: до 25 сделок в день
    MIN_SIGNAL_STRENGTH = 0.3        # СНИЖЕНО: 0.3 минимальная сила сигнала
    
    # АДАПТИВНЫЕ ЛИМИТЫ (меняются в зависимости от активности)
    ADAPTIVE_CONFIDENCE_MIN = 0.35   # Минимальный порог при низкой активности
    ADAPTIVE_CONFIDENCE_MAX = 0.65   # Максимальный порог при высокой активности
    ACTIVITY_CHECK_WINDOW = 100      # Окно для проверки активности
    MIN_TRADES_IN_WINDOW = 3         # Минимум сделок в окне
    
    # Управление рисками
    MAX_CONSECUTIVE_HOLDS = 80       # СНИЖЕНО: меньше терпения к hold
    FORCE_TRADE_EVERY_N_STEPS = 150  # СНИЖЕНО: принуждение к торговле
    MAX_DRAWDOWN_STOP = 0.18         # 18% макс. просадка
    MIN_TRADES_PER_EPISODE = 8       # Минимум сделок за эпизод
    
    # НОВОЕ: Мониторинг и адаптация
    ENABLE_ADAPTIVE_THRESHOLDS = True  # Включить адаптивные пороги
    ACTIVITY_BOOST_FACTOR = 0.9       # Фактор снижения порогов при низкой активности
    QUALITY_PENALTY_FACTOR = 1.1      # Фактор повышения порогов при плохом качестве


class DynamicOptimizedConfig:
    """Конфигурация с ДИНАМИЧЕСКОЙ САМООПТИМИЗАЦИЕЙ (Этап 3)"""
    
    # АДАПТИВНЫЕ БАЗОВЫЕ ПАРАМЕТРЫ
    BASE_RISK_PER_TRADE = 0.012       # 1.2% базовый риск
    DYNAMIC_RISK_MULTIPLIER = 1.4     # Более агрессивное увеличение
    MIN_POSITION_SIZE = 0.04          # 4% минимум (ниже для большей активности)
    MAX_POSITION_MULTIPLIER = 0.45    # 45% максимум (выше для возможностей)
    
    # ДИНАМИЧЕСКИЕ ПОРОГИ (стартовые значения)
    BASE_CONFIDENCE_THRESHOLD = 0.35  # СНИЖЕНО: 35% базовый уровень
    CONFIDENCE_ADAPTATION_RATE = 0.1  # Скорость адаптации порогов
    
    # СООТНОШЕНИЕ РИСК/ПРИБЫЛЬ
    STOP_LOSS_PERCENTAGE = 0.015      # 1.5% стоп-лосс
    TAKE_PROFIT_PERCENTAGE = 0.035    # 3.5% тейк-профит (1:2.3)
    ENABLE_STOP_LOSS = True
    ENABLE_TAKE_PROFIT = True
    
    # Комиссии
    COMMISSION_RATE = 0.001           # 0.1% комиссия
    
    # ДИНАМИЧЕСКИЕ ЛИМИТЫ АКТИВНОСТИ (автоматически адаптируются)
    BASE_MIN_TIME_BETWEEN_TRADES = 10      # СНИЖЕНО: 10 шагов базовый
    BASE_MAX_DAILY_TRADES = 40             # УВЕЛИЧЕНО: 40 сделок базовый лимит
    ADAPTIVE_LIMIT_ADJUSTMENT = True       # Включить автоадаптацию лимитов
    
    # УМНЫЕ СИГНАЛЫ
    BASE_MIN_SIGNAL_STRENGTH = 0.3    # СНИЖЕНО: 30% базовая сила сигнала
    SIGNAL_ADAPTATION_ENABLED = True  # Автоадаптация силы сигналов
    
    # УПРАВЛЕНИЕ РИСКАМИ
    MAX_CONSECUTIVE_HOLDS = 150       # Увеличено для терпения
    FORCE_TRADE_EVERY_N_STEPS = 250   # Принуждение к торговле реже
    MAX_DRAWDOWN_STOP = 0.20          # 20% макс. просадка
    MIN_TRADES_PER_EPISODE = 8        # Минимум сделок за эпизод
    
    # НОВОЕ: ПАРАМЕТРЫ ДИНАМИЧЕСКОЙ ОПТИМИЗАЦИИ
    PERFORMANCE_WINDOW = 100          # Окно для анализа производительности
    ADAPTATION_SENSITIVITY = 0.15     # Чувствительность адаптации
    MIN_TRADES_FOR_ADAPTATION = 5     # Минимум сделок для адаптации
    
    # НОВОЕ: РЕЖИМЫ РАБОТЫ
    AGGRESSIVE_MODE_THRESHOLD = 0.8   # Порог перехода в агрессивный режим
    CONSERVATIVE_MODE_THRESHOLD = 0.3 # Порог перехода в консервативный режим
    
    # НОВОЕ: АДАПТИВНЫЕ КОЭФФИЦИЕНТЫ
    PROFITABILITY_WEIGHT = 0.4        # Вес прибыльности в адаптации
    ACTIVITY_WEIGHT = 0.3             # Вес активности в адаптации  
    RISK_WEIGHT = 0.3                 # Вес риска в адаптации


class DynamicRewardConfig:
    """Система наград для динамической оптимизации"""
    
    # Базовые награды (адаптируются)
    BALANCE_CHANGE_MULTIPLIER = 100.0
    ACTION_REWARD = 0.1
    
    # Торговые награды
    PROFITABLE_TRADE_BONUS = 2.0
    HIGH_PROFIT_BONUS = 5.0
    EXCELLENT_TRADE_BONUS = 10.0
    LOSING_TRADE_PENALTY = -1.5
    SMALL_LOSS_PROTECTION = -0.2
    
    # Умные награды
    TREND_FOLLOWING_BONUS = 1.0
    COUNTER_TREND_PENALTY = -0.5
    
    # Адаптивные награды за качество
    SELECTIVE_TRADING_BONUS = 1.5
    PATIENCE_BONUS = 0.8
    MARKET_TIMING_BONUS = 2.0        # НОВОЕ: за правильный тайминг
    
    # Штрафы (смягченные для большей активности) 
    INACTIVITY_PENALTY = -0.05       # СНИЖЕНО
    LONG_INACTIVITY_PENALTY = -0.3   # СНИЖЕНО  
    OVERTRADING_PENALTY = -0.5       # СНИЖЕНО
    
    # Управление рисками
    DRAWDOWN_PENALTY_MULTIPLIER = 50.0


# ПЕРЕКЛЮЧАТЕЛЬ КОНФИГУРАЦИИ
USE_AGGRESSIVE_CONFIG = False  # True = агрессивная, False = консервативная

# ПЕРЕКЛЮЧАТЕЛЬ НА ВЫСОКОПРИБЫЛЬНУЮ КОНФИГУРАЦИЮ
USE_HIGH_PROFIT_MODE = False  # True = максимальная прибыльность

# ЭТАП 2: ПЕРЕКЛЮЧАТЕЛЬ НА ИНТЕЛЛЕКТУАЛЬНУЮ БАЛАНСИРОВКУ
USE_SMART_BALANCED_MODE = False  # True = адаптивная балансировка активности/качества

# ЭТАП 3: ПЕРЕКЛЮЧАТЕЛЬ НА ДИНАМИЧЕСКУЮ ОПТИМИЗАЦИЮ
USE_DYNAMIC_OPTIMIZATION = True  # True = динамическая самооптимизация

# Выбираем активную конфигурацию
if USE_DYNAMIC_OPTIMIZATION:
    ActiveTradingConfig = DynamicOptimizedConfig  # ЭТАП 3: ДИНАМИЧЕСКАЯ САМООПТИМИЗАЦИЯ
    ActiveRewardConfig = DynamicRewardConfig     # Адаптивная система наград
elif USE_SMART_BALANCED_MODE:
    ActiveTradingConfig = SmartBalancedConfig  # ЭТАП 2: ИНТЕЛЛЕКТУАЛЬНАЯ БАЛАНСИРОВКА
    ActiveRewardConfig = HighProfitRewardConfig  # Продолжаем использовать качественные награды
elif USE_HIGH_PROFIT_MODE:
    ActiveTradingConfig = HighProfitConfig
    ActiveRewardConfig = HighProfitRewardConfig
elif USE_AGGRESSIVE_CONFIG:
    ActiveTradingConfig = TradingConfigFixed
    ActiveRewardConfig = ImprovedRewardConfig
else:
    ActiveTradingConfig = ProfitOptimizedConfig  # Новая оптимизированная
    ActiveRewardConfig = ImprovedRewardConfig


class MLConfig:
    """Настройки машинного обучения"""
    
    # Архитектура модели (ВОЗМОЖНАЯ ПРОБЛЕМА: ПЕРЕОБУЧЕНИЕ)
    LSTM_HIDDEN_SIZE = 256             # УМЕНЬШЕНО с 512 до 256
    LSTM_NUM_LAYERS = 2                # УМЕНЬШЕНО с 3 до 2
    LSTM_DROPOUT = 0.2                 # Добавлен dropout
    
    # PPO параметры
    LEARNING_RATE = 5e-4               # УВЕЛИЧЕНО с 3e-4 до 5e-4
    TOTAL_TIMESTEPS = 150000           # УВЕЛИЧЕНО с 100000 до 150000
    PPO_ENT_COEF = 0.1                 # КРИТИЧНО: УВЕЛИЧЕНО с 0.05 до 0.1 для максимального исследования
    
    # Ранняя остановка (более терпеливая)
    ENABLE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 100     # УВЕЛИЧЕНО с 75 до 100
    MIN_EPISODES_BEFORE_STOPPING = 200 # УВЕЛИЧЕНО с 150 до 200  
    IMPROVEMENT_THRESHOLD = 0.001      # УМЕНЬШЕНО с 0.003 для большей чувствительности
    
    
class DataConfig:
    """Настройки данных и индикаторов"""
    
    # Файлы и пути
    DATA_FOLDER = "data/"
    DATA_FILE = "BTC_5_96w.csv"
    
    # Параметры окна
    WINDOW_SIZE = 50
    TRAIN_TEST_SPLIT = 0.8             # 80% на обучение
    
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
    
    
class SystemConfig:
    """Системные настройки"""
    
    # Начальный баланс
    INITIAL_BALANCE = 10000
    
    # Устройство вычислений
    AUTO_DEVICE = True
    FORCE_CPU = False
    DEVICE = "cpu"
    
    # Визуализация
    FIGURE_SIZE = (16, 10)
    SAVE_PLOTS = True
    PLOT_DPI = 300
    
    # Логирование
    LOG_LEVEL = "INFO"
    LOG_TO_FILE = True
    LOG_FILE = "trading_log.log"
    
    
class FutureIntegrationsConfig:
    """Настройки для будущих интеграций"""
    
    # Binance API (пока заглушки)
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
    BINANCE_TESTNET = True
    
    # Telegram Bot (пока заглушки)
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Уведомления
    ENABLE_TELEGRAM_ALERTS = False
    ALERT_ON_TRADE = True
    ALERT_ON_PROFIT_TARGET = True
    ALERT_ON_DRAWDOWN = True
    

def get_config() -> Dict[str, Any]:
    """Получить все конфигурации в одном словаре"""
    return {
        'trading': ActiveTradingConfig,
        'reward': ActiveRewardConfig,
        'ml': MLConfig,
        'data': DataConfig,
        'system': SystemConfig,
        'integrations': FutureIntegrationsConfig
    }


def validate_config():
    """Валидация критических параметров"""
    warnings = []
    
    # Проверка риск-параметров
    if ActiveTradingConfig.BASE_RISK_PER_TRADE > 0.02:  # >2%
        warnings.append("⚠️ ВЫСОКИЙ РИСК: BASE_RISK_PER_TRADE > 2%")
    
    if ActiveTradingConfig.MAX_POSITION_MULTIPLIER > 1.0:  # >100%
        warnings.append("⚠️ ОПАСНО: MAX_POSITION_MULTIPLIER > 100%")
    
    # Проверка лимита просадки (универсальная)
    max_drawdown = getattr(ActiveTradingConfig, 'MAX_DRAWDOWN_STOP', 
                          getattr(ActiveTradingConfig, 'MAX_DRAWDOWN_LIMIT', 0.15))
    if max_drawdown > 0.15:  # >15%
        warnings.append(f"⚠️ ВЫСОКАЯ ПРОСАДКА: {max_drawdown*100:.1f}% > 15%")
    
    # Проверка наград
    if ActiveRewardConfig.BALANCE_CHANGE_MULTIPLIER > 100:
        warnings.append("⚠️ ВЫСОКИЕ НАГРАДЫ: Может привести к нестабильности")
    
    return warnings


# === ДИАГНОСТИКА ПРОБЛЕМ С ПРИБЫЛЬНОСТЬЮ ===

class ProfitabilityIssues:
    """
    АНАЛИЗ ПРОБЛЕМ С ПРИБЫЛЬНОСТЬЮ:
    
    1. ПЕРЕОБУЧЕНИЕ:
       - Слишком сложная LSTM модель (512 hidden, 3 layers)
       - Недостаточно данных для обучения
       - Отсутствие регуляризации
    
    2. НЕПРАВИЛЬНЫЕ НАГРАДЫ:
       - BALANCE_CHANGE_MULTIPLIER = 200 (слишком высоко!)
       - Искусственные бонусы искажают обучение
       - Награды не отражают реальную прибыль
    
    3. АГРЕССИВНЫЙ РИСК-МЕНЕДЖМЕНТ:
       - 3% риска на сделку (слишком много!)
       - 120% максимальная позиция (опасно!)
       - Слишком широкие стоп-лоссы
    
    4. ПРОБЛЕМЫ ДАННЫХ:
       - Возможная утечка данных из будущего
       - Неправильная нормализация цен
       - Отсутствие walk-forward валидации
    
    5. ТОРГОВАЯ ЛОГИКА:
       - Только 3 действия (Hold, Buy, Sell)
       - Нет частичного закрытия позиций
       - Отсутствие фильтров качества сигналов
    
    РЕКОМЕНДАЦИИ:
    1. Упростить модель и уменьшить риски
    2. Переработать систему наград
    3. Добавить proper backtesting
    4. Реализовать walk-forward validation
    5. Добавить больше торговых действий
    """
    pass 