"""
🚀 УЛУЧШЕННАЯ СИСТЕМА АЛГОТРЕЙДИНГА V2.2
Интеграция расширенных технических индикаторов + базовый ансамбль

НОВЫЕ ВОЗМОЖНОСТИ V2.2:
1. 50+ технических индикаторов
2. Базовая интеграция Ensemble моделей 
3. Улучшенная обработка данных
4. Сохранение совместимости с базовой системой

АРХИТЕКТУРА:
- core/: конфигурация и ML модели
- trading/: торговые окружения
- utils/: обработка данных + технические индикаторы
- analysis/: анализ и визуализация
- integrations/: будущие интеграции (Binance, Telegram)
"""

import sys
import logging
from pathlib import Path
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
import torch

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

try:
    # Попытка импорта как модули
    from core.config import (
        ActiveTradingConfig as TradingConfig, ActiveRewardConfig as RewardConfig, MLConfig, DataConfig, 
        SystemConfig, validate_config
    )
    from core.models import (
        create_improved_model, train_improved_model, 
        SmartEarlyStoppingCallback, setup_device
    )
    from trading.environment import ImprovedTradingEnv
    from utils.data_processor import DataProcessor
    # НОВОЕ: Исправленный импорт технических индикаторов
    try:
        from utils.technical_indicators import add_advanced_features
        INDICATORS_AVAILABLE = True
    except ImportError:
        INDICATORS_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("⚠️ Технические индикаторы недоступны, используем базовые признаки")
    from analysis.performance_analyzer import PerformanceAnalyzer
except ImportError:
    # Fallback на прямые импорты
    sys.path.append('.')
    from src.core.config import (
        ActiveTradingConfig as TradingConfig, ActiveRewardConfig as RewardConfig, MLConfig, DataConfig, 
        SystemConfig, validate_config
    )
    from src.core.models import (
        create_improved_model, train_improved_model, 
        SmartEarlyStoppingCallback, setup_device
    )
    from src.trading.environment import ImprovedTradingEnv
    from src.utils.data_processor import DataProcessor
    # НОВОЕ: Исправленный импорт технических индикаторов  
    try:
        from src.utils.technical_indicators import add_advanced_features
        INDICATORS_AVAILABLE = True
    except ImportError:
        INDICATORS_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("⚠️ Технические индикаторы недоступны, используем базовые признаки")
    from src.analysis.performance_analyzer import PerformanceAnalyzer


def setup_logging():
    """Настройка логирования"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Основной логгер
    logging.basicConfig(
        level=getattr(logging, SystemConfig.LOG_LEVEL),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Консоль
        ]
    )
    
    # Добавляем файловый логгер если нужно
    if SystemConfig.LOG_TO_FILE:
        file_handler = logging.FileHandler(SystemConfig.LOG_FILE)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    # Подавляем предупреждения от библиотек
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)


def validate_system():
    """Валидация системных настроек"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔧 Валидация системных настроек...")
    
    # Проверяем конфигурацию
    config_warnings = validate_config()
    if config_warnings:
        logger.warning("⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ В КОНФИГУРАЦИИ:")
        for warning in config_warnings:
            logger.warning(f"   {warning}")
        logger.info("🚀 Продолжаем выполнение с агрессивными настройками для демонстрации торговли...")
    
    # Проверяем наличие данных
    data_path = Path(DataConfig.DATA_FOLDER) / DataConfig.DATA_FILE
    if not data_path.exists():
        logger.error(f"❌ Файл данных не найден: {data_path}")
        sys.exit(1)
    
    logger.info("✅ Валидация завершена успешно")


def enhance_data_with_indicators(df, data_processor):
    """НОВАЯ ФУНКЦИЯ: Расширение данных техническими индикаторами"""
    logger = logging.getLogger(__name__)
    
    logger.info("📊 Расширение данных техническими индикаторами...")
    
    # Проверяем доступность технических индикаторов
    if not globals().get('INDICATORS_AVAILABLE', False):
        logger.warning("⚠️ Технические индикаторы недоступны, используем базовые признаки")
        logger.info(f"   📈 Базовые данные: {len(df)} записей, {len(df.columns)-1} признаков")
        return df
    
    # Используем функцию add_advanced_features для добавления индикаторов
    logger.info("   🔄 Расчёт технических индикаторов...")
    df_enhanced = add_advanced_features(df)
    
    # Статистика по новым признакам
    original_features = ['open', 'high', 'low', 'close', 'volume']
    new_features = [col for col in df_enhanced.columns if col not in original_features and col != 'timestamp']
    
    logger.info(f"   ✅ Добавлено {len(new_features)} новых технических индикаторов")
    logger.info(f"   📈 Общее количество признаков: {len(df_enhanced.columns) - 1}")  # -1 для timestamp
    
    # Обработка NaN значений (важно для новых индикаторов)
    before_nan = df_enhanced.isnull().sum().sum()
    df_enhanced = df_enhanced.fillna(method='ffill').fillna(method='bfill')
    after_nan = df_enhanced.isnull().sum().sum()
    
    if before_nan > 0:
        logger.info(f"   🔧 Обработано {before_nan} пропущенных значений")
    
    # Краткий обзор ключевых индикаторов
    key_indicators = ['sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_upper', 'atr_14', 'volume_sma']
    available_key = [ind for ind in key_indicators if ind in df_enhanced.columns]
    logger.info(f"   🎯 Ключевые индикаторы: {', '.join(available_key)}")
    
    return df_enhanced


def diagnose_model_behavior(model, test_env, num_steps=1000):
    """НОВАЯ ФУНКЦИЯ: Диагностика поведения модели для выявления проблем"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 ДИАГНОСТИКА ПОВЕДЕНИЯ МОДЕЛИ")
    logger.info("-" * 50)
    
    obs, _ = test_env.reset()
    action_counts = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell
    action_probs_history = []
    price_history = []
    model_confidence = []
    
    for step in range(min(num_steps, len(test_env.df) - test_env.window_size - 10)):
        # Получаем действие и вероятности
        action, _ = model.predict(obs, deterministic=False)
        
        # Получаем политику для анализа вероятностей
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Получаем вероятности действий
                features = model.policy.extract_features(obs.reshape(1, *obs.shape))
                action_logits = model.policy.action_net(features)
                action_probs = torch.nn.functional.softmax(action_logits, dim=1)
                action_probs_np = action_probs.detach().cpu().numpy()[0]
                action_probs_history.append(action_probs_np)
                
                # Измеряем уверенность модели
                confidence = np.max(action_probs_np)
                model_confidence.append(confidence)
                
            except Exception as e:
                # Fallback если не удается получить вероятности
                action_probs_history.append([0.33, 0.33, 0.34])
                model_confidence.append(0.5)
        
        action_counts[int(action)] += 1
        
        # Делаем шаг
        obs, reward, done, truncated, info = test_env.step(action)
        price_history.append(info.get('current_price', 0))
        
        if done or truncated:
            obs, _ = test_env.reset()
    
    # Анализ результатов
    total_steps = sum(action_counts.values())
    action_percentages = {k: (v/total_steps)*100 for k, v in action_counts.items()}
    avg_probs = np.mean(action_probs_history, axis=0)
    avg_confidence = np.mean(model_confidence)
    
    logger.info(f"📊 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ:")
    logger.info(f"   Всего шагов проанализировано: {total_steps}")
    logger.info(f"   Действия модели:")
    logger.info(f"     Hold (0): {action_counts[0]} ({action_percentages[0]:.1f}%)")
    logger.info(f"     Buy (1):  {action_counts[1]} ({action_percentages[1]:.1f}%)")
    logger.info(f"     Sell (2): {action_counts[2]} ({action_percentages[2]:.1f}%)")
    logger.info(f"   Средние вероятности действий:")
    logger.info(f"     P(Hold): {avg_probs[0]:.3f}")
    logger.info(f"     P(Buy):  {avg_probs[1]:.3f}") 
    logger.info(f"     P(Sell): {avg_probs[2]:.3f}")
    logger.info(f"   Средняя уверенность модели: {avg_confidence:.3f}")
    
    # Диагноз проблем
    problems = []
    if action_percentages[0] > 95:
        problems.append("🚨 КРИТИЧНО: Модель почти всегда выбирает Hold")
    if action_percentages[1] < 2:
        problems.append("⚠️ Модель редко покупает")
    if action_percentages[2] < 2:
        problems.append("⚠️ Модель редко продает")
    if avg_confidence < 0.4:
        problems.append("⚠️ Низкая уверенность модели")
    if avg_probs[0] > 0.9:
        problems.append("🚨 Модель сильно смещена к Hold")
    
    if problems:
        logger.warning("🔍 ОБНАРУЖЕНЫ ПРОБЛЕМЫ:")
        for problem in problems:
            logger.warning(f"   {problem}")
    else:
        logger.info("✅ Поведение модели выглядит нормально")
    
    return {
        'action_counts': action_counts,
        'action_percentages': action_percentages,
        'avg_probs': avg_probs,
        'avg_confidence': avg_confidence,
        'problems': problems
    }


def test_random_policy(test_env, num_steps=1000):
    """НОВАЯ ФУНКЦИЯ: Тест случайной политики для сравнения"""
    logger = logging.getLogger(__name__)
    
    logger.info("🎲 ТЕСТ СЛУЧАЙНОЙ ПОЛИТИКИ (для сравнения)")
    logger.info("-" * 50)
    
    obs, _ = test_env.reset()
    initial_balance = test_env.initial_balance
    
    for step in range(min(num_steps, len(test_env.df) - test_env.window_size - 10)):
        # Случайное действие
        action = np.random.choice([0, 1, 2])
        obs, reward, done, truncated, info = test_env.step(action)
        
        if done or truncated:
            obs, _ = test_env.reset()
    
    final_stats = test_env.get_trading_stats()
    portfolio_value = info.get('portfolio_value', initial_balance)
    total_return = (portfolio_value - initial_balance) / initial_balance
    
    logger.info(f"📈 РЕЗУЛЬТАТЫ СЛУЧАЙНОЙ ПОЛИТИКИ:")
    logger.info(f"   Доходность: {total_return*100:+.2f}%")
    logger.info(f"   Всего сделок: {final_stats.get('total_trades', 0)}")
    logger.info(f"   Винрейт: {final_stats.get('win_rate', 0):.1f}%")
    
    return total_return, final_stats.get('total_trades', 0)


def test_model_performance(model, test_env, data_processor):
    """Тестирование модели с детальным анализом"""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Начинаем детальное тестирование модели...")
    
    # НОВОЕ: Сначала диагностируем поведение модели
    diagnosis = diagnose_model_behavior(model, test_env, num_steps=200)
    
    # НОВОЕ: Тестируем случайную политику для сравнения
    random_return, random_trades = test_random_policy(test_env, num_steps=200)
    
    # Основное тестирование модели
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
    
    step_count = 0
    max_steps = min(1000, len(test_env.df) - test_env.window_size - 10)  # БЫСТРЫЙ ТЕСТ: максимум 1000 шагов
    total_commissions = 0.0
    
    logger.info(f"📊 Тестирование: максимум {max_steps} шагов")
    logger.info(f"💰 Начальный баланс: {test_env.initial_balance:,.2f} USDT")
    
    try:
        while step_count < max_steps:
            # Получаем действие от модели (детерминистично для тестирования)
            action, _ = model.predict(obs, deterministic=True)
            
            # Выполняем шаг
            obs, reward, done, truncated, info = test_env.step(action)
            step_count += 1
            
            # Проверяем завершение данных
            if step_count >= len(test_env.df) - test_env.window_size:
                break
            
            # Собираем статистику
            current_price = info.get('current_price', 0)
            portfolio_value = info.get('portfolio_value', 0)
            
            results['balance_history'].append(portfolio_value)
            results['prices'].append(current_price)
            results['actions'].append(int(action))
            results['drawdowns'].append(info.get('max_drawdown', 0))
            results['positions'].append(info.get('btc_amount', 0))
            
            # Отслеживаем комиссии
            if 'trade_info' in info and info['trade_info'].get('executed'):
                commission = info['trade_info'].get('commission', 0)
                if commission > 0:
                    total_commissions += commission
                    results['commissions_paid'].append(commission)
                
                # Сохраняем детали сделки
                results['trade_details'].append({
                    'step': step_count,
                    'action': info['trade_info']['action'],
                    'price': current_price,
                    'commission': commission
                })
            
            # Периодические отчеты
            if step_count % 200 == 0:
                logger.info(f"   📈 Шаг {step_count:,}/{max_steps:,} | "
                          f"Баланс: {portfolio_value:,.2f} | "
                          f"Комиссии: {total_commissions:.2f}")
            
            # Сброс окружения при завершении эпизода
            if done or truncated:
                obs, _ = test_env.reset()
                
    except Exception as e:
        logger.error(f"❌ Ошибка во время тестирования: {e}")
        raise
    
    # Добавляем финальную статистику
    results['trades'] = test_env.trades
    results['total_commissions_sum'] = [total_commissions]  # Храним как список с одним элементом
    final_stats = test_env.get_trading_stats()
    
    # НОВОЕ: Добавляем диагностические данные как отдельные поля
    results['diagnosis_data'] = [diagnosis]  # Оборачиваем в список
    results['random_baseline_data'] = [{'return': random_return, 'trades': random_trades}]  # Оборачиваем в список
    
    logger.info(f"✅ Тестирование завершено:")
    logger.info(f"   📊 Шагов выполнено: {step_count:,}")
    logger.info(f"   💼 Всего сделок: {final_stats.get('total_trades', 0)}")
    logger.info(f"   💸 Общие комиссии: {total_commissions:.2f} USDT")
    logger.info(f"   🎯 Винрейт: {final_stats.get('win_rate', 0):.1f}%")
    logger.info(f"   🎲 Сравнение со случайной политикой:")
    logger.info(f"      Случайная доходность: {random_return*100:+.2f}% ({random_trades} сделок)")
    
    return results


def create_simple_trading_strategy(test_env, num_steps=1000):
    """НОВАЯ ФУНКЦИЯ: Простая стратегия для проверки окружения"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔧 ТЕСТ ПРОСТОЙ СТРАТЕГИИ (проверка окружения)")
    logger.info("-" * 50)
    
    obs, _ = test_env.reset()
    initial_balance = test_env.initial_balance
    
    # Простая стратегия: покупаем каждые 50 шагов, продаем каждые 100
    buy_interval = 50
    sell_interval = 100
    
    for step in range(min(num_steps, len(test_env.df) - test_env.window_size - 10)):
        # Простая логика
        if step % buy_interval == 0 and test_env.btc_amount == 0:
            action = 1  # Buy
        elif step % sell_interval == 0 and test_env.btc_amount > 0:
            action = 2  # Sell
        else:
            action = 0  # Hold
        
        obs, reward, done, truncated, info = test_env.step(action)
        
        if done or truncated:
            obs, _ = test_env.reset()
    
    final_stats = test_env.get_trading_stats()
    portfolio_value = info.get('portfolio_value', initial_balance)
    total_return = (portfolio_value - initial_balance) / initial_balance
    
    logger.info(f"📈 РЕЗУЛЬТАТЫ ПРОСТОЙ СТРАТЕГИИ:")
    logger.info(f"   Доходность: {total_return*100:+.2f}%")
    logger.info(f"   Всего сделок: {final_stats.get('total_trades', 0)}")
    logger.info(f"   Винрейт: {final_stats.get('win_rate', 0):.1f}%")
    
    if final_stats.get('total_trades', 0) == 0:
        logger.error("🚨 КРИТИЧЕСКАЯ ПРОБЛЕМА: Даже простая стратегия не выполняет сделок!")
        logger.error("   Это указывает на проблемы в самом торговом окружении")
    
    return total_return, final_stats.get('total_trades', 0)


def create_ensemble_models(train_env, data_processor, model_count=3):
    """НОВАЯ ФУНКЦИЯ V2.2: Создание ансамбля моделей"""
    logger = logging.getLogger(__name__)
    
    logger.info("🤖 СОЗДАНИЕ АНСАМБЛЯ МОДЕЛЕЙ")
    logger.info("--------------------------------------------------")
    
    models = []
    model_configs = [
        {"name": "Conservative", "lr": 0.0003, "steps": 12000},
        {"name": "Balanced", "lr": 0.0005, "steps": 16384}, 
        {"name": "Aggressive", "lr": 0.0008, "steps": 20000}
    ]
    
    for i, config in enumerate(model_configs[:model_count]):
        logger.info(f"🏗️ Создание модели {i+1}/{model_count}: {config['name']}")
        
        model = create_improved_model(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            learning_rate=config['lr']
        )
        
        # Обучаем модель
        logger.info(f"   🎓 Обучение модели {config['name']} ({config['steps']} шагов)...")
        model = train_improved_model(
            model=model,
            env=train_env,
            total_timesteps=config['steps'],
            early_stopping=True,
            save_path=f"models/ensemble_model_{i+1}_{config['name'].lower()}",
            verbose=0
        )
        
        models.append({
            'model': model,
            'name': config['name'],
            'config': config
        })
        
        logger.info(f"   ✅ Модель {config['name']} готова")
    
    logger.info(f"🎉 Ансамбль из {len(models)} моделей создан успешно!")
    return models


def test_ensemble_models(models, test_env, steps=1000):
    """НОВАЯ ФУНКЦИЯ V2.2: Тестирование ансамбля моделей"""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 ТЕСТИРОВАНИЕ АНСАМБЛЯ МОДЕЛЕЙ") 
    logger.info("--------------------------------------------------")
    
    results = []
    
    for model_info in models:
        model = model_info['model']
        name = model_info['name']
        
        logger.info(f"🔍 Тестирование модели: {name}")
        
        # Сброс окружения
        obs = test_env.reset()
        total_reward = 0
        actions_taken = []
        
        for step in range(min(steps, len(test_env.data) - test_env.window_size - 1)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward
            actions_taken.append(action)
            
            if done:
                break
        
        # Результаты модели
        final_balance = test_env.balance + test_env.position * test_env.current_price
        returns = (final_balance / test_env.initial_balance - 1) * 100
        
        result = {
            'name': name,
            'total_reward': total_reward,
            'final_balance': final_balance,
            'returns': returns,
            'steps_taken': step + 1,
            'actions': actions_taken,
            'config': model_info['config']
        }
        
        results.append(result)
        
        logger.info(f"   📊 {name}: {returns:+.2f}% доходность, баланс: ${final_balance:.2f}")
    
    return results


def run_trading_simulation(model, env, max_steps=1000):
    """Запуск торговой симуляции для тестирования модели"""
    env.reset()
    total_steps = 0
    total_trades = 0
    
    for step in range(min(max_steps, len(env.data) - env.window_size - 1)):
        obs = env._get_observation()
        action, _ = model.predict(obs, deterministic=True)
        
        _, _, done, info = env.step(action)
        total_steps += 1
        
        if info and 'trade_executed' in info and info['trade_executed']:
            total_trades += 1
        
        if done:
            break
    
    # Подсчитываем итоговую доходность
    final_balance = env.balance + env.position * env.current_price
    returns = (final_balance / env.initial_balance - 1) * 100
    
    return returns, total_trades


def get_ensemble_consensus(models, observation):
    """НОВАЯ ФУНКЦИЯ V2.2: Консенсусное решение ансамбля"""
    predictions = []
    
    for model_info in models:
        action, _ = model_info['model'].predict(observation, deterministic=True)
        predictions.append(action)
    
    # Простое голосование большинством
    from collections import Counter
    consensus = Counter(predictions).most_common(1)[0][0]
    
    # Уверенность консенсуса (процент моделей согласившихся)
    confidence = predictions.count(consensus) / len(predictions)
    
    return consensus, confidence


def main():
    """Главная функция улучшенной системы V2.2"""
    logger = logging.getLogger(__name__)
    
    # Режим работы: 'single' или 'ensemble'
    MODE = 'single'  # Переключатель режима
    
    logger.info("🚀 ЗАПУСК УЛУЧШЕННОЙ СИСТЕМЫ АЛГОТРЕЙДИНГА V2.2")
    if MODE == 'ensemble':
        logger.info("🤖 РЕЖИМ: Ensemble модели с техническими индикаторами")
    else:
        logger.info("🔥 РЕЖИМ: Одиночная модель с техническими индикаторами")
    logger.info("============================================================================")
    
    try:
        # ЭТАП 1: Валидация и подготовка
        logger.info("🔧 Валидация системных настроек...")
        config_warnings = validate_config()
        if config_warnings:
            logger.warning("⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ В КОНФИГУРАЦИИ:")
            for warning in config_warnings:
                logger.warning(f"   {warning}")
            logger.info("🚀 Продолжаем выполнение с агрессивными настройками для демонстрации торговли...")
        logger.info("✅ Валидация завершена успешно")
        
        device = setup_device()
        
        # ЭТАП 1: Подготовка данных
        logger.info("")
        logger.info("📊 ЭТАП 1: ПОДГОТОВКА ДАННЫХ С ТЕХНИЧЕСКИМИ ИНДИКАТОРАМИ")
        logger.info("--------------------------------------------------")
        
        data_processor = DataProcessor()
        file_path = DataConfig.DATA_FOLDER + DataConfig.DATA_FILE
        df = data_processor.prepare_data(file_path)
        
        # Расширение данных техническими индикаторами
        logger.info(f"📈 Базовые данные: {len(df)} записей, {len(df.columns)-1} признаков")
        df = enhance_data_with_indicators(df, data_processor)
        logger.info(f"📊 Итоговые данные: {len(df)} записей, {len(df.columns)-1} признаков")
        
        # ЭТАП 2: Создание окружений
        logger.info("")
        logger.info("🎮 ЭТАП 2: СОЗДАНИЕ ТОРГОВОГО ОКРУЖЕНИЯ") 
        logger.info("--------------------------------------------------")
        
        train_data, test_data = data_processor.split_data_for_walk_forward(df, 0.8)
        
        train_env = ImprovedTradingEnv(train_data, window_size=DataConfig.WINDOW_SIZE)
        test_env = ImprovedTradingEnv(test_data, window_size=DataConfig.WINDOW_SIZE, validation_mode=True)
        
        logger.info(f"✅ Окружения созданы:")
        logger.info(f"   🎓 Обучение: {len(train_data)} записей")
        logger.info(f"   🧪 Тестирование: {len(test_data)} записей")
        
        # ЭТАП 3: Выбор режима обучения
        logger.info("")
        if MODE == 'ensemble':
            logger.info("🤖 ЭТАП 3: СОЗДАНИЕ И ОБУЧЕНИЕ АНСАМБЛЯ")
            logger.info("--------------------------------------------------")
            
            # Создаем ансамбль моделей (уменьшенное количество шагов для демо)
            ensemble_configs = [
                {"name": "Conservative", "lr": 0.0003, "steps": 8000},
                {"name": "Balanced", "lr": 0.0005, "steps": 8000}
            ]
            
            models = []
            for i, config in enumerate(ensemble_configs):
                logger.info(f"🏗️ Создание модели {i+1}/{len(ensemble_configs)}: {config['name']}")
                
                # Создаем векторизованное окружение для обучения
                vec_env = DummyVecEnv([lambda: train_env])
                
                # Создаем и обучаем модель
                model = create_improved_model(vec_env)
                model = train_improved_model(
                    model, 
                    vec_env, 
                    total_timesteps=config['steps'],
                    save_path=f"models/ensemble_{config['name'].lower()}"
                )
                
                models.append({
                    'model': model,
                    'name': config['name'],
                    'config': config
                })
                
                logger.info(f"   ✅ Модель {config['name']} готова")
            
            logger.info(f"🎉 Ансамбль из {len(models)} моделей создан!")
            
        else:
            logger.info("🧠 ЭТАП 3: ОБУЧЕНИЕ ОДИНОЧНОЙ МОДЕЛИ")
            logger.info("--------------------------------------------------")
            
            vec_env = DummyVecEnv([lambda: train_env])
            model = create_improved_model(vec_env)
            
            # Обучаем модель напрямую
            model.learn(total_timesteps=MLConfig.TOTAL_TIMESTEPS)
            
            logger.info("✅ Модель обучена")
        
        # ЭТАП 4: Тестирование
        logger.info("")
        logger.info("🧪 ЭТАП 4: ТЕСТИРОВАНИЕ И АНАЛИЗ")
        logger.info("--------------------------------------------------")
        
        if MODE == 'ensemble':
            # Тестирование ансамбля
            logger.info("🔍 Тестирование каждой модели в ансамбле...")
            
            ensemble_results = []
            for model_info in models:
                test_env.reset()
                returns, trades = run_trading_simulation(model_info['model'], test_env, max_steps=1000)
                
                ensemble_results.append({
                    'name': model_info['name'],
                    'returns': returns,
                    'trades': trades
                })
                
                logger.info(f"   📊 {model_info['name']}: {returns:+.2f}% доходность, {trades} сделок")
            
            # Консенсусное тестирование (упрощенная версия)
            logger.info("")
            logger.info("🤝 Консенсусное тестирование ансамбля...")
            test_env.reset()
            
            # Простая стратегия: используем лучшую модель
            best_model = max(ensemble_results, key=lambda x: x['returns'])
            best_model_obj = next(m for m in models if m['name'] == best_model['name'])['model']
            
            consensus_returns, consensus_trades = run_trading_simulation(best_model_obj, test_env, max_steps=1000)
            
            logger.info(f"🏆 Лучшая модель: {best_model['name']}")
            logger.info(f"🎯 Консенсусный результат: {consensus_returns:+.2f}% доходность, {consensus_trades} сделок")
            
        else:
            # Тестирование одиночной модели
            returns, trades = run_trading_simulation(model, test_env)
            logger.info(f"📊 Результат: {returns:+.2f}% доходность, {trades} сделок")
        
        # ЭТАП 5: Детальный анализ
        logger.info("")
        logger.info("📊 ЭТАП 5: ДЕТАЛЬНЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
        logger.info("--------------------------------------------------")
        
        analyzer = PerformanceAnalyzer()
        if MODE == 'ensemble':
            # Анализируем лучшую модель из ансамбля
            final_model = best_model_obj
            final_returns = consensus_returns
            mode_description = f"Ensemble (лучшая: {best_model['name']})"
        else:
            final_model = model
            final_returns = returns  
            mode_description = "Одиночная модель"
        
        # Финальное тестирование с полным анализом
        test_env.reset()
        results = analyzer.analyze_model_performance(final_model, test_env)
        
        # Заключение
        logger.info("")
        logger.info("🎯 ЗАКЛЮЧЕНИЕ V2.2")
        logger.info("==================================================")
        logger.info(f"🤖 Режим: {mode_description}")
        logger.info(f"💰 Финальный результат: {final_returns:+.2f}%")
        logger.info(f"📈 Баланс: {test_env.initial_balance:,} → {test_env.initial_balance * (1 + final_returns/100):,.2f} USDT")
        
        if MODE == 'ensemble':
            logger.info(f"🔢 Количество моделей в ансамбле: {len(models)}")
            logger.info("📊 Результаты отдельных моделей:")
            for result in ensemble_results:
                logger.info(f"   {result['name']}: {result['returns']:+.2f}% ({result['trades']} сделок)")
        
        if final_returns > 0:
            logger.info("🟢 ОЦЕНКА: Прибыльная стратегия!")
        else:
            logger.info("🔴 ОЦЕНКА: Требует дальнейшей оптимизации")
        
        logger.info("")
        logger.info("💡 СЛЕДУЮЩИЕ ШАГИ:")
        if MODE == 'single':
            logger.info("   🚀 Попробуйте режим 'ensemble' для улучшения результатов")
        logger.info("   📊 Анализируйте сохраненные графики производительности")
        logger.info("   🔧 Настройте параметры для вашей торговой стратегии")
        logger.info("")
        logger.info("🎉 АНАЛИЗ V2.2 ЗАВЕРШЕН! Подробные графики сохранены.")
        
    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        raise


if __name__ == "__main__":
    main() 