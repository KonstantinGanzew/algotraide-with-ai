"""
🚀 ПРОДВИНУТАЯ СИСТЕМА АЛГОТРЕЙДИНГА V3.0
Интеграция всех улучшений: ensemble модели, walk-forward анализ, 
многомасштабный анализ, расширенные индикаторы

НОВЫЕ ВОЗМОЖНОСТИ:
✅ 7+ лет исторических данных (789K+ записей)
✅ 50+ технических индикаторов
✅ Ensemble ML + RL модели
✅ Walk-forward валидация
✅ Многомасштабный анализ (5m-1d)
✅ Рыночные режимы и консенсус сигналы
✅ Продвинутая диагностика и отчетность
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import time

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

try:
    # Основные модули
    from core.config import (
        ActiveTradingConfig as TradingConfig, 
        ActiveRewardConfig as RewardConfig, 
        MLConfig, DataConfig, SystemConfig, validate_config
    )
    from core.models import (
        create_improved_model, train_improved_model, 
        SmartEarlyStoppingCallback, setup_device
    )
    from trading.environment import ImprovedTradingEnv
    from utils.data_processor import DataProcessor
    from analysis.performance_analyzer import PerformanceAnalyzer
    
    # Новые модули
    from utils.data_loader import HistoricalDataLoader
    from utils.technical_indicators import add_advanced_features
    from core.ensemble_models import create_ensemble_system, HybridEnsemble
    from analysis.walk_forward_analysis import run_walk_forward_analysis, WalkForwardValidator
    from analysis.multi_timeframe_analyzer import analyze_multi_timeframe_market, MultiTimeframeAnalyzer
    
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Ошибка импорта: {e}")
    sys.exit(1)

from stable_baselines3.common.vec_env import DummyVecEnv


def setup_advanced_logging():
    """Расширенная настройка логирования"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Основной логгер
    logging.basicConfig(
        level=getattr(logging, SystemConfig.LOG_LEVEL),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Специальный логгер для результатов
    results_logger = logging.getLogger('RESULTS')
    results_logger.setLevel(logging.INFO)
    
    if SystemConfig.LOG_TO_FILE:
        # Создаем папку для логов
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Файловые логгеры
        main_handler = logging.FileHandler(log_dir / f'advanced_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        main_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(main_handler)
        
        results_handler = logging.FileHandler(log_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        results_handler.setFormatter(logging.Formatter(log_format))
        results_logger.addHandler(results_handler)
    
    # Подавляем предупреждения
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)


def prepare_advanced_data(use_extended_dataset: bool = True) -> pd.DataFrame:
    """Подготовка расширенного датасета с техническими индикаторами"""
    logger = logging.getLogger(__name__)
    
    logger.info("📊 ПОДГОТОВКА РАСШИРЕННОГО ДАТАСЕТА")
    logger.info("=" * 50)
    
    if use_extended_dataset:
        # Проверяем наличие расширенного датасета
        extended_file = Path('data/BTC_5m_6years_extended.csv')
        
        if extended_file.exists():
            logger.info(f"📈 Загрузка расширенного датасета: {extended_file}")
            df = pd.read_csv(extended_file)
            logger.info(f"✅ Загружено {len(df):,} записей из расширенного файла")
        else:
            logger.info("📥 Создание расширенного датасета...")
            loader = HistoricalDataLoader()
            df = loader.create_extended_dataset()
            
            if df.empty:
                logger.warning("⚠️ Не удалось создать расширенный датасет, используем базовый")
                df = pd.read_csv(DataConfig.DATA_FOLDER + DataConfig.DATA_FILE)
    else:
        # Используем базовый датасет
        logger.info("📊 Загрузка базового датасета...")
        df = pd.read_csv(DataConfig.DATA_FOLDER + DataConfig.DATA_FILE)
    
    logger.info(f"📊 Исходные данные: {len(df):,} записей")
    
    # Добавляем расширенные технические индикаторы
    logger.info("🔧 Добавление расширенных технических индикаторов...")
    start_time = time.time()
    
    df = add_advanced_features(df)
    
    indicator_time = time.time() - start_time
    logger.info(f"✅ Добавлено {len(df.columns) - 6} технических индикаторов за {indicator_time:.1f}с")
    
    # Обработка данных
    data_processor = DataProcessor()
    # Простая обработка - добавляем целевую переменную
    df['future_return'] = df['close'].pct_change(5).shift(-5).fillna(0)
    
    # Базовая статистика данных
    quality_report = {
        'total_records': len(df),
        'features_count': len(df.columns) - 6,  # Исключаем базовые OHLCV колонки
        'price_statistics': {
            'min_price': df['close'].min(),
            'max_price': df['close'].max()
        }
    }
    logger.info(f"📈 Качество данных:")
    logger.info(f"   📊 Всего записей: {quality_report['total_records']:,}")
    logger.info(f"   🔧 Признаков: {quality_report['features_count']}")
    logger.info(f"   💰 Диапазон цен: ${quality_report['price_statistics']['min_price']:.2f} - ${quality_report['price_statistics']['max_price']:.2f}")
    
    return df


def create_advanced_environments(df: pd.DataFrame) -> tuple:
    """Создание продвинутых торговых окружений"""
    logger = logging.getLogger(__name__)
    
    logger.info("🎮 СОЗДАНИЕ ПРОДВИНУТЫХ ТОРГОВЫХ ОКРУЖЕНИЙ")
    logger.info("-" * 50)
    
    # Разделяем данные с учетом времени
    data_processor = DataProcessor()
    train_df, test_df = data_processor.split_data_for_walk_forward(df, 0.8)
    
    logger.info(f"📊 Разделение данных:")
    logger.info(f"   🎓 Обучение: {len(train_df):,} записей")
    logger.info(f"   🧪 Тестирование: {len(test_df):,} записей")
    
    # Создаем основные окружения
    train_env = ImprovedTradingEnv(
        train_df,
        window_size=DataConfig.WINDOW_SIZE,
        initial_balance=SystemConfig.INITIAL_BALANCE,
        validation_mode=False
    )
    
    test_env = ImprovedTradingEnv(
        test_df,
        window_size=DataConfig.WINDOW_SIZE,
        initial_balance=SystemConfig.INITIAL_BALANCE,
        validation_mode=True
    )
    
    # Создаем ensemble окружения с разными параметрами
    ensemble_envs = []
    
    # Консервативное окружение
    conservative_env = ImprovedTradingEnv(
        train_df,
        window_size=DataConfig.WINDOW_SIZE,
        initial_balance=SystemConfig.INITIAL_BALANCE,
        validation_mode=False
    )
    conservative_env.max_position_size = 0.5  # Меньший размер позиции
    ensemble_envs.append(conservative_env)
    
    # Агрессивное окружение
    aggressive_env = ImprovedTradingEnv(
        train_df,
        window_size=DataConfig.WINDOW_SIZE,
        initial_balance=SystemConfig.INITIAL_BALANCE,
        validation_mode=False
    )
    aggressive_env.max_position_size = 0.95  # Больший размер позиции
    ensemble_envs.append(aggressive_env)
    
    # Окружение для скальпинга
    scalping_env = ImprovedTradingEnv(
        train_df,
        window_size=DataConfig.WINDOW_SIZE // 2,  # Меньшее окно
        initial_balance=SystemConfig.INITIAL_BALANCE,
        validation_mode=False
    )
    ensemble_envs.append(scalping_env)
    
    # Векторизуем для stable-baselines3
    train_env_vec = DummyVecEnv([lambda: train_env])
    ensemble_envs_vec = [DummyVecEnv([lambda env=env: env]) for env in ensemble_envs]
    
    logger.info(f"✅ Создано окружений:")
    logger.info(f"   🎯 Основное обучающее: 1")
    logger.info(f"   🧪 Тестовое: 1") 
    logger.info(f"   🤖 Ensemble: {len(ensemble_envs)}")
    
    return train_env_vec, test_env, ensemble_envs_vec, train_df, test_df


def train_advanced_ensemble_system(train_data: pd.DataFrame, ensemble_envs: list) -> HybridEnsemble:
    """Обучение продвинутой ensemble системы"""
    logger = logging.getLogger(__name__)
    
    logger.info("🧠 ОБУЧЕНИЕ ПРОДВИНУТОЙ ENSEMBLE СИСТЕМЫ")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    # Создаем ensemble систему
    hybrid_ensemble = create_ensemble_system(train_data, ensemble_envs)
    
    training_time = time.time() - start_time
    logger.info(f"✅ Ensemble система обучена за {training_time:.1f}с")
    
    return hybrid_ensemble


def run_multi_timeframe_analysis(df: pd.DataFrame) -> dict:
    """Запуск многомасштабного анализа"""
    logger = logging.getLogger(__name__)
    
    logger.info("🕰️ МНОГОМАСШТАБНЫЙ АНАЛИЗ РЫНКА")
    logger.info("=" * 50)
    
    # Анализируем последние данные для актуального состояния рынка
    recent_data = df.tail(5000).copy()  # Последние ~17 дней данных
    
    timeframes = ['15m', '1h', '4h', '1d']
    
    start_time = time.time()
    analysis_result = analyze_multi_timeframe_market(recent_data, timeframes)
    analysis_time = time.time() - start_time
    
    if analysis_result:
        consensus = analysis_result.get('consensus', {})
        market_regime = analysis_result.get('market_regime', {})
        
        logger.info(f"🎯 Результаты многомасштабного анализа (за {analysis_time:.1f}с):")
        logger.info(f"   📊 Консенсусное действие: {['Hold', 'Buy', 'Sell'][consensus.get('action', 0)]}")
        logger.info(f"   🎯 Уверенность: {consensus.get('confidence', 0):.1%}")
        logger.info(f"   🏛️ Рыночный режим: {market_regime.get('regime', 'unknown')}")
        logger.info(f"   📈 Долгосрочный тренд: {market_regime.get('long_term', 'unknown')}")
        
    return analysis_result


def run_walk_forward_validation(df: pd.DataFrame, ensemble_system: HybridEnsemble) -> dict:
    """Запуск walk-forward валидации"""
    logger = logging.getLogger(__name__)
    
    logger.info("📈 WALK-FORWARD ВАЛИДАЦИЯ")
    logger.info("=" * 50)
    
    def model_factory(train_data):
        """Фабрика моделей для walk-forward теста"""
        try:
            # Создаем быструю версию ensemble для валидации
            mini_ensemble = HybridEnsemble()
            mini_ensemble.ml_ensemble.train_base_models(train_data)
            mini_ensemble.is_trained = True
            return mini_ensemble
        except Exception as e:
            logger.error(f"❌ Ошибка создания модели: {e}")
            return None
    
    def environment_factory(test_data, **kwargs):
        """Фабрика окружений для walk-forward теста"""
        return ImprovedTradingEnv(
            test_data,
            window_size=DataConfig.WINDOW_SIZE,
            initial_balance=kwargs.get('initial_balance', SystemConfig.INITIAL_BALANCE),
            validation_mode=True
        )
    
    # Используем более быстрые параметры для демонстрации
    validator = WalkForwardValidator(
        train_window_months=6,  # 6 месяцев обучения
        test_window_months=1,   # 1 месяц тестирования
        step_months=1,          # Шаг 1 месяц
        min_trades=5            # Минимум 5 сделок
    )
    
    start_time = time.time()
    
    # Запускаем walk-forward анализ
    results = validator.run_walk_forward_test(
        df, 
        model_factory, 
        environment_factory,
        {'initial_balance': SystemConfig.INITIAL_BALANCE, 'commission': 0.001}
    )
    
    validation_time = time.time() - start_time
    
    if results:
        report = validator.generate_report('walk_forward_results')
        
        logger.info(f"📊 Walk-Forward результаты (за {validation_time:.1f}с):")
        logger.info(f"   📈 Периодов протестировано: {report['total_periods']}")
        logger.info(f"   💰 Средняя доходность: {report['avg_return']:+.2%}")
        logger.info(f"   🎯 Винрейт периодов: {report['win_rate_periods']:.1%}")
        logger.info(f"   📊 Кумулятивная доходность: {report['cumulative_return']:+.2%}")
        
        return report
    
    logger.warning("⚠️ Walk-Forward валидация не дала результатов")
    return {}


def run_comprehensive_backtesting(ensemble_system: HybridEnsemble, test_env, 
                                 multi_tf_analysis: dict, data_processor: DataProcessor) -> dict:
    """Комплексное тестирование с использованием всех улучшений"""
    logger = logging.getLogger(__name__)
    results_logger = logging.getLogger('RESULTS')
    
    logger.info("🧪 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ПРОДВИНУТОЙ СИСТЕМЫ")
    logger.info("=" * 50)
    
    obs, _ = test_env.reset()
    results = {
        'balance_history': [],
        'prices': [],
        'actions': [],
        'trades': [],
        'ensemble_signals': [],
        'multi_tf_signals': [],
        'market_regimes': [],
        'risk_levels': []
    }
    
    step_count = 0
    max_steps = min(2000, len(test_env.df) - test_env.window_size - 10)
    
    # Инициализируем многомасштабный анализатор
    tf_analyzer = multi_tf_analysis.get('analyzer')
    
    logger.info(f"🎯 Запуск комплексного тестирования: {max_steps} шагов")
    
    while step_count < max_steps:
        try:
            # === ПОЛУЧЕНИЕ СИГНАЛОВ ===
            
            # 1. Ensemble сигнал
            current_data = test_env.df.iloc[test_env.current_step-10:test_env.current_step+1]
            
            if len(current_data) > 5 and ensemble_system.is_trained:
                ensemble_action = ensemble_system.predict_hybrid_action(obs, current_data)
            else:
                ensemble_action = 0  # Hold
            
            # 2. Многомасштабный сигнал (каждые 10 шагов для экономии времени)
            if step_count % 10 == 0 and tf_analyzer:
                recent_data = test_env.df.iloc[max(0, test_env.current_step-200):test_env.current_step+1]
                if len(recent_data) > 50:
                    tf_signals = tf_analyzer.analyze_multiple_timeframes(recent_data, ['15m', '1h'])
                    tf_consensus = tf_analyzer.generate_consensus_signal(tf_signals)
                    tf_action = tf_consensus.get('action', 0)
                    market_regime = tf_analyzer.get_market_regime(tf_signals)
                else:
                    tf_action = 0
                    tf_consensus = {'action': 0, 'confidence': 0, 'risk_level': 'medium'}
                    market_regime = {'regime': 'unknown'}
            else:
                # Используем последний сигнал
                tf_action = results['multi_tf_signals'][-1] if results['multi_tf_signals'] else 0
                tf_consensus = {'risk_level': 'medium'}
                market_regime = {'regime': 'unknown'}
            
            # === ПРИНЯТИЕ РЕШЕНИЯ ===
            
            # Комбинируем сигналы (приоритет ensemble)
            if ensemble_action != 0:
                final_action = ensemble_action
            elif tf_action != 0:
                final_action = tf_action
            else:
                final_action = 0  # Hold
            
            # Модификация действия на основе риска
            risk_level = tf_consensus.get('risk_level', 'medium')
            if risk_level == 'high' and final_action != 0:
                # В условиях высокого риска более консервативный подход
                if np.random.random() < 0.3:  # 30% шанс отмены торговли
                    final_action = 0
            
            # === ВЫПОЛНЕНИЕ ДЕЙСТВИЯ ===
            
            obs, reward, done, truncated, info = test_env.step(final_action)
            step_count += 1
            
            # === СБОР СТАТИСТИКИ ===
            
            portfolio_value = info.get('portfolio_value', test_env.initial_balance)
            current_price = info.get('current_price', 0)
            
            results['balance_history'].append(portfolio_value)
            results['prices'].append(current_price)
            results['actions'].append(final_action)
            results['ensemble_signals'].append(ensemble_action)
            results['multi_tf_signals'].append(tf_action)
            results['market_regimes'].append(market_regime.get('regime', 'unknown'))
            results['risk_levels'].append(risk_level)
            
            # Отслеживаем сделки
            if 'trade_info' in info and info['trade_info'].get('executed'):
                trade_info = info['trade_info'].copy()
                trade_info['step'] = step_count
                trade_info['market_regime'] = market_regime.get('regime', 'unknown')
                trade_info['risk_level'] = risk_level
                results['trades'].append(trade_info)
            
            # Периодические отчеты
            if step_count % 500 == 0:
                current_return = (portfolio_value - test_env.initial_balance) / test_env.initial_balance
                logger.info(f"   📊 Шаг {step_count:,}/{max_steps:,} | "
                          f"Доходность: {current_return:+.2%} | "
                          f"Режим: {market_regime.get('regime', 'unknown')}")
            
            if done or truncated:
                obs, _ = test_env.reset()
                
        except Exception as e:
            logger.error(f"❌ Ошибка на шаге {step_count}: {e}")
            break
    
    # === ФИНАЛЬНАЯ СТАТИСТИКА ===
    
    final_balance = results['balance_history'][-1] if results['balance_history'] else test_env.initial_balance
    total_return = (final_balance - test_env.initial_balance) / test_env.initial_balance
    
    # Анализ сигналов
    ensemble_trades = sum(1 for action in results['ensemble_signals'] if action != 0)
    tf_trades = sum(1 for action in results['multi_tf_signals'] if action != 0)
    
    # Анализ рыночных режимов
    regime_counts = {}
    for regime in results['market_regimes']:
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else 'unknown'
    
    results_logger.info("🎯 РЕЗУЛЬТАТЫ КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ:")
    results_logger.info(f"💰 Итоговая доходность: {total_return:+.2%}")
    results_logger.info(f"💼 Финальный баланс: {final_balance:,.2f} USDT")
    results_logger.info(f"🤖 Ensemble сигналов: {ensemble_trades}")
    results_logger.info(f"🕰️ Многомасштабных сигналов: {tf_trades}")
    results_logger.info(f"🏛️ Доминирующий режим: {dominant_regime}")
    results_logger.info(f"🔄 Всего сделок: {len(results['trades'])}")
    
    # Добавляем метрики в результаты
    results.update({
        'final_balance': final_balance,
        'total_return': total_return,
        'ensemble_signals_count': ensemble_trades,
        'tf_signals_count': tf_trades,
        'dominant_regime': dominant_regime,
        'regime_distribution': regime_counts
    })
    
    return results


def generate_comprehensive_report(test_results: dict, walk_forward_results: dict, 
                                multi_tf_analysis: dict) -> dict:
    """Генерация комплексного отчета"""
    logger = logging.getLogger(__name__)
    results_logger = logging.getLogger('RESULTS')
    
    logger.info("📊 ГЕНЕРАЦИЯ КОМПЛЕКСНОГО ОТЧЕТА")
    logger.info("=" * 50)
    
    # Анализируем результаты
    analyzer = PerformanceAnalyzer()
    detailed_analysis = analyzer.generate_full_analysis(test_results, SystemConfig.INITIAL_BALANCE)
    
    # Создаем итоговый отчет
    comprehensive_report = {
        'timestamp': datetime.now().isoformat(),
        'system_version': '3.0_Advanced',
        
        # Основные метрики
        'performance': {
            'total_return': test_results.get('total_return', 0),
            'final_balance': test_results.get('final_balance', SystemConfig.INITIAL_BALANCE),
            'sharpe_ratio': detailed_analysis.get('sharpe_ratio', 0),
            'max_drawdown': detailed_analysis.get('max_drawdown', 0),
            'win_rate': detailed_analysis.get('win_rate', 0),
            'total_trades': len(test_results.get('trades', []))
        },
        
        # Walk-forward метрики
        'walk_forward': {
            'periods_tested': walk_forward_results.get('total_periods', 0),
            'avg_period_return': walk_forward_results.get('avg_return', 0),
            'period_win_rate': walk_forward_results.get('win_rate_periods', 0),
            'cumulative_return': walk_forward_results.get('cumulative_return', 0)
        },
        
        # Многомасштабный анализ
        'multi_timeframe': {
            'consensus_action': multi_tf_analysis.get('consensus', {}).get('action', 0),
            'consensus_confidence': multi_tf_analysis.get('consensus', {}).get('confidence', 0),
            'market_regime': multi_tf_analysis.get('market_regime', {}).get('regime', 'unknown'),
            'dominant_regime': test_results.get('dominant_regime', 'unknown')
        },
        
        # Системная информация
        'system_info': {
            'ensemble_signals': test_results.get('ensemble_signals_count', 0),
            'tf_signals': test_results.get('tf_signals_count', 0),
            'data_records': len(test_results.get('balance_history', [])),
            'regime_distribution': test_results.get('regime_distribution', {})
        }
    }
    
    # Выводим итоговый отчет
    results_logger.info("\n🎯 ИТОГОВЫЙ ОТЧЕТ ПРОДВИНУТОЙ СИСТЕМЫ АЛГОТРЕЙДИНГА V3.0")
    results_logger.info("=" * 80)
    
    results_logger.info("📈 ПРОИЗВОДИТЕЛЬНОСТЬ:")
    results_logger.info(f"   💰 Общая доходность: {comprehensive_report['performance']['total_return']:+.2%}")
    results_logger.info(f"   💼 Финальный баланс: {comprehensive_report['performance']['final_balance']:,.2f} USDT")
    results_logger.info(f"   📊 Sharpe Ratio: {comprehensive_report['performance']['sharpe_ratio']:.3f}")
    results_logger.info(f"   ⚠️  Макс. просадка: {comprehensive_report['performance']['max_drawdown']:.1%}")
    results_logger.info(f"   🎯 Винрейт: {comprehensive_report['performance']['win_rate']:.1%}")
    results_logger.info(f"   🔄 Всего сделок: {comprehensive_report['performance']['total_trades']}")
    
    if comprehensive_report['walk_forward']['periods_tested'] > 0:
        results_logger.info("\n📈 WALK-FORWARD ВАЛИДАЦИЯ:")
        results_logger.info(f"   📊 Периодов: {comprehensive_report['walk_forward']['periods_tested']}")
        results_logger.info(f"   💰 Средняя доходность: {comprehensive_report['walk_forward']['avg_period_return']:+.2%}")
        results_logger.info(f"   🎯 Винрейт периодов: {comprehensive_report['walk_forward']['period_win_rate']:.1%}")
        results_logger.info(f"   📈 Кумулятивная доходность: {comprehensive_report['walk_forward']['cumulative_return']:+.2%}")
    
    results_logger.info("\n🕰️ МНОГОМАСШТАБНЫЙ АНАЛИЗ:")
    action_names = ['Hold', 'Buy', 'Sell']
    results_logger.info(f"   🎯 Консенсус: {action_names[comprehensive_report['multi_timeframe']['consensus_action']]}")
    results_logger.info(f"   📊 Уверенность: {comprehensive_report['multi_timeframe']['consensus_confidence']:.1%}")
    results_logger.info(f"   🏛️ Рыночный режим: {comprehensive_report['multi_timeframe']['market_regime']}")
    
    results_logger.info("\n🤖 СИСТЕМА:")
    results_logger.info(f"   🧠 Ensemble сигналов: {comprehensive_report['system_info']['ensemble_signals']}")
    results_logger.info(f"   🕰️ Многомасштабных сигналов: {comprehensive_report['system_info']['tf_signals']}")
    results_logger.info(f"   📊 Обработано данных: {comprehensive_report['system_info']['data_records']:,}")
    
    # Оценка качества системы
    score = 0
    if comprehensive_report['performance']['total_return'] > 0:
        score += 20
    if comprehensive_report['performance']['sharpe_ratio'] > 0.5:
        score += 20
    if comprehensive_report['performance']['max_drawdown'] < 0.2:
        score += 20
    if comprehensive_report['walk_forward']['period_win_rate'] > 0.5:
        score += 20
    if comprehensive_report['multi_timeframe']['consensus_confidence'] > 0.5:
        score += 20
    
    results_logger.info(f"\n🏆 ОБЩАЯ ОЦЕНКА СИСТЕМЫ: {score}/100")
    
    if score >= 80:
        results_logger.info("🟢 ОТЛИЧНО! Система показывает высокую производительность")
    elif score >= 60:
        results_logger.info("🟡 ХОРОШО! Система работает стабильно")
    elif score >= 40:
        results_logger.info("🟠 УДОВЛЕТВОРИТЕЛЬНО! Есть потенциал для улучшения")
    else:
        results_logger.info("🔴 ТРЕБУЕТ ДОРАБОТКИ! Необходимо пересмотреть стратегию")
    
    return comprehensive_report


def main():
    """
    🚀 ГЛАВНАЯ ФУНКЦИЯ ПРОДВИНУТОЙ СИСТЕМЫ АЛГОТРЕЙДИНГА
    """
    # === ИНИЦИАЛИЗАЦИЯ ===
    setup_advanced_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 ЗАПУСК ПРОДВИНУТОЙ СИСТЕМЫ АЛГОТРЕЙДИНГА V3.0")
    logger.info("=" * 80)
    logger.info("✨ Новые возможности:")
    logger.info("   📊 7+ лет данных (789K+ записей)")
    logger.info("   🔧 50+ технических индикаторов") 
    logger.info("   🧠 Ensemble ML + RL модели")
    logger.info("   📈 Walk-forward валидация")
    logger.info("   🕰️ Многомасштабный анализ")
    logger.info("   🏛️ Анализ рыночных режимов")
    logger.info("=" * 80)
    
    total_start_time = time.time()
    
    try:
        # === ЭТАП 1: ПОДГОТОВКА ДАННЫХ ===
        logger.info("\n📊 ЭТАП 1: ПОДГОТОВКА РАСШИРЕННОГО ДАТАСЕТА")
        df = prepare_advanced_data(use_extended_dataset=True)
        
        # === ЭТАП 2: СОЗДАНИЕ ОКРУЖЕНИЙ ===
        logger.info("\n🎮 ЭТАП 2: СОЗДАНИЕ ПРОДВИНУТЫХ ОКРУЖЕНИЙ")
        train_env_vec, test_env, ensemble_envs, train_df, test_df = create_advanced_environments(df)
        
        # === ЭТАП 3: ОБУЧЕНИЕ ENSEMBLE СИСТЕМЫ ===
        logger.info("\n🧠 ЭТАП 3: ОБУЧЕНИЕ ENSEMBLE СИСТЕМЫ")
        ensemble_system = train_advanced_ensemble_system(train_df, ensemble_envs)
        
        # === ЭТАП 4: МНОГОМАСШТАБНЫЙ АНАЛИЗ ===
        logger.info("\n🕰️ ЭТАП 4: МНОГОМАСШТАБНЫЙ АНАЛИЗ")
        multi_tf_analysis = run_multi_timeframe_analysis(df)
        
        # === ЭТАП 5: WALK-FORWARD ВАЛИДАЦИЯ ===
        logger.info("\n📈 ЭТАП 5: WALK-FORWARD ВАЛИДАЦИЯ")
        walk_forward_results = run_walk_forward_validation(df, ensemble_system)
        
        # === ЭТАП 6: КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ===
        logger.info("\n🧪 ЭТАП 6: КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ")
        data_processor = DataProcessor()
        test_results = run_comprehensive_backtesting(
            ensemble_system, test_env, multi_tf_analysis, data_processor
        )
        
        # === ЭТАП 7: ГЕНЕРАЦИЯ ОТЧЕТА ===
        logger.info("\n📊 ЭТАП 7: ИТОГОВЫЙ ОТЧЕТ")
        comprehensive_report = generate_comprehensive_report(
            test_results, walk_forward_results, multi_tf_analysis
        )
        
        # === ЗАКЛЮЧЕНИЕ ===
        total_time = time.time() - total_start_time
        
        logger.info(f"\n🎉 ПРОДВИНУТАЯ СИСТЕМА АЛГОТРЕЙДИНГА ЗАВЕРШЕНА!")
        logger.info(f"⏱️ Общее время выполнения: {total_time:.1f}с")
        logger.info(f"📊 Создано файлов отчетов: walk_forward_results/, multi_timeframe_analysis.png, trading_analysis.png")
        logger.info(f"💾 Логи сохранены в папке: logs/")
        
        return {
            'ensemble_system': ensemble_system,
            'test_results': test_results,
            'walk_forward_results': walk_forward_results,
            'multi_tf_analysis': multi_tf_analysis,
            'comprehensive_report': comprehensive_report
        }
        
    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    main() 