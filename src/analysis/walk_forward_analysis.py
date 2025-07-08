"""
📈 WALK-FORWARD АНАЛИЗ
Правильная валидация моделей алготрейдинга на временных рядах
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
# sharpe_ratio не существует в sklearn, будем вычислять вручную
import warnings

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Результат walk-forward теста"""
    period_start: datetime
    period_end: datetime
    returns: List[float]
    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    model_stats: Dict[str, Any]


class WalkForwardValidator:
    """Валидатор для walk-forward анализа алготрейдинга"""
    
    def __init__(self, train_window_months: int = 12, test_window_months: int = 3, 
                 step_months: int = 1, min_trades: int = 10):
        """
        Args:
            train_window_months: Размер окна обучения в месяцах
            test_window_months: Размер окна тестирования в месяцах  
            step_months: Шаг сдвига окна в месяцах
            min_trades: Минимальное количество сделок для валидного периода
        """
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_months = step_months
        self.min_trades = min_trades
        
        self.results = []
        self.cumulative_returns = []
        self.model_performance_history = []
    
    def split_data_by_date(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Разделение данных по датам для walk-forward анализа"""
        
        logger.info("📅 Создание walk-forward разделений...")
        
        # Преобразуем timestamp в datetime
        if df[timestamp_col].dtype != 'datetime64[ns]':
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
        
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        start_date = df[timestamp_col].min()
        end_date = df[timestamp_col].max()
        
        logger.info(f"📊 Период данных: {start_date.date()} - {end_date.date()}")
        logger.info(f"🔄 Параметры: train={self.train_window_months}м, test={self.test_window_months}м, step={self.step_months}м")
        
        splits = []
        current_date = start_date + pd.DateOffset(months=self.train_window_months)
        
        while current_date + pd.DateOffset(months=self.test_window_months) <= end_date:
            # Определяем границы окон
            train_start = current_date - pd.DateOffset(months=self.train_window_months)
            train_end = current_date
            test_start = current_date
            test_end = current_date + pd.DateOffset(months=self.test_window_months)
            
            # Фильтруем данные
            train_mask = (df[timestamp_col] >= train_start) & (df[timestamp_col] < train_end)
            test_mask = (df[timestamp_col] >= test_start) & (df[timestamp_col] < test_end)
            
            train_data = df[train_mask].copy()
            test_data = df[test_mask].copy()
            
            if len(train_data) > 0 and len(test_data) > 0:
                splits.append((train_data, test_data))
                logger.info(f"📈 Период {len(splits)}: train={len(train_data)} записей, test={len(test_data)} записей")
            
            # Сдвигаем окно
            current_date += pd.DateOffset(months=self.step_months)
        
        logger.info(f"✅ Создано {len(splits)} walk-forward разделений")
        return splits
    
    def run_walk_forward_test(self, df: pd.DataFrame, model_factory: Callable, 
                            environment_factory: Callable = None, 
                            trading_params: Dict = None) -> List[WalkForwardResult]:
        """
        Запуск walk-forward теста
        
        Args:
            df: Исходные данные
            model_factory: Функция создания модели (train_data, **kwargs) -> model
            environment_factory: Функция создания торгового окружения (optional)
            trading_params: Параметры торговли
        """
        
        logger.info("🚀 ЗАПУСК WALK-FORWARD АНАЛИЗА")
        logger.info("=" * 50)
        
        if trading_params is None:
            trading_params = {
                'initial_balance': 10000,
                'commission': 0.001,
                'max_position_size': 0.95
            }
        
        # Создаем разделения данных
        data_splits = self.split_data_by_date(df)
        
        if not data_splits:
            logger.error("❌ Не удалось создать разделения данных")
            return []
        
        self.results = []
        cumulative_balance = trading_params['initial_balance']
        
        for i, (train_data, test_data) in enumerate(data_splits):
            logger.info(f"\n📊 ПЕРИОД {i+1}/{len(data_splits)}")
            logger.info("-" * 30)
            
            period_start = test_data['timestamp'].min()
            period_end = test_data['timestamp'].max()
            
            logger.info(f"📅 Тест период: {period_start.date()} - {period_end.date()}")
            logger.info(f"📈 Обучение: {len(train_data)} записей, Тест: {len(test_data)} записей")
            
            try:
                # Обучаем модель
                logger.info("🎓 Обучение модели...")
                model = model_factory(train_data)
                
                # Тестируем модель
                logger.info("🧪 Тестирование модели...")
                period_result = self._test_period(
                    model, test_data, period_start, period_end, 
                    trading_params, environment_factory
                )
                
                if period_result and period_result.total_trades >= self.min_trades:
                    self.results.append(period_result)
                    
                    # Обновляем кумулятивный баланс
                    cumulative_balance *= (1 + period_result.total_return)
                    self.cumulative_returns.append(cumulative_balance)
                    
                    logger.info(f"✅ Результат периода:")
                    logger.info(f"   💰 Доходность: {period_result.total_return:+.2%}")
                    logger.info(f"   📊 Sharpe: {period_result.sharpe_ratio:.3f}")
                    logger.info(f"   ⚠️  Просадка: {period_result.max_drawdown:.1%}")
                    logger.info(f"   🎯 Винрейт: {period_result.win_rate:.1%}")
                    logger.info(f"   🔄 Сделок: {period_result.total_trades}")
                    logger.info(f"   💼 Кумулятивный баланс: {cumulative_balance:,.2f}")
                else:
                    logger.warning(f"⚠️ Период пропущен: недостаточно сделок ({period_result.total_trades if period_result else 0} < {self.min_trades})")
                
            except Exception as e:
                logger.error(f"❌ Ошибка в периоде {i+1}: {e}")
                continue
        
        logger.info(f"\n🎯 ИТОГИ WALK-FORWARD АНАЛИЗА")
        logger.info("=" * 50)
        logger.info(f"📊 Протестированных периодов: {len(self.results)}")
        
        if self.results:
            total_return = (cumulative_balance / trading_params['initial_balance']) - 1
            avg_period_return = np.mean([r.total_return for r in self.results])
            win_rate = np.mean([r.total_return > 0 for r in self.results])
            
            logger.info(f"💰 Общая доходность: {total_return:+.2%}")
            logger.info(f"📈 Средняя доходность периода: {avg_period_return:+.2%}")
            logger.info(f"🎯 Винрейт периодов: {win_rate:.1%}")
            logger.info(f"💼 Финальный баланс: {cumulative_balance:,.2f}")
        
        return self.results
    
    def _test_period(self, model: Any, test_data: pd.DataFrame, 
                    period_start: datetime, period_end: datetime,
                    trading_params: Dict, environment_factory: Callable = None) -> Optional[WalkForwardResult]:
        """Тестирование модели на одном периоде"""
        
        try:
            if environment_factory:
                # Используем торговое окружение если предоставлено
                env = environment_factory(test_data, **trading_params)
                return self._test_with_environment(model, env, period_start, period_end)
            else:
                # Простое тестирование по сигналам
                return self._test_with_signals(model, test_data, period_start, period_end, trading_params)
        
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования периода: {e}")
            return None
    
    def _test_with_environment(self, model: Any, env: Any, 
                              period_start: datetime, period_end: datetime) -> WalkForwardResult:
        """Тестирование с торговым окружением"""
        
        obs, _ = env.reset()
        returns = []
        balance_history = [env.initial_balance]
        
        done = False
        while not done:
            try:
                # Получаем действие от модели
                if hasattr(model, 'predict'):
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # Для простых моделей
                    action = 0  # Hold
                
                # Выполняем шаг
                obs, reward, done, truncated, info = env.step(action)
                
                # Собираем статистику
                portfolio_value = info.get('portfolio_value', env.initial_balance)
                balance_history.append(portfolio_value)
                
                if len(balance_history) > 1:
                    period_return = (balance_history[-1] - balance_history[-2]) / balance_history[-2]
                    returns.append(period_return)
                
                if done or truncated:
                    break
                    
            except Exception as e:
                logger.error(f"❌ Ошибка в шаге: {e}")
                break
        
        # Вычисляем метрики
        total_return = (balance_history[-1] - balance_history[0]) / balance_history[0]
        volatility = np.std(returns) * np.sqrt(288) if returns else 0  # Для 5-мин данных
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(288) if returns and np.std(returns) > 0 else 0
        
        # Максимальная просадка
        peak = balance_history[0]
        max_dd = 0
        for balance in balance_history:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Статистика сделок
        trades = getattr(env, 'trades', [])
        win_rate = np.mean([trade.get('profit', 0) > 0 for trade in trades]) if trades else 0
        avg_trade_return = np.mean([trade.get('profit', 0) for trade in trades]) if trades else 0
        
        return WalkForwardResult(
            period_start=period_start,
            period_end=period_end,
            returns=returns,
            total_return=total_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_trade_return=avg_trade_return,
            model_stats={}
        )
    
    def _test_with_signals(self, model: Any, test_data: pd.DataFrame, 
                          period_start: datetime, period_end: datetime,
                          trading_params: Dict) -> WalkForwardResult:
        """Простое тестирование по торговым сигналам"""
        
        balance = trading_params['initial_balance']
        position = 0  # 0 = no position, 1 = long
        trades = []
        returns = []
        balance_history = [balance]
        
        commission = trading_params.get('commission', 0.001)
        max_position_size = trading_params.get('max_position_size', 0.95)
        
        for i in range(len(test_data) - 1):
            current_row = test_data.iloc[i]
            next_row = test_data.iloc[i + 1]
            
            current_price = current_row['close']
            next_price = next_row['close']
            
            try:
                # Получаем сигнал от модели
                if hasattr(model, 'predict'):
                    # Для ML моделей
                    signal = model.predict(test_data.iloc[i:i+1])
                    action = 1 if signal > 0.01 else (2 if signal < -0.01 else 0)
                else:
                    # Простая стратегия
                    action = 0
                
                # Торговая логика
                if action == 1 and position == 0:  # Buy signal
                    # Открываем позицию
                    position_size = balance * max_position_size
                    shares = position_size / (current_price * (1 + commission))
                    cost = shares * current_price * (1 + commission)
                    
                    if cost <= balance:
                        balance -= cost
                        position = shares
                        trades.append({
                            'type': 'buy',
                            'price': current_price,
                            'shares': shares,
                            'timestamp': current_row['timestamp']
                        })
                
                elif action == 2 and position > 0:  # Sell signal
                    # Закрываем позицию
                    revenue = position * current_price * (1 - commission)
                    balance += revenue
                    
                    # Вычисляем прибыль сделки
                    if trades and trades[-1]['type'] == 'buy':
                        buy_cost = trades[-1]['shares'] * trades[-1]['price'] * (1 + commission)
                        trade_profit = revenue - buy_cost
                        trades.append({
                            'type': 'sell',
                            'price': current_price,
                            'shares': position,
                            'profit': trade_profit,
                            'timestamp': current_row['timestamp']
                        })
                    
                    position = 0
                
                # Вычисляем текущую стоимость портфеля
                portfolio_value = balance + (position * next_price if position > 0 else 0)
                balance_history.append(portfolio_value)
                
                # Доходность
                if len(balance_history) > 1:
                    period_return = (balance_history[-1] - balance_history[-2]) / balance_history[-2]
                    returns.append(period_return)
                    
            except Exception as e:
                logger.warning(f"⚠️ Ошибка в торговле: {e}")
                continue
        
        # Закрываем позицию в конце периода
        if position > 0:
            final_price = test_data.iloc[-1]['close']
            revenue = position * final_price * (1 - commission)
            balance += revenue
            
            if trades and trades[-1]['type'] == 'buy':
                buy_cost = trades[-1]['shares'] * trades[-1]['price'] * (1 + commission)
                trade_profit = revenue - buy_cost
                trades.append({
                    'type': 'sell',
                    'price': final_price,
                    'shares': position,
                    'profit': trade_profit,
                    'timestamp': test_data.iloc[-1]['timestamp']
                })
        
        # Метрики
        final_balance = balance_history[-1] if balance_history else trading_params['initial_balance']
        total_return = (final_balance - trading_params['initial_balance']) / trading_params['initial_balance']
        
        volatility = np.std(returns) * np.sqrt(288) if returns else 0
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(288) if returns and np.std(returns) > 0 else 0
        
        # Максимальная просадка
        peak = trading_params['initial_balance']
        max_dd = 0
        for balance_val in balance_history:
            if balance_val > peak:
                peak = balance_val
            dd = (peak - balance_val) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Статистика сделок
        completed_trades = [t for t in trades if 'profit' in t]
        win_rate = np.mean([t['profit'] > 0 for t in completed_trades]) if completed_trades else 0
        avg_trade_return = np.mean([t['profit'] for t in completed_trades]) if completed_trades else 0
        
        return WalkForwardResult(
            period_start=period_start,
            period_end=period_end,
            returns=returns,
            total_return=total_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=len(completed_trades),
            avg_trade_return=avg_trade_return,
            model_stats={}
        )
    
    def generate_report(self, save_path: str = None) -> Dict[str, Any]:
        """Генерация отчета по walk-forward анализу"""
        
        if not self.results:
            logger.error("❌ Нет результатов для генерации отчета")
            return {}
        
        logger.info("📊 Генерация walk-forward отчета...")
        
        # Агрегированные метрики
        returns = [r.total_return for r in self.results]
        sharpe_ratios = [r.sharpe_ratio for r in self.results]
        max_drawdowns = [r.max_drawdown for r in self.results]
        win_rates = [r.win_rate for r in self.results]
        
        report = {
            'total_periods': len(self.results),
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'return_std': np.std(returns),
            'best_period': max(returns),
            'worst_period': min(returns),
            'positive_periods': sum(1 for r in returns if r > 0),
            'win_rate_periods': np.mean([r > 0 for r in returns]),
            
            'avg_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'best_sharpe': max(sharpe_ratios),
            'worst_sharpe': min(sharpe_ratios),
            
            'avg_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': max(max_drawdowns),
            
            'avg_win_rate': np.mean(win_rates),
            'total_trades': sum(r.total_trades for r in self.results),
            
            'cumulative_return': (self.cumulative_returns[-1] / self.cumulative_returns[0] - 1) if self.cumulative_returns else 0,
            'periods_data': self.results
        }
        
        # Создаем визуализации
        if save_path:
            self._create_visualizations(report, save_path)
        
        # Выводим сводку
        logger.info("📈 СВОДКА WALK-FORWARD АНАЛИЗА:")
        logger.info(f"📊 Периодов: {report['total_periods']}")
        logger.info(f"💰 Средняя доходность: {report['avg_return']:+.2%}")
        logger.info(f"📈 Кумулятивная доходность: {report['cumulative_return']:+.2%}")
        logger.info(f"🎯 Винрейт периодов: {report['win_rate_periods']:.1%}")
        logger.info(f"📊 Средний Sharpe: {report['avg_sharpe']:.3f}")
        logger.info(f"⚠️  Средняя просадка: {report['avg_max_drawdown']:.1%}")
        
        return report
    
    def _create_visualizations(self, report: Dict, save_path: str) -> None:
        """Создание графиков walk-forward анализа"""
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        plt.style.use('default')
        
        # 1. Кумулятивная доходность
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        if self.cumulative_returns:
            ax1.plot(self.cumulative_returns, linewidth=2, color='blue')
            ax1.set_title('Кумулятивная стоимость портфеля', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Стоимость портфеля')
            ax1.grid(True, alpha=0.3)
        
        # 2. Доходность по периодам
        returns = [r.total_return * 100 for r in self.results]
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax2.bar(range(len(returns)), returns, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Доходность по периодам (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Доходность (%)')
        ax2.set_xlabel('Период')
        ax2.grid(True, alpha=0.3)
        
        # 3. Sharpe ratio по периодам
        sharpe_ratios = [r.sharpe_ratio for r in self.results]
        ax3.plot(sharpe_ratios, marker='o', linewidth=2, markersize=4)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
        ax3.set_title('Sharpe Ratio по периодам', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_xlabel('Период')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Максимальная просадка
        drawdowns = [r.max_drawdown * 100 for r in self.results]
        ax4.bar(range(len(drawdowns)), drawdowns, color='red', alpha=0.7)
        ax4.set_title('Максимальная просадка по периодам (%)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Просадка (%)')
        ax4.set_xlabel('Период')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'walk_forward_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Распределение доходности
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=np.mean(returns), color='red', linestyle='--', label=f'Среднее: {np.mean(returns):.2f}%')
        ax1.set_title('Распределение доходности периодов', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Доходность (%)')
        ax1.set_ylabel('Частота')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot метрик
        metrics_data = [returns, sharpe_ratios, [d*100 for d in drawdowns]]
        ax2.boxplot(metrics_data, labels=['Доходность (%)', 'Sharpe Ratio', 'Просадка (%)'])
        ax2.set_title('Распределение ключевых метрик', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'walk_forward_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"💾 Графики сохранены в {save_path}")


def run_walk_forward_analysis(df: pd.DataFrame, model_factory: Callable, 
                            train_months: int = 12, test_months: int = 3, 
                            step_months: int = 1) -> WalkForwardValidator:
    """Удобная функция для запуска walk-forward анализа"""
    
    validator = WalkForwardValidator(
        train_window_months=train_months,
        test_window_months=test_months,
        step_months=step_months
    )
    
    results = validator.run_walk_forward_test(df, model_factory)
    
    if results:
        report = validator.generate_report('walk_forward_results')
        logger.info("✅ Walk-forward анализ завершен!")
    
    return validator


if __name__ == "__main__":
    # Демонстрация использования
    logging.basicConfig(level=logging.INFO)
    
    print("📈 Walk-Forward анализ готов к использованию!")
    print("Для использования: from src.analysis.walk_forward_analysis import run_walk_forward_analysis") 