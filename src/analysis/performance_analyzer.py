"""
Анализ производительности торговых алгоритмов

УЛУЧШЕНИЯ:
1. Более детальные метрики
2. Сравнение с бенчмарками  
3. Риск-анализ
4. Статистические тесты
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from ..core.config import SystemConfig


class PerformanceAnalyzer:
    """
    УЛУЧШЕННЫЙ анализатор производительности торгового алгоритма
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Настраиваем стиль графиков
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def calculate_comprehensive_metrics(self, results: Dict[str, Any], 
                                      initial_balance: float) -> Dict[str, Any]:
        """
        РАСШИРЕННЫЙ расчет торговых метрик
        """
        if not results.get('balance_history'):
            self.logger.warning("Нет истории баланса для анализа")
            return {}
        
        balance_history = np.array(results['balance_history'])
        final_balance = balance_history[-1]
        trades = results.get('trades', [])
        
        # === БАЗОВЫЕ МЕТРИКИ ===
        total_return = (final_balance - initial_balance) / initial_balance
        max_balance = np.max(balance_history)
        max_drawdown = np.max(results.get('drawdowns', [0]))
        
        # === ТОРГОВЫЕ МЕТРИКИ ===
        if trades:
            profitable_trades = [t for t in trades if t > 0]
            losing_trades = [t for t in trades if t <= 0]
            
            win_rate = len(profitable_trades) / len(trades) * 100
            avg_profit = np.mean(profitable_trades) if profitable_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(sum(profitable_trades) / sum(losing_trades)) if losing_trades else float('inf')
            
            # Максимальные серии
            win_streak, loss_streak = self._calculate_streaks(trades)
            
            # Коэффициент восстановления
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf')
        else:
            win_rate = avg_profit = avg_loss = profit_factor = 0
            win_streak = loss_streak = 0
            recovery_factor = 0
        
        # === РИСК МЕТРИКИ ===
        returns = np.diff(balance_history) / balance_history[:-1]
        
        if len(returns) > 1:
            # Sharpe Ratio (предполагаем 252 торговых дня)
            annual_return = np.mean(returns) * 252
            annual_volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Sortino Ratio (только отрицательная волатильность)
            negative_returns = returns[returns < 0]
            downside_volatility = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
            
            # Calmar Ratio
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            # Value at Risk (VaR 95%)
            var_95 = np.percentile(returns, 5) * initial_balance
            
            # Maximum Drawdown Duration
            dd_duration = self._calculate_drawdown_duration(balance_history)
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = var_95 = dd_duration = 0
            annual_return = annual_volatility = 0
        
        # === ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ ===
        # Ожидаемая прибыль на сделку
        expectancy = np.mean(trades) if trades else 0
        
        # R-квадрат (стабильность роста)
        r_squared = self._calculate_r_squared(balance_history)
        
        # Стабильность
        stability = self._calculate_stability(returns) if len(returns) > 1 else 0
        
        metrics = {
            # Доходность
            'total_return': total_return,
            'annual_return': annual_return,
            'final_balance': final_balance,
            'max_balance': max_balance,
            
            # Риск
            'max_drawdown': max_drawdown,
            'annual_volatility': annual_volatility,
            'var_95': var_95,
            'drawdown_duration': dd_duration,
            
            # Торговые метрики
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'win_streak': win_streak,
            'loss_streak': loss_streak,
            
            # Риск-скорректированная доходность
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,
            
            # Качество
            'r_squared': r_squared,
            'stability': stability,
            
            # Комиссии
            'total_commission': results.get('total_commissions', 0)
        }
        
        return metrics
    
    def _calculate_streaks(self, trades: List[float]) -> tuple:
        """Расчет максимальных серий побед и поражений"""
        if not trades:
            return 0, 0
        
        win_streak = loss_streak = 0
        current_win = current_loss = 0
        
        for trade in trades:
            if trade > 0:
                current_win += 1
                current_loss = 0
                win_streak = max(win_streak, current_win)
            else:
                current_loss += 1
                current_win = 0
                loss_streak = max(loss_streak, current_loss)
        
        return win_streak, loss_streak
    
    def _calculate_drawdown_duration(self, balance_history: np.ndarray) -> int:
        """Расчет максимальной продолжительности просадки"""
        peak = np.maximum.accumulate(balance_history)
        drawdown = (peak - balance_history) / peak
        
        duration = 0
        max_duration = 0
        
        for dd in drawdown:
            if dd > 0:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0
        
        return max_duration
    
    def _calculate_r_squared(self, balance_history: np.ndarray) -> float:
        """Расчет R² для стабильности роста"""
        if len(balance_history) < 2:
            return 0
        
        x = np.arange(len(balance_history))
        correlation_matrix = np.corrcoef(x, balance_history)
        return correlation_matrix[0, 1] ** 2
    
    def _calculate_stability(self, returns: np.ndarray) -> float:
        """Оценка стабильности доходности"""
        if len(returns) < 2:
            return 0
        
        # Стабильность как обратная величина коэффициента вариации
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if abs(mean_return) < 1e-8 or std_return == 0:
            return 0
        
        cv = std_return / abs(mean_return)
        return 1 / (1 + cv)  # Нормализованная стабильность от 0 до 1
    
    def compare_with_benchmark(self, results: Dict[str, Any], 
                             benchmark_return: float = 0.0) -> Dict[str, Any]:
        """
        Сравнение с бенчмарком (например, HODL стратегия)
        """
        balance_history = np.array(results['balance_history'])
        initial_balance = balance_history[0]
        final_balance = balance_history[-1]
        
        algo_return = (final_balance - initial_balance) / initial_balance
        
        # Если бенчмарк не задан, используем HODL
        if benchmark_return == 0.0:
            prices = results.get('prices', [])
            if prices:
                benchmark_return = (prices[-1] - prices[0]) / prices[0]
        
        alpha = algo_return - benchmark_return
        
        return {
            'algorithm_return': algo_return,
            'benchmark_return': benchmark_return,
            'alpha': alpha,
            'outperformed': alpha > 0
        }
    
    def create_comprehensive_report(self, results: Dict[str, Any], 
                                  initial_balance: float) -> str:
        """
        Создание детального текстового отчета
        """
        metrics = self.calculate_comprehensive_metrics(results, initial_balance)
        benchmark = self.compare_with_benchmark(results)
        
        report = f"""
🚀 ДЕТАЛЬНЫЙ ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ АЛГОТРЕЙДИНГ СИСТЕМЫ
{'='*80}

📈 ДОХОДНОСТЬ:
   Итоговый баланс:      {metrics['final_balance']:,.2f} USDT
   Общая доходность:     {metrics['total_return']:+.2%}
   Годовая доходность:   {metrics['annual_return']:+.2%}
   Максимальный баланс:  {metrics['max_balance']:,.2f} USDT

⚠️  РИСК:
   Максимальная просадка:      {metrics['max_drawdown']:.2%}
   Годовая волатильность:      {metrics['annual_volatility']:.2%}
   VaR (95%):                  {metrics['var_95']:,.2f} USDT
   Продолжительность просадки: {metrics['drawdown_duration']} периодов

💼 ТОРГОВАЯ АКТИВНОСТЬ:
   Всего сделок:         {metrics['total_trades']}
   Винрейт:              {metrics['win_rate']:.1f}%
   Средняя прибыль:      {metrics['avg_profit']:+.2f} USDT
   Средний убыток:       {metrics['avg_loss']:+.2f} USDT
   Фактор прибыли:       {metrics['profit_factor']:.2f}
   Ожидаемая прибыль:    {metrics['expectancy']:+.2f} USDT
   Макс. серия побед:    {metrics['win_streak']}
   Макс. серия поражений: {metrics['loss_streak']}

📊 РИСК-СКОРРЕКТИРОВАННЫЕ МЕТРИКИ:
   Коэффициент Шарпа:    {metrics['sharpe_ratio']:.3f}
   Коэффициент Сортино:  {metrics['sortino_ratio']:.3f}
   Коэффициент Калмара:  {metrics['calmar_ratio']:.3f}
   Фактор восстановления: {metrics['recovery_factor']:.2f}

🎯 КАЧЕСТВО СТРАТЕГИИ:
   R²:                   {metrics['r_squared']:.3f}
   Стабильность:         {metrics['stability']:.3f}

💰 ИЗДЕРЖКИ:
   Общие комиссии:       {metrics['total_commission']:.2f} USDT

🔍 СРАВНЕНИЕ С БЕНЧМАРКОМ:
   Доходность алгоритма: {benchmark['algorithm_return']:+.2%}
   Доходность бенчмарка: {benchmark['benchmark_return']:+.2%}
   Альфа:                {benchmark['alpha']:+.2%}
   Превосходство:        {'✅ ДА' if benchmark['outperformed'] else '❌ НЕТ'}

🚨 ОЦЕНКА КАЧЕСТВА:
"""
        
        # Добавляем оценку качества
        if metrics['sharpe_ratio'] > 1.0:
            report += "   Sharpe Ratio:     ✅ ОТЛИЧНЫЙ (>1.0)\n"
        elif metrics['sharpe_ratio'] > 0.5:
            report += "   Sharpe Ratio:     ⚠️  ХОРОШИЙ (0.5-1.0)\n"
        else:
            report += "   Sharpe Ratio:     ❌ СЛАБЫЙ (<0.5)\n"
        
        if metrics['max_drawdown'] < 0.1:
            report += "   Просадка:         ✅ НИЗКАЯ (<10%)\n"
        elif metrics['max_drawdown'] < 0.2:
            report += "   Просадка:         ⚠️  УМЕРЕННАЯ (10-20%)\n"
        else:
            report += "   Просадка:         ❌ ВЫСОКАЯ (>20%)\n"
        
        if metrics['win_rate'] > 60:
            report += "   Винрейт:          ✅ ВЫСОКИЙ (>60%)\n"
        elif metrics['win_rate'] > 40:
            report += "   Винрейт:          ⚠️  СРЕДНИЙ (40-60%)\n"
        else:
            report += "   Винрейт:          ❌ НИЗКИЙ (<40%)\n"
        
        return report
    
    def create_advanced_visualization(self, results: Dict[str, Any], 
                                    analysis: Dict[str, Any],
                                    save_path: str = "trading_analysis.png"):
        """
        ПРОДВИНУТАЯ визуализация результатов
        """
        if not results.get('balance_history'):
            self.logger.warning("Нет данных для визуализации")
            return
        
        # Создаем фигуру с подграфиками
        fig = plt.figure(figsize=SystemConfig.FIGURE_SIZE)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. График эквити кривой
        ax1 = fig.add_subplot(gs[0, :2])
        balance_history = results['balance_history']
        ax1.plot(balance_history, linewidth=2, color='blue', label='Баланс портфеля')
        ax1.axhline(y=balance_history[0], color='gray', linestyle='--', alpha=0.7, label='Начальный баланс')
        ax1.set_title("Эквити кривая", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Баланс (USDT)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Добавляем аннотации
        final_balance = balance_history[-1]
        initial_balance = balance_history[0]
        total_return = (final_balance - initial_balance) / initial_balance
        ax1.annotate(f'Итоговый доход: {total_return:+.2%}', 
                    xy=(len(balance_history)-1, final_balance),
                    xytext=(len(balance_history)*0.7, final_balance*1.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, fontweight='bold')
        
        # 2. График цены и позиций
        ax2 = fig.add_subplot(gs[1, :2])
        if 'prices' in results and 'positions' in results:
            prices = results['prices']
            positions = results['positions']
            
            ax2.plot(prices, color='orange', alpha=0.7, label='Цена BTC')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(positions, color='green', alpha=0.7, label='Размер позиции')
            ax2_twin.fill_between(range(len(positions)), positions, alpha=0.3, color='green')
            
            ax2.set_title("Цена BTC и торговые позиции", fontsize=14, fontweight='bold')
            ax2.set_ylabel("Цена BTC (USDT)", color='orange')
            ax2_twin.set_ylabel("Размер позиции (BTC)", color='green')
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
        
        # 3. График просадки
        ax3 = fig.add_subplot(gs[2, :2])
        if 'drawdowns' in results:
            drawdowns = np.array(results['drawdowns']) * 100  # В процентах
            ax3.fill_between(range(len(drawdowns)), drawdowns, color='red', alpha=0.7)
            ax3.set_title("Просадка портфеля", fontsize=14, fontweight='bold')
            ax3.set_ylabel("Просадка (%)")
            ax3.set_xlabel("Время")
            ax3.grid(True, alpha=0.3)
            
            # Добавляем максимальную просадку
            max_dd = np.max(drawdowns)
            ax3.axhline(y=max_dd, color='darkred', linestyle='--', linewidth=2)
            ax3.text(len(drawdowns)*0.1, max_dd*1.1, f'Макс. просадка: {max_dd:.1f}%', 
                    fontweight='bold', color='darkred')
        
        # 4. Гистограмма прибылей/убытков
        ax4 = fig.add_subplot(gs[0, 2])
        if 'trades' in results and results['trades']:
            trades = results['trades']
            num_bins = max(1, min(30, len(trades)//2)) if len(trades) > 2 else len(trades) 
            ax4.hist(trades, bins=max(1, num_bins), alpha=0.7, color='purple', edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax4.set_title("Распределение P&L", fontsize=12, fontweight='bold')
            ax4.set_xlabel("Прибыль/Убыток")
            ax4.set_ylabel("Количество сделок")
            ax4.grid(True, alpha=0.3)
        
        # 5. Круговая диаграмма успешности
        ax5 = fig.add_subplot(gs[1, 2])
        if analysis.get('total_trades', 0) > 0:
            win_rate = analysis.get('win_rate', 0)
            loss_rate = 100 - win_rate
            
            sizes = [win_rate, loss_rate]
            labels = [f'Прибыльные\n{win_rate:.1f}%', f'Убыточные\n{loss_rate:.1f}%']
            colors = ['lightgreen', 'lightcoral']
            
            ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax5.set_title("Соотношение сделок", fontsize=12, fontweight='bold')
        
        # 6. Ключевые метрики
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        metrics_text = f"""КЛЮЧЕВЫЕ МЕТРИКИ:

Sharpe Ratio: {analysis.get('sharpe_ratio', 0):.3f}
Макс. просадка: {analysis.get('max_drawdown', 0):.1%}
Фактор прибыли: {analysis.get('profit_factor', 0):.2f}
Винрейт: {analysis.get('win_rate', 0):.1f}%
Всего сделок: {analysis.get('total_trades', 0)}

Оценка: {'🟢 ХОРОШО' if analysis.get('sharpe_ratio', 0) > 0.5 else '🔴 ПЛОХО'}
"""
        
        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Общий заголовок
        fig.suptitle("📊 ПРОДВИНУТЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ АЛГОТРЕЙДИНГ СИСТЕМЫ", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Сохраняем график
        plt.savefig(save_path, dpi=SystemConfig.PLOT_DPI, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        if SystemConfig.SAVE_PLOTS:
            self.logger.info(f"📈 Расширенная визуализация сохранена: {save_path}")
        
        plt.close()  # Освобождаем память
    
    def generate_full_analysis(self, results: Dict[str, Any], 
                             initial_balance: float) -> Dict[str, Any]:
        """
        ПОЛНЫЙ анализ с отчетом и визуализацией
        """
        self.logger.info("🔍 Генерация полного анализа производительности...")
        
        # Рассчитываем метрики
        metrics = self.calculate_comprehensive_metrics(results, initial_balance)
        benchmark = self.compare_with_benchmark(results)
        
        # Объединяем все данные
        full_analysis = {**metrics, **benchmark}
        
        # Создаем отчет
        report = self.create_comprehensive_report(results, initial_balance)
        
        # Создаем визуализацию
        self.create_advanced_visualization(results, full_analysis)
        
        full_analysis['text_report'] = report
        
        self.logger.info("✅ Полный анализ завершен")
        return full_analysis


# Фабричные функции для обратной совместимости
def analyze_results(results: Dict[str, Any], initial_balance: float) -> Dict[str, Any]:
    """Функция для обратной совместимости"""
    analyzer = PerformanceAnalyzer()
    return analyzer.calculate_comprehensive_metrics(results, initial_balance)


def visualize_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Функция для обратной совместимости"""
    analyzer = PerformanceAnalyzer()
    analyzer.create_advanced_visualization(results, analysis) 