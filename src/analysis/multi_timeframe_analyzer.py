"""
🕰️ МНОГОМАСШТАБНЫЙ АНАЛИЗ
Анализ рынка на разных таймфреймах для улучшения точности прогнозов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class TimeframeSignal:
    """Сигнал с определенного таймфрейма"""
    timeframe: str
    signal_strength: float  # -1 to 1
    confidence: float  # 0 to 1
    trend_direction: str  # 'up', 'down', 'sideways'
    volatility_level: str  # 'low', 'medium', 'high'
    volume_confirmation: bool
    support_resistance_distance: float


class MultiTimeframeAnalyzer:
    """Анализатор на множественных таймфреймах"""
    
    def __init__(self, base_timeframe: str = '5m'):
        """
        Args:
            base_timeframe: Базовый таймфрейм данных (5m, 15m, 1h, etc.)
        """
        self.base_timeframe = base_timeframe
        self.timeframe_multipliers = {
            '5m': 1,
            '15m': 3,
            '30m': 6,
            '1h': 12,
            '4h': 48,
            '1d': 288,
            '1w': 2016
        }
        
        self.timeframe_weights = {
            '5m': 0.1,   # Краткосрочный шум
            '15m': 0.15,
            '30m': 0.2,
            '1h': 0.25,  # Основной для скальпинга
            '4h': 0.2,   # Средний тренд
            '1d': 0.15   # Долгосрочный тренд
        }
        
        self.signals_history = []
        self.current_market_state = {}
    
    def resample_to_timeframe(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Пересэмплирование данных в нужный таймфрейм"""
        
        if 'timestamp' not in df.columns:
            logger.error("❌ Колонка timestamp отсутствует")
            return df
        
        # Преобразуем timestamp в datetime
        df_copy = df.copy()
        if df_copy['timestamp'].dtype != 'datetime64[ns]':
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='ms')
        
        df_copy = df_copy.set_index('timestamp')
        
        # Определяем правила агрегации
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Добавляем правила для технических индикаторов
        for col in df_copy.columns:
            if col not in agg_rules:
                if any(indicator in col.lower() for indicator in ['rsi', 'cci', 'williams', 'stoch', 'adx']):
                    agg_rules[col] = 'last'  # Осцилляторы - последнее значение
                elif any(indicator in col.lower() for indicator in ['sma', 'ema', 'ma', 'bb', 'kc']):
                    agg_rules[col] = 'last'  # Средние - последнее значение
                elif any(indicator in col.lower() for indicator in ['volume', 'obv', 'ad']):
                    agg_rules[col] = 'sum'   # Объемные - сумма
                else:
                    agg_rules[col] = 'last'  # По умолчанию
        
        # Ресэмплируем
        timeframe_map = {
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w'
        }
        
        pandas_tf = timeframe_map.get(target_timeframe, '1h')
        resampled = df_copy.resample(pandas_tf).agg(agg_rules)
        
        # Удаляем NaN строки
        resampled = resampled.dropna()
        
        # Возвращаем timestamp как колонку
        resampled = resampled.reset_index()
        resampled['timestamp'] = resampled['timestamp'].astype(int) // 10**6  # Обратно в миллисекунды
        
        logger.info(f"📊 Ресэмплирование {target_timeframe}: {len(df)} -> {len(resampled)} записей")
        
        return resampled
    
    def analyze_timeframe_trend(self, df: pd.DataFrame, timeframe: str) -> TimeframeSignal:
        """Анализ тренда на конкретном таймфрейме"""
        
        if len(df) < 50:  # Минимум данных для анализа
            logger.warning(f"⚠️ Недостаточно данных для анализа {timeframe}")
            return TimeframeSignal(
                timeframe=timeframe,
                signal_strength=0,
                confidence=0,
                trend_direction='sideways',
                volatility_level='medium',
                volume_confirmation=False,
                support_resistance_distance=0
            )
        
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # === АНАЛИЗ ТРЕНДА ===
            
            # Средние для определения тренда
            ema_short = close.ewm(span=10).mean()
            ema_long = close.ewm(span=50).mean()
            
            # Направление тренда
            current_price = close.iloc[-1]
            short_ma = ema_short.iloc[-1]
            long_ma = ema_long.iloc[-1]
            
            # Сила тренда
            if short_ma > long_ma:
                trend_direction = 'up'
                trend_strength = min((short_ma - long_ma) / long_ma * 100, 1.0)
            elif short_ma < long_ma:
                trend_direction = 'down'
                trend_strength = min((long_ma - short_ma) / long_ma * 100, 1.0)
            else:
                trend_direction = 'sideways'
                trend_strength = 0
            
            # Подтверждение трендом цены
            price_above_short = current_price > short_ma
            price_above_long = current_price > long_ma
            
            if trend_direction == 'up' and price_above_short and price_above_long:
                signal_strength = trend_strength
            elif trend_direction == 'down' and not price_above_short and not price_above_long:
                signal_strength = -trend_strength
            else:
                signal_strength = trend_strength * 0.5  # Слабый сигнал при противоречии
            
            # === ВОЛАТИЛЬНОСТЬ ===
            
            # ATR для волатильности
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Классификация волатильности
            atr_sma = true_range.rolling(50).mean().iloc[-1]
            if atr > atr_sma * 1.5:
                volatility_level = 'high'
            elif atr < atr_sma * 0.7:
                volatility_level = 'low'
            else:
                volatility_level = 'medium'
            
            # === ОБЪЕМНЫЙ АНАЛИЗ ===
            
            # Подтверждение объемом
            volume_sma = volume.rolling(20).mean()
            current_volume = volume.iloc[-1]
            avg_volume = volume_sma.iloc[-1]
            
            volume_confirmation = current_volume > avg_volume * 1.2
            
            # === ПОДДЕРЖКА/СОПРОТИВЛЕНИЕ ===
            
            # Простое определение уровней через локальные экстремумы
            lookback = min(20, len(df) // 4)
            recent_highs = high.tail(lookback)
            recent_lows = low.tail(lookback)
            
            resistance_level = recent_highs.quantile(0.8)
            support_level = recent_lows.quantile(0.2)
            
            # Расстояние до ближайшего уровня
            dist_to_resistance = (resistance_level - current_price) / current_price
            dist_to_support = (current_price - support_level) / current_price
            
            support_resistance_distance = min(abs(dist_to_resistance), abs(dist_to_support))
            
            # === УВЕРЕННОСТЬ ===
            
            # Уверенность зависит от согласованности сигналов
            confidence_factors = []
            
            # 1. Согласованность MA
            if (trend_direction == 'up' and price_above_short and price_above_long) or \
               (trend_direction == 'down' and not price_above_short and not price_above_long):
                confidence_factors.append(0.3)
            
            # 2. Подтверждение объемом
            if volume_confirmation:
                confidence_factors.append(0.2)
            
            # 3. Низкая волатильность = больше уверенности
            if volatility_level == 'low':
                confidence_factors.append(0.2)
            elif volatility_level == 'medium':
                confidence_factors.append(0.1)
            
            # 4. Расстояние от S/R уровней
            if support_resistance_distance > 0.02:  # Далеко от уровней
                confidence_factors.append(0.2)
            
            # 5. Длительность тренда
            trend_length = 0
            for i in range(2, min(10, len(ema_short))):
                if trend_direction == 'up' and ema_short.iloc[-i] < ema_long.iloc[-i]:
                    break
                elif trend_direction == 'down' and ema_short.iloc[-i] > ema_long.iloc[-i]:
                    break
                trend_length += 1
            
            if trend_length >= 5:
                confidence_factors.append(0.1)
            
            confidence = min(sum(confidence_factors), 1.0)
            
            return TimeframeSignal(
                timeframe=timeframe,
                signal_strength=signal_strength,
                confidence=confidence,
                trend_direction=trend_direction,
                volatility_level=volatility_level,
                volume_confirmation=volume_confirmation,
                support_resistance_distance=support_resistance_distance
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа {timeframe}: {e}")
            return TimeframeSignal(
                timeframe=timeframe,
                signal_strength=0,
                confidence=0,
                trend_direction='sideways',
                volatility_level='medium',
                volume_confirmation=False,
                support_resistance_distance=0
            )
    
    def analyze_multiple_timeframes(self, df: pd.DataFrame, 
                                  timeframes: List[str] = None) -> Dict[str, TimeframeSignal]:
        """Анализ на множественных таймфреймах"""
        
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']
        
        logger.info(f"🕰️ Многомасштабный анализ на {len(timeframes)} таймфреймах...")
        
        signals = {}
        
        for tf in timeframes:
            try:
                # Ресэмплируем данные для таймфрейма
                if tf == self.base_timeframe:
                    tf_data = df.copy()
                else:
                    tf_data = self.resample_to_timeframe(df, tf)
                
                # Анализируем таймфрейм
                if len(tf_data) > 10:  # Минимальная проверка
                    signal = self.analyze_timeframe_trend(tf_data, tf)
                    signals[tf] = signal
                    
                    logger.info(f"📊 {tf}: {signal.trend_direction} "
                              f"(сила: {signal.signal_strength:+.3f}, "
                              f"уверенность: {signal.confidence:.3f})")
                else:
                    logger.warning(f"⚠️ Недостаточно данных для {tf}")
            
            except Exception as e:
                logger.error(f"❌ Ошибка анализа {tf}: {e}")
        
        self.current_market_state = signals
        return signals
    
    def generate_consensus_signal(self, signals: Dict[str, TimeframeSignal] = None) -> Dict[str, Any]:
        """Генерация консенсусного сигнала на основе всех таймфреймов"""
        
        if signals is None:
            signals = self.current_market_state
        
        if not signals:
            logger.warning("⚠️ Нет сигналов для генерации консенсуса")
            return {
                'action': 0,  # Hold
                'confidence': 0,
                'consensus_strength': 0,
                'dominant_timeframe': None,
                'risk_level': 'high'
            }
        
        logger.info("🎯 Генерация консенсусного сигнала...")
        
        # Взвешенные сигналы
        weighted_signals = []
        total_weight = 0
        
        for tf, signal in signals.items():
            weight = self.timeframe_weights.get(tf, 0.1)
            weighted_strength = signal.signal_strength * signal.confidence * weight
            weighted_signals.append(weighted_strength)
            total_weight += weight
        
        # Консенсусная сила
        consensus_strength = sum(weighted_signals) / total_weight if total_weight > 0 else 0
        
        # Определяем действие
        action = 0  # Hold по умолчанию
        if consensus_strength > 0.1:
            action = 1  # Buy
        elif consensus_strength < -0.1:
            action = 2  # Sell
        
        # Уверенность консенсуса
        agreement_signals = []
        for signal in signals.values():
            if consensus_strength > 0.1 and signal.signal_strength > 0:
                agreement_signals.append(signal.confidence)
            elif consensus_strength < -0.1 and signal.signal_strength < 0:
                agreement_signals.append(signal.confidence)
            elif abs(consensus_strength) <= 0.1 and abs(signal.signal_strength) <= 0.1:
                agreement_signals.append(signal.confidence)
        
        consensus_confidence = np.mean(agreement_signals) if agreement_signals else 0
        
        # Доминирующий таймфрейм
        dominant_tf = None
        max_weighted_signal = 0
        
        for tf, signal in signals.items():
            weight = self.timeframe_weights.get(tf, 0.1)
            weighted_signal = abs(signal.signal_strength * signal.confidence * weight)
            if weighted_signal > max_weighted_signal:
                max_weighted_signal = weighted_signal
                dominant_tf = tf
        
        # Уровень риска
        risk_factors = []
        
        # 1. Высокая волатильность = высокий риск
        high_vol_timeframes = [tf for tf, sig in signals.items() if sig.volatility_level == 'high']
        if len(high_vol_timeframes) >= len(signals) // 2:
            risk_factors.append('high_volatility')
        
        # 2. Низкая уверенность = высокий риск
        if consensus_confidence < 0.5:
            risk_factors.append('low_confidence')
        
        # 3. Противоречивые сигналы = высокий риск
        positive_signals = sum(1 for sig in signals.values() if sig.signal_strength > 0.1)
        negative_signals = sum(1 for sig in signals.values() if sig.signal_strength < -0.1)
        if min(positive_signals, negative_signals) > 0:
            risk_factors.append('contradictory_signals')
        
        # 4. Близость к S/R уровням
        close_to_sr = [tf for tf, sig in signals.items() if sig.support_resistance_distance < 0.01]
        if len(close_to_sr) >= len(signals) // 2:
            risk_factors.append('near_support_resistance')
        
        # Определяем итоговый уровень риска
        if len(risk_factors) >= 3:
            risk_level = 'high'
        elif len(risk_factors) >= 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        result = {
            'action': action,
            'confidence': consensus_confidence,
            'consensus_strength': consensus_strength,
            'dominant_timeframe': dominant_tf,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'timeframe_agreement': len(agreement_signals) / len(signals) if signals else 0,
            'detailed_signals': signals
        }
        
        # Логируем результат
        action_names = ['Hold', 'Buy', 'Sell']
        logger.info(f"🎯 Консенсусный сигнал:")
        logger.info(f"   Действие: {action_names[action]} (сила: {consensus_strength:+.3f})")
        logger.info(f"   Уверенность: {consensus_confidence:.3f}")
        logger.info(f"   Доминирующий ТФ: {dominant_tf}")
        logger.info(f"   Уровень риска: {risk_level}")
        logger.info(f"   Согласованность: {result['timeframe_agreement']:.1%}")
        
        return result
    
    def create_market_state_dashboard(self, signals: Dict[str, TimeframeSignal] = None, 
                                    save_path: str = None) -> None:
        """Создание дашборда состояния рынка"""
        
        if signals is None:
            signals = self.current_market_state
        
        if not signals:
            logger.warning("⚠️ Нет сигналов для создания дашборда")
            return
        
        logger.info("📊 Создание дашборда многомасштабного анализа...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Сила сигналов по таймфреймам
        timeframes = list(signals.keys())
        strengths = [signals[tf].signal_strength for tf in timeframes]
        confidences = [signals[tf].confidence for tf in timeframes]
        
        colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in strengths]
        bars = ax1.barh(timeframes, strengths, color=colors, alpha=0.7)
        
        # Добавляем уверенность как размер точек
        for i, (tf, strength, conf) in enumerate(zip(timeframes, strengths, confidences)):
            ax1.scatter(strength, i, s=conf*500, color='black', alpha=0.5)
        
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Сила сигнала')
        ax1.set_title('Сигналы по таймфреймам\n(размер точки = уверенность)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Направление тренда
        trend_counts = {'up': 0, 'down': 0, 'sideways': 0}
        for signal in signals.values():
            trend_counts[signal.trend_direction] += 1
        
        ax2.pie(trend_counts.values(), labels=trend_counts.keys(), autopct='%1.1f%%',
                colors=['green', 'red', 'gray'])
        ax2.set_title('Распределение трендов', fontweight='bold')
        
        # 3. Волатильность по таймфреймам
        volatility_data = {}
        for tf, signal in signals.items():
            if signal.volatility_level not in volatility_data:
                volatility_data[signal.volatility_level] = []
            volatility_data[signal.volatility_level].append(tf)
        
        vol_levels = ['low', 'medium', 'high']
        vol_colors = ['green', 'yellow', 'red']
        vol_counts = [len(volatility_data.get(level, [])) for level in vol_levels]
        
        ax3.bar(vol_levels, vol_counts, color=vol_colors, alpha=0.7)
        ax3.set_ylabel('Количество таймфреймов')
        ax3.set_title('Уровни волатильности', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Матрица консенсуса
        consensus_data = []
        labels = []
        
        for tf, signal in signals.items():
            weight = self.timeframe_weights.get(tf, 0.1)
            weighted_signal = signal.signal_strength * signal.confidence * weight
            consensus_data.append([weighted_signal])
            labels.append(f"{tf}")
        
        if consensus_data:
            im = ax4.imshow(consensus_data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
            ax4.set_yticks(range(len(labels)))
            ax4.set_yticklabels(labels)
            ax4.set_xticks([0])
            ax4.set_xticklabels(['Взвешенный сигнал'])
            ax4.set_title('Консенсус-матрица', fontweight='bold')
            
            # Добавляем colorbar
            plt.colorbar(im, ax=ax4, label='Сила сигнала')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Дашборд сохранен: {save_path}")
        else:
            plt.savefig('multi_timeframe_dashboard.png', dpi=300, bbox_inches='tight')
            logger.info("💾 Дашборд сохранен: multi_timeframe_dashboard.png")
        
        plt.close()
    
    def get_market_regime(self, signals: Dict[str, TimeframeSignal] = None) -> Dict[str, str]:
        """Определение рыночного режима на основе многомасштабного анализа"""
        
        if signals is None:
            signals = self.current_market_state
        
        if not signals:
            return {'regime': 'unknown', 'confidence': 'low'}
        
        # Анализируем тренды на разных масштабах
        short_term_trends = []  # 5m, 15m, 30m
        medium_term_trends = []  # 1h, 4h
        long_term_trends = []   # 1d, 1w
        
        for tf, signal in signals.items():
            if tf in ['5m', '15m', '30m']:
                short_term_trends.append(signal.trend_direction)
            elif tf in ['1h', '4h']:
                medium_term_trends.append(signal.trend_direction)
            elif tf in ['1d', '1w']:
                long_term_trends.append(signal.trend_direction)
        
        # Определяем режимы
        def get_dominant_trend(trends):
            if not trends:
                return 'unknown'
            trend_counts = {'up': 0, 'down': 0, 'sideways': 0}
            for trend in trends:
                trend_counts[trend] += 1
            return max(trend_counts.items(), key=lambda x: x[1])[0]
        
        short_regime = get_dominant_trend(short_term_trends)
        medium_regime = get_dominant_trend(medium_term_trends)
        long_regime = get_dominant_trend(long_term_trends)
        
        # Классифицируем общий режим
        if long_regime == 'up' and medium_regime == 'up':
            if short_regime == 'up':
                regime = 'strong_bull_trend'
            elif short_regime == 'down':
                regime = 'bull_trend_with_pullback'
            else:
                regime = 'bull_trend_consolidation'
        
        elif long_regime == 'down' and medium_regime == 'down':
            if short_regime == 'down':
                regime = 'strong_bear_trend'
            elif short_regime == 'up':
                regime = 'bear_trend_with_bounce'
            else:
                regime = 'bear_trend_consolidation'
        
        elif long_regime == 'sideways' or medium_regime == 'sideways':
            if short_regime in ['up', 'down']:
                regime = 'range_bound_with_breakout_attempt'
            else:
                regime = 'range_bound_consolidation'
        
        else:
            regime = 'mixed_signals'
        
        # Уверенность в режиме
        total_signals = len([s for s in signals.values() if s.confidence > 0.3])
        high_confidence_signals = len([s for s in signals.values() if s.confidence > 0.7])
        
        if high_confidence_signals >= total_signals * 0.7:
            confidence = 'high'
        elif high_confidence_signals >= total_signals * 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'regime': regime,
            'confidence': confidence,
            'short_term': short_regime,
            'medium_term': medium_regime,
            'long_term': long_regime
        }


def analyze_multi_timeframe_market(df: pd.DataFrame, 
                                 timeframes: List[str] = None) -> Dict[str, Any]:
    """Удобная функция для полного многомасштабного анализа"""
    
    analyzer = MultiTimeframeAnalyzer()
    
    # Анализируем все таймфреймы
    signals = analyzer.analyze_multiple_timeframes(df, timeframes)
    
    if not signals:
        logger.error("❌ Не удалось получить сигналы")
        return {}
    
    # Генерируем консенсусный сигнал
    consensus = analyzer.generate_consensus_signal(signals)
    
    # Определяем рыночный режим
    market_regime = analyzer.get_market_regime(signals)
    
    # Создаем дашборд
    analyzer.create_market_state_dashboard(signals, 'multi_timeframe_analysis.png')
    
    return {
        'signals': signals,
        'consensus': consensus,
        'market_regime': market_regime,
        'analyzer': analyzer
    }


if __name__ == "__main__":
    # Демонстрация использования
    logging.basicConfig(level=logging.INFO)
    
    print("🕰️ Многомасштабный анализ готов к использованию!")
    print("Для использования: from src.analysis.multi_timeframe_analyzer import analyze_multi_timeframe_market") 