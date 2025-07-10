"""
📊 ПРИМЕР АНАЛИЗА МУЛЬТИТАЙМФРЕЙМ ДАННЫХ
Демонстрация работы с загруженными данными с Binance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


class MultiTimeframeAnalyzer:
    """
    Анализатор мультитаймфрейм данных
    """
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = Path(data_folder)
        self.timeframes = ['5m', '1h', '4h', '1d']
        self.data = {}
        
    def load_all_data(self):
        """Загрузка всех таймфреймов"""
        print("📂 Загрузка мультитаймфрейм данных...")
        
        for tf in self.timeframes:
            filename = f"BTCUSDT_{tf}_2y.csv"
            filepath = self.data_folder / filename
            
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                self.data[tf] = df
                print(f"✅ {tf:>3}: {len(df):>7,} записей | "
                      f"{df['datetime'].min().strftime('%Y-%m-%d')} - "
                      f"{df['datetime'].max().strftime('%Y-%m-%d')}")
            else:
                print(f"❌ {tf}: файл не найден")
        
        print(f"\n📊 Загружено {len(self.data)} таймфреймов")
        
    def analyze_price_statistics(self):
        """Анализ ценовой статистики"""
        print("\n" + "="*60)
        print("📈 ЦЕНОВАЯ СТАТИСТИКА ПО ТАЙМФРЕЙМАМ")
        print("="*60)
        
        for tf, df in self.data.items():
            volatility = ((df['high'] - df['low']) / df['close'] * 100).mean()
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)
            
            print(f"\n🔸 {tf.upper()} ТАЙМФРЕЙМ:")
            print(f"   💰 Цена: ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
            print(f"   📈 Изменение: {price_change:+.2f}%")
            print(f"   📊 Мин/Макс: ${df['low'].min():.2f} / ${df['high'].max():.2f}")
            print(f"   ⚡ Средняя волатильность: {volatility:.2f}%")
            print(f"   📊 Средний объем: {df['volume'].mean():.2f}")
            
    def calculate_technical_indicators(self, timeframe: str = '1h'):
        """Расчет технических индикаторов"""
        if timeframe not in self.data:
            print(f"❌ Данные для {timeframe} не найдены")
            return None
            
        print(f"\n📊 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ ({timeframe.upper()})")
        print("-" * 50)
        
        df = self.data[timeframe].copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        
        # Выводим последние значения
        latest = df.iloc[-1]
        print(f"🔹 RSI: {latest['rsi']:.2f}")
        print(f"🔹 MACD: {latest['macd']:.2f}")
        print(f"🔹 MACD Signal: {latest['macd_signal']:.2f}")
        print(f"🔹 BB Upper: ${latest['bb_upper']:.2f}")
        print(f"🔹 BB Lower: ${latest['bb_lower']:.2f}")
        
        return df
        
    def compare_timeframes(self):
        """Сравнение корреляции между таймфреймами"""
        print("\n📊 КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
        print("-" * 30)
        
        # Синхронизируем данные по времени (используем дневные закрытия)
        daily_closes = {}
        
        for tf, df in self.data.items():
            # Группируем по дням и берем последнее закрытие дня
            df_daily = df.set_index('datetime').resample('D')['close'].last().dropna()
            daily_closes[tf] = df_daily
            
        # Создаем общий DataFrame
        correlation_data = pd.DataFrame(daily_closes)
        correlation_data = correlation_data.dropna()
        
        # Вычисляем корреляцию
        correlation_matrix = correlation_data.corr()
        
        print("Матрица корреляции дневных закрытий:")
        print(correlation_matrix.round(4))
        
        return correlation_matrix
        
    def detect_trading_opportunities(self, timeframe: str = '1h'):
        """Поиск торговых возможностей"""
        if timeframe not in self.data:
            return
            
        print(f"\n🎯 ТОРГОВЫЕ СИГНАЛЫ ({timeframe.upper()})")
        print("-" * 40)
        
        df = self.calculate_technical_indicators(timeframe)
        if df is None:
            return
            
        latest = df.iloc[-1]
        signals = []
        
        # RSI сигналы
        if latest['rsi'] < 30:
            signals.append("🟢 RSI OVERSOLD - возможность покупки")
        elif latest['rsi'] > 70:
            signals.append("🔴 RSI OVERBOUGHT - возможность продажи")
            
        # MACD сигналы
        if latest['macd'] > latest['macd_signal']:
            signals.append("🟢 MACD BULLISH - восходящий тренд")
        else:
            signals.append("🔴 MACD BEARISH - нисходящий тренд")
            
        # Bollinger Bands сигналы
        if latest['close'] < latest['bb_lower']:
            signals.append("🟢 PRICE BELOW BB LOWER - возможность покупки")
        elif latest['close'] > latest['bb_upper']:
            signals.append("🔴 PRICE ABOVE BB UPPER - возможность продажи")
            
        if signals:
            for signal in signals:
                print(f"   {signal}")
        else:
            print("   📊 Нет четких сигналов")
            
    def export_summary(self):
        """Экспорт сводки в файл"""
        summary_path = self.data_folder / "data_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("📊 СВОДКА ПО МУЛЬТИТАЙМФРЕЙМ ДАННЫМ BTC/USDT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for tf, df in self.data.items():
                f.write(f"🔸 {tf.upper()} ТАЙМФРЕЙМ:\n")
                f.write(f"   📊 Записей: {len(df):,}\n")
                f.write(f"   📅 Период: {df['datetime'].min()} - {df['datetime'].max()}\n")
                f.write(f"   💰 Цена: ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}\n")
                
                volatility = ((df['high'] - df['low']) / df['close'] * 100).mean()
                f.write(f"   ⚡ Волатильность: {volatility:.2f}%\n\n")
                
        print(f"\n📄 Сводка сохранена: {summary_path}")


def main():
    """Основная функция демонстрации"""
    print("🚀 АНАЛИЗ МУЛЬТИТАЙМФРЕЙМ ДАННЫХ BTC/USDT")
    print("=" * 50)
    
    # Создаем анализатор
    analyzer = MultiTimeframeAnalyzer()
    
    # Загружаем данные
    analyzer.load_all_data()
    
    if not analyzer.data:
        print("❌ Нет данных для анализа")
        return
        
    # Анализ статистики
    analyzer.analyze_price_statistics()
    
    # Технические индикаторы
    analyzer.calculate_technical_indicators('1h')
    
    # Корреляционный анализ
    analyzer.compare_timeframes()
    
    # Торговые сигналы
    analyzer.detect_trading_opportunities('1h')
    
    # Экспорт сводки
    analyzer.export_summary()
    
    print("\n✅ Анализ завершен!")
    print("\n💡 Советы по использованию:")
    print("   - Используйте разные таймфреймы для подтверждения сигналов")
    print("   - 5m данные для точных входов")
    print("   - 1h/4h для основного анализа тренда")  
    print("   - 1d для долгосрочного направления")


if __name__ == "__main__":
    main() 