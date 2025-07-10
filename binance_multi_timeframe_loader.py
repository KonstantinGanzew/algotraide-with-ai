"""
🚀 ЗАГРУЗЧИК ДАННЫХ С BINANCE - МУЛЬТИТАЙМФРЕЙМ
Загрузка реальных данных с Binance API за последние 2 года
Поддерживаемые таймфреймы: 5m, 1h, 4h, 1d
"""

import pandas as pd
import numpy as np
import ccxt
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceMultiTimeframeLoader:
    """
    Загрузчик данных с Binance для мультитаймфрейм анализа
    """
    
    # Поддерживаемые таймфреймы
    TIMEFRAMES = {
        '5m': '5m',
        '1h': '1h', 
        '4h': '4h',
        '1d': '1d'
    }
    
    # Лимиты запросов для разных таймфреймов
    REQUEST_LIMITS = {
        '5m': 1000,  # Макс 1000 свечей за запрос
        '1h': 1000,
        '4h': 1000, 
        '1d': 1000
    }
    
    def __init__(self, data_folder: str = "data"):
        """
        Инициализация загрузчика
        
        Args:
            data_folder: Папка для сохранения данных
        """
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
        
        # Инициализация Binance exchange
        try:
            self.exchange = ccxt.binance({
                'apiKey': '',  # Публичные данные не требуют API ключей
                'secret': '',
                'sandbox': False,
                'rateLimit': 1200,  # Задержка между запросами (мс)
                'enableRateLimit': True,
            })
            logger.info("✅ Binance exchange инициализирован")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Binance: {e}")
            raise e
    
    def calculate_timeframe_period(self, timeframe: str) -> tuple:
        """
        Вычисление периода для загрузки данных за 2 года
        
        Args:
            timeframe: Таймфрейм ('5m', '1h', '4h', '1d')
            
        Returns:
            tuple: (start_timestamp, end_timestamp)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 года
        
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        return start_ts, end_ts
    
    def fetch_ohlcv_batch(self, symbol: str, timeframe: str, start_ts: int, limit: int) -> List:
        """
        Загрузка одной порции OHLCV данных
        
        Args:
            symbol: Торговая пара (например, 'BTC/USDT')
            timeframe: Таймфрейм
            start_ts: Начальный timestamp
            limit: Количество свечей
            
        Returns:
            List: OHLCV данные
        """
        try:
            data = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=start_ts,
                limit=limit
            )
            return data
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {symbol} {timeframe}: {e}")
            return []
    
    def download_timeframe_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Загрузка всех данных для конкретного таймфрейма
        
        Args:
            symbol: Торговая пара 
            timeframe: Таймфрейм
            
        Returns:
            pd.DataFrame: Загруженные данные
        """
        logger.info(f"📊 Загрузка {symbol} {timeframe} за последние 2 года...")
        
        start_ts, end_ts = self.calculate_timeframe_period(timeframe)
        limit = self.REQUEST_LIMITS[timeframe]
        
        all_data = []
        current_ts = start_ts
        request_count = 0
        
        while current_ts < end_ts:
            # Загружаем порцию данных
            batch_data = self.fetch_ohlcv_batch(symbol, timeframe, current_ts, limit)
            
            if not batch_data:
                logger.warning(f"⚠️ Пустые данные для {timeframe} с timestamp {current_ts}")
                break
            
            # Добавляем к общему списку
            all_data.extend(batch_data)
            
            # Обновляем timestamp для следующего запроса
            last_candle_time = batch_data[-1][0]
            current_ts = last_candle_time + self.get_timeframe_ms(timeframe)
            
            request_count += 1
            
            # Лог прогресса
            if request_count % 10 == 0:
                logger.info(f"📈 {timeframe}: загружено {len(all_data)} свечей (запросов: {request_count})")
            
            # Соблюдаем rate limits
            time.sleep(0.1)
            
            # Проверяем, не превышаем ли end_ts
            if last_candle_time >= end_ts:
                break
        
        if not all_data:
            logger.warning(f"⚠️ Нет данных для {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Конвертируем в DataFrame
        df = pd.DataFrame(all_data)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Удаляем дубликаты и сортируем
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        # Фильтруем по временному диапазону
        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
        
        logger.info(f"✅ {symbol} {timeframe}: загружено {len(df)} свечей")
        logger.info(f"📅 Период: {pd.to_datetime(df['timestamp'].min(), unit='ms')} - {pd.to_datetime(df['timestamp'].max(), unit='ms')}")
        
        return df
    
    def get_timeframe_ms(self, timeframe: str) -> int:
        """
        Получение продолжительности таймфрейма в миллисекундах
        
        Args:
            timeframe: Таймфрейм
            
        Returns:
            int: Миллисекунды
        """
        timeframe_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return timeframe_ms.get(timeframe, 60 * 1000)
    
    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Сохранение данных в CSV файл
        
        Args:
            df: DataFrame для сохранения
            filename: Имя файла
        """
        if df.empty:
            logger.warning(f"⚠️ Попытка сохранить пустые данные: {filename}")
            return
            
        filepath = self.data_folder / filename
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Сохранено: {filename} ({len(df)} записей, {filepath.stat().st_size / 1024 / 1024:.1f} MB)")
    
    def validate_data(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Валидация загруженных данных
        
        Args:
            df: DataFrame для проверки
            timeframe: Таймфрейм
            
        Returns:
            Dict: Результаты валидации
        """
        if df.empty:
            return {'status': 'empty', 'errors': ['Пустой датасет']}
        
        errors = []
        warnings = []
        
        # Проверка основных столбцов
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Отсутствуют столбцы: {missing_columns}")
        
        # Проверка OHLC логики
        ohlc_errors = (
            (df['high'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low'])
        ).sum()
        
        if ohlc_errors > 0:
            warnings.append(f"OHLC ошибки: {ohlc_errors} записей")
        
        # Проверка на отрицательные цены
        negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).sum().sum()
        if negative_prices > 0:
            errors.append(f"Отрицательные цены: {negative_prices}")
        
        # Проверка временных промежутков
        expected_interval_ms = self.get_timeframe_ms(timeframe)
        time_diffs = df['timestamp'].diff().dropna()
        irregular_intervals = (time_diffs != expected_interval_ms).sum()
        
        if irregular_intervals > len(df) * 0.1:  # Более 10% нерегулярных интервалов
            warnings.append(f"Нерегулярные временные интервалы: {irregular_intervals} из {len(df)}")
        
        # Общая статистика
        stats = {
            'total_records': len(df),
            'date_range': {
                'start': pd.to_datetime(df['timestamp'].min(), unit='ms').strftime('%Y-%m-%d %H:%M:%S'),
                'end': pd.to_datetime(df['timestamp'].max(), unit='ms').strftime('%Y-%m-%d %H:%M:%S')
            },
            'missing_values': df.isnull().sum().to_dict(),
            'price_range': {
                'min_price': df['low'].min(),
                'max_price': df['high'].max(),
                'avg_price': df['close'].mean()
            },
            'volume_stats': {
                'total_volume': df['volume'].sum(),
                'avg_volume': df['volume'].mean(),
                'zero_volume_count': (df['volume'] == 0).sum()
            }
        }
        
        return {
            'status': 'ok' if not errors else 'error',
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }
    
    def download_all_timeframes(self, symbol: str = 'BTC/USDT') -> Dict[str, pd.DataFrame]:
        """
        Загрузка данных для всех таймфреймов
        
        Args:
            symbol: Торговая пара
            
        Returns:
            Dict: Словарь с данными по таймфреймам
        """
        logger.info("🚀 НАЧИНАЕМ ЗАГРУЗКУ МУЛЬТИТАЙМФРЕЙМ ДАННЫХ")
        logger.info("=" * 60)
        logger.info(f"🎯 Символ: {symbol}")
        logger.info(f"📅 Период: последние 2 года") 
        logger.info(f"⏰ Таймфреймы: {list(self.TIMEFRAMES.keys())}")
        logger.info("=" * 60)
        
        results = {}
        
        for timeframe in self.TIMEFRAMES.keys():
            try:
                logger.info(f"\n🔄 Загрузка {timeframe}...")
                
                # Загружаем данные
                df = self.download_timeframe_data(symbol, timeframe)
                
                if not df.empty:
                    # Валидация данных
                    validation = self.validate_data(df, timeframe)
                    
                    if validation['status'] == 'ok':
                        # Сохраняем данные
                        symbol_clean = symbol.replace('/', '')
                        filename = f"{symbol_clean}_{timeframe}_2y.csv"
                        self.save_data(df, filename)
                        
                        results[timeframe] = df
                        
                        # Выводим статистику
                        stats = validation['stats']
                        logger.info(f"📊 Статистика {timeframe}:")
                        logger.info(f"   📈 Записей: {stats['total_records']:,}")
                        logger.info(f"   📅 Период: {stats['date_range']['start']} - {stats['date_range']['end']}")
                        logger.info(f"   💰 Цены: ${stats['price_range']['min_price']:.2f} - ${stats['price_range']['max_price']:.2f}")
                        logger.info(f"   📊 Средняя цена: ${stats['price_range']['avg_price']:.2f}")
                        
                        if validation['warnings']:
                            logger.warning(f"⚠️ Предупреждения: {validation['warnings']}")
                    else:
                        logger.error(f"❌ Ошибки валидации {timeframe}: {validation['errors']}")
                else:
                    logger.error(f"❌ Не удалось загрузить данные для {timeframe}")
                    
            except Exception as e:
                logger.error(f"❌ Критическая ошибка при загрузке {timeframe}: {e}")
                continue
            
            # Пауза между таймфреймами
            time.sleep(1)
        
        # Итоговая статистика
        logger.info("\n" + "=" * 60)
        logger.info("📋 ИТОГОВАЯ СТАТИСТИКА ЗАГРУЗКИ")
        logger.info("=" * 60)
        
        for timeframe, df in results.items():
            size_mb = len(df) * df.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(f"✅ {timeframe:>3}: {len(df):>7,} записей ({size_mb:.1f} MB)")
        
        logger.info(f"📁 Общее количество файлов: {len(results)}")
        logger.info("🎉 ЗАГРУЗКА ЗАВЕРШЕНА!")
        
        return results


def main():
    """
    Основная функция для запуска загрузки
    """
    print("🚀 BINANCE МУЛЬТИТАЙМФРЕЙМ ЗАГРУЗЧИК")
    print("📊 Загрузка данных BTC/USDT за последние 2 года")
    print("⏰ Таймфреймы: 5m, 1h, 4h, 1d")
    print("=" * 50)
    
    try:
        # Создаем загрузчик
        loader = BinanceMultiTimeframeLoader()
        
        # Загружаем данные для всех таймфреймов
        results = loader.download_all_timeframes('BTC/USDT')
        
        if results:
            print(f"\n✅ Успешно загружены данные для {len(results)} таймфреймов")
            print("📁 Файлы сохранены в папку 'data/'")
        else:
            print("❌ Не удалось загрузить данные")
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # Обновляем requirements.txt если нужно
    requirements_path = Path("requirements.txt")
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            content = f.read()
        
        if 'ccxt' not in content:
            with open(requirements_path, 'a') as f:
                f.write('\n# Для работы с криптобиржами\nccxt>=4.0.0\n')
            print("📝 Добавлен ccxt в requirements.txt")
    
    # Запускаем загрузку
    success = main()
    
    if success:
        print("\n🎯 Для использования загруженных данных:")
        print("   - Данные сохранены в CSV формате в папке data/")
        print("   - Формат: timestamp,open,high,low,close,volume")
        print("   - Можно использовать в существующих торговых алгоритмах") 