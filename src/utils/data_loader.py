"""
📊 МОДУЛЬ ЗАГРУЗКИ ИСТОРИЧЕСКИХ ДАННЫХ
Поддержка различных источников данных для расширения до 6+ лет истории
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import ccxt
import yfinance as yf

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """Загрузчик исторических данных из различных источников"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
        
        # Инициализация бирж
        self.exchanges = {
            'binance': ccxt.binance(),
            'okx': ccxt.okx(),
            'bybit': ccxt.bybit()
        }
    
    def load_yahoo_finance_data(self, symbol: str = "BTC-USD", period: str = "6y", interval: str = "5m") -> pd.DataFrame:
        """
        Загрузка данных из Yahoo Finance
        
        Args:
            symbol: Символ (BTC-USD, ETH-USD)
            period: Период (6y, 5y, 3y, 1y)
            interval: Интервал (1m, 5m, 15m, 1h, 1d)
        """
        try:
            logger.info(f"📥 Загрузка данных {symbol} с Yahoo Finance...")
            
            ticker = yf.Ticker(symbol)
            
            # Yahoo Finance ограничения для разных интервалов
            if interval in ['1m', '2m', '5m']:
                # Для минутных данных максимум 7 дней
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            elif interval in ['15m', '30m', '1h']:
                # Для часовых данных максимум 60 дней
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                # Для дневных данных можно больше
                data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"⚠️ Нет данных для {symbol}")
                return pd.DataFrame()
            
            # Преобразование в стандартный формат
            df = pd.DataFrame()
            df['timestamp'] = data.index.astype(int) // 10**6  # в миллисекундах
            df['open'] = data['Open'].values
            df['high'] = data['High'].values
            df['low'] = data['Low'].values
            df['close'] = data['Close'].values
            df['volume'] = data['Volume'].values
            
            logger.info(f"✅ Загружено {len(df)} записей из Yahoo Finance")
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def load_binance_historical(self, symbol: str = "BTCUSDT", interval: str = "5m", 
                               start_date: str = "2018-01-01", limit: int = 1000) -> pd.DataFrame:
        """
        Загрузка исторических данных с Binance
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал (1m, 5m, 15m, 1h, 1d)
            start_date: Начальная дата (YYYY-MM-DD)
            limit: Количество записей за запрос (макс 1000)
        """
        try:
            logger.info(f"📥 Загрузка данных {symbol} с Binance...")
            
            exchange = self.exchanges['binance']
            
            # Конвертация даты в timestamp
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            current_ts = int(datetime.now().timestamp() * 1000)
            
            all_data = []
            
            while start_ts < current_ts:
                try:
                    # Загружаем данные частями
                    ohlcv = exchange.fetch_ohlcv(
                        symbol, interval, since=start_ts, limit=limit
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    start_ts = ohlcv[-1][0] + 1  # Следующий timestamp
                    
                    logger.info(f"📊 Загружено {len(ohlcv)} записей, всего: {len(all_data)}")
                    time.sleep(0.1)  # Соблюдаем rate limits
                    
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка при загрузке: {e}")
                    break
            
            if not all_data:
                return pd.DataFrame()
            
            # Преобразование в DataFrame
            df = pd.DataFrame(all_data)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            logger.info(f"✅ Загружено {len(df)} записей с Binance")
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки Binance: {e}")
            return pd.DataFrame()
    
    def load_multiple_timeframes(self, symbol: str = "BTCUSDT", 
                               timeframes: List[str] = ["5m", "15m", "1h", "4h", "1d"]) -> Dict[str, pd.DataFrame]:
        """Загрузка данных на разных таймфреймах"""
        
        logger.info(f"📈 Загрузка мультитаймфрейм данных для {symbol}")
        
        data = {}
        for tf in timeframes:
            try:
                df = self.load_binance_historical(symbol, tf, "2018-01-01")
                if not df.empty:
                    data[tf] = df
                    self.save_data(df, f"{symbol}_{tf}_6y.csv")
                    logger.info(f"✅ {tf}: {len(df)} записей")
                else:
                    logger.warning(f"⚠️ {tf}: Нет данных")
            except Exception as e:
                logger.error(f"❌ {tf}: {e}")
        
        return data
    
    def combine_datasets(self, files: List[str]) -> pd.DataFrame:
        """Объединение нескольких файлов данных"""
        
        logger.info("🔄 Объединение датасетов...")
        
        combined_data = []
        for file in files:
            file_path = self.data_folder / file
            if file_path.exists():
                df = pd.read_csv(file_path)
                combined_data.append(df)
                logger.info(f"📁 {file}: {len(df)} записей")
        
        if not combined_data:
            return pd.DataFrame()
        
        # Объединяем и удаляем дубликаты
        result = pd.concat(combined_data, ignore_index=True)
        result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        logger.info(f"✅ Объединено: {len(result)} уникальных записей")
        return result
    
    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """Сохранение данных в CSV"""
        
        file_path = self.data_folder / filename
        df.to_csv(file_path, index=False)
        logger.info(f"💾 Сохранено: {filename} ({len(df)} записей)")
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Проверка качества данных"""
        
        if df.empty:
            return {'status': 'empty'}
        
        quality = {
            'total_records': len(df),
            'date_range': {
                'start': pd.to_datetime(df['timestamp'].min(), unit='ms'),
                'end': pd.to_datetime(df['timestamp'].max(), unit='ms')
            },
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated(subset=['timestamp']).sum(),
            'price_anomalies': {
                'negative_prices': (df[['open', 'high', 'low', 'close']] < 0).sum().sum(),
                'zero_volume': (df['volume'] == 0).sum(),
                'ohlc_errors': ((df['high'] < df['low']) | 
                              (df['close'] > df['high']) | 
                              (df['close'] < df['low']) |
                              (df['open'] > df['high']) | 
                              (df['open'] < df['low'])).sum()
            }
        }
        
        return quality
    
    def create_extended_dataset(self) -> pd.DataFrame:
        """Создание расширенного 6-летнего датасета"""
        
        logger.info("🚀 СОЗДАНИЕ РАСШИРЕННОГО ДАТАСЕТА (6 ЛЕТ)")
        logger.info("=" * 50)
        
        # 1. Загружаем данные с разных источников
        datasets = []
        
        # Binance данные (основной источник)
        try:
            binance_data = self.load_binance_historical("BTCUSDT", "5m", "2018-01-01")
            if not binance_data.empty:
                datasets.append(binance_data)
                logger.info(f"📈 Binance: {len(binance_data)} записей")
        except Exception as e:
            logger.error(f"❌ Binance error: {e}")
        
        # Существующие данные
        existing_files = ["BTC_5_96w.csv", "BTC_5_2w.csv"]
        for file in existing_files:
            file_path = self.data_folder / file
            if file_path.exists():
                df = pd.read_csv(file_path)
                datasets.append(df)
                logger.info(f"📁 {file}: {len(df)} записей")
        
        # Объединяем все данные
        if datasets:
            combined = pd.concat(datasets, ignore_index=True)
            combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Проверка качества
            quality = self.validate_data_quality(combined)
            logger.info(f"📊 ИТОГО: {quality['total_records']} записей")
            logger.info(f"📅 Период: {quality['date_range']['start']} - {quality['date_range']['end']}")
            
            # Сохраняем расширенный датасет
            self.save_data(combined, "BTC_5m_6years_extended.csv")
            
            return combined
        
        logger.warning("⚠️ Не удалось создать расширенный датасет")
        return pd.DataFrame()


# Удобные функции для быстрого использования
def download_6year_data():
    """Быстрая загрузка 6-летних данных"""
    loader = HistoricalDataLoader()
    return loader.create_extended_dataset()


def get_multi_timeframe_data():
    """Загрузка данных на разных таймфреймах"""
    loader = HistoricalDataLoader()
    return loader.load_multiple_timeframes()


if __name__ == "__main__":
    # Демонстрация использования
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Загрузка расширенных исторических данных...")
    
    loader = HistoricalDataLoader()
    extended_data = loader.create_extended_dataset()
    
    if not extended_data.empty:
        print(f"✅ Успешно создан расширенный датасет: {len(extended_data)} записей")
    else:
        print("❌ Не удалось создать расширенный датасет") 