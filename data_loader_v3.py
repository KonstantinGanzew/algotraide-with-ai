"""
🚀 DATA LOADER V3.0 - Мульти-активы торговая система
Загрузчик данных для множественных криптовалют с корреляционным анализом
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import os


class MultiAssetDataLoader:
    """Загрузчик данных для множественных криптовалют"""
    
    # Поддерживаемые активы
    SUPPORTED_ASSETS = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD', 
        'BNB': 'BNB-USD',
        'ADA': 'ADA-USD',
        'SOL': 'SOL-USD',
        'MATIC': 'MATIC-USD',
        'DOT': 'DOT-USD',
        'AVAX': 'AVAX-USD'
    }
    
    def __init__(self, data_folder: str = "data/"):
        self.data_folder = data_folder
        self.ensure_data_folder()
    
    def ensure_data_folder(self):
        """Создание папки для данных если не существует"""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
    
    def download_crypto_data(self, symbol: str, period: str = "2y", interval: str = "5m") -> pd.DataFrame:
        """
        Загрузка данных криптовалюты через Yahoo Finance
        
        Args:
            symbol: Символ криптовалюты (BTC, ETH, etc.)
            period: Период данных (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Интервал (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        if symbol not in self.SUPPORTED_ASSETS:
            raise ValueError(f"Неподдерживаемый актив: {symbol}")
        
        ticker = self.SUPPORTED_ASSETS[symbol]
        print(f"📊 Загружаю данные {symbol} ({ticker})...")
        
        try:
            # Загрузка через yfinance
            crypto = yf.Ticker(ticker)
            data = crypto.history(period=period, interval=interval)
            
            if data.empty:
                raise Exception(f"Пустые данные для {symbol}")
            
            # Стандартизация колонок
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Добавление символа
            data['symbol'] = symbol
            
            print(f"✅ Загружено {len(data)} записей для {symbol}")
            return data
            
        except Exception as e:
            print(f"❌ Ошибка загрузки {symbol}: {e}")
            return pd.DataFrame()
    
    def download_all_assets(self, period: str = "2y", interval: str = "5m") -> Dict[str, pd.DataFrame]:
        """Загрузка данных всех поддерживаемых активов"""
        all_data = {}
        
        for symbol in self.SUPPORTED_ASSETS.keys():
            data = self.download_crypto_data(symbol, period, interval)
            if not data.empty:
                all_data[symbol] = data
                # Сохранение в файл
                filename = f"{symbol}_{interval}_{period}.csv"
                filepath = os.path.join(self.data_folder, filename)
                data.to_csv(filepath)
                print(f"💾 Сохранено: {filepath}")
            
            # Задержка между запросами
            time.sleep(1)
        
        return all_data
    
    def load_existing_data(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Загрузка существующих данных из файлов"""
        if symbols is None:
            symbols = list(self.SUPPORTED_ASSETS.keys())
        
        data = {}
        for symbol in symbols:
            # Поиск файлов с данными для символа
            files = [f for f in os.listdir(self.data_folder) if f.startswith(symbol)]
            
            if files:
                # Берем самый новый файл
                latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(self.data_folder, x)))
                filepath = os.path.join(self.data_folder, latest_file)
                
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    data[symbol] = df
                    print(f"📂 Загружен {symbol}: {len(df)} записей из {latest_file}")
                except Exception as e:
                    print(f"❌ Ошибка чтения {symbol}: {e}")
        
        return data
    
    def calculate_correlations(self, data: Dict[str, pd.DataFrame], window: int = 24*7) -> pd.DataFrame:
        """
        Расчет корреляций между активами
        
        Args:
            data: Словарь с данными активов
            window: Окно для скользящей корреляции (в периодах)
        """
        if len(data) < 2:
            print("⚠️ Недостаточно активов для корреляционного анализа")
            return pd.DataFrame()
        
        # Объединение данных по цене закрытия
        price_data = pd.DataFrame()
        for symbol, df in data.items():
            if 'close' in df.columns:
                price_data[symbol] = df['close']
        
        # Расчет корреляций
        correlations = price_data.corr()
        
        # Скользящие корреляции
        rolling_corr = {}
        symbols = list(price_data.columns)
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i+1:], i+1):
                pair = f"{sym1}-{sym2}"
                rolling_corr[pair] = price_data[sym1].rolling(window).corr(price_data[sym2])
        
        rolling_corr_df = pd.DataFrame(rolling_corr)
        
        print("📊 Корреляционная матрица:")
        print(correlations.round(3))
        
        return correlations, rolling_corr_df
    
    def prepare_multi_asset_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Подготовка объединенных признаков для всех активов
        """
        combined_features = pd.DataFrame()
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # Базовые признаки
            features = pd.DataFrame(index=df.index)
            features[f'{symbol}_close'] = df['close']
            features[f'{symbol}_volume'] = df['volume']
            features[f'{symbol}_high'] = df['high']
            features[f'{symbol}_low'] = df['low']
            
            # Технические индикаторы
            features[f'{symbol}_returns'] = df['close'].pct_change()
            features[f'{symbol}_volatility'] = features[f'{symbol}_returns'].rolling(24).std()
            
            # EMA
            features[f'{symbol}_ema_12'] = df['close'].ewm(span=12).mean()
            features[f'{symbol}_ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features[f'{symbol}_rsi'] = 100 - (100 / (1 + rs))
            
            # Объединение
            if combined_features.empty:
                combined_features = features
            else:
                combined_features = combined_features.join(features, how='outer')
        
        # Межактивные признаки
        symbols = list(data.keys())
        if len(symbols) >= 2:
            # Спреды между активами
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols[i+1:], i+1):
                    if f'{sym1}_close' in combined_features.columns and f'{sym2}_close' in combined_features.columns:
                        spread = combined_features[f'{sym1}_close'] / combined_features[f'{sym2}_close']
                        combined_features[f'{sym1}_{sym2}_spread'] = spread
                        combined_features[f'{sym1}_{sym2}_spread_ma'] = spread.rolling(24).mean()
        
        # Удаление NaN
        combined_features = combined_features.dropna()
        
        print(f"🔧 Подготовлено {len(combined_features.columns)} признаков для {len(symbols)} активов")
        print(f"📈 Данные: {len(combined_features)} записей")
        
        return combined_features


def quick_test():
    """Быстрый тест загрузчика"""
    print("🧪 Тест Multi-Asset Data Loader V3.0")
    print("=" * 50)
    
    loader = MultiAssetDataLoader()
    
    # Тест загрузки одного актива
    print("\n1️⃣ Тест загрузки BTC...")
    btc_data = loader.download_crypto_data('BTC', period='5d', interval='1h')
    print(f"Результат: {len(btc_data)} записей")
    
    # Тест загрузки нескольких активов
    print("\n2️⃣ Тест загрузки ETH и BNB...")
    multi_data = {}
    for symbol in ['ETH', 'BNB']:
        data = loader.download_crypto_data(symbol, period='5d', interval='1h')
        if not data.empty:
            multi_data[symbol] = data
    
    # Тест корреляций
    if len(multi_data) >= 2:
        print("\n3️⃣ Тест корреляционного анализа...")
        correlations, rolling_corr = loader.calculate_correlations(multi_data)
    
    # Тест объединенных признаков
    if multi_data:
        print("\n4️⃣ Тест подготовки признаков...")
        features = loader.prepare_multi_asset_features(multi_data)
        print(f"Признаки: {list(features.columns[:10])}")
    
    print("\n✅ Тест завершен!")


if __name__ == "__main__":
    quick_test() 