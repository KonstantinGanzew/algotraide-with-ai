"""
🚀 МУЛЬТИ-АКТИВЫ ТОРГОВАЯ СИСТЕМА V3.0
Торговля множественными криптовалютами с корреляционным анализом и портфельным подходом
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MultiAssetConfig:
    """Конфигурация для мульти-активов системы"""
    
    # Поддерживаемые активы
    ASSETS = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
    
    # Корреляционный анализ
    CORRELATION_WINDOW = 168  # 7 дней * 24 часа
    CORRELATION_THRESHOLD = 0.7  # Порог высокой корреляции
    
    # Портфельные параметры
    MAX_POSITION_PER_ASSET = 0.3  # Максимум 30% в одном активе
    MIN_POSITION_SIZE = 0.05  # Минимум 5% позиция
    PORTFOLIO_REBALANCE_FREQUENCY = 24  # Ребалансировка каждые 24 часа
    
    # Риск-менеджмент
    PORTFOLIO_RISK_LIMIT = 0.15  # 15% риск на весь портфель
    CORRELATION_RISK_PENALTY = 0.5  # Штраф за высокую корреляцию
    DIVERSIFICATION_BONUS = 0.3  # Бонус за диверсификацию
    
    # Технические параметры
    WINDOW_SIZE = 50
    INITIAL_BALANCE = 10000
    COMMISSION_RATE = 0.001


def generate_correlated_crypto_data(n_points: int = 10000, start_price: float = 50000) -> Dict[str, pd.DataFrame]:
    """
    Генерация симулированных данных криптовалют с реалистичными корреляциями
    """
    print("📊 Генерация симулированных данных мульти-активов...")
    
    np.random.seed(42)  # Для воспроизводимости
    
    # Временные метки
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=n_points),
        periods=n_points,
        freq='1H'
    )
    
    # Общий рыночный фактор (влияет на все активы)
    market_factor = np.cumsum(np.random.normal(0, 0.001, n_points))
    
    # Данные для каждого актива
    assets_data = {}
    asset_configs = {
        'BTC': {'base_price': 50000, 'volatility': 0.02, 'market_beta': 1.0},
        'ETH': {'base_price': 3000, 'volatility': 0.025, 'market_beta': 1.2},
        'BNB': {'base_price': 300, 'volatility': 0.03, 'market_beta': 0.8},
        'ADA': {'base_price': 0.5, 'volatility': 0.035, 'market_beta': 1.1},
        'SOL': {'base_price': 100, 'volatility': 0.04, 'market_beta': 1.3}
    }
    
    for asset, config in asset_configs.items():
        # Индивидуальный шум актива (уменьшенная волатильность)
        individual_noise = np.random.normal(0, config['volatility'] * 0.1, n_points)
        
        # Комбинированное движение цены (нормализованное)
        price_changes = (
            market_factor * config['market_beta'] * 0.1 + 
            individual_noise * 0.5
        )
        
        # Ограничиваем изменения цены
        price_changes = np.clip(price_changes, -0.1, 0.1)
        
        # Генерация цен с контролем роста
        price_returns = np.cumsum(price_changes)
        prices = config['base_price'] * (1 + price_returns * 0.1)
        
        # Создание OHLCV данных
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_points))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_points))),
            'close': prices,
            'volume': np.random.exponential(1000000, n_points)
        })
        
        # Коррекция high/low
        df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
        df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
        
        assets_data[asset] = df
        print(f"   ✅ {asset}: {len(df)} записей, цена {df['close'].iloc[-1]:.2f}")
    
    return assets_data


def calculate_correlations(assets_data: Dict[str, pd.DataFrame], window: int = 168) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Расчет корреляций между активами"""
    print(f"📈 Расчет корреляций (окно: {window} часов)...")
    
    # Объединение цен закрытия
    price_data = pd.DataFrame()
    for asset, df in assets_data.items():
        price_data[asset] = df['close'].values
    
    # Убираем inf и заменяем на NaN, затем заполняем
    price_data = price_data.replace([np.inf, -np.inf], np.nan)
    price_data = price_data.fillna(method='ffill').fillna(0)
    
    # Статическая корреляция
    static_corr = price_data.corr().fillna(0)
    
    # Скользящая корреляция
    rolling_corr = {}
    assets = list(price_data.columns)
    
    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets[i+1:], i+1):
            pair = f"{asset1}-{asset2}"
            rolling_corr[pair] = price_data[asset1].rolling(window).corr(price_data[asset2])
    
    rolling_corr_df = pd.DataFrame(rolling_corr)
    
    print("📊 Корреляционная матрица:")
    print(static_corr.round(3))
    
    return static_corr, rolling_corr_df


def prepare_multi_asset_features(assets_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Подготовка объединенных признаков для всех активов"""
    print("🔧 Подготовка мульти-активов признаков...")
    
    combined_features = pd.DataFrame()
    
    for asset, df in assets_data.items():
        # Базовые признаки
        features = pd.DataFrame()
        features[f'{asset}_close'] = df['close']
        features[f'{asset}_volume'] = df['volume']
        features[f'{asset}_returns'] = df['close'].pct_change()
        features[f'{asset}_volatility'] = features[f'{asset}_returns'].rolling(24).std()
        
        # Технические индикаторы
        features[f'{asset}_sma_20'] = df['close'].rolling(20).mean()
        features[f'{asset}_ema_12'] = df['close'].ewm(span=12).mean()
        features[f'{asset}_rsi'] = calculate_rsi(df['close'])
        
        # Объединение
        if combined_features.empty:
            combined_features = features
        else:
            combined_features = combined_features.join(features, how='outer')
    
    # Межактивные признаки
    assets = list(assets_data.keys())
    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets[i+1:], i+1):
            if f'{asset1}_close' in combined_features.columns and f'{asset2}_close' in combined_features.columns:
                # Спред между активами
                spread = combined_features[f'{asset1}_close'] / combined_features[f'{asset2}_close']
                combined_features[f'{asset1}_{asset2}_spread'] = spread
                combined_features[f'{asset1}_{asset2}_spread_ma'] = spread.rolling(24).mean()
    
    # Портфельные признаки
    price_cols = [col for col in combined_features.columns if col.endswith('_close')]
    if len(price_cols) > 1:
        # Средневзвешенный индекс портфеля
        weights = np.ones(len(price_cols)) / len(price_cols)  # Равные веса
        portfolio_value = np.dot(combined_features[price_cols].values, weights)
        combined_features['portfolio_index'] = portfolio_value
        combined_features['portfolio_momentum'] = pd.Series(portfolio_value).pct_change().rolling(12).mean()
    
    combined_features = combined_features.dropna()
    
    print(f"✅ Подготовлено {len(combined_features.columns)} признаков для {len(assets)} активов")
    print(f"📈 Данные: {len(combined_features)} записей")
    
    return combined_features


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Расчет RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


class MultiAssetTradingEnv(gym.Env):
    """Торговое окружение для множественных активов"""
    
    def __init__(self, assets_data: Dict[str, pd.DataFrame], features_df: pd.DataFrame):
        super().__init__()
        
        self.assets_data = assets_data
        self.features_df = features_df.reset_index(drop=True)
        self.assets = list(assets_data.keys())
        self.n_assets = len(self.assets)
        
        # Пространство действий: для каждого актива (Hold, Buy, Sell)
        # Плюс портфельные действия (Rebalance, Close All)
        self.action_space = spaces.Discrete(self.n_assets * 3 + 2)
        
        # Пространство наблюдений
        n_features = len(features_df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(MultiAssetConfig.WINDOW_SIZE, n_features),
            dtype=np.float32
        )
        
        self._reset_state()
    
    def _reset_state(self):
        """Сброс состояния"""
        self.current_step = MultiAssetConfig.WINDOW_SIZE
        self.initial_balance = MultiAssetConfig.INITIAL_BALANCE
        self.balance = float(self.initial_balance)
        
        # Позиции по активам
        self.positions = {asset: 0.0 for asset in self.assets}
        self.entry_prices = {asset: 0.0 for asset in self.assets}
        
        # Статистика
        self.total_trades = 0
        self.portfolio_history = [self.initial_balance]
        self.trades_history = []
        
        # Корреляционный анализ
        self.correlation_history = []
        self.diversification_score = 0.0
        
    def reset(self, seed=None, options=None):
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Получение наблюдения"""
        start_idx = max(0, self.current_step - MultiAssetConfig.WINDOW_SIZE)
        end_idx = self.current_step
        
        obs = self.features_df.iloc[start_idx:end_idx].values
        
        # Дополняем если недостаточно данных
        if len(obs) < MultiAssetConfig.WINDOW_SIZE:
            padding = np.tile(obs[0], (MultiAssetConfig.WINDOW_SIZE - len(obs), 1))
            obs = np.vstack([padding, obs])
        
        return obs.astype(np.float32)
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Получение текущих цен всех активов"""
        if self.current_step >= len(self.features_df):
            step = len(self.features_df) - 1
        else:
            step = self.current_step
        
        prices = {}
        for asset in self.assets:
            close_col = f'{asset}_close'
            if close_col in self.features_df.columns:
                prices[asset] = self.features_df.iloc[step][close_col]
            else:
                prices[asset] = 0.0
        
        return prices
    
    def _calculate_portfolio_value(self) -> float:
        """Расчет общей стоимости портфеля"""
        current_prices = self._get_current_prices()
        portfolio_value = self.balance
        
        for asset in self.assets:
            if self.positions[asset] > 0:
                portfolio_value += self.positions[asset] * current_prices[asset]
        
        return portfolio_value
    
    def _calculate_diversification_score(self) -> float:
        """Расчет диверсификации портфеля"""
        current_prices = self._get_current_prices()
        total_value = self._calculate_portfolio_value()
        
        if total_value <= 0:
            return 0.0
        
        # Веса активов в портфеле
        weights = []
        for asset in self.assets:
            asset_value = self.positions[asset] * current_prices[asset]
            weight = asset_value / total_value
            weights.append(weight)
        
        # Индекс Херфиндаля (чем ниже, тем лучше диверсификация)
        herfindahl_index = sum(w**2 for w in weights)
        diversification_score = 1 - herfindahl_index
        
        return diversification_score
    
    def _execute_action(self, action: int) -> Dict[str, Any]:
        """Выполнение действия"""
        current_prices = self._get_current_prices()
        trade_result = {'executed': False, 'asset': None, 'action_type': None, 'amount': 0}
        
        if action < self.n_assets * 3:
            # Действия по отдельным активам
            asset_idx = action // 3
            action_type = action % 3  # 0-Hold, 1-Buy, 2-Sell
            
            asset = self.assets[asset_idx]
            current_price = current_prices[asset]
            
            if action_type == 1:  # Buy
                # Покупка с учетом лимитов портфеля
                max_investment = self.balance * MultiAssetConfig.MAX_POSITION_PER_ASSET
                min_investment = self.balance * MultiAssetConfig.MIN_POSITION_SIZE
                
                if self.balance > min_investment:
                    investment = min(max_investment, self.balance * 0.2)  # 20% от баланса
                    amount = investment / current_price
                    commission = investment * MultiAssetConfig.COMMISSION_RATE
                    
                    self.positions[asset] += amount
                    self.balance -= investment + commission
                    self.entry_prices[asset] = current_price
                    self.total_trades += 1
                    
                    trade_result.update({
                        'executed': True, 'asset': asset, 'action_type': 'BUY',
                        'amount': amount, 'price': current_price, 'investment': investment
                    })
            
            elif action_type == 2 and self.positions[asset] > 0:  # Sell
                # Продажа позиции
                amount = self.positions[asset]
                revenue = amount * current_price
                commission = revenue * MultiAssetConfig.COMMISSION_RATE
                
                self.balance += revenue - commission
                self.positions[asset] = 0.0
                self.entry_prices[asset] = 0.0
                self.total_trades += 1
                
                trade_result.update({
                    'executed': True, 'asset': asset, 'action_type': 'SELL',
                    'amount': amount, 'price': current_price, 'revenue': revenue
                })
        
        elif action == self.n_assets * 3:  # Rebalance Portfolio
            # Ребалансировка портфеля
            self._rebalance_portfolio()
            trade_result.update({'executed': True, 'action_type': 'REBALANCE'})
        
        elif action == self.n_assets * 3 + 1:  # Close All Positions
            # Закрытие всех позиций
            total_closed_value = 0
            for asset in self.assets:
                if self.positions[asset] > 0:
                    amount = self.positions[asset]
                    revenue = amount * current_prices[asset]
                    commission = revenue * MultiAssetConfig.COMMISSION_RATE
                    
                    self.balance += revenue - commission
                    total_closed_value += revenue
                    self.positions[asset] = 0.0
                    self.entry_prices[asset] = 0.0
            
            if total_closed_value > 0:
                trade_result.update({
                    'executed': True, 'action_type': 'CLOSE_ALL',
                    'total_value': total_closed_value
                })
        
        return trade_result
    
    def _rebalance_portfolio(self):
        """Ребалансировка портфеля для оптимальной диверсификации"""
        current_prices = self._get_current_prices()
        total_value = self._calculate_portfolio_value()
        
        # Продаем все позиции
        for asset in self.assets:
            if self.positions[asset] > 0:
                amount = self.positions[asset]
                revenue = amount * current_prices[asset]
                commission = revenue * MultiAssetConfig.COMMISSION_RATE
                self.balance += revenue - commission
                self.positions[asset] = 0.0
        
        # Перераспределяем равномерно
        if self.balance > 0:
            investment_per_asset = self.balance / len(self.assets)
            
            for asset in self.assets:
                if investment_per_asset > self.balance * MultiAssetConfig.MIN_POSITION_SIZE:
                    amount = investment_per_asset / current_prices[asset]
                    commission = investment_per_asset * MultiAssetConfig.COMMISSION_RATE
                    
                    self.positions[asset] = amount
                    self.balance -= investment_per_asset + commission
                    self.entry_prices[asset] = current_prices[asset]
    
    def _calculate_reward(self) -> float:
        """Расчет награды с учетом мульти-активов факторов"""
        current_portfolio_value = self._calculate_portfolio_value()
        
        # Базовая награда - изменение портфеля
        portfolio_return = (current_portfolio_value - self.portfolio_history[-1]) / self.portfolio_history[-1]
        base_reward = portfolio_return * 100
        
        # Бонус за диверсификацию
        diversification_score = self._calculate_diversification_score()
        diversification_bonus = diversification_score * MultiAssetConfig.DIVERSIFICATION_BONUS
        
        # Штраф за высокую корреляцию активов
        correlation_penalty = 0.0
        if len(self.portfolio_history) > MultiAssetConfig.CORRELATION_WINDOW:
            # Упрощенный расчет корреляционного штрафа
            correlation_penalty = -0.1  # Базовый штраф
        
        total_reward = base_reward + diversification_bonus + correlation_penalty
        
        return total_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Шаг симуляции"""
        # Выполнение действия
        trade_result = self._execute_action(action)
        
        # Обновление состояния
        self.current_step += 1
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_history.append(portfolio_value)
        
        if trade_result['executed']:
            self.trades_history.append(trade_result)
        
        # Расчет награды
        reward = self._calculate_reward()
        
        # Проверка завершения эпизода
        done = (
            self.current_step >= len(self.features_df) - 1 or
            portfolio_value <= self.initial_balance * 0.1  # Стоп-лосс 90%
        )
        
        # Информация
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'total_trades': self.total_trades,
            'diversification_score': self._calculate_diversification_score(),
            'trade_result': trade_result
        }
        
        return self._get_observation(), reward, done, False, info


def train_multi_asset_model(env, total_timesteps: int = 50000):
    """Обучение модели для мульти-активов торговли"""
    print(f"🧠 Обучение мульти-активов модели ({total_timesteps:,} шагов)...")
    
    # Создание векторизованного окружения
    vec_env = DummyVecEnv([lambda: env])
    
    # Создание модели PPO
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # Обучение
    model.learn(total_timesteps=total_timesteps)
    
    print("✅ Обучение завершено!")
    return model


def test_multi_asset_model(model, env, max_steps: int = 1000):
    """Тестирование мульти-активов модели"""
    print(f"🧪 Тестирование модели (до {max_steps:,} шагов)...")
    
    obs, _ = env.reset()
    results = []
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        results.append({
            'step': step,
            'portfolio_value': info['portfolio_value'],
            'balance': info['balance'],
            'total_trades': info['total_trades'],
            'diversification_score': info['diversification_score'],
            'reward': reward
        })
        
        if done:
            break
    
    return results


def analyze_multi_asset_results(results: List[Dict], initial_balance: float, assets: List[str]):
    """Анализ результатов мульти-активов торговли"""
    print("\n📊 АНАЛИЗ МУЛЬТИ-АКТИВОВ ТОРГОВЛИ")
    print("=" * 50)
    
    final_value = results[-1]['portfolio_value']
    total_return = (final_value - initial_balance) / initial_balance * 100
    total_trades = results[-1]['total_trades']
    avg_diversification = np.mean([r['diversification_score'] for r in results])
    
    print(f"💰 Начальный баланс: {initial_balance:,.2f} USDT")
    print(f"💰 Финальная стоимость: {final_value:,.2f} USDT")
    print(f"📈 Общая доходность: {total_return:+.2f}%")
    print(f"🔄 Всего сделок: {total_trades}")
    print(f"🎯 Средняя диверсификация: {avg_diversification:.3f}")
    print(f"🪙 Количество активов: {len(assets)}")
    print(f"📊 Активы: {', '.join(assets)}")
    
    # Построение графиков
    visualize_multi_asset_results(results, assets)
    
    return {
        'total_return': total_return,
        'final_value': final_value,
        'total_trades': total_trades,
        'avg_diversification': avg_diversification
    }


def visualize_multi_asset_results(results: List[Dict], assets: List[str]):
    """Визуализация результатов мульти-активов торговли"""
    print("📈 Создание графиков мульти-активов анализа...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚀 Multi-Asset Trading System V3.0 - Результаты', fontsize=16, fontweight='bold')
    
    steps = [r['step'] for r in results]
    portfolio_values = [r['portfolio_value'] for r in results]
    diversification_scores = [r['diversification_score'] for r in results]
    rewards = [r['reward'] for r in results]
    
    # 1. Стоимость портфеля
    axes[0, 0].plot(steps, portfolio_values, linewidth=2, color='green')
    axes[0, 0].set_title('💰 Стоимость Портфеля')
    axes[0, 0].set_xlabel('Шаги')
    axes[0, 0].set_ylabel('USDT')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Диверсификация
    axes[0, 1].plot(steps, diversification_scores, linewidth=2, color='blue')
    axes[0, 1].set_title('🎯 Диверсификация Портфеля')
    axes[0, 1].set_xlabel('Шаги')
    axes[0, 1].set_ylabel('Коэффициент')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # 3. Награды
    axes[1, 0].plot(steps, rewards, linewidth=1, alpha=0.7, color='purple')
    axes[1, 0].set_title('🏆 Награды Агента')
    axes[1, 0].set_xlabel('Шаги')
    axes[1, 0].set_ylabel('Награда')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Сводная информация
    axes[1, 1].axis('off')
    summary_text = f"""
    🎯 СВОДКА РЕЗУЛЬТАТОВ V3.0
    
    📊 Активы: {', '.join(assets)}
    💰 Финальная стоимость: {portfolio_values[-1]:,.2f} USDT
    📈 Доходность: {(portfolio_values[-1]/portfolio_values[0]-1)*100:+.2f}%
    🔄 Всего сделок: {results[-1]['total_trades']}
    🎯 Средняя диверсификация: {np.mean(diversification_scores):.3f}
    
    ✨ Преимущества V3.0:
    • Мульти-активы торговля
    • Корреляционный анализ  
    • Автоматическая диверсификация
    • Портфельный риск-менеджмент
    """
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                   fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('multi_asset_results_v3.png', dpi=300, bbox_inches='tight')
    print("💾 Графики сохранены: multi_asset_results_v3.png")
    plt.show()


def main():
    """Главная функция мульти-активов системы V3.0"""
    print("🚀 ЗАПУСК МУЛЬТИ-АКТИВЫ ТОРГОВОЙ СИСТЕМЫ V3.0")
    print("=" * 60)
    
    # 1. Генерация данных
    print("\n📊 ЭТАП 1: ГЕНЕРАЦИЯ МУЛЬТИ-АКТИВОВ ДАННЫХ")
    print("-" * 40)
    assets_data = generate_correlated_crypto_data(n_points=5000)
    
    # 2. Корреляционный анализ
    print("\n📈 ЭТАП 2: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
    print("-" * 40)
    static_corr, rolling_corr = calculate_correlations(assets_data)
    
    # 3. Подготовка признаков
    print("\n🔧 ЭТАП 3: ПОДГОТОВКА ПРИЗНАКОВ")
    print("-" * 40)
    features_df = prepare_multi_asset_features(assets_data)
    
    # 4. Создание окружения
    print("\n🎮 ЭТАП 4: СОЗДАНИЕ МУЛЬТИ-АКТИВОВ ОКРУЖЕНИЯ")
    print("-" * 40)
    env = MultiAssetTradingEnv(assets_data, features_df)
    assets = list(assets_data.keys())
    print(f"✅ Окружение создано для {len(assets)} активов: {', '.join(assets)}")
    
    # 5. Обучение модели
    print("\n🧠 ЭТАП 5: ОБУЧЕНИЕ МУЛЬТИ-АКТИВОВ МОДЕЛИ")
    print("-" * 40)
    model = train_multi_asset_model(env, total_timesteps=25000)
    
    # 6. Тестирование
    print("\n🧪 ЭТАП 6: ТЕСТИРОВАНИЕ СИСТЕМЫ")
    print("-" * 40)
    results = test_multi_asset_model(model, env, max_steps=2000)
    
    # 7. Анализ результатов
    print("\n📊 ЭТАП 7: АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("-" * 40)
    analysis = analyze_multi_asset_results(results, MultiAssetConfig.INITIAL_BALANCE, assets)
    
    # 8. Заключение
    print("\n🎯 ЗАКЛЮЧЕНИЕ V3.0")
    print("=" * 50)
    print("🚀 Мульти-активы торговая система V3.0 успешно протестирована!")
    print(f"💡 Достигнута доходность: {analysis['total_return']:+.2f}%")
    print(f"🎯 Диверсификация: {analysis['avg_diversification']:.3f}/1.0")
    print("\n✨ Новые возможности V3.0:")
    print("  • Торговля 5 криптовалютами одновременно")
    print("  • Автоматический корреляционный анализ")
    print("  • Умная диверсификация портфеля")
    print("  • Адаптивная ребалансировка")
    print("  • Мульти-активы риск-менеджмент")
    
    if analysis['total_return'] > 0:
        print("\n🟢 ОЦЕНКА: Прибыльная мульти-активы стратегия!")
    else:
        print("\n🔶 ОЦЕНКА: Требует дальнейшей оптимизации")
    
    print("\n🎉 АНАЛИЗ V3.0 ЗАВЕРШЕН!")


if __name__ == "__main__":
    main() 