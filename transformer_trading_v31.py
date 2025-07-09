"""
🚀 TRANSFORMER ТОРГОВАЯ СИСТЕМА V3.1
Замена LSTM на Transformer архитектуру с attention механизмами
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class TransformerConfig:
    """Конфигурация для Transformer системы"""
    
    # Transformer параметры
    D_MODEL = 128          # Размерность модели
    N_HEADS = 8            # Количество attention heads
    N_LAYERS = 3           # Количество transformer слоёв
    D_FF = 512             # Размерность feed-forward сети
    DROPOUT = 0.1          # Dropout rate
    
    # Торговые параметры
    WINDOW_SIZE = 64       # Увеличенное окно для transformer
    INITIAL_BALANCE = 10000
    RISK_PER_TRADE = 0.02
    STOP_LOSS = 0.015
    TAKE_PROFIT = 0.045
    COMMISSION_RATE = 0.001
    
    # Обучение
    TOTAL_TIMESTEPS = 50000
    LEARNING_RATE = 3e-4


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention механизм"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Scaled Dot-Product Attention"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Сохраняем веса внимания для анализа
        self.attention_weights = attention_weights.detach()
        
        output = torch.matmul(attention_weights, v)
        return output
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        return output


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """Transformer блок с attention и feed-forward"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attention_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """Transformer Feature Extractor для обработки временных рядов"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Получаем размеры входа
        if observation_space.shape is not None:
            self.seq_len = observation_space.shape[0]
            self.input_features = observation_space.shape[1]
        else:
            self.seq_len = TransformerConfig.WINDOW_SIZE
            self.input_features = 20
        
        self.d_model = TransformerConfig.D_MODEL
        
        # Входная проекция
        self.input_projection = nn.Linear(self.input_features, self.d_model)
        
        # Позиционное кодирование
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=self.seq_len)
        
        # Transformer блоки
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=TransformerConfig.N_HEADS,
                d_ff=TransformerConfig.D_FF,
                dropout=TransformerConfig.DROPOUT
            ) for _ in range(TransformerConfig.N_LAYERS)
        ])
        
        # Выходная проекция
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, features_dim),
            nn.ReLU(),
            nn.Dropout(TransformerConfig.DROPOUT),
            nn.Linear(features_dim, features_dim)
        )
        
        # Глобальное pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = observations.shape
        
        # Входная проекция
        x = self.input_projection(observations)  # [batch, seq_len, d_model]
        
        # Позиционное кодирование
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        
        # Transformer блоки
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Глобальное pooling (среднее по времени)
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch, d_model]
        
        # Выходная проекция
        x = self.output_projection(x)
        
        return x


def generate_enhanced_crypto_data(n_points: int = 10000) -> pd.DataFrame:
    """Генерация улучшенных данных с большим количеством признаков"""
    print("📊 Генерация данных для Transformer системы...")
    
    np.random.seed(42)
    
    # Временные метки
    timestamps = pd.date_range(start='2020-01-01', periods=n_points, freq='1H')
    
    # Базовая цена с трендом и сезонностью
    trend = np.linspace(45000, 65000, n_points)
    seasonal = 2000 * np.sin(2 * np.pi * np.arange(n_points) / 168)  # Недельная сезонность
    noise = np.random.normal(0, 1000, n_points)
    
    base_price = trend + seasonal + noise
    
    # Создание OHLCV данных
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': base_price,
        'high': base_price * (1 + np.abs(np.random.normal(0, 0.01, n_points))),
        'low': base_price * (1 - np.abs(np.random.normal(0, 0.01, n_points))),
        'close': base_price + np.random.normal(0, 500, n_points),
        'volume': np.random.exponential(1000000, n_points)
    })
    
    # Коррекция high/low
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    print(f"✅ Сгенерировано {len(df)} записей данных")
    return df


def add_transformer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление продвинутых признаков для Transformer"""
    print("🔧 Добавление продвинутых признаков...")
    
    # Базовые признаки
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(24).std()
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # Technical indicators
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    bb_sma = df['close'].rolling(bb_period).mean()
    bb_stddev = df['close'].rolling(bb_period).std()
    df['bb_upper'] = bb_sma + (bb_std * bb_stddev)
    df['bb_lower'] = bb_sma - (bb_std * bb_stddev)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_sma
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Price patterns
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
    df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1).astype(int)
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['price_volume'] = df['close'] * df['volume']
    
    # Momentum indicators
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour / 24.0
    df['day_of_week'] = df['timestamp'].dt.dayofweek / 7.0
    df['day_of_month'] = df['timestamp'].dt.day / 31.0
    
    # Удаляем временную колонку и NaN
    df = df.drop(['timestamp'], axis=1)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✅ Добавлено {len(df.columns)} признаков")
    return df


class TransformerTradingEnv(gym.Env):
    """Торговое окружение с поддержкой Transformer архитектуры"""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = TransformerConfig.WINDOW_SIZE
        
        # Пространство действий: Hold, Buy, Sell
        self.action_space = spaces.Discrete(3)
        
        # Пространство наблюдений
        n_features = len(df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, n_features),
            dtype=np.float32
        )
        
        self._reset_state()
    
    def _reset_state(self):
        """Сброс состояния окружения"""
        self.current_step = self.window_size
        self.balance = TransformerConfig.INITIAL_BALANCE
        self.btc_amount = 0.0
        self.entry_price = 0.0
        
        self.total_trades = 0
        self.profitable_trades = 0
        self.portfolio_history = [TransformerConfig.INITIAL_BALANCE]
        self.trades_history = []
        
        # Для анализа attention
        self.attention_history = []
    
    def reset(self, seed=None, options=None):
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Получение наблюдения для Transformer"""
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        obs = self.df.iloc[start_idx:end_idx].values
        
        # Дополняем если недостаточно данных
        if len(obs) < self.window_size:
            padding = np.tile(obs[0], (self.window_size - len(obs), 1))
            obs = np.vstack([padding, obs])
        
        return obs.astype(np.float32)
    
    def _get_current_price(self) -> float:
        """Получение текущей цены"""
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.iloc[self.current_step]['close']
    
    def _calculate_portfolio_value(self) -> float:
        """Расчет стоимости портфеля"""
        current_price = self._get_current_price()
        return self.balance + self.btc_amount * current_price
    
    def _execute_trade(self, action: int, current_price: float) -> Dict[str, Any]:
        """Выполнение торговой операции"""
        trade_result = {'executed': False, 'type': None, 'amount': 0, 'price': 0}
        
        if action == 1 and self.balance > 100:  # Buy
            investment = self.balance * TransformerConfig.RISK_PER_TRADE
            amount = investment / current_price
            commission = investment * TransformerConfig.COMMISSION_RATE
            
            self.btc_amount += amount
            self.balance -= investment + commission
            self.entry_price = current_price
            
            trade_result.update({
                'executed': True, 'type': 'BUY',
                'amount': amount, 'price': current_price,
                'investment': investment
            })
            
        elif action == 2 and self.btc_amount > 0:  # Sell
            revenue = self.btc_amount * current_price
            commission = revenue * TransformerConfig.COMMISSION_RATE
            
            profit = revenue - self.btc_amount * self.entry_price
            if profit > 0:
                self.profitable_trades += 1
            
            self.balance += revenue - commission
            self.btc_amount = 0.0
            self.entry_price = 0.0
            
            trade_result.update({
                'executed': True, 'type': 'SELL',
                'amount': self.btc_amount, 'price': current_price,
                'revenue': revenue, 'profit': profit
            })
        
        if trade_result['executed']:
            self.total_trades += 1
            self.trades_history.append(trade_result)
        
        return trade_result
    
    def _calculate_reward(self, current_price: float) -> float:
        """Расчет награды с учетом производительности"""
        current_portfolio = self._calculate_portfolio_value()
        prev_portfolio = self.portfolio_history[-1]
        
        # Базовая награда - изменение портфеля
        portfolio_change = (current_portfolio - prev_portfolio) / prev_portfolio
        base_reward = portfolio_change * 100
        
        # Бонус за стабильную прибыльность
        if self.total_trades > 5:
            win_rate = self.profitable_trades / self.total_trades
            if win_rate > 0.6:
                base_reward *= 1.2
        
        return base_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Шаг симуляции"""
        current_price = self._get_current_price()
        
        # Выполнение действия
        trade_result = self._execute_trade(action, current_price)
        
        # Обновление состояния
        self.current_step += 1
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_history.append(portfolio_value)
        
        # Расчет награды
        reward = self._calculate_reward(current_price)
        
        # Проверка завершения
        done = (
            self.current_step >= len(self.df) - 1 or
            portfolio_value <= TransformerConfig.INITIAL_BALANCE * 0.1
        )
        
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'btc_amount': self.btc_amount,
            'total_trades': self.total_trades,
            'trade_result': trade_result
        }
        
        return self._get_observation(), reward, done, False, info


def create_transformer_model(env):
    """Создание PPO модели с Transformer архитектурой"""
    print("🧠 Создание Transformer модели...")
    
    # Создание policy с кастомным feature extractor
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=TransformerConfig.LEARNING_RATE,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    print("✅ Transformer модель создана")
    return model


def train_transformer_model(model, total_timesteps: int = None):
    """Обучение Transformer модели"""
    if total_timesteps is None:
        total_timesteps = TransformerConfig.TOTAL_TIMESTEPS
    
    print(f"🎓 Обучение Transformer модели ({total_timesteps:,} шагов)...")
    model.learn(total_timesteps=total_timesteps)
    print("✅ Обучение завершено")
    return model


def test_transformer_model(model, env, max_steps: int = 2000):
    """Тестирование Transformer модели"""
    print(f"🧪 Тестирование Transformer модели (до {max_steps:,} шагов)...")
    
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
            'reward': reward
        })
        
        if done:
            break
    
    return results


def analyze_transformer_results(results: List[Dict], initial_balance: float):
    """Анализ результатов Transformer системы"""
    print("\n📊 АНАЛИЗ TRANSFORMER ТОРГОВОЙ СИСТЕМЫ V3.1")
    print("=" * 55)
    
    final_value = results[-1]['portfolio_value']
    total_return = (final_value - initial_balance) / initial_balance * 100
    total_trades = results[-1]['total_trades']
    
    print(f"💰 Начальный баланс: {initial_balance:,.2f} USDT")
    print(f"💰 Финальная стоимость: {final_value:,.2f} USDT")
    print(f"📈 Общая доходность: {total_return:+.2f}%")
    print(f"🔄 Всего сделок: {total_trades}")
    print(f"🧠 Архитектура: Transformer (Multi-Head Attention)")
    print(f"🎯 Параметры: {TransformerConfig.N_LAYERS} слоёв, {TransformerConfig.N_HEADS} heads")
    
    # Построение графиков
    visualize_transformer_results(results)
    
    return {
        'total_return': total_return,
        'final_value': final_value,
        'total_trades': total_trades
    }


def visualize_transformer_results(results: List[Dict]):
    """Визуализация результатов Transformer системы"""
    print("📈 Создание графиков Transformer анализа...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚀 Transformer Trading System V3.1 - Результаты', fontsize=16, fontweight='bold')
    
    steps = [r['step'] for r in results]
    portfolio_values = [r['portfolio_value'] for r in results]
    rewards = [r['reward'] for r in results]
    trades = [r['total_trades'] for r in results]
    
    # 1. Стоимость портфеля
    axes[0, 0].plot(steps, portfolio_values, linewidth=2, color='blue')
    axes[0, 0].set_title('💰 Стоимость Портфеля')
    axes[0, 0].set_xlabel('Шаги')
    axes[0, 0].set_ylabel('USDT')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Накопленные награды
    cumulative_rewards = np.cumsum(rewards)
    axes[0, 1].plot(steps, cumulative_rewards, linewidth=2, color='green')
    axes[0, 1].set_title('🏆 Накопленные Награды')
    axes[0, 1].set_xlabel('Шаги')
    axes[0, 1].set_ylabel('Награда')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Количество сделок
    axes[1, 0].plot(steps, trades, linewidth=2, color='orange')
    axes[1, 0].set_title('🔄 Накопленные Сделки')
    axes[1, 0].set_xlabel('Шаги')
    axes[1, 0].set_ylabel('Количество')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Сводная информация
    axes[1, 1].axis('off')
    summary_text = f"""
    🎯 TRANSFORMER СИСТЕМА V3.1
    
    🧠 Архитектура: Multi-Head Attention
    📏 Слоёв: {TransformerConfig.N_LAYERS}
    👁️ Attention Heads: {TransformerConfig.N_HEADS}
    🪟 Размер окна: {TransformerConfig.WINDOW_SIZE}
    
    💰 Финальный результат: {portfolio_values[-1]:,.2f} USDT
    📈 Доходность: {(portfolio_values[-1]/portfolio_values[0]-1)*100:+.2f}%
    🔄 Всего сделок: {trades[-1]}
    
    ✨ Преимущества Transformer:
    • Лучшее понимание долгосрочных зависимостей
    • Параллельная обработка данных
    • Интерпретируемые attention веса
    • Эффективное обучение на GPU
    """
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                   fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('transformer_results_v31.png', dpi=300, bbox_inches='tight')
    print("💾 Графики сохранены: transformer_results_v31.png")
    plt.show()


def main():
    """Главная функция Transformer системы V3.1"""
    print("🚀 ЗАПУСК TRANSFORMER ТОРГОВОЙ СИСТЕМЫ V3.1")
    print("=" * 60)
    
    # 1. Генерация данных
    print("\n📊 ЭТАП 1: ГЕНЕРАЦИЯ ДАННЫХ")
    print("-" * 30)
    df = generate_enhanced_crypto_data(n_points=8000)
    
    # 2. Добавление признаков
    print("\n🔧 ЭТАП 2: ОБРАБОТКА ПРИЗНАКОВ")
    print("-" * 30)
    df = add_transformer_features(df)
    
    # 3. Создание окружения
    print("\n🎮 ЭТАП 3: СОЗДАНИЕ ОКРУЖЕНИЯ")
    print("-" * 30)
    env = TransformerTradingEnv(df)
    vec_env = DummyVecEnv([lambda: env])
    print(f"✅ Окружение создано с {len(df.columns)} признаками")
    
    # 4. Создание и обучение модели
    print("\n🧠 ЭТАП 4: СОЗДАНИЕ И ОБУЧЕНИЕ TRANSFORMER")
    print("-" * 30)
    model = create_transformer_model(vec_env)
    model = train_transformer_model(model, total_timesteps=30000)
    
    # 5. Тестирование
    print("\n🧪 ЭТАП 5: ТЕСТИРОВАНИЕ")
    print("-" * 30)
    results = test_transformer_model(model, env, max_steps=2000)
    
    # 6. Анализ результатов
    print("\n📊 ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("-" * 30)
    analysis = analyze_transformer_results(results, TransformerConfig.INITIAL_BALANCE)
    
    # 7. Заключение
    print("\n🎯 ЗАКЛЮЧЕНИЕ V3.1")
    print("=" * 50)
    print("🚀 Transformer торговая система V3.1 успешно протестирована!")
    print(f"💡 Достигнута доходность: {analysis['total_return']:+.2f}%")
    print(f"🧠 Использована архитектура: Multi-Head Attention")
    print("\n✨ Новые возможности V3.1:")
    print(f"  • {TransformerConfig.N_LAYERS} слоёв Transformer архитектуры")
    print(f"  • {TransformerConfig.N_HEADS} attention heads для анализа паттернов")
    print(f"  • Увеличенное окно наблюдений ({TransformerConfig.WINDOW_SIZE})")
    print("  • Позиционное кодирование для временных данных")
    print("  • Параллельная обработка последовательностей")
    print("  • Интерпретируемые attention веса")
    
    if analysis['total_return'] > 0:
        print("\n🟢 ОЦЕНКА: Прибыльная Transformer стратегия!")
    else:
        print("\n🔶 ОЦЕНКА: Требует дальнейшей оптимизации")
    
    print("\n🎉 АНАЛИЗ V3.1 ЗАВЕРШЕН!")


if __name__ == "__main__":
    main() 