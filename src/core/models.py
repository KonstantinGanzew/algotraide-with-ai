"""
ML модели для алготрейдинг системы

ПРОБЛЕМЫ В ОРИГИНАЛЬНОЙ МОДЕЛИ:
1. Слишком сложная LSTM (512 hidden, 3 layers) - переобучение
2. Отсутствие регуляризации
3. Неправильные гиперпараметры PPO

ИСПРАВЛЕНИЯ:
1. Упрощенная LSTM архитектура
2. Добавлен dropout и batch normalization  
3. Консервативные настройки PPO
4. Улучшенный early stopping
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import gymnasium as gym
from typing import Dict, Any, Optional
import logging

from .config import MLConfig, SystemConfig


class ImprovedLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    УЛУЧШЕННЫЙ LSTM Feature Extractor с защитой от переобучения
    
    Изменения от оригинала:
    1. Уменьшен размер модели (256 -> 128)
    2. Добавлен dropout и batch norm
    3. Более простая архитектура
    4. Лучшая инициализация весов
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        logger = logging.getLogger(__name__)
        
        # Получаем размеры входа
        if observation_space.shape is not None:
            n_input_features = observation_space.shape[-1]
            sequence_length = observation_space.shape[0]
        else:
            # Fallback значения
            n_input_features = 20
            sequence_length = 50
        
        logger.info(f"LSTM модель: вход={n_input_features}, последовательность={sequence_length}, выход={features_dim}")
        
        # УПРОЩЕННАЯ LSTM архитектура для предотвращения переобучения
        self.lstm_hidden_size = MLConfig.LSTM_HIDDEN_SIZE  # 256 вместо 512
        self.lstm_layers = MLConfig.LSTM_NUM_LAYERS        # 2 вместо 3
        
        # Входной слой нормализации
        self.input_norm = nn.BatchNorm1d(n_input_features)
        
        # LSTM слои с dropout
        self.lstm = nn.LSTM(
            input_size=n_input_features,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=MLConfig.LSTM_DROPOUT if self.lstm_layers > 1 else 0,
            bidirectional=False  # Убираем bidirectional для простоты
        )
        
        # Слои после LSTM
        self.feature_net = nn.Sequential(
            nn.Dropout(MLConfig.LSTM_DROPOUT),
            nn.Linear(self.lstm_hidden_size, features_dim),
            nn.ReLU(),
            nn.BatchNorm1d(features_dim),
            nn.Dropout(MLConfig.LSTM_DROPOUT * 0.5),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Правильная инициализация весов для LSTM"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Инициализация linear слоев
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Прямой проход через модель"""
        batch_size = observations.shape[0]
        seq_len = observations.shape[1]
        features = observations.shape[2]
        
        # Нормализация входа
        # Reshape для batch norm: (batch * seq, features)
        obs_reshaped = observations.view(-1, features)
        obs_normalized = self.input_norm(obs_reshaped)
        obs_normalized = obs_normalized.view(batch_size, seq_len, features)
        
        # LSTM обработка
        lstm_out, (hidden, cell) = self.lstm(obs_normalized)
        
        # Берем выход последнего временного шага
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Финальные слои
        features = self.feature_net(last_output)
        
        return features


class SmartEarlyStoppingCallback(BaseCallback):
    """
    УЛУЧШЕННЫЙ callback с ранней остановкой
    
    Изменения:
    1. Более консервативные критерии остановки
    2. Лучшее отслеживание метрик
    3. Защита от ложных срабатываний
    """
    
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        
        # Счетчики
        self.step_count = 0
        self.episode_count = 0
        
        # Отслеживание производительности
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.episode_rewards_history = []
        
        # Параметры из config
        self.patience = MLConfig.EARLY_STOPPING_PATIENCE
        self.min_episodes = MLConfig.MIN_EPISODES_BEFORE_STOPPING
        self.improvement_threshold = MLConfig.IMPROVEMENT_THRESHOLD
        self.window_size = 20  # Окно для сглаживания
        
        # Интервалы отчетности
        self.progress_interval = 5000
        
        logger = logging.getLogger(__name__)
        logger.info(f"Early Stopping: терпение={self.patience}, мин_эпизодов={self.min_episodes}")
    
    def _on_step(self) -> bool:
        """Callback на каждом шаге обучения"""
        self.step_count += 1
        
        # Обрабатываем завершенные эпизоды
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_reward = info['episode'].get('r', 0)
                    self.episode_count += 1
                    self.episode_rewards_history.append(episode_reward)
                    
                    # Проверяем улучшение только после накопления достаточной истории
                    if len(self.episode_rewards_history) >= self.window_size:
                        # Используем скользящее среднее для сглаживания
                        recent_avg = np.mean(self.episode_rewards_history[-self.window_size:])
                        
                        # Проверяем значимое улучшение
                        if recent_avg > self.best_reward + self.improvement_threshold:
                            improvement = recent_avg - self.best_reward
                            self.best_reward = recent_avg
                            self.episodes_without_improvement = 0
                            
                            if self.verbose >= 1:
                                logger = logging.getLogger(__name__)
                                logger.info(f"🚀 [Эпизод {self.episode_count}] Новый рекорд: {recent_avg:.4f} (+{improvement:.4f})")
                        else:
                            self.episodes_without_improvement += 1
                            
                            # Периодические отчеты
                            if self.episode_count % 10 == 0 and self.verbose >= 1:
                                logger = logging.getLogger(__name__)
                                logger.info(f"📊 [Эпизод {self.episode_count}] Среднее: {recent_avg:.4f}, Лучшее: {self.best_reward:.4f}, Без улучшения: {self.episodes_without_improvement}")
        
        # Прогресс по шагам
        if self.step_count % self.progress_interval == 0 and self.verbose >= 1:
            logger = logging.getLogger(__name__)
            logger.info(f"⏱️ [{self.step_count:,} шагов] Эпизодов: {self.episode_count}")
        
        # Проверка условий ранней остановки
        if (MLConfig.ENABLE_EARLY_STOPPING and 
            self.episode_count >= self.min_episodes and
            self.episodes_without_improvement >= self.patience):
            
            logger = logging.getLogger(__name__)
            logger.info(f"\n🛑 РАННЯЯ ОСТАНОВКА!")
            logger.info(f"   Эпизодов без улучшения: {self.episodes_without_improvement}")
            logger.info(f"   Лучшая средняя награда: {self.best_reward:.4f}")
            logger.info(f"   Всего эпизодов: {self.episode_count}")
            logger.info(f"   Всего шагов: {self.step_count:,}")
            return False  # Останавливаем обучение
        
        return True  # Продолжаем обучение


def create_improved_model(env, device: str = "cpu") -> PPO:
    """
    Создание УЛУЧШЕННОЙ модели PPO с консервативными настройками
    
    Изменения от оригинала:
    1. Более простая архитектура
    2. Консервативные гиперпараметры
    3. Лучшая регуляризация
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"🎯 Создание улучшенной PPO модели на устройстве: {device}")
    logger.info(f"📚 Настройки: lr={MLConfig.LEARNING_RATE}, steps={MLConfig.TOTAL_TIMESTEPS:,}")
    
    # УЛУЧШЕННЫЕ настройки политики
    policy_kwargs = {
        "features_extractor_class": ImprovedLSTMFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": MLConfig.LSTM_HIDDEN_SIZE},
        # Более простая архитектура сети
        "net_arch": [dict(pi=[256, 128], vf=[256, 128])],  # Уменьшено от [512, 256]
        "activation_fn": nn.ReLU,
        "normalize_images": False
    }
    
    # Создание модели с АГРЕССИВНЫМИ параметрами для исследования
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=MLConfig.LEARNING_RATE,    # 5e-4 - увеличено
        n_steps=8192,                            # УВЕЛИЧЕНО с 4096 до 8192
        batch_size=256,                          # УВЕЛИЧЕНО с 128 до 256  
        n_epochs=8,                              # УВЕЛИЧЕНО с 6 до 8
        gamma=0.99,                              # Стандартный
        gae_lambda=0.95,                         # Стандартный
        clip_range=0.3,                          # УВЕЛИЧЕНО с 0.2 до 0.3 для большего обновления
        clip_range_vf=None,
        ent_coef=MLConfig.PPO_ENT_COEF,         # 0.1 - максимальное исследование!
        vf_coef=0.25,                           # УМЕНЬШЕНО с 0.5 до 0.25
        max_grad_norm=1.0,                      # УВЕЛИЧЕНО с 0.5 до 1.0
        device=device,
        verbose=1,
        seed=42                                 # Фиксированный seed для воспроизводимости
    )
    
    logger.info("✅ Модель создана с улучшенными настройками")
    return model


def train_improved_model(model: PPO, callback: Optional[SmartEarlyStoppingCallback] = None) -> PPO:
    """
    Обучение улучшенной модели с лучшим контролем
    """
    logger = logging.getLogger(__name__)
    
    if callback is None:
        callback = SmartEarlyStoppingCallback()
    
    logger.info(f"🚀 НАЧАЛО УЛУЧШЕННОГО ОБУЧЕНИЯ")
    logger.info(f"🎯 Цель: {MLConfig.TOTAL_TIMESTEPS:,} шагов")
    logger.info(f"🧠 Ранняя остановка: {'Включена' if MLConfig.ENABLE_EARLY_STOPPING else 'Отключена'}")
    
    try:
        model.learn(
            total_timesteps=MLConfig.TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=False  # Отключаем встроенный прогресс-бар
        )
        logger.info("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ ОБУЧЕНИЕ ПРЕРВАНО ПОЛЬЗОВАТЕЛЕМ!")
    except Exception as e:
        logger.error(f"❌ ОШИБКА ОБУЧЕНИЯ: {e}")
        raise
    
    # Финальная статистика
    if hasattr(callback, 'episode_count'):
        logger.info(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
        logger.info(f"   Эпизодов: {callback.episode_count}")
        logger.info(f"   Шагов: {callback.step_count:,}")
        logger.info(f"   Лучшая награда: {callback.best_reward:.4f}")
    
    return model


def setup_device() -> str:
    """Настройка устройства вычислений"""
    logger = logging.getLogger(__name__)
    
    if SystemConfig.FORCE_CPU:
        device = "cpu"
        logger.info("🔧 Принудительно используется CPU")
    elif SystemConfig.AUTO_DEVICE:
        if torch.cuda.is_available():
            device = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🚀 GPU обнаружен: {gpu_name} ({gpu_memory:.1f} GB)")
            except:
                logger.info("🚀 Используется GPU")
        else:
            device = "cpu"
            logger.warning("⚠️ GPU недоступно, используется CPU")
    else:
        device = SystemConfig.DEVICE
        logger.info(f"🎯 Используется заданное устройство: {device}")
    
    return device 