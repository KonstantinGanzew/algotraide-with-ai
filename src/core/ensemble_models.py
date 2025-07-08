"""
🧠 ENSEMBLE МОДЕЛИ ДЛЯ АЛГОТРЕЙДИНГА
Комбинирование нескольких моделей для повышения точности предсказаний
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble предиктор для комбинирования разных типов моделей"""
    
    def __init__(self, models_config: Dict[str, Dict] = None):
        self.models = {}
        self.weights = {}
        self.performance_history = {}
        self.is_trained = False
        
        # Конфигурация по умолчанию
        if models_config is None:
            self.models_config = {
                'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
                'ridge': {'alpha': 1.0},
                'svr': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
                'linear': {}
            }
        else:
            self.models_config = models_config
    
    def _prepare_features(self, data: pd.DataFrame, target_col: str = 'future_return') -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка признаков для обучения"""
        # Исключаем целевую переменную и нерелевантные колонки
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', target_col]
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols].fillna(0).values
        y = data[target_col].fillna(0).values if target_col in data.columns else None
        
        return X, y, feature_cols
    
    def train_base_models(self, train_data: pd.DataFrame, target_col: str = 'future_return') -> Dict[str, float]:
        """Обучение базовых моделей"""
        logger.info("🎯 Обучение базовых моделей ensemble...")
        
        X_train, y_train, self.feature_names = self._prepare_features(train_data, target_col)
        
        if y_train is None:
            # Создаем целевую переменную - будущий доходность
            y_train = train_data['close'].pct_change(5).shift(-5).fillna(0).values
        
        performance = {}
        
        # Random Forest
        try:
            self.models['random_forest'] = RandomForestRegressor(**self.models_config['random_forest'])
            self.models['random_forest'].fit(X_train, y_train)
            rf_pred = self.models['random_forest'].predict(X_train)
            performance['random_forest'] = mean_squared_error(y_train, rf_pred)
            logger.info(f"✅ Random Forest MSE: {performance['random_forest']:.6f}")
        except Exception as e:
            logger.error(f"❌ Random Forest error: {e}")
        
        # Ridge Regression
        try:
            self.models['ridge'] = Ridge(**self.models_config['ridge'])
            self.models['ridge'].fit(X_train, y_train)
            ridge_pred = self.models['ridge'].predict(X_train)
            performance['ridge'] = mean_squared_error(y_train, ridge_pred)
            logger.info(f"✅ Ridge MSE: {performance['ridge']:.6f}")
        except Exception as e:
            logger.error(f"❌ Ridge error: {e}")
        
        # Support Vector Regression
        try:
            # Используем только подвыборку для SVR (он медленный)
            sample_size = min(10000, len(X_train))
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            
            self.models['svr'] = SVR(**self.models_config['svr'])
            self.models['svr'].fit(X_train[indices], y_train[indices])
            svr_pred = self.models['svr'].predict(X_train[indices])
            performance['svr'] = mean_squared_error(y_train[indices], svr_pred)
            logger.info(f"✅ SVR MSE: {performance['svr']:.6f}")
        except Exception as e:
            logger.error(f"❌ SVR error: {e}")
        
        # Linear Regression
        try:
            self.models['linear'] = LinearRegression(**self.models_config['linear'])
            self.models['linear'].fit(X_train, y_train)
            linear_pred = self.models['linear'].predict(X_train)
            performance['linear'] = mean_squared_error(y_train, linear_pred)
            logger.info(f"✅ Linear MSE: {performance['linear']:.6f}")
        except Exception as e:
            logger.error(f"❌ Linear error: {e}")
        
        # Вычисляем веса на основе производительности
        self._calculate_weights(performance)
        self.is_trained = True
        
        return performance
    
    def _calculate_weights(self, performance: Dict[str, float]) -> None:
        """Вычисление весов моделей на основе их производительности"""
        if not performance:
            return
        
        # Инвертируем MSE (меньше = лучше)
        inverse_mse = {name: 1.0 / (mse + 1e-8) for name, mse in performance.items()}
        total_weight = sum(inverse_mse.values())
        
        # Нормализуем веса
        self.weights = {name: weight / total_weight for name, weight in inverse_mse.items()}
        
        logger.info("📊 Веса моделей:")
        for name, weight in self.weights.items():
            logger.info(f"   {name}: {weight:.3f}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Prediction using ensemble"""
        if not self.is_trained:
            logger.error("❌ Модель не обучена!")
            return np.zeros(len(data))
        
        X, _, _ = self._prepare_features(data)
        predictions = {}
        
        # Получаем предсказания от каждой модели
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка предсказания {name}: {e}")
                predictions[name] = np.zeros(len(X))
        
        # Взвешенное усреднение
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0)
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def save_models(self, save_path: str) -> None:
        """Сохранение ensemble моделей"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        ensemble_data = {
            'models': self.models,
            'weights': self.weights,
            'feature_names': getattr(self, 'feature_names', []),
            'is_trained': self.is_trained
        }
        
        with open(save_path / 'ensemble_models.pkl', 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        logger.info(f"💾 Ensemble модели сохранены в {save_path}")
    
    def load_models(self, load_path: str) -> None:
        """Загрузка ensemble моделей"""
        load_path = Path(load_path) / 'ensemble_models.pkl'
        
        if not load_path.exists():
            logger.error(f"❌ Файл моделей не найден: {load_path}")
            return
        
        with open(load_path, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.models = ensemble_data['models']
        self.weights = ensemble_data['weights']
        self.feature_names = ensemble_data.get('feature_names', [])
        self.is_trained = ensemble_data['is_trained']
        
        logger.info(f"📥 Ensemble модели загружены из {load_path}")


class MultiAgentEnsemble:
    """Ensemble из нескольких RL агентов"""
    
    def __init__(self, agent_configs: List[Dict] = None):
        self.agents = {}
        self.agent_weights = {}
        self.training_envs = {}
        
        # Конфигурация агентов по умолчанию
        if agent_configs is None:
            self.agent_configs = [
                {'algorithm': 'PPO', 'policy': 'MlpPolicy', 'learning_rate': 3e-4},
                {'algorithm': 'PPO', 'policy': 'MlpPolicy', 'learning_rate': 1e-4},
                {'algorithm': 'A2C', 'policy': 'MlpPolicy', 'learning_rate': 3e-4},
            ]
        else:
            self.agent_configs = agent_configs
    
    def train_agents(self, environments: List[Any], total_timesteps: int = 10000) -> Dict[str, float]:
        """Обучение нескольких агентов"""
        logger.info("🤖 Обучение multi-agent ensemble...")
        
        performance = {}
        
        for i, (env, config) in enumerate(zip(environments, self.agent_configs)):
            agent_name = f"{config['algorithm']}_agent_{i}"
            
            try:
                # Создаем агента
                if config['algorithm'] == 'PPO':
                    agent = PPO(
                        config['policy'], 
                        env, 
                        learning_rate=config['learning_rate'],
                        verbose=0
                    )
                elif config['algorithm'] == 'A2C':
                    agent = A2C(
                        config['policy'], 
                        env, 
                        learning_rate=config['learning_rate'],
                        verbose=0
                    )
                elif config['algorithm'] == 'SAC':
                    agent = SAC(
                        config['policy'], 
                        env, 
                        learning_rate=config['learning_rate'],
                        verbose=0
                    )
                else:
                    logger.error(f"❌ Неизвестный алгоритм: {config['algorithm']}")
                    continue
                
                # Обучаем агента
                logger.info(f"🎓 Обучение {agent_name}...")
                agent.learn(total_timesteps=total_timesteps // len(self.agent_configs))
                
                # Оцениваем производительность
                obs, _ = env.reset()
                total_reward = 0
                steps = 0
                
                for _ in range(100):  # Тестируем на 100 шагах
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    if done or truncated:
                        obs, _ = env.reset()
                
                avg_reward = total_reward / steps if steps > 0 else 0
                performance[agent_name] = avg_reward
                
                self.agents[agent_name] = agent
                logger.info(f"✅ {agent_name} - средняя награда: {avg_reward:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Ошибка обучения {agent_name}: {e}")
        
        # Вычисляем веса агентов
        self._calculate_agent_weights(performance)
        
        return performance
    
    def _calculate_agent_weights(self, performance: Dict[str, float]) -> None:
        """Вычисление весов агентов"""
        if not performance:
            return
        
        # Нормализуем награды (выше = лучше)
        min_reward = min(performance.values())
        if min_reward < 0:
            # Сдвигаем в положительную область
            shifted_rewards = {name: reward - min_reward + 1 for name, reward in performance.items()}
        else:
            shifted_rewards = performance
        
        total_weight = sum(shifted_rewards.values())
        
        if total_weight > 0:
            self.agent_weights = {name: weight / total_weight for name, weight in shifted_rewards.items()}
        else:
            # Равные веса если все агенты одинаково плохие
            self.agent_weights = {name: 1.0 / len(performance) for name in performance.keys()}
        
        logger.info("🤖 Веса агентов:")
        for name, weight in self.agent_weights.items():
            logger.info(f"   {name}: {weight:.3f}")
    
    def predict_ensemble_action(self, observation: np.ndarray, method: str = 'weighted_vote') -> int:
        """Предсказание действия ensemble агентов"""
        if not self.agents:
            logger.error("❌ Нет обученных агентов!")
            return 0  # Hold by default
        
        actions = {}
        
        # Получаем действия от каждого агента
        for name, agent in self.agents.items():
            try:
                action, _ = agent.predict(observation, deterministic=True)
                actions[name] = int(action)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка предсказания {name}: {e}")
                actions[name] = 0
        
        # Комбинируем действия
        if method == 'weighted_vote':
            # Взвешенное голосование
            action_scores = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell
            
            for name, action in actions.items():
                weight = self.agent_weights.get(name, 0)
                action_scores[action] += weight
            
            # Возвращаем действие с максимальным счетом
            return max(action_scores.items(), key=lambda x: x[1])[0]
        
        elif method == 'majority_vote':
            # Простое большинство
            from collections import Counter
            vote_counts = Counter(actions.values())
            return vote_counts.most_common(1)[0][0]
        
        elif method == 'best_agent':
            # Только лучший агент
            if self.agent_weights:
                best_agent = max(self.agent_weights.items(), key=lambda x: x[1])[0]
                return actions.get(best_agent, 0)
        
        return 0  # Default to Hold


class HybridEnsemble:
    """Гибридная ensemble система, комбинирующая ML и RL модели"""
    
    def __init__(self):
        self.ml_ensemble = EnsemblePredictor()
        self.rl_ensemble = MultiAgentEnsemble()
        self.hybrid_weights = {'ml': 0.3, 'rl': 0.7}  # Больше веса RL
        self.is_trained = False
    
    def train_hybrid_system(self, train_data: pd.DataFrame, rl_environments: List[Any], 
                          timesteps: int = 10000) -> Dict[str, Any]:
        """Обучение гибридной системы"""
        logger.info("🔗 Обучение гибридной ensemble системы...")
        
        results = {}
        
        # Обучаем ML модели
        logger.info("📊 Этап 1: Обучение ML моделей...")
        ml_performance = self.ml_ensemble.train_base_models(train_data)
        results['ml_performance'] = ml_performance
        
        # Обучаем RL агентов
        logger.info("🤖 Этап 2: Обучение RL агентов...")
        rl_performance = self.rl_ensemble.train_agents(rl_environments, timesteps)
        results['rl_performance'] = rl_performance
        
        # Адаптивная настройка весов
        self._adjust_hybrid_weights(ml_performance, rl_performance)
        results['hybrid_weights'] = self.hybrid_weights
        
        self.is_trained = True
        logger.info("✅ Гибридная система обучена!")
        
        return results
    
    def _adjust_hybrid_weights(self, ml_performance: Dict[str, float], rl_performance: Dict[str, float]) -> None:
        """Адаптивная настройка весов между ML и RL"""
        
        # Средняя производительность ML (инвертируем MSE)
        ml_avg = np.mean([1.0 / (mse + 1e-8) for mse in ml_performance.values()]) if ml_performance else 0
        
        # Средняя производительность RL
        rl_avg = np.mean(list(rl_performance.values())) if rl_performance else 0
        
        # Нормализуем веса
        if ml_avg + rl_avg > 0:
            total = ml_avg + rl_avg
            self.hybrid_weights['ml'] = ml_avg / total
            self.hybrid_weights['rl'] = rl_avg / total
        
        logger.info(f"🔗 Гибридные веса: ML={self.hybrid_weights['ml']:.3f}, RL={self.hybrid_weights['rl']:.3f}")
    
    def predict_hybrid_action(self, observation: np.ndarray, market_data: pd.DataFrame) -> int:
        """Предсказание гибридного действия"""
        if not self.is_trained:
            logger.error("❌ Гибридная система не обучена!")
            return 0
        
        # Получаем предсказания от ML
        ml_prediction = 0
        if len(market_data) > 0:
            try:
                ml_signal = self.ml_ensemble.predict(market_data.tail(1))
                if len(ml_signal) > 0:
                    # Конвертируем сигнал в действие
                    if ml_signal[0] > 0.01:  # Покупка при положительном сигнале
                        ml_prediction = 1
                    elif ml_signal[0] < -0.01:  # Продажа при отрицательном сигнале
                        ml_prediction = 2
                    else:
                        ml_prediction = 0  # Hold
            except Exception as e:
                logger.warning(f"⚠️ ML prediction error: {e}")
                ml_prediction = 0
        
        # Получаем предсказания от RL
        rl_prediction = self.rl_ensemble.predict_ensemble_action(observation)
        
        # Комбинируем предсказания
        ml_weight = self.hybrid_weights['ml']
        rl_weight = self.hybrid_weights['rl']
        
        # Простое взвешенное голосование
        if ml_weight > rl_weight and ml_prediction != 0:
            return ml_prediction
        else:
            return rl_prediction
    
    def save_hybrid_system(self, save_path: str) -> None:
        """Сохранение гибридной системы"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Сохраняем ML модели
        self.ml_ensemble.save_models(save_path / 'ml_models')
        
        # Сохраняем конфигурацию
        config = {
            'hybrid_weights': self.hybrid_weights,
            'is_trained': self.is_trained
        }
        
        with open(save_path / 'hybrid_config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"💾 Гибридная система сохранена в {save_path}")


def create_ensemble_system(train_data: pd.DataFrame, rl_environments: List[Any] = None) -> HybridEnsemble:
    """Удобная функция для создания ensemble системы"""
    
    logger.info("🎯 Создание ensemble системы...")
    
    hybrid = HybridEnsemble()
    
    if rl_environments is None:
        logger.info("⚠️ RL окружения не предоставлены, используем только ML ensemble")
        hybrid.ml_ensemble.train_base_models(train_data)
        hybrid.hybrid_weights = {'ml': 1.0, 'rl': 0.0}
    else:
        hybrid.train_hybrid_system(train_data, rl_environments, timesteps=5000)
    
    return hybrid


if __name__ == "__main__":
    # Демонстрация использования
    logging.basicConfig(level=logging.INFO)
    
    print("🧠 Ensemble модули готовы к использованию!")
    print("Для использования: from src.core.ensemble_models import create_ensemble_system") 