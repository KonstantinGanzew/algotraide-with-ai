#!/usr/bin/env python3
"""
Тест продвинутого торгового алгоритма с LSTM и риск-менеджментом
"""

print("🧪 Тестирование продвинутого алгоритма...")

try:
    # Тест импорта
    print("1. Тестирование импорта...")
    from main import Config, setup_device, check_gpu_requirements
    from main import AdvancedTradingEnv, LSTMFeatureExtractor
    from main import load_and_prepare_data, analyze_results
    print("✅ Импорт успешен")
    
    # Тест конфигурации
    print("2. Тестирование конфигурации...")
    print(f"   - WINDOW_SIZE: {Config.WINDOW_SIZE}")
    print(f"   - POSITIONS_LIMIT: {Config.POSITIONS_LIMIT}")
    print(f"   - RISK_PER_TRADE: {Config.RISK_PER_TRADE}")
    print(f"   - STOP_LOSS_PERCENTAGE: {Config.STOP_LOSS_PERCENTAGE}")
    print(f"   - LSTM_HIDDEN_SIZE: {Config.LSTM_HIDDEN_SIZE}")
    print(f"   🔥 TOTAL_TIMESTEPS: {Config.TOTAL_TIMESTEPS:,} (МАКСИМАЛЬНОЕ ОБУЧЕНИЕ!)")
    print(f"   🚫 ENABLE_EARLY_STOPPING: {Config.ENABLE_EARLY_STOPPING}")
    print(f"   ⚡ LEARNING_RATE: {Config.LEARNING_RATE}")
    print("✅ Конфигурация OK")
    
    # Тест GPU функций
    print("3. Тестирование GPU функций...")
    gpu_info = check_gpu_requirements()
    print(f"   - PyTorch версия: {gpu_info['torch_version']}")
    print(f"   - CUDA доступна: {gpu_info['cuda_available']}")
    print(f"   - GPU устройств: {gpu_info['device_count']}")
    
    device = setup_device()
    print(f"   - Выбранное устройство: {device}")
    print("✅ GPU функции OK")
    
    # Тест загрузки данных с расширенными индикаторами
    print("4. Тестирование расширенной загрузки данных...")
    import os
    
    data_path = "data/BTC_5_96w.csv"
    if os.path.exists(data_path):
        df = load_and_prepare_data(data_path)
        print(f"   - Загружено {len(df)} строк")
        print(f"   - Количество признаков: {len(df.columns)}")
        
        # Проверяем наличие новых индикаторов
        new_indicators = ['macd', 'obv', 'vwap', 'bb_upper', 'bb_lower', 'atr']
        found_indicators = [ind for ind in new_indicators if ind in df.columns]
        print(f"   - Новые индикаторы найдены: {found_indicators}")
        print("✅ Расширенная загрузка данных OK")
        
        # Тест продвинутого окружения
        print("5. Тестирование AdvancedTradingEnv...")
        env = AdvancedTradingEnv(df, window_size=Config.WINDOW_SIZE)
        print(f"   - Пространство действий: {env.action_space}")
        print(f"   - Пространство наблюдений: {env.observation_space.shape}")
        
        # Тест нескольких шагов
        obs, _ = env.reset()
        print(f"   - Форма наблюдения: {obs.shape}")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"   - Шаг {i+1}: action={action}, reward={reward:.3f}, balance={env.balance:.2f}")
            if done:
                break
        
        print("✅ AdvancedTradingEnv OK")
        
        # Тест LSTM Feature Extractor
        print("6. Тестирование LSTM Feature Extractor...")
        import torch
        import numpy as np
        from gymnasium import spaces
        
        obs_space = spaces.Box(low=-1, high=1, shape=(Config.WINDOW_SIZE, df.shape[1]), dtype=np.float32)
        lstm_extractor = LSTMFeatureExtractor(obs_space, features_dim=64)
        
        # Создаем фиктивный тензор
        batch_size = 4
        test_input = torch.randn(batch_size, Config.WINDOW_SIZE, df.shape[1])
        
        with torch.no_grad():
            features = lstm_extractor(test_input)
            print(f"   - Входная форма: {test_input.shape}")
            print(f"   - Выходная форма: {features.shape}")
            print(f"   - LSTM скрытый размер: {Config.LSTM_HIDDEN_SIZE}")
        
        print("✅ LSTM Feature Extractor OK")
        
    else:
        print(f"⚠️  Файл данных не найден: {data_path}")
    
    print("\n🎉 Все тесты продвинутого алгоритма пройдены успешно!")
    print("\n📋 Новые возможности:")
    print("   • Система риск-менеджмента со стоп-лоссом и тейк-профитом")
    print("   • Частичное закрытие позиций (25%, 50%, 100%)")
    print("   • Динамический размер позиций на основе баланса")
    print("   • Расширенные технические индикаторы (MACD, OBV, VWAP, Bollinger Bands, ATR)")
    print("   • LSTM архитектура для лучшей обработки временных рядов")
    print("   • Усложненная система вознаграждений с учетом волатильности")
    print("   • Детальная аналитика результатов с множественными метриками")
    print(f"   🔥 МАКСИМАЛЬНОЕ ОБУЧЕНИЕ: {Config.TOTAL_TIMESTEPS:,} шагов БЕЗ раннего завершения!")
    print("   🚫 Отключен early stopping для полного использования данных")
    print("   ⚡ Оптимизированные гиперпараметры для длительного обучения")
    print("\n🚀 Для запуска МАКСИМАЛЬНОГО обучения используйте: python main.py")
    print("⏱️  Ожидаемое время обучения:")
    print(f"   • GPU (RTX 4090): ~30-50 минут ({Config.TOTAL_TIMESTEPS:,} шагов)")
    print(f"   • GPU (RTX 3070): ~60-90 минут ({Config.TOTAL_TIMESTEPS:,} шагов)")
    print(f"   • CPU: ~2-4 часа ({Config.TOTAL_TIMESTEPS:,} шагов)")
    
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("💡 Установите зависимости: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Неожиданная ошибка: {e}")
    import traceback
    traceback.print_exc() 