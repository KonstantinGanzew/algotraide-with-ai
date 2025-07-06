#!/usr/bin/env python3
"""
Быстрый тест исправленного main.py
"""

print("🧪 Тестирование исправлений...")

try:
    # Тест импорта
    print("1. Тестирование импорта...")
    from main import Config, setup_device, check_gpu_requirements
    print("✅ Импорт успешен")
    
    # Тест конфигурации
    print("2. Тестирование конфигурации...")
    print(f"   - WINDOW_SIZE: {Config.WINDOW_SIZE}")
    print(f"   - INITIAL_BALANCE: {Config.INITIAL_BALANCE}")
    print(f"   - DATA_FILE: {Config.DATA_FILE}")
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
    
    # Тест загрузки данных
    print("4. Тестирование загрузки данных...")
    from main import load_and_prepare_data
    import os
    
    data_path = "data/BTC_5_96w.csv"
    if os.path.exists(data_path):
        df = load_and_prepare_data(data_path)
        print(f"   - Загружено {len(df)} строк")
        print(f"   - Колонки: {list(df.columns)}")
        print("✅ Загрузка данных OK")
    else:
        print(f"⚠️  Файл данных не найден: {data_path}")
    
    print("\n🎉 Все тесты пройдены успешно!")
    print("Теперь можно запускать полное обучение командой: python main.py")
    
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
except Exception as e:
    print(f"❌ Неожиданная ошибка: {e}")
    import traceback
    traceback.print_exc() 