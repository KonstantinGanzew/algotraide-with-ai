#!/usr/bin/env python3
"""
🚀 УДАЛЕННЫЙ GPU ТРЕНЕР
Простой скрипт для обучения на удаленном сервере с NVIDIA GPU
Пароль запрашивается только один раз
"""

import subprocess
import os
import getpass
import time
import json
from pathlib import Path

print("🚀 УДАЛЕННЫЙ GPU ТРЕНЕР")
print("=" * 50)

# Выбор сервера
print("🖥️ ВЫБЕРИТЕ СЕРВЕР:")
print("   1 - 192.168.88.218 (локальный сервер)")
print("   2 - kureed.ml (облачный сервер)")

server_choice = input("Выбор сервера [1]: ").strip() or "1"

if server_choice == "2":
    SERVER_IP = "kureed.ml"
    USERNAME = "kureed"
    PASSWORD = "123qweknz"
    print(f"✅ Выбран сервер: {USERNAME}@{SERVER_IP}")
else:
    SERVER_IP = "192.168.88.218"
    USERNAME = input("👤 Username: ").strip()
    if not USERNAME:
        print("❌ Username обязателен!")
        exit(1)
    
    # Запрос пароля ОДИН РАЗ для локального сервера
    print(f"🔑 Введите пароль для {USERNAME}@{SERVER_IP}")
    PASSWORD = getpass.getpass("Пароль: ")

# Настройки
REMOTE_PATH = f"/home/{USERNAME}/gpu_training"
CONNECTION = f"{USERNAME}@{SERVER_IP}"

# Файлы для синхронизации
FILES_TO_SYNC = [
    "sentiment_trading_v69_remote.py",
    "requirements-gpu.txt",
    "data/BTCUSDT_5m_2y.csv", 
    "data/BTCUSDT_1h_2y.csv",
    "data/BTCUSDT_4h_2y.csv",
    "data/BTCUSDT_1d_2y.csv"
]

def run_ssh_cmd(command, capture_output=True):
    """Выполнение SSH команды с паролем и стабильными опциями"""
    cmd = [
        "sshpass", f"-p{PASSWORD}",
        "ssh", 
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ControlMaster=no",
        "-o", "LogLevel=ERROR",
        CONNECTION, command
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, timeout=300)
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout if capture_output else '',
            'stderr': result.stderr if capture_output else ''
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def run_scp_cmd(source, destination):
    """Копирование файлов с стабильными опциями"""
    cmd = [
        "sshpass", f"-p{PASSWORD}",
        "scp", 
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", 
        "-o", "ControlMaster=no",
        "-o", "LogLevel=ERROR",
        source, destination
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0
    except:
        return False

def test_connection():
    """Тест подключения"""
    print("🔗 Тест подключения...")
    result = run_ssh_cmd("echo 'Подключение успешно'")
    
    if result['success']:
        print("✅ Подключение установлено")
        return True
    else:
        print("❌ Ошибка подключения")
        print(f"Ошибка: {result.get('stderr', result.get('error', 'Unknown'))}")
        return False

def check_gpu():
    """Проверка GPU"""
    print("🎮 Проверка GPU...")
    result = run_ssh_cmd("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader")
    
    if result['success']:
        gpus = []
        for line in result['stdout'].strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpus.append({
                        'name': parts[0],
                        'memory_total': parts[1], 
                        'memory_free': parts[2]
                    })
        
        print(f"✅ Найдено GPU: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu['name']} ({gpu['memory_free']} / {gpu['memory_total']})")
        return True
    else:
        print("⚠️ GPU не найдены")
        return False

def setup_environment():
    """Настройка среды"""
    print("🔧 Настройка среды...")
    
    commands = [
        f"mkdir -p {REMOTE_PATH}",
        f"mkdir -p {REMOTE_PATH}/data",
        f"mkdir -p {REMOTE_PATH}/logs",
        f"mkdir -p {REMOTE_PATH}/models",
        f"mkdir -p {REMOTE_PATH}/results"
    ]
    
    for cmd in commands:
        result = run_ssh_cmd(cmd)
        if not result['success']:
            print(f"❌ Ошибка: {cmd}")
            return False
    
    print("✅ Среда настроена")
    return True

def sync_files():
    """Синхронизация файлов"""
    print("📁 Синхронизация файлов...")
    
    success = True
    for file_path in FILES_TO_SYNC:
        local_file = Path(file_path)
        
        if not local_file.exists():
            print(f"⚠️ Файл не найден: {file_path}")
            continue
        
        if 'data/' in file_path:
            remote_file = f"{CONNECTION}:{REMOTE_PATH}/data/{local_file.name}"
        else:
            remote_file = f"{CONNECTION}:{REMOTE_PATH}/{local_file.name}"
        
        print(f"📤 {file_path} -> удаленный сервер")
        if not run_scp_cmd(str(local_file), remote_file):
            print(f"❌ Ошибка копирования: {file_path}")
            success = False
        else:
            print("✅ Скопировано")
    
    return success

def install_dependencies():
    """Установка зависимостей"""
    print("📦 Установка зависимостей...")
    
    commands = [
        "python3 --version",
        f"cd {REMOTE_PATH} && python3 -m venv venv",
        f"cd {REMOTE_PATH} && source venv/bin/activate && pip install --upgrade pip",
        f"cd {REMOTE_PATH} && source venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        f"cd {REMOTE_PATH} && source venv/bin/activate && pip install stable-baselines3 gymnasium numpy pandas matplotlib ccxt scikit-learn tqdm"
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"📦 Шаг {i}/{len(commands)}...")
        result = run_ssh_cmd(cmd, capture_output=False)
        
        if not result['success']:
            print(f"❌ Ошибка на шаге {i}")
            return False
    
    print("✅ Зависимости установлены")
    return True

def test_gpu_setup():
    """Тест GPU конфигурации"""
    print("🧪 Тест GPU конфигурации...")
    
    test_script = '''
import torch
print("🎮 GPU ТЕСТ")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU устройств: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
    print("✅ GPU готов к обучению!")
else:
    print("❌ CUDA недоступна")
    '''
    
    # Создание и отправка тестового файла
    with open('temp_gpu_test.py', 'w') as f:
        f.write(test_script)
    
    try:
        # Копирование и запуск теста
        if run_scp_cmd('temp_gpu_test.py', f'{CONNECTION}:{REMOTE_PATH}/gpu_test.py'):
            result = run_ssh_cmd(f"cd {REMOTE_PATH} && source venv/bin/activate && python gpu_test.py")
            
            if result['success']:
                print("✅ GPU тест успешен")
                print("\n" + "="*40)
                print("🎮 РЕЗУЛЬТАТ GPU ТЕСТА:")
                print("="*40)
                print(result['stdout'])
                return True
            else:
                print("❌ GPU тест не прошел")
                print(result.get('stderr', 'Ошибка'))
                return False
        else:
            print("❌ Ошибка копирования тестового файла")
            return False
    finally:
        # Удаление временного файла
        if os.path.exists('temp_gpu_test.py'):
            os.remove('temp_gpu_test.py')
    
    return False

def start_training():
    """Запуск обучения"""
    print("🚀 Запуск обучения...")
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = f"logs/training_{timestamp}.log"
    
    cmd = (f"cd {REMOTE_PATH} && "
           f"source venv/bin/activate && "
           f"nohup python sentiment_trading_v69_remote.py > {log_file} 2>&1 & "
           f"echo $!")
    
    result = run_ssh_cmd(cmd)
    
    if result['success']:
        pid = result['stdout'].strip()
        print(f"✅ Обучение запущено!")
        print(f"📋 PID: {pid}")
        print(f"📄 Лог: {log_file}")
        return {'success': True, 'pid': pid, 'log_file': log_file}
    else:
        print("❌ Ошибка запуска")
        print(result.get('stderr', 'Unknown error'))
        return {'success': False}

def monitor_training():
    """Мониторинг обучения"""
    print("📊 Поиск последнего лога...")
    
    # Найти последний лог файл
    result = run_ssh_cmd(f"ls -t {REMOTE_PATH}/logs/training_*.log | head -1")
    
    if result['success'] and result['stdout'].strip():
        log_file = result['stdout'].strip()
        print(f"📋 Мониторинг: {log_file}")
        print("⏹️ Ctrl+C для выхода")
        
        try:
            # Мониторинг через tail -f
            cmd = [
                "sshpass", f"-p{PASSWORD}",
                "ssh", 
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ControlMaster=no",
                "-o", "LogLevel=ERROR",
                CONNECTION, f"tail -f {log_file}"
            ]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n⏹️ Мониторинг остановлен")
    else:
        print("❌ Лог файл не найден")

def get_status():
    """Статус обучения"""
    print("📋 Проверка статуса...")
    
    # Проверка процессов
    result = run_ssh_cmd("pgrep -f 'python.*sentiment_trading' || echo 'No processes'")
    
    processes = []
    if result['success'] and 'No processes' not in result['stdout']:
        processes = result['stdout'].strip().split('\n')
    
    # Последние логи
    log_result = run_ssh_cmd(f"ls -t {REMOTE_PATH}/logs/training_*.log | head -1 | xargs tail -5 || echo 'No logs'")
    logs = log_result['stdout'] if log_result['success'] else 'No logs'
    
    status = {
        'processes': processes,
        'latest_logs': logs,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(json.dumps(status, indent=2, ensure_ascii=False))

def main():
    """Главная функция"""
    
    # Проверка sshpass
    try:
        subprocess.run(['which', 'sshpass'], check=True, capture_output=True)
    except:
        print("❌ sshpass не установлен!")
        print("💡 Установите: sudo apt install sshpass")
        return
    
    # Тест подключения
    if not test_connection():
        return
    
    print(f"\n📋 Сервер: {SERVER_IP}")
    print(f"👤 Пользователь: {USERNAME}")
    print(f"📁 Удаленный путь: {REMOTE_PATH}")
    
    # Главное меню
    while True:
        print("\n📋 ВЫБЕРИТЕ ДЕЙСТВИЕ:")
        print("   1 - Полная настройка и запуск обучения")
        print("   2 - Проверить GPU")
        print("   3 - Настроить среду")
        print("   4 - Синхронизировать файлы") 
        print("   5 - Установить зависимости")
        print("   6 - Тест GPU")
        print("   7 - Запустить обучение")
        print("   8 - Мониторинг")
        print("   9 - Статус")
        print("   0 - Выход")
        
        choice = input("Выбор: ").strip()
        
        if choice == "1":
            print("\n🚀 ПОЛНАЯ НАСТРОЙКА И ЗАПУСК...")
            steps = [
                ("🎮 Проверка GPU", check_gpu),
                ("🔧 Настройка среды", setup_environment), 
                ("📁 Синхронизация файлов", sync_files),
                ("📦 Установка зависимостей", install_dependencies),
                ("🧪 Тест GPU", test_gpu_setup)
            ]
            
            success = True
            for name, func in steps:
                print(f"\n{name}...")
                if not func():
                    print(f"❌ Ошибка на этапе: {name}")
                    success = False
                    break
            
            if success:
                print("\n🚀 Запуск обучения...")
                result = start_training()
                if result['success']:
                    print("🎉 Все готово! Обучение запущено!")
                    print(f"🔍 Для мониторинга выберите пункт 8")
                else:
                    print("❌ Ошибка запуска обучения")
        
        elif choice == "2":
            check_gpu()
        elif choice == "3":
            setup_environment()
        elif choice == "4":
            sync_files()
        elif choice == "5":
            install_dependencies()
        elif choice == "6":
            test_gpu_setup()
        elif choice == "7":
            result = start_training()
            if result['success']:
                print("🎉 Обучение запущено!")
        elif choice == "8":
            monitor_training()
        elif choice == "9":
            get_status()
        elif choice == "0":
            print("👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор")

if __name__ == "__main__":
    main() 