"""
🚀 НАСТРОЙКА УДАЛЕННОГО ОБУЧЕНИЯ НА GPU СЕРВЕРЕ
Скрипт для синхронизации и запуска обучения на удаленном сервере
Сервер: 192.168.88.218 (NVIDIA GPU + CUDA 12.8)
"""

import os
import subprocess
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RemoteTrainingManager:
    """Менеджер для удаленного обучения на GPU сервере"""
    
    def __init__(self, 
                 server_ip: str = "192.168.88.218",
                 username: str = "kureed",  # Замените на ваш username
                 remote_path: str = "/home/kureed/training",  # Замените на ваш путь
                 ssh_key_path: Optional[str] = None):
        """
        Инициализация менеджера удаленного обучения
        
        Args:
            server_ip: IP адрес сервера
            username: Имя пользователя на сервере  
            remote_path: Путь к рабочей директории на сервере
            ssh_key_path: Путь к SSH ключу (если используется)
        """
        self.server_ip = server_ip
        self.username = username
        self.remote_path = remote_path
        self.ssh_key_path = ssh_key_path
        self.connection_string = f"{username}@{server_ip}"
        
        # Локальные файлы для синхронизации
        self.sync_files = [
            "sentiment_trading_v69.py",
            "data/BTCUSDT_5m_2y.csv",
            "data/BTCUSDT_1h_2y.csv", 
            "data/BTCUSDT_4h_2y.csv",
            "data/BTCUSDT_1d_2y.csv",
            "requirements-gpu.txt"
        ]
        
    def _run_ssh_command(self, command: str, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Выполнение команды через SSH"""
        ssh_cmd = ["ssh"]
        
        if self.ssh_key_path:
            ssh_cmd.extend(["-i", self.ssh_key_path])
            
        ssh_cmd.extend([self.connection_string, command])
        
        logger.info(f"🔧 Выполнение SSH команды: {' '.join(ssh_cmd[2:])}")
        
        try:
            result = subprocess.run(ssh_cmd, capture_output=capture_output, text=True, timeout=300)
            return result
        except subprocess.TimeoutExpired:
            logger.error("⏰ Тайм-аут SSH команды")
            raise
        except Exception as e:
            logger.error(f"❌ Ошибка SSH: {e}")
            raise
    
    def _run_scp_command(self, source: str, destination: str) -> bool:
        """Копирование файлов через SCP"""
        scp_cmd = ["scp"]
        
        if self.ssh_key_path:
            scp_cmd.extend(["-i", self.ssh_key_path])
            
        scp_cmd.extend(["-r", source, destination])
        
        logger.info(f"📁 Копирование: {source} -> {destination}")
        
        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                logger.info("✅ Копирование успешно")
                return True
            else:
                logger.error(f"❌ Ошибка копирования: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка SCP: {e}")
            return False
    
    def check_connection(self) -> bool:
        """Проверка подключения к серверу"""
        logger.info(f"🔗 Проверка подключения к {self.server_ip}...")
        
        try:
            result = self._run_ssh_command("echo 'Connected successfully'")
            if result.returncode == 0:
                logger.info("✅ Подключение установлено")
                return True
            else:
                logger.error(f"❌ Ошибка подключения: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Не удалось подключиться: {e}")
            return False
    
    def check_gpu_status(self) -> Dict[str, Any]:
        """Проверка статуса GPU на сервере"""
        logger.info("🎮 Проверка GPU статуса...")
        
        try:
            # Проверка NVIDIA-SMI
            result = self._run_ssh_command("nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits")
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split('\n')
                gpu_data = []
                
                for i, gpu_line in enumerate(gpu_info):
                    parts = gpu_line.split(', ')
                    if len(parts) >= 4:
                        gpu_data.append({
                            'id': i,
                            'name': parts[0],
                            'memory_total': int(parts[1]),
                            'memory_free': int(parts[2]),
                            'temperature': int(parts[3])
                        })
                
                # Проверка CUDA версии
                cuda_result = self._run_ssh_command("nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-")
                cuda_version = cuda_result.stdout.strip() if cuda_result.returncode == 0 else "Unknown"
                
                logger.info(f"✅ Найдено {len(gpu_data)} GPU(s)")
                logger.info(f"🔧 CUDA версия: {cuda_version}")
                
                return {
                    'available': True,
                    'cuda_version': cuda_version,
                    'gpus': gpu_data
                }
            else:
                logger.warning("⚠️ NVIDIA-SMI не найден или недоступен")
                return {'available': False, 'error': result.stderr}
                
        except Exception as e:
            logger.error(f"❌ Ошибка проверки GPU: {e}")
            return {'available': False, 'error': str(e)}
    
    def setup_remote_environment(self) -> bool:
        """Настройка удаленной среды"""
        logger.info("🔧 Настройка удаленной среды...")
        
        commands = [
            f"mkdir -p {self.remote_path}",
            f"mkdir -p {self.remote_path}/data",
            f"mkdir -p {self.remote_path}/logs",
            f"mkdir -p {self.remote_path}/models",
        ]
        
        for cmd in commands:
            result = self._run_ssh_command(cmd)
            if result.returncode != 0:
                logger.error(f"❌ Ошибка выполнения: {cmd}")
                return False
        
        logger.info("✅ Удаленная среда настроена")
        return True
    
    def sync_files(self) -> bool:
        """Синхронизация файлов на сервер"""
        logger.info("📁 Синхронизация файлов...")
        
        success = True
        
        for file_path in self.sync_files:
            local_path = Path(file_path)
            
            if not local_path.exists():
                logger.warning(f"⚠️ Файл не найден: {file_path}")
                continue
            
            if local_path.is_file():
                # Для файлов
                remote_file_path = f"{self.connection_string}:{self.remote_path}/{file_path}"
                if not self._run_scp_command(str(local_path), remote_file_path):
                    success = False
            else:
                # Для директорий
                remote_dir_path = f"{self.connection_string}:{self.remote_path}/"
                if not self._run_scp_command(str(local_path), remote_dir_path):
                    success = False
        
        if success:
            logger.info("✅ Все файлы синхронизированы")
        else:
            logger.error("❌ Некоторые файлы не удалось синхронизировать")
            
        return success
    
    def install_dependencies(self) -> bool:
        """Установка зависимостей на сервере"""
        logger.info("📦 Установка зависимостей...")
        
        commands = [
            # Проверка Python
            "python3 --version",
            
            # Создание виртуальной среды если её нет
            f"cd {self.remote_path} && python3 -m venv venv || true",
            
            # Активация и установка зависимостей
            f"cd {self.remote_path} && source venv/bin/activate && pip install --upgrade pip",
            f"cd {self.remote_path} && source venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            f"cd {self.remote_path} && source venv/bin/activate && pip install -r requirements-gpu.txt",
        ]
        
        for cmd in commands:
            logger.info(f"🔧 {cmd.split('&&')[-1].strip()}")
            result = self._run_ssh_command(cmd, capture_output=False)
            
            if result.returncode != 0:
                logger.error(f"❌ Ошибка установки: {cmd}")
                return False
                
        logger.info("✅ Зависимости установлены")
        return True
    
    def start_training(self, script_name: str = "sentiment_trading_v69.py", 
                      background: bool = True) -> bool:
        """Запуск обучения на сервере"""
        logger.info("🚀 Запуск обучения на удаленном сервере...")
        
        if background:
            # Запуск в фоне с логированием
            cmd = (f"cd {self.remote_path} && "
                   f"source venv/bin/activate && "
                   f"nohup python {script_name} > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 & "
                   f"echo $!")
        else:
            # Интерактивный запуск
            cmd = (f"cd {self.remote_path} && "
                   f"source venv/bin/activate && "
                   f"python {script_name}")
        
        try:
            result = self._run_ssh_command(cmd, capture_output=background)
            
            if background and result.returncode == 0:
                process_id = result.stdout.strip()
                logger.info(f"✅ Обучение запущено в фоне (PID: {process_id})")
                logger.info(f"📋 Для мониторинга: ssh {self.connection_string} 'tail -f {self.remote_path}/logs/training_*.log'")
                return True
            elif not background:
                return result.returncode == 0
            else:
                logger.error(f"❌ Ошибка запуска: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка запуска обучения: {e}")
            return False
    
    def monitor_training(self) -> None:
        """Мониторинг обучения в реальном времени"""
        logger.info("📊 Запуск мониторинга обучения...")
        
        try:
            # Найти последний лог файл
            find_cmd = f"ls -t {self.remote_path}/logs/training_*.log | head -1"
            result = self._run_ssh_command(find_cmd)
            
            if result.returncode == 0 and result.stdout.strip():
                log_file = result.stdout.strip()
                
                # Мониторинг в реальном времени
                monitor_cmd = f"tail -f {log_file}"
                logger.info(f"📋 Мониторинг: {log_file}")
                
                self._run_ssh_command(monitor_cmd, capture_output=False)
            else:
                logger.error("❌ Лог файл не найден")
                
        except KeyboardInterrupt:
            logger.info("⏹️ Мониторинг остановлен")
        except Exception as e:
            logger.error(f"❌ Ошибка мониторинга: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Получение статуса обучения"""
        try:
            # Проверка запущенных процессов Python
            result = self._run_ssh_command("pgrep -f 'python.*sentiment_trading' || echo 'No processes found'")
            
            processes = []
            if result.returncode == 0 and result.stdout.strip() != "No processes found":
                pids = result.stdout.strip().split('\n')
                processes = [{'pid': pid.strip()} for pid in pids if pid.strip()]
            
            # Последние логи
            log_result = self._run_ssh_command(f"ls -t {self.remote_path}/logs/training_*.log 2>/dev/null | head -1 | xargs tail -5 2>/dev/null || echo 'No logs'")
            latest_logs = log_result.stdout.strip() if log_result.returncode == 0 else "No logs available"
            
            return {
                'processes': processes,
                'latest_logs': latest_logs,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {'error': str(e)}


def main():
    """Главная функция для управления удаленным обучением"""
    print("🚀 МЕНЕДЖЕР УДАЛЕННОГО ОБУЧЕНИЯ")
    print("=" * 50)
    
    # Конфигурация (ИЗМЕНИТЕ НА ВАШИ ДАННЫЕ)
    config = {
        'server_ip': '192.168.88.218',
        'username': 'user',  # ЗАМЕНИТЕ НА ВАШ USERNAME
        'remote_path': '/home/user/training',  # ЗАМЕНИТЕ НА ВАШ ПУТЬ
        'ssh_key_path': None  # Укажите путь к SSH ключу если используется
    }
    
    print("📋 КОНФИГУРАЦИЯ:")
    print(f"   🖥️  Сервер: {config['server_ip']}")
    print(f"   👤 Пользователь: {config['username']}")
    print(f"   📁 Удаленный путь: {config['remote_path']}")
    print()
    
    # Создание менеджера
    manager = RemoteTrainingManager(**config)
    
    # Пошаговая настройка
    steps = [
        ("🔗 Проверка подключения", manager.check_connection),
        ("🎮 Проверка GPU", lambda: print(json.dumps(manager.check_gpu_status(), indent=2)) or True),
        ("🔧 Настройка среды", manager.setup_remote_environment),
        ("📁 Синхронизация файлов", manager.sync_files),
        ("📦 Установка зависимостей", manager.install_dependencies),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"❌ Ошибка на этапе: {step_name}")
            return False
    
    print("\n✅ Настройка завершена!")
    print("\n🎯 ДОСТУПНЫЕ КОМАНДЫ:")
    print("   1. Запуск обучения: manager.start_training()")
    print("   2. Мониторинг: manager.monitor_training()")
    print("   3. Статус: manager.get_training_status()")
    
    # Интерактивное меню
    while True:
        print("\n📋 ВЫБЕРИТЕ ДЕЙСТВИЕ:")
        print("   1 - Запустить обучение")
        print("   2 - Мониторинг обучения")
        print("   3 - Статус обучения")
        print("   4 - Проверить GPU")
        print("   0 - Выход")
        
        choice = input("Ваш выбор: ").strip()
        
        if choice == "1":
            print("\n🚀 Запуск обучения...")
            if manager.start_training():
                print("✅ Обучение запущено!")
            else:
                print("❌ Ошибка запуска")
                
        elif choice == "2":
            print("\n📊 Мониторинг (Ctrl+C для выхода)...")
            manager.monitor_training()
            
        elif choice == "3":
            print("\n📋 Статус обучения...")
            status = manager.get_training_status()
            print(json.dumps(status, indent=2))
            
        elif choice == "4":
            print("\n🎮 Статус GPU...")
            gpu_status = manager.check_gpu_status()
            print(json.dumps(gpu_status, indent=2))
            
        elif choice == "0":
            print("👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор")


if __name__ == "__main__":
    main() 