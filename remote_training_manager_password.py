"""
🚀 МЕНЕДЖЕР УДАЛЕННОГО ОБУЧЕНИЯ С ПАРОЛЬНОЙ АУТЕНТИФИКАЦИЕЙ
Поддержка подключения к GPU серверу по паролю
Сервер: 192.168.88.218 (NVIDIA GPU + CUDA 12.8)
"""

import paramiko
import scp
import os
import json
import time
import getpass
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import threading
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RemoteGPUTrainingManager:
    """Менеджер для удаленного обучения на GPU сервере с парольной аутентификацией"""
    
    def __init__(self, 
                 server_ip: str = "192.168.88.218",
                 username: str = "user",
                 password: Optional[str] = None,
                 remote_path: str = "/home/user/gpu_training",
                 port: int = 22):
        """
        Инициализация менеджера удаленного обучения
        
        Args:
            server_ip: IP адрес сервера
            username: Имя пользователя на сервере  
            password: Пароль (если None - будет запрошен интерактивно)
            remote_path: Путь к рабочей директории на сервере
            port: SSH порт
        """
        self.server_ip = server_ip
        self.username = username
        self.password = password
        self.remote_path = remote_path
        self.port = port
        
        # SSH клиент
        self.ssh_client = None
        self.scp_client = None
        
        # Локальные файлы для синхронизации
        self.sync_files = [
            "sentiment_trading_v69_remote.py",
            "requirements-gpu.txt",
            "data/BTCUSDT_5m_2y.csv",
            "data/BTCUSDT_1h_2y.csv", 
            "data/BTCUSDT_4h_2y.csv",
            "data/BTCUSDT_1d_2y.csv"
        ]
        
    def _get_password(self) -> str:
        """Получение пароля (интерактивно или из переменной)"""
        if self.password:
            return self.password
        
        # Попробовать получить из переменной окружения
        env_password = os.getenv('REMOTE_SERVER_PASSWORD')
        if env_password:
            return env_password
            
        # Запросить интерактивно
        return getpass.getpass(f"Введите пароль для {self.username}@{self.server_ip}: ")
    
    def connect(self) -> bool:
        """Установка SSH подключения"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            password = self._get_password()
            
            logger.info(f"🔗 Подключение к {self.server_ip}...")
            self.ssh_client.connect(
                hostname=self.server_ip,
                port=self.port,
                username=self.username,
                password=password,
                timeout=30,
                look_for_keys=False,  # Отключить поиск SSH ключей
                allow_agent=False     # Отключить SSH агент
            )
            
            # Создание SCP клиента
            transport = self.ssh_client.get_transport()
            if transport:
                self.scp_client = scp.SCPClient(transport)
            else:
                raise Exception("Не удалось получить транспорт SSH")
            
            logger.info("✅ SSH подключение установлено")
            return True
            
        except paramiko.AuthenticationException:
            logger.error("❌ Ошибка аутентификации - неверный пароль")
            return False
        except paramiko.SSHException as e:
            logger.error(f"❌ SSH ошибка: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка подключения: {e}")
            return False
    
    def disconnect(self):
        """Закрытие подключения"""
        if self.scp_client:
            self.scp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
        logger.info("🔌 Подключение закрыто")
    
    def execute_command(self, command: str, timeout: int = 300) -> Dict[str, Any]:
        """Выполнение команды на удаленном сервере"""
        if not self.ssh_client:
            return {'success': False, 'error': 'No SSH connection'}
        
        try:
            logger.info(f"🔧 Выполнение: {command}")
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
            
            # Чтение вывода
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            exit_code = stdout.channel.recv_exit_status()
            
            result = {
                'success': exit_code == 0,
                'exit_code': exit_code,
                'stdout': stdout_text,
                'stderr': stderr_text
            }
            
            if exit_code == 0:
                logger.info("✅ Команда выполнена успешно")
            else:
                logger.error(f"❌ Команда завершилась с ошибкой (код: {exit_code})")
                if stderr_text:
                    logger.error(f"Ошибка: {stderr_text.strip()}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения команды: {e}")
            return {'success': False, 'error': str(e)}
    
    def check_gpu_status(self) -> Dict[str, Any]:
        """Проверка статуса GPU на сервере"""
        logger.info("🎮 Проверка GPU статуса...")
        
        # Проверка NVIDIA-SMI
        result = self.execute_command("nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits")
        
        if result['success']:
            gpu_info = result['stdout'].strip().split('\n')
            gpu_data = []
            
            for i, gpu_line in enumerate(gpu_info):
                if gpu_line.strip():
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
            cuda_result = self.execute_command("nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-")
            cuda_version = cuda_result['stdout'].strip() if cuda_result['success'] else "Unknown"
            
            logger.info(f"✅ Найдено {len(gpu_data)} GPU(s)")
            logger.info(f"🔧 CUDA версия: {cuda_version}")
            
            return {
                'available': True,
                'cuda_version': cuda_version,
                'gpus': gpu_data
            }
        else:
            logger.warning("⚠️ NVIDIA-SMI не найден или недоступен")
            return {'available': False, 'error': result.get('stderr', 'Unknown error')}
    
    def setup_environment(self) -> bool:
        """Настройка удаленной среды"""
        logger.info("🔧 Настройка удаленной среды...")
        
        commands = [
            f"mkdir -p {self.remote_path}",
            f"mkdir -p {self.remote_path}/data",
            f"mkdir -p {self.remote_path}/logs",
            f"mkdir -p {self.remote_path}/models",
            f"mkdir -p {self.remote_path}/results"
        ]
        
        for cmd in commands:
            result = self.execute_command(cmd)
            if not result['success']:
                logger.error(f"❌ Ошибка создания директории: {cmd}")
                return False
        
        logger.info("✅ Удаленная среда настроена")
        return True
    
    def sync_files(self) -> bool:
        """Синхронизация файлов на сервер"""
        if not self.scp_client:
            logger.error("❌ SCP клиент не подключен")
            return False
        
        logger.info("📁 Синхронизация файлов...")
        success = True
        
        for file_path in self.sync_files:
            local_path = Path(file_path)
            
            if not local_path.exists():
                logger.warning(f"⚠️ Файл не найден: {file_path}")
                continue
            
            try:
                if 'data/' in file_path:
                    remote_file_path = f"{self.remote_path}/data/{local_path.name}"
                else:
                    remote_file_path = f"{self.remote_path}/{local_path.name}"
                
                logger.info(f"📤 Копирование: {file_path} -> {remote_file_path}")
                self.scp_client.put(str(local_path), remote_file_path)
                logger.info("✅ Файл скопирован")
                
            except Exception as e:
                logger.error(f"❌ Ошибка копирования {file_path}: {e}")
                success = False
        
        if success:
            logger.info("✅ Все файлы синхронизированы")
        
        return success
    
    def install_dependencies(self) -> bool:
        """Установка зависимостей на сервере"""
        logger.info("📦 Установка зависимостей...")
        
        commands = [
            # Проверка Python
            "python3 --version",
            
            # Создание виртуальной среды
            f"cd {self.remote_path} && python3 -m venv venv || true",
            
            # Активация и обновление pip
            f"cd {self.remote_path} && source venv/bin/activate && pip install --upgrade pip",
            
            # Установка PyTorch с CUDA
            f"cd {self.remote_path} && source venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            
            # Установка основных зависимостей
            f"cd {self.remote_path} && source venv/bin/activate && pip install stable-baselines3>=2.2.1 gymnasium>=0.29.0",
            
            # Установка остальных библиотек
            f"cd {self.remote_path} && source venv/bin/activate && pip install numpy pandas scikit-learn matplotlib seaborn tqdm psutil ccxt"
        ]
        
        for i, cmd in enumerate(commands, 1):
            logger.info(f"🔧 Шаг {i}/{len(commands)}: {cmd.split('&&')[-1].strip()}")
            result = self.execute_command(cmd, timeout=600)  # Увеличенный тайм-аут для установки
            
            if not result['success']:
                logger.error(f"❌ Ошибка на шаге {i}")
                logger.error(f"Команда: {cmd}")
                logger.error(f"Ошибка: {result.get('stderr', 'Unknown error')}")
                return False
                
        logger.info("✅ Зависимости установлены")
        return True
    
    def test_gpu_setup(self) -> bool:
        """Тест GPU конфигурации"""
        logger.info("🧪 Тестирование GPU конфигурации...")
        
        test_script = '''
import torch
print("=" * 50)
print("🎮 ТЕСТ GPU КОНФИГУРАЦИИ")
print("=" * 50)
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"Количество GPU: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Память: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Мультипроцессоров: {props.multi_processor_count}")
    
    # Быстрый тест производительности
    device = torch.device('cuda:0')
    x = torch.randn(1000, 1000, device=device)
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print("✅ GPU тест прошел успешно!")
else:
    print("❌ CUDA недоступна")
        '''
        
        # Создание тестового файла
        with open('temp_gpu_test.py', 'w') as f:
            f.write(test_script)
        
        try:
            # Копирование тестового файла
            self.scp_client.put('temp_gpu_test.py', f'{self.remote_path}/gpu_test.py')
            
            # Запуск теста
            result = self.execute_command(f"cd {self.remote_path} && source venv/bin/activate && python gpu_test.py")
            
            if result['success']:
                logger.info("✅ GPU тест прошел успешно")
                print("\n" + "="*50)
                print("🎮 РЕЗУЛЬТАТ GPU ТЕСТА:")
                print("="*50)
                print(result['stdout'])
                return True
            else:
                logger.error("❌ GPU тест не прошел")
                logger.error(result.get('stderr', 'Unknown error'))
                return False
                
        finally:
            # Удаление временного файла
            if os.path.exists('temp_gpu_test.py'):
                os.remove('temp_gpu_test.py')
        
        return False
    
    def start_training(self, script_name: str = "sentiment_trading_v69_remote.py") -> Dict[str, Any]:
        """Запуск обучения на сервере"""
        logger.info("🚀 Запуск обучения на удаленном GPU сервере...")
        
        # Создание уникального имени лога
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/training_{timestamp}.log"
        
        # Команда для запуска в фоне
        cmd = (f"cd {self.remote_path} && "
               f"source venv/bin/activate && "
               f"nohup python {script_name} > {log_file} 2>&1 & "
               f"echo $!")
        
        result = self.execute_command(cmd)
        
        if result['success']:
            process_id = result['stdout'].strip()
            logger.info(f"✅ Обучение запущено в фоне (PID: {process_id})")
            logger.info(f"📋 Лог файл: {log_file}")
            
            return {
                'success': True,
                'process_id': process_id,
                'log_file': log_file,
                'timestamp': timestamp
            }
        else:
            logger.error(f"❌ Ошибка запуска: {result.get('stderr')}")
            return {
                'success': False,
                'error': result.get('stderr', 'Unknown error')
            }
    
    def monitor_training(self, log_file: str = None) -> None:
        """Мониторинг обучения в реальном времени"""
        logger.info("📊 Запуск мониторинга обучения...")
        
        if not log_file:
            # Найти последний лог файл
            result = self.execute_command(f"ls -t {self.remote_path}/logs/training_*.log | head -1")
            if result['success'] and result['stdout'].strip():
                log_file = result['stdout'].strip()
            else:
                logger.error("❌ Лог файл не найден")
                return
        else:
            log_file = f"{self.remote_path}/{log_file}"
        
        try:
            logger.info(f"📋 Мониторинг: {log_file}")
            logger.info("⏹️ Нажмите Ctrl+C для выхода")
            
            # Мониторинг с помощью tail -f
            command = f"tail -f {log_file}"
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            
            # Читаем вывод в реальном времени
            for line in iter(stdout.readline, ""):
                print(line.rstrip())
                
        except KeyboardInterrupt:
            logger.info("⏹️ Мониторинг остановлен")
        except Exception as e:
            logger.error(f"❌ Ошибка мониторинга: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Получение статуса обучения"""
        try:
            # Проверка запущенных процессов Python
            result = self.execute_command("pgrep -f 'python.*sentiment_trading' || echo 'No processes'")
            
            processes = []
            if result['success'] and 'No processes' not in result['stdout']:
                pids = result['stdout'].strip().split('\n')
                processes = [{'pid': pid.strip()} for pid in pids if pid.strip()]
            
            # Последние логи
            log_result = self.execute_command(f"ls -t {self.remote_path}/logs/training_*.log 2>/dev/null | head -1 | xargs tail -10 2>/dev/null || echo 'No logs'")
            latest_logs = log_result['stdout'].strip() if log_result['success'] else "No logs available"
            
            return {
                'processes': processes,
                'latest_logs': latest_logs,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def download_results(self, local_dir: str = "./remote_results") -> bool:
        """Скачивание результатов обучения"""
        logger.info("📥 Скачивание результатов...")
        
        # Создание локальной директории
        Path(local_dir).mkdir(exist_ok=True)
        
        try:
            # Список файлов для скачивания
            files_to_download = [
                "model_gpu_*.zip",
                "results_*.png", 
                "stats_*.txt",
                "training.log",
                "logs/training_*.log"
            ]
            
            success = True
            for pattern in files_to_download:
                # Найти файлы по паттерну
                result = self.execute_command(f"find {self.remote_path} -name '{pattern}' -type f 2>/dev/null || true")
                
                if result['success'] and result['stdout'].strip():
                    files = result['stdout'].strip().split('\n')
                    
                    for remote_file in files:
                        if remote_file.strip():
                            local_file = Path(local_dir) / Path(remote_file).name
                            try:
                                logger.info(f"📥 Скачивание: {remote_file}")
                                self.scp_client.get(remote_file, str(local_file))
                                logger.info(f"✅ Сохранено: {local_file}")
                            except Exception as e:
                                logger.error(f"❌ Ошибка скачивания {remote_file}: {e}")
                                success = False
            
            if success:
                logger.info(f"✅ Результаты скачаны в {local_dir}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Ошибка скачивания результатов: {e}")
            return False


def main():
    """Главная функция для интерактивного управления"""
    print("🚀 МЕНЕДЖЕР УДАЛЕННОГО GPU ОБУЧЕНИЯ (С ПАРОЛЕМ)")
    print("=" * 60)
    
    # Получение параметров подключения
    server_ip = input("🖥️  IP сервера [192.168.88.218]: ").strip() or "192.168.88.218"
    username = input("👤 Username: ").strip()
    if not username:
        print("❌ Username обязателен!")
        return
    
    remote_path = input(f"📁 Удаленный путь [/home/{username}/gpu_training]: ").strip()
    if not remote_path:
        remote_path = f"/home/{username}/gpu_training"
    
    print(f"\n📋 КОНФИГУРАЦИЯ:")
    print(f"   🖥️  Сервер: {server_ip}")
    print(f"   👤 Пользователь: {username}")
    print(f"   📁 Удаленный путь: {remote_path}")
    print()
    
    # Создание менеджера
    manager = RemoteGPUTrainingManager(
        server_ip=server_ip,
        username=username,
        remote_path=remote_path
    )
    
    try:
        # Подключение
        if not manager.connect():
            print("❌ Не удалось подключиться к серверу")
            return
        
        # Интерактивное меню
        while True:
            print("\n📋 ВЫБЕРИТЕ ДЕЙСТВИЕ:")
            print("   1 - Полная настройка и запуск обучения")
            print("   2 - Проверить GPU статус")
            print("   3 - Настроить среду")
            print("   4 - Синхронизировать файлы")
            print("   5 - Установить зависимости")
            print("   6 - Тест GPU")
            print("   7 - Запустить обучение")
            print("   8 - Мониторинг обучения")
            print("   9 - Статус обучения")
            print("   10 - Скачать результаты")
            print("   0 - Выход")
            
            choice = input("Ваш выбор: ").strip()
            
            if choice == "1":
                print("\n🚀 ПОЛНАЯ НАСТРОЙКА И ЗАПУСК...")
                steps = [
                    ("🎮 Проверка GPU", lambda: manager.check_gpu_status()),
                    ("🔧 Настройка среды", manager.setup_environment),
                    ("📁 Синхронизация файлов", manager.sync_files),
                    ("📦 Установка зависимостей", manager.install_dependencies),
                    ("🧪 Тест GPU", manager.test_gpu_setup),
                ]
                
                success = True
                for step_name, step_func in steps:
                    print(f"\n{step_name}...")
                    if not step_func():
                        print(f"❌ Ошибка на этапе: {step_name}")
                        success = False
                        break
                
                if success:
                    print("\n🚀 Запуск обучения...")
                    result = manager.start_training()
                    if result['success']:
                        print("✅ Обучение запущено!")
                        print(f"📋 PID: {result['process_id']}")
                        print(f"📄 Лог: {result['log_file']}")
                    else:
                        print("❌ Ошибка запуска обучения")
                
            elif choice == "2":
                print("\n🎮 Проверка GPU...")
                gpu_status = manager.check_gpu_status()
                print(json.dumps(gpu_status, indent=2, ensure_ascii=False))
                
            elif choice == "3":
                print("\n🔧 Настройка среды...")
                if manager.setup_environment():
                    print("✅ Среда настроена")
                else:
                    print("❌ Ошибка настройки")
                    
            elif choice == "4":
                print("\n📁 Синхронизация файлов...")
                if manager.sync_files():
                    print("✅ Файлы синхронизированы")
                else:
                    print("❌ Ошибка синхронизации")
                    
            elif choice == "5":
                print("\n📦 Установка зависимостей...")
                if manager.install_dependencies():
                    print("✅ Зависимости установлены")
                else:
                    print("❌ Ошибка установки")
                    
            elif choice == "6":
                print("\n🧪 Тест GPU...")
                if manager.test_gpu_setup():
                    print("✅ GPU тест прошел успешно")
                else:
                    print("❌ GPU тест не прошел")
                    
            elif choice == "7":
                print("\n🚀 Запуск обучения...")
                result = manager.start_training()
                if result['success']:
                    print("✅ Обучение запущено!")
                    print(f"📋 PID: {result['process_id']}")
                    print(f"📄 Лог: {result['log_file']}")
                else:
                    print("❌ Ошибка запуска")
                    
            elif choice == "8":
                print("\n📊 Мониторинг (Ctrl+C для выхода)...")
                manager.monitor_training()
                
            elif choice == "9":
                print("\n📋 Статус обучения...")
                status = manager.get_training_status()
                print(json.dumps(status, indent=2, ensure_ascii=False))
                
            elif choice == "10":
                print("\n📥 Скачивание результатов...")
                if manager.download_results():
                    print("✅ Результаты скачаны")
                else:
                    print("❌ Ошибка скачивания")
                    
            elif choice == "0":
                print("👋 До свидания!")
                break
            else:
                print("❌ Неверный выбор")
    
    finally:
        manager.disconnect()


if __name__ == "__main__":
    main() 