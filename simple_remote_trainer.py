#!/usr/bin/env python3
"""
🚀 ПРОСТОЙ МЕНЕДЖЕР УДАЛЕННОГО ОБУЧЕНИЯ
Использует sshpass для парольной аутентификации
Сервер: 192.168.88.218 (NVIDIA GPU + CUDA 12.8)
"""

import subprocess
import os
import getpass
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleRemoteTrainer:
    """Простой менеджер для удаленного обучения"""
    
    def __init__(self, server_ip: str = "192.168.88.218", username: str = "user", password: Optional[str] = None):
        self.server_ip = server_ip
        self.username = username
        self.remote_path = f"/home/{username}/gpu_training"
        self.connection = f"{username}@{server_ip}"
        
        # Запрос пароля один раз при инициализации
        self.password = self._get_password_once(password)
        
        # Файлы для синхронизации
        self.files_to_sync = [
            "sentiment_trading_v69_remote.py",
            "requirements-gpu.txt",
            "data/BTCUSDT_5m_2y.csv",
            "data/BTCUSDT_1h_2y.csv",
            "data/BTCUSDT_4h_2y.csv", 
            "data/BTCUSDT_1d_2y.csv"
        ]
    
    def _get_password_once(self, provided_password: Optional[str]) -> str:
        """Получение пароля один раз при инициализации"""
        if provided_password:
            return provided_password
        
        env_password = os.getenv('REMOTE_PASSWORD')
        if env_password:
            return env_password
            
        return getpass.getpass(f"🔑 Пароль для {self.connection}: ")
    
    def _get_password(self) -> str:
        """Возвращает сохраненный пароль"""
        return self.password
    
    def _run_ssh_command(self, command: str, capture_output: bool = True) -> Dict:
        """Выполнение SSH команды через sshpass"""
        password = self._get_password()
        
        # Проверка установки sshpass
        if not self._check_sshpass():
            return {'success': False, 'error': 'sshpass не установлен'}
        
        ssh_cmd = [
            "sshpass", f"-p{password}",
            "ssh", "-o", "StrictHostKeyChecking=no",
            self.connection, command
        ]
        
        logger.info(f"🔧 Выполнение: {command}")
        
        try:
            result = subprocess.run(
                ssh_cmd, 
                capture_output=capture_output, 
                text=True, 
                timeout=300
            )
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout if capture_output else '',
                'stderr': result.stderr if capture_output else ''
            }
            
        except subprocess.TimeoutExpired:
            logger.error("⏰ Тайм-аут команды")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_scp_command(self, source: str, destination: str) -> bool:
        """Копирование файлов через scp с паролем"""
        password = self._get_password()
        
        scp_cmd = [
            "sshpass", f"-p{password}",
            "scp", "-o", "StrictHostKeyChecking=no",
            source, destination
        ]
        
        logger.info(f"📁 Копирование: {source} -> {destination}")
        
        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("✅ Файл скопирован")
                return True
            else:
                logger.error(f"❌ Ошибка копирования: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка SCP: {e}")
            return False
    
    def _check_sshpass(self) -> bool:
        """Проверка установки sshpass"""
        try:
            result = subprocess.run(['which', 'sshpass'], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def install_sshpass(self) -> bool:
        """Установка sshpass если не установлен"""
        if self._check_sshpass():
            logger.info("✅ sshpass уже установлен")
            return True
        
        logger.info("📦 Установка sshpass...")
        
        # Определение пакетного менеджера
        managers = [
            (['apt', '-y', 'install', 'sshpass'], 'apt'),
            (['yum', '-y', 'install', 'sshpass'], 'yum'),
            (['pacman', '-S', '--noconfirm', 'sshpass'], 'pacman'),
            (['brew', 'install', 'hudochenkov/sshpass/sshpass'], 'brew')
        ]
        
        for cmd, manager in managers:
            try:
                result = subprocess.run(['which', manager.split()[0]], capture_output=True)
                if result.returncode == 0:
                    logger.info(f"🔧 Используется {manager}")
                    install_result = subprocess.run(['sudo'] + cmd, capture_output=True, text=True)
                    
                    if install_result.returncode == 0:
                        logger.info("✅ sshpass установлен")
                        return True
                    else:
                        logger.error(f"❌ Ошибка установки: {install_result.stderr}")
            except:
                continue
        
        logger.error("❌ Не удалось установить sshpass")
        logger.error("💡 Установите вручную:")
        logger.error("   Ubuntu/Debian: sudo apt install sshpass")
        logger.error("   CentOS/RHEL: sudo yum install sshpass")
        logger.error("   macOS: brew install hudochenkov/sshpass/sshpass")
        return False
    
    def test_connection(self) -> bool:
        """Тест подключения"""
        logger.info(f"🔗 Тест подключения к {self.server_ip}...")
        
        result = self._run_ssh_command("echo 'Подключение успешно'")
        
        if result['success']:
            logger.info("✅ Подключение установлено")
            logger.info(f"Ответ сервера: {result['stdout'].strip()}")
            return True
        else:
            logger.error("❌ Ошибка подключения")
            logger.error(f"Ошибка: {result.get('stderr', result.get('error', 'Unknown'))}")
            return False
    
    def check_gpu(self) -> Dict:
        """Проверка GPU на сервере"""
        logger.info("🎮 Проверка GPU...")
        
        result = self._run_ssh_command("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader")
        
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
            
            logger.info(f"✅ Найдено GPU: {len(gpus)}")
            return {'success': True, 'gpus': gpus}
        else:
            logger.warning("⚠️ GPU не найдены или nvidia-smi недоступен")
            return {'success': False, 'error': result.get('stderr', 'GPU not available')}
    
    def setup_environment(self) -> bool:
        """Настройка удаленной среды"""
        logger.info("🔧 Настройка среды...")
        
        commands = [
            f"mkdir -p {self.remote_path}",
            f"mkdir -p {self.remote_path}/data",
            f"mkdir -p {self.remote_path}/logs",
            f"mkdir -p {self.remote_path}/models",
            f"mkdir -p {self.remote_path}/results"
        ]
        
        for cmd in commands:
            result = self._run_ssh_command(cmd)
            if not result['success']:
                logger.error(f"❌ Ошибка: {cmd}")
                return False
        
        logger.info("✅ Среда настроена")
        return True
    
    def sync_files(self) -> bool:
        """Синхронизация файлов"""
        logger.info("📁 Синхронизация файлов...")
        
        success = True
        for file_path in self.files_to_sync:
            local_file = Path(file_path)
            
            if not local_file.exists():
                logger.warning(f"⚠️ Файл не найден: {file_path}")
                continue
            
            if 'data/' in file_path:
                remote_file = f"{self.connection}:{self.remote_path}/data/{local_file.name}"
            else:
                remote_file = f"{self.connection}:{self.remote_path}/{local_file.name}"
            
            if not self._run_scp_command(str(local_file), remote_file):
                success = False
        
        if success:
            logger.info("✅ Файлы синхронизированы")
        
        return success
    
    def install_dependencies(self) -> bool:
        """Установка зависимостей"""
        logger.info("📦 Установка зависимостей...")
        
        commands = [
            "python3 --version",
            f"cd {self.remote_path} && python3 -m venv venv",
            f"cd {self.remote_path} && source venv/bin/activate && pip install --upgrade pip",
            f"cd {self.remote_path} && source venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            f"cd {self.remote_path} && source venv/bin/activate && pip install stable-baselines3 gymnasium numpy pandas matplotlib ccxt"
        ]
        
        for i, cmd in enumerate(commands, 1):
            logger.info(f"📦 Шаг {i}/{len(commands)}")
            result = self._run_ssh_command(cmd, capture_output=False)
            
            if not result['success']:
                logger.error(f"❌ Ошибка на шаге {i}")
                return False
        
        logger.info("✅ Зависимости установлены")
        return True
    
    def start_training(self) -> Dict:
        """Запуск обучения"""
        logger.info("🚀 Запуск обучения...")
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/training_{timestamp}.log"
        
        cmd = (f"cd {self.remote_path} && "
               f"source venv/bin/activate && "
               f"nohup python sentiment_trading_v69_remote.py > {log_file} 2>&1 & "
               f"echo $!")
        
        result = self._run_ssh_command(cmd)
        
        if result['success']:
            pid = result['stdout'].strip()
            logger.info(f"✅ Обучение запущено (PID: {pid})")
            logger.info(f"📄 Лог: {log_file}")
            
            return {
                'success': True,
                'pid': pid,
                'log_file': log_file,
                'timestamp': timestamp
            }
        else:
            logger.error("❌ Ошибка запуска")
            return {'success': False, 'error': result.get('stderr', 'Unknown error')}
    
    def monitor_training(self, log_file: str = None) -> None:
        """Мониторинг обучения"""
        if not log_file:
            # Найти последний лог
            result = self._run_ssh_command(f"ls -t {self.remote_path}/logs/training_*.log | head -1")
            if result['success'] and result['stdout'].strip():
                log_file = result['stdout'].strip()
            else:
                logger.error("❌ Лог файл не найден")
                return
        else:
            log_file = f"{self.remote_path}/{log_file}"
        
        logger.info(f"📊 Мониторинг: {log_file}")
        logger.info("⏹️ Ctrl+C для выхода")
        
        try:
            # Мониторинг через tail -f
            password = self._get_password()
            cmd = [
                "sshpass", f"-p{password}",
                "ssh", "-o", "StrictHostKeyChecking=no",
                self.connection, f"tail -f {log_file}"
            ]
            
            subprocess.run(cmd)
            
        except KeyboardInterrupt:
            logger.info("⏹️ Мониторинг остановлен")
        except Exception as e:
            logger.error(f"❌ Ошибка мониторинга: {e}")
    
    def get_status(self) -> Dict:
        """Статус обучения"""
        result = self._run_ssh_command("pgrep -f 'python.*sentiment_trading' || echo 'No processes'")
        
        processes = []
        if result['success'] and 'No processes' not in result['stdout']:
            processes = result['stdout'].strip().split('\n')
        
        log_result = self._run_ssh_command(f"ls -t {self.remote_path}/logs/training_*.log | head -1 | xargs tail -5 || echo 'No logs'")
        logs = log_result['stdout'] if log_result['success'] else 'No logs'
        
        return {
            'processes': processes,
            'logs': logs,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }


def main():
    """Главная функция"""
    print("🚀 ПРОСТОЙ МЕНЕДЖЕР УДАЛЕННОГО GPU ОБУЧЕНИЯ")
    print("=" * 60)
    
    # Получение параметров
    server_ip = input("🖥️  IP сервера [192.168.88.218]: ").strip() or "192.168.88.218"
    username = input("👤 Username: ").strip()
    
    if not username:
        print("❌ Username обязателен!")
        return
    
    print(f"\n📋 Подключение: {username}@{server_ip}")
    
    # Создание тренера (пароль будет запрошен в конструкторе)
    trainer = SimpleRemoteTrainer(server_ip, username, None)
    
    # Проверка sshpass
    if not trainer.install_sshpass():
        return
    
    # Тест подключения
    if not trainer.test_connection():
        return
    
    # Меню
    while True:
        print("\n📋 ВЫБЕРИТЕ ДЕЙСТВИЕ:")
        print("   1 - Полная настройка и запуск")
        print("   2 - Проверить GPU")
        print("   3 - Настроить среду") 
        print("   4 - Синхронизировать файлы")
        print("   5 - Установить зависимости")
        print("   6 - Запустить обучение")
        print("   7 - Мониторинг")
        print("   8 - Статус")
        print("   0 - Выход")
        
        choice = input("Выбор: ").strip()
        
        if choice == "1":
            print("\n🚀 ПОЛНАЯ НАСТРОЙКА...")
            steps = [
                ("🎮 Проверка GPU", trainer.check_gpu),
                ("🔧 Настройка среды", trainer.setup_environment),
                ("📁 Синхронизация", trainer.sync_files),
                ("📦 Установка зависимостей", trainer.install_dependencies)
            ]
            
            success = True
            for name, func in steps:
                print(f"\n{name}...")
                if not func():
                    success = False
                    break
            
            if success:
                print("\n🚀 Запуск обучения...")
                result = trainer.start_training()
                if result['success']:
                    print(f"✅ Запущено! PID: {result['pid']}")
                else:
                    print("❌ Ошибка запуска")
                    
        elif choice == "2":
            gpu_status = trainer.check_gpu()
            print(json.dumps(gpu_status, indent=2, ensure_ascii=False))
            
        elif choice == "3":
            trainer.setup_environment()
            
        elif choice == "4":
            trainer.sync_files()
            
        elif choice == "5":
            trainer.install_dependencies()
            
        elif choice == "6":
            result = trainer.start_training()
            if result['success']:
                print(f"✅ Запущено! PID: {result['pid']}")
            else:
                print("❌ Ошибка запуска")
                
        elif choice == "7":
            trainer.monitor_training()
            
        elif choice == "8":
            status = trainer.get_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            
        elif choice == "0":
            print("👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор")


if __name__ == "__main__":
    main() 