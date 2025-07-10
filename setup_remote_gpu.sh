#!/bin/bash

# 🚀 УСТАНОВОЧНЫЙ СКРИПТ ДЛЯ УДАЛЕННОГО GPU СЕРВЕРА
# Сервер: 192.168.88.218 (NVIDIA GPU + CUDA 12.8)

echo "🚀 Настройка обучения на GPU сервере..."
echo "======================================"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка NVIDIA GPU
log_info "Проверка NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    log_success "NVIDIA GPU обнаружен"
else
    log_error "NVIDIA GPU не найден или драйверы не установлены"
    exit 1
fi

# Проверка CUDA
log_info "Проверка CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    log_success "CUDA версия: $CUDA_VERSION"
else
    log_warning "NVCC не найден в PATH, но CUDA может быть установлена"
fi

# Проверка Python
log_info "Проверка Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    log_success "$PYTHON_VERSION"
else
    log_error "Python3 не найден"
    exit 1
fi

# Создание рабочей директории
WORK_DIR="$HOME/gpu_training"
log_info "Создание рабочей директории: $WORK_DIR"
mkdir -p $WORK_DIR
mkdir -p $WORK_DIR/data
mkdir -p $WORK_DIR/logs
mkdir -p $WORK_DIR/models
mkdir -p $WORK_DIR/results

cd $WORK_DIR

# Создание виртуальной среды
log_info "Создание виртуальной среды..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_success "Виртуальная среда создана"
else
    log_warning "Виртуальная среда уже существует"
fi

# Активация виртуальной среды
log_info "Активация виртуальной среды..."
source venv/bin/activate

# Обновление pip
log_info "Обновление pip..."
pip install --upgrade pip

# Установка PyTorch с CUDA
log_info "Установка PyTorch с поддержкой CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Проверка PyTorch CUDA
log_info "Проверка PyTorch CUDA..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}'); print(f'GPU устройств: {torch.cuda.device_count()}')"

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    log_success "PyTorch с CUDA настроен успешно"
else
    log_error "PyTorch CUDA не работает"
    exit 1
fi

# Установка остальных зависимостей
log_info "Установка остальных зависимостей..."
pip install stable-baselines3>=2.2.1
pip install gymnasium>=0.29.0
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install tqdm psutil scipy
pip install tensorboard wandb
pip install ccxt>=4.0.0
pip install optuna joblib

log_success "Все зависимости установлены"

# Создание тестового скрипта для проверки GPU
log_info "Создание тестового скрипта..."
cat > test_gpu.py << 'EOF'
#!/usr/bin/env python3
"""
Тест GPU конфигурации для обучения
"""
import torch
import time

print("🎮 ТЕСТ GPU КОНФИГУРАЦИИ")
print("=" * 40)

# Основная информация
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"Количество GPU: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Память: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Мультипроцессоров: {props.multi_processor_count}")
        print(f"  CUDA возможности: {props.major}.{props.minor}")
    
    # Тест производительности
    print("\n🚀 Тест производительности...")
    device = torch.device('cuda:0')
    
    # Создание большой матрицы на GPU
    size = 5000
    print(f"Создание матрицы {size}x{size} на GPU...")
    
    start_time = time.time()
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Матричное умножение
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # Ожидание завершения
    
    gpu_time = time.time() - start_time
    print(f"Время на GPU: {gpu_time:.2f} секунд")
    
    # Освобождение памяти
    del a, b, c
    torch.cuda.empty_cache()
    
    print("✅ GPU тест завершен успешно!")
else:
    print("❌ CUDA недоступна")
EOF

# Запуск теста GPU
log_info "Запуск теста GPU..."
python3 test_gpu.py

# Создание скрипта для мониторинга GPU
cat > monitor_gpu.py << 'EOF'
#!/usr/bin/env python3
"""
Мониторинг GPU в реальном времени
"""
import subprocess
import time
import os

def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        return result.stdout.strip().split('\n')
    except:
        return None

def monitor_gpu():
    print("🎮 МОНИТОРИНГ GPU (Ctrl+C для выхода)")
    print("=" * 50)
    
    try:
        while True:
            os.system('clear')
            print("🎮 МОНИТОРИНГ GPU")
            print("=" * 50)
            
            gpu_info = get_gpu_info()
            if gpu_info:
                for i, info in enumerate(gpu_info):
                    if info.strip():
                        util, mem_used, mem_total, temp = info.split(', ')
                        mem_percent = (int(mem_used) / int(mem_total)) * 100
                        
                        print(f"GPU {i}:")
                        print(f"  Использование: {util}%")
                        print(f"  Память: {mem_used}MB / {mem_total}MB ({mem_percent:.1f}%)")
                        print(f"  Температура: {temp}°C")
                        print()
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n👋 Мониторинг остановлен")

if __name__ == "__main__":
    monitor_gpu()
EOF

log_success "Тестовые скрипты созданы"

# Финальная информация
echo ""
log_success "✅ НАСТРОЙКА ЗАВЕРШЕНА!"
echo "======================================"
echo ""
echo "📁 Рабочая директория: $WORK_DIR"
echo "🔧 Активация среды: source $WORK_DIR/venv/bin/activate"
echo "🧪 Тест GPU: python3 test_gpu.py"
echo "📊 Мониторинг GPU: python3 monitor_gpu.py"
echo ""
echo "🚀 Для запуска обучения:"
echo "   1. Скопируйте файлы обучения в $WORK_DIR"
echo "   2. Активируйте среду: source venv/bin/activate"
echo "   3. Запустите: python3 sentiment_trading_v69_remote.py"
echo ""

# Сохранение информации о системе
cat > system_info.txt << EOF
GPU Training Environment Setup
==============================
Date: $(date)
Host: $(hostname)
User: $(whoami)
Working Directory: $WORK_DIR

GPU Information:
$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv)

CUDA Version:
$(nvcc --version 2>/dev/null || echo "NVCC not in PATH")

Python Version:
$(python3 --version)

PyTorch Version:
$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "PyTorch not installed")
EOF

log_info "Информация о системе сохранена в system_info.txt"
log_success "Настройка завершена успешно! 🎉" 