#!/bin/bash

echo "========================================"
echo "Установка GPU версии торговой системы (Linux)"
echo "========================================"

# Проверка доступности GPU
GPU_TYPE="none"
GPU_AVAILABLE=false

echo "🔍 Проверка доступных GPU..."

# Проверка NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ NVIDIA GPU обнаружен!"
        GPU_TYPE="nvidia"
        GPU_AVAILABLE=true
        nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
    fi
fi

# Проверка AMD GPU (ROCm)
if command -v rocm-smi &> /dev/null; then
    rocm-smi > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ AMD ROCm GPU обнаружен!"
        if [ "$GPU_TYPE" = "none" ]; then
            GPU_TYPE="amd"
            GPU_AVAILABLE=true
        fi
    fi
fi

# Проверка AMD GPU через lspci
if [ "$GPU_TYPE" = "none" ] && lspci | grep -i "amd\|ati" | grep -i "vga\|3d\|display" > /dev/null; then
    echo "⚠️ AMD GPU найден, но ROCm не установлен"
    echo "💡 Для использования AMD GPU установите ROCm: https://rocmdocs.amd.com/en/latest/deploy/linux/quick_start.html"
fi

if [ "$GPU_AVAILABLE" = false ]; then
    echo "❌ GPU не обнаружен, будет использован CPU"
    echo
fi

# Определение файла требований
if [ "$GPU_AVAILABLE" = false ]; then
    echo "📦 Устанавливаю CPU версию..."
    REQUIREMENTS_FILE="requirements.txt"
else
    echo "🚀 Устанавливаю GPU версию ($GPU_TYPE)..."
    REQUIREMENTS_FILE="requirements-gpu-linux.txt"
fi

echo "📁 Создание виртуального окружения..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Ошибка создания виртуального окружения"
    echo "💡 Убедитесь, что установлен python3-venv: sudo apt install python3-venv"
    exit 1
fi

echo "🔄 Активация виртуального окружения..."
source venv/bin/activate

echo "⬆️ Обновление pip..."
python -m pip install --upgrade pip

echo "📚 Установка зависимостей из $REQUIREMENTS_FILE..."
pip install -r $REQUIREMENTS_FILE
if [ $? -ne 0 ]; then
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "⚠️ Ошибка установки GPU версии, переключаемся на CPU..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "❌ Ошибка установки!"
            exit 1
        fi
        echo "✅ CPU версия установлена!"
    else
        echo "❌ Ошибка установки!"
        exit 1
    fi
else
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "✅ GPU версия успешно установлена!"
        
        # Тестирование GPU
        echo "🧪 Тестирование GPU..."
        python3 -c "
import torch
print(f'PyTorch версия: {torch.__version__}')
if torch.cuda.is_available():
    print(f'✅ CUDA доступен: {torch.cuda.get_device_name(0)}')
    print(f'CUDA версия: {torch.version.cuda}')
elif hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
    print('✅ ROCm/HIP доступен')
else:
    print('⚠️ GPU не обнаружен PyTorch, будет использован CPU')
"
    else
        echo "✅ CPU версия установлена!"
    fi
fi

echo
echo "🎯 Установка завершена!"
echo "🚀 Для запуска используйте:"
echo "   source venv/bin/activate"
echo "   python sentiment_trading_v63.py"

if [ "$GPU_AVAILABLE" = true ]; then
    echo
    echo "📊 Для мониторинга GPU используйте:"
    if [ "$GPU_TYPE" = "nvidia" ]; then
        echo "   nvidia-smi"
        echo "   watch -n 1 nvidia-smi"
    elif [ "$GPU_TYPE" = "amd" ]; then
        echo "   rocm-smi"
        echo "   watch -n 1 rocm-smi"
    fi
fi 