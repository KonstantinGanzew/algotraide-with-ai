#!/bin/bash

echo "========================================"
echo "Установка GPU версии торгового алгоритма"
echo "========================================"

echo "Проверка NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ NVIDIA GPU обнаружен!"
        GPU_AVAILABLE=true
    else
        echo "❌ NVIDIA GPU недоступен!"
        GPU_AVAILABLE=false
    fi
else
    echo "❌ nvidia-smi не найден!"
    GPU_AVAILABLE=false
fi

echo

if [ "$GPU_AVAILABLE" = false ]; then
    echo "Устанавливаю CPU версию..."
    REQUIREMENTS_FILE="requirements.txt"
else
    echo "Устанавливаю GPU версию..."
    REQUIREMENTS_FILE="requirements-gpu.txt"
fi

echo "Создание виртуального окружения..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Ошибка создания виртуального окружения"
    exit 1
fi

echo "Активация виртуального окружения..."
source venv/bin/activate

echo "Обновление pip..."
python -m pip install --upgrade pip

echo "Установка зависимостей из $REQUIREMENTS_FILE..."
pip install -r $REQUIREMENTS_FILE
if [ $? -ne 0 ]; then
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "⚠️ Ошибка установки GPU версии, переключаемся на CPU..."
        pip install -r requirements.txt
    else
        echo "❌ Ошибка установки!"
        exit 1
    fi
fi

if [ "$GPU_AVAILABLE" = true ]; then
    echo "✅ GPU версия успешно установлена!"
else
    echo "✅ CPU версия установлена!"
fi

echo "Для запуска используйте: ./run.sh"
echo
echo "Установка завершена!" 