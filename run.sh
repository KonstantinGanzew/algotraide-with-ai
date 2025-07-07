#!/bin/bash

echo "🚀 Запуск продвинутого торгового алгоритма с LSTM..."
echo

# Проверка и активация виртуального окружения
if [ -f "venv/bin/activate" ]; then
    echo "📁 Активация виртуального окружения..."
    source venv/bin/activate
else
    echo "⚠️  Виртуальное окружение не найдено, используем системный Python"
fi

echo
echo "🧪 Быстрая проверка системы..."
python test_advanced.py

if [ $? -eq 0 ]; then
    echo
    echo "✅ Тест пройден! Запускаем основное обучение..."
    echo
    python main.py
    
    if [ $? -eq 0 ]; then
        echo
        echo "🎉 Обучение завершено успешно!"
    else
        echo
        echo "❌ Ошибка во время обучения"
    fi
else
    echo
    echo "❌ Тест не пройден. Проверьте установку зависимостей:"
    echo "pip install -r requirements.txt"
    echo
fi

echo
echo "💡 Для выхода из виртуального окружения наберите: deactivate" 