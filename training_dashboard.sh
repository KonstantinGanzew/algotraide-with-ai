#!/bin/bash
clear
echo "🚀 ДАШБОРД ОБУЧЕНИЯ RL АГЕНТА"
echo "=================================================="
echo "⏰ Время: $(date '+%H:%M:%S')"
echo

# Статус процесса
if ps -p 177228 > /dev/null 2>&1; then
    UPTIME=$(ps -p 177228 -o etime= | tr -d ' ')
    CPU=$(ps -p 177228 -o %cpu= | tr -d ' ')
    MEM=$(ps -p 177228 -o %mem= | tr -d ' ')
    echo "✅ ПРОЦЕСС АКТИВЕН:"
    echo "   🕐 Время работы: $UPTIME"
    echo "   🔥 CPU: $CPU%"
    echo "   💾 RAM: $MEM%"
else
    echo "❌ ПРОЦЕСС ЗАВЕРШЕН"
    exit 1
fi

echo

# GPU статус
echo "🎯 GPU СТАТУС:"
GPU_INFO=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits)
if [ $? -eq 0 ]; then
    echo "   $GPU_INFO" | while IFS=', ' read -r temp power util mem; do
        echo "   🌡️  Температура: ${temp}°C"
        echo "   ⚡ Мощность: ${power}W" 
        echo "   🎯 Загрузка: ${util}%"
        echo "   💾 VRAM: ${mem}MB"
    done
else
    echo "   ❌ Ошибка получения данных GPU"
fi

echo
echo "📊 Из прикрепленных логов видно:"
echo "   • Итерация: ~90+"
echo "   • Временных шагов: ~184,320+"
echo "   • FPS: ~175-178"
echo "   • Обучение активно продолжается!"

echo
echo "=================================================="
echo "🔄 Обновление каждые 5 сек... (Ctrl+C для выхода)"
