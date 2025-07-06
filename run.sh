#!/bin/bash

echo "Активация виртуального окружения..."
source venv/bin/activate

echo "Запуск торгового алгоритма..."
python main.py

echo ""
echo "Для выхода из виртуального окружения наберите: deactivate"
bash 