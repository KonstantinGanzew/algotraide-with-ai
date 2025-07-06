@echo off
echo Активация виртуального окружения...
call venv\Scripts\activate

echo Запуск торгового алгоритма...
python main.py

echo.
echo Для выхода из виртуального окружения наберите: deactivate
cmd /k 