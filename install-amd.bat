@echo off
echo ========================================
echo Установка AMD GPU версии торгового алгоритма
echo ========================================

echo Проверка DirectML поддержки...
python -c "import platform; print('Windows версия:', platform.version())" 2>nul
if errorlevel 1 (
    echo ❌ Python не найден!
    pause
    exit /b 1
)

echo.
echo Создание виртуального окружения...
python -m venv venv
if errorlevel 1 (
    echo ❌ Ошибка создания виртуального окружения
    pause
    exit /b 1
)

echo Активация виртуального окружения...
call venv\Scripts\activate.bat

echo Обновление pip...
python -m pip install --upgrade pip

echo Установка зависимостей для AMD GPU...
pip install -r requirements-amd.txt
if errorlevel 1 (
    echo ⚠️ Ошибка установки AMD версии, переключаемся на CPU...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Ошибка установки!
        pause
        exit /b 1
    )
    echo ✅ CPU версия установлена!
) else (
    echo ✅ AMD GPU версия успешно установлена!
)

echo.
echo Тестирование DirectML...
python -c "import torch_directml; print('DirectML устройство:', torch_directml.device())" 2>nul
if errorlevel 1 (
    echo ⚠️ DirectML не доступен, будет использован CPU
) else (
    echo ✅ DirectML готов к работе!
)

echo.
echo Для запуска используйте: python sentiment_trading_v60.py
echo.
echo Установка завершена!
pause 