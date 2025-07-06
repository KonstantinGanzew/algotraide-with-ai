@echo off
echo ========================================
echo Установка GPU версии торгового алгоритма
echo ========================================

echo Проверка NVIDIA GPU...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ NVIDIA GPU или драйверы не найдены!
    echo Устанавливаю CPU версию...
    goto cpu_install
)

echo ✅ NVIDIA GPU обнаружен!
echo.

echo Создание виртуального окружения...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Ошибка создания виртуального окружения
    pause
    exit /b 1
)

echo Активация виртуального окружения...
call venv\Scripts\activate

echo Обновление pip...
python -m pip install --upgrade pip

echo Установка GPU зависимостей...
pip install -r requirements-gpu.txt
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️ Ошибка установки GPU версии, переключаемся на CPU...
    goto cpu_install
)

echo ✅ GPU версия успешно установлена!
echo Для запуска используйте: run.bat
goto end

:cpu_install
pip install -r requirements.txt
echo ✅ CPU версия установлена!

:end
echo.
echo Установка завершена!
pause 