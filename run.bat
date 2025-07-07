@echo off
echo 🚀 Запуск продвинутого торгового алгоритма с LSTM...
echo.

REM Проверка активации виртуального окружения
if exist "venv\Scripts\activate.bat" (
    echo 📁 Активация виртуального окружения...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  Виртуальное окружение не найдено, используем системный Python
)

echo.
echo 🧪 Быстрая проверка системы...
python test_advanced.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Тест пройден! Запускаем основное обучение...
    echo.
    python main.py
) else (
    echo.
    echo ❌ Тест не пройден. Проверьте установку зависимостей:
    echo pip install -r requirements.txt
    echo.
    pause
)

echo.
echo 🎉 Выполнение завершено!
pause 