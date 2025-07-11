# 🚀 Быстрый старт

## Для тех, кто хочет быстро запустить проект:

### 🚀 С GPU ускорением (рекомендуется):

**Windows:**
1. Выполните: `install-gpu.bat` (автоматически определит GPU)
2. Затем: `run.bat`

**macOS/Linux:**
1. Выполните: `chmod +x install-gpu.sh && ./install-gpu.sh`
2. Затем: `chmod +x run.sh && ./run.sh`

### 🐌 Только CPU (если нет GPU):

**Windows:**
1. Откройте командную строку в папке проекта
2. Выполните: `run.bat`
3. Ждите завершения обучения и результатов

**macOS/Linux:**
1. Откройте терминал в папке проекта  
2. Выполните: 
   ```bash
   chmod +x run.sh
   ./run.sh
   ```
3. Ждите завершения обучения и результатов

## Если возникли проблемы:

### 1. Проверьте установку Python:
```bash
python --version
```
Должен быть Python 3.8 или выше.

### 2. Ручная установка:
```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Активация (macOS/Linux) 
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск
python main.py
```

### 3. Деактивация окружения:
```bash
deactivate
```

## ⚡ Время обучения:
- **GPU**: 3-10 минут (зависит от видеокарты)
- **CPU**: 20-30 минут

## Настройка параметров:

Откройте `main.py` и измените значения в классе `Config`:
- `INITIAL_BALANCE` - начальный баланс (по умолчанию: 10000)
- `TOTAL_TIMESTEPS` - время обучения (по умолчанию: 50000)
- `DATA_FILE` - файл с данными (по умолчанию: "BTC_5_96w.csv")
- `FORCE_CPU` - принудительно использовать CPU (по умолчанию: False)

## Результат:

После завершения вы увидите:
- Статистику торговли
- Графики баланса и сигналов
- Процент прибыли/убытка
- Винрейт

⚠️ **Внимание**: Это образовательный проект. Реальная торговля требует дополнительного анализа и тестирования! 