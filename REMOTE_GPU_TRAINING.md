# 🚀 Удаленное GPU Обучение - Инструкция

Настройка и запуск обучения на удаленном сервере с NVIDIA GPU.

**Сервер:** `192.168.88.218` (NVIDIA GPU + CUDA 12.8)

## 📋 Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Подробная настройка](#подробная-настройка)
3. [Запуск обучения](#запуск-обучения)
4. [Мониторинг](#мониторинг)
5. [Решение проблем](#решение-проблем)

---

## 🚀 Быстрый старт

### 1. Автоматическая настройка с помощью Python

```bash
# Запуск менеджера удаленного обучения
python remote_training_setup.py
```

**Важно:** Отредактируйте в `remote_training_setup.py`:
- `username` - ваш логин на сервере
- `remote_path` - путь к рабочей директории на сервере
- `ssh_key_path` - путь к SSH ключу (если используется)

### 2. Автоматическая настройка через SSH

```bash
# Копируем и запускаем установочный скрипт на сервере
scp setup_remote_gpu.sh user@192.168.88.218:~/
ssh user@192.168.88.218 'bash ~/setup_remote_gpu.sh'
```

---

## 🔧 Подробная настройка

### Шаг 1: Подготовка файлов

Убедитесь, что у вас есть все необходимые файлы:

```
algotraide-with-ai/
├── remote_training_setup.py      # Менеджер удаленного обучения
├── sentiment_trading_v69_remote.py  # Оптимизированный скрипт для GPU
├── setup_remote_gpu.sh           # Автоматический установщик
├── requirements-gpu.txt          # Зависимости для GPU
└── data/                         # Данные для обучения
    ├── BTCUSDT_5m_2y.csv
    ├── BTCUSDT_1h_2y.csv
    ├── BTCUSDT_4h_2y.csv
    └── BTCUSDT_1d_2y.csv
```

### Шаг 2: Настройка SSH подключения

```bash
# Проверка подключения
ssh user@192.168.88.218

# Настройка SSH ключей (рекомендуется)
ssh-copy-id user@192.168.88.218
```

### Шаг 3: Ручная настройка на сервере

Если автоматическая настройка не сработала:

```bash
# Подключение к серверу
ssh user@192.168.88.218

# Создание рабочей директории
mkdir -p ~/gpu_training/{data,logs,models,results}
cd ~/gpu_training

# Создание виртуальной среды
python3 -m venv venv
source venv/bin/activate

# Установка PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Установка остальных зависимостей
pip install stable-baselines3 gymnasium numpy pandas matplotlib
pip install ccxt scikit-learn tqdm psutil seaborn
```

---

## 🎓 Запуск обучения

### Метод 1: Через менеджер Python

```python
from remote_training_setup import RemoteTrainingManager

# Создание менеджера
manager = RemoteTrainingManager(
    server_ip='192.168.88.218',
    username='ваш_username',
    remote_path='/home/ваш_username/gpu_training'
)

# Полная настройка и запуск
manager.check_connection()
manager.setup_remote_environment()
manager.sync_files()
manager.install_dependencies()
manager.start_training('sentiment_trading_v69_remote.py')
```

### Метод 2: Ручной запуск

```bash
# Копирование файлов
scp sentiment_trading_v69_remote.py user@192.168.88.218:~/gpu_training/
scp data/*.csv user@192.168.88.218:~/gpu_training/data/

# Подключение и запуск
ssh user@192.168.88.218
cd ~/gpu_training
source venv/bin/activate

# Запуск в фоне с логированием
nohup python sentiment_trading_v69_remote.py > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Метод 3: Интерактивный запуск

```bash
ssh user@192.168.88.218
cd ~/gpu_training
source venv/bin/activate
python sentiment_trading_v69_remote.py
```

---

## 📊 Мониторинг

### 1. Мониторинг обучения

```bash
# Просмотр логов в реальном времени
ssh user@192.168.88.218 'tail -f ~/gpu_training/logs/training_*.log'

# Проверка запущенных процессов
ssh user@192.168.88.218 'pgrep -f python'
```

### 2. Мониторинг GPU

```bash
# Статус GPU
ssh user@192.168.88.218 'nvidia-smi'

# Непрерывный мониторинг
ssh user@192.168.88.218 'watch -n 2 nvidia-smi'

# Использование скрипта мониторинга
ssh user@192.168.88.218 'cd ~/gpu_training && python monitor_gpu.py'
```

### 3. Python мониторинг

```python
manager = RemoteTrainingManager(...)

# Статус обучения
status = manager.get_training_status()
print(status)

# Мониторинг в реальном времени
manager.monitor_training()  # Ctrl+C для выхода
```

---

## 📈 Получение результатов

### Автоматическое скачивание

```python
# Через менеджер
manager.download_results()  # Скачивает модели, логи, графики
```

### Ручное скачивание

```bash
# Скачивание модели
scp user@192.168.88.218:~/gpu_training/model_gpu_*.zip ./

# Скачивание результатов
scp user@192.168.88.218:~/gpu_training/results_*.png ./
scp user@192.168.88.218:~/gpu_training/stats_*.txt ./

# Скачивание логов
scp user@192.168.88.218:~/gpu_training/logs/training_*.log ./
```

---

## 🔧 Оптимизации для удаленного обучения

### Особенности GPU версии:

1. **Автоматический выбор GPU** - выбирается GPU с наибольшим количеством свободной памяти
2. **Увеличенный batch size** - оптимизирован для GPU (128 вместо 64)
3. **Больше шагов обучения** - 2M timesteps для лучшего использования GPU
4. **Headless режим** - matplotlib настроен для работы без дисплея
5. **Расширенное логирование** - все результаты сохраняются в файлы

### Конфигурация для разных GPU:

```python
# Для GPU с 8GB+ памяти
BATCH_SIZE = 128
N_STEPS = 4096
TOTAL_TIMESTEPS = 2000000

# Для GPU с 4-8GB памяти  
BATCH_SIZE = 64
N_STEPS = 2048
TOTAL_TIMESTEPS = 1000000

# Для GPU с менее 4GB памяти
BATCH_SIZE = 32
N_STEPS = 1024
TOTAL_TIMESTEPS = 500000
```

---

## 🚨 Решение проблем

### Проблема: CUDA не найдена

```bash
# Проверка установки CUDA
ssh user@192.168.88.218 'nvidia-smi'
ssh user@192.168.88.218 'nvcc --version'

# Если nvcc не найден, добавьте в PATH:
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
```

### Проблема: PyTorch не видит GPU

```bash
# Тест PyTorch CUDA
ssh user@192.168.88.218 'cd ~/gpu_training && python -c "import torch; print(torch.cuda.is_available())"'

# Переустановка PyTorch
ssh user@192.168.88.218 'cd ~/gpu_training && source venv/bin/activate && pip uninstall torch torchvision torchaudio'
ssh user@192.168.88.218 'cd ~/gpu_training && source venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
```

### Проблема: Недостаточно памяти GPU

```bash
# Проверка использования памяти
ssh user@192.168.88.218 'nvidia-smi'

# Уменьшение batch size в коде
# Отредактируйте TrendTraderConfig.BATCH_SIZE
```

### Проблема: SSH подключение прерывается

```bash
# Использование screen/tmux для длительных процессов
ssh user@192.168.88.218
screen -S training
cd ~/gpu_training && source venv/bin/activate
python sentiment_trading_v69_remote.py

# Отключение: Ctrl+A, D
# Подключение: screen -r training
```

---

## 📊 Ожидаемые результаты

### Производительность на GPU:

- **Время обучения:** ~30-60 минут (vs 4-8 часов на CPU)
- **Использование GPU:** 70-90%
- **Память GPU:** 2-6GB (зависит от batch size)
- **Timesteps:** 2M за сессию

### Выходные файлы:

```
~/gpu_training/
├── model_gpu_YYYYMMDD_HHMMSS.zip    # Обученная модель
├── results_YYYYMMDD_HHMMSS.png      # График результатов
├── stats_YYYYMMDD_HHMMSS.txt        # Статистика
├── training.log                     # Основной лог
└── logs/
    └── training_YYYYMMDD_HHMMSS.log # Детальный лог
```

---

## 🎯 Команды для копирования

```bash
# === БЫСТРАЯ НАСТРОЙКА ===

# 1. Копирование файлов на сервер
scp remote_training_setup.py setup_remote_gpu.sh requirements-gpu.txt user@192.168.88.218:~/

# 2. Копирование данных
scp data/BTCUSDT_*.csv user@192.168.88.218:~/gpu_training/data/

# 3. Запуск настройки
ssh user@192.168.88.218 'bash ~/setup_remote_gpu.sh'

# 4. Копирование и запуск обучения  
scp sentiment_trading_v69_remote.py user@192.168.88.218:~/gpu_training/
ssh user@192.168.88.218 'cd ~/gpu_training && source venv/bin/activate && nohup python sentiment_trading_v69_remote.py > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &'

# 5. Мониторинг
ssh user@192.168.88.218 'tail -f ~/gpu_training/logs/training_*.log'
```

---

## 📞 Поддержка

Если возникли проблемы:

1. Проверьте подключение: `ping 192.168.88.218`
2. Проверьте SSH: `ssh user@192.168.88.218 'echo "OK"'`
3. Проверьте GPU: `ssh user@192.168.88.218 'nvidia-smi'`
4. Проверьте Python: `ssh user@192.168.88.218 'python3 --version'`

**Удачного обучения! 🚀** 