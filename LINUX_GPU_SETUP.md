# 🐧 Настройка GPU для торговой системы на Linux

## 📋 Поддерживаемые GPU

### NVIDIA GPU (рекомендуется)
- **GTX 1060 6GB** и новее 
- **RTX серия** (20xx, 30xx, 40xx)
- **Профессиональные карты**: Quadro, Tesla

### AMD GPU  
- **RX 500 серия** и новее (RX 580, RX 6000, RX 7000)
- **Профессиональные карты**: Radeon Pro

## 🚀 Быстрая установка

### Автоматическая установка (рекомендуется)
```bash
chmod +x install-gpu-linux.sh
./install-gpu-linux.sh
```

### Ручная установка

#### 1. NVIDIA GPU
```bash
# Проверка драйверов
nvidia-smi

# Если драйверы не установлены:
sudo apt update
sudo apt install nvidia-driver-535  # или новее

# Создание окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements-gpu-linux.txt
```

#### 2. AMD GPU (ROCm)
```bash
# Установка ROCm (Ubuntu 22.04)
wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
sudo amdgpu-install --usecase=rocm

# Проверка
rocm-smi

# Создание окружения
python3 -m venv venv
source venv/bin/activate

# Редактирование requirements-gpu-linux.txt
# Раскомментируйте строки ROCm и закомментируйте CUDA
pip install -r requirements-gpu-linux.txt
```

## ⚡ Производительность

### Ожидаемое ускорение обучения:
- **Без GPU**: 3-6 часов (1M timesteps)
- **NVIDIA GTX 1660**: 45-60 минут  
- **NVIDIA RTX 3070**: 25-35 минут
- **NVIDIA RTX 4090**: 15-20 минут
- **AMD RX 6800 XT**: 35-50 минут

### Рекомендации по памяти:
- **Минимум**: 6 GB VRAM
- **Комфортно**: 8+ GB VRAM
- **Для больших моделей**: 12+ GB VRAM

## 🔧 Оптимизация настроек

### В файле `sentiment_trading_v63.py`:

```python
class TrendTraderConfig:
    # Быстрое тестирование
    TOTAL_TIMESTEPS = 100000  # ~5-10 минут
    
    # Средняя производительность
    TOTAL_TIMESTEPS = 500000  # ~25-45 минут
    
    # Полное обучение
    TOTAL_TIMESTEPS = 1000000  # ~45-90 минут
```

### Уменьшение использования памяти:
```python
# В main() функции измените:
n_steps=1024,      # вместо 2048
batch_size=32,     # вместо 64
```

## 📊 Мониторинг GPU

### NVIDIA:
```bash
# Текущее состояние
nvidia-smi

# Постоянный мониторинг
watch -n 1 nvidia-smi

# Детальная информация
nvidia-smi dmon -s pucvmet
```

### AMD:
```bash
# Текущее состояние  
rocm-smi

# Постоянный мониторинг
watch -n 1 rocm-smi

# Температура и частоты
rocm-smi -T -f
```

### Системный мониторинг:
```bash
# Установка htop и gpustat
pip install gpustat
sudo apt install htop

# Мониторинг
gpustat -i 1
htop
```

## 🚨 Устранение проблем

### NVIDIA: CUDA недоступен
```bash
# Проверка версии драйвера
nvidia-smi

# Проверка CUDA
nvcc --version

# Переустановка PyTorch с правильной версией CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### AMD: ROCm недоступен
```bash
# Проверка поддержки карты
rocminfo

# Проверка переменных окружения
echo $HIP_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=0

# Переустановка PyTorch с ROCm
pip uninstall torch torchvision  
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
```

### Ошибки памяти
```python
# Уменьшите batch_size в коде:
batch_size=16,  # или даже 8

# Или уменьшите размер модели:
features_dim=128,  # вместо 256
net_arch=dict(pi=[128, 64], vf=[128, 64])
```

### Медленная работа
```bash
# Проверьте загрузку GPU
nvidia-smi  # или rocm-smi

# Если GPU не используется, проверьте устройство в коде:
python3 -c "
import torch
print('CUDA доступен:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Устройство:', torch.cuda.get_device_name(0))
"
```

## ✅ Проверка корректной работы

Запустите обучение и убедитесь, что видите:

### NVIDIA:
```
🚀 Используется NVIDIA CUDA: GeForce RTX 3070
   Устройство: cuda:0
   CUDA версия: 11.8
📊 GPU память: 2.1GB использовано / 8.0GB всего
```

### AMD:
```
🚀 Используется AMD ROCm/HIP
   Устройство: cuda:0
```

### CPU (если GPU недоступен):
```
💻 Используется CPU: cpu
💡 Для GPU ускорения установите:
   NVIDIA: драйверы + CUDA toolkit
   AMD: ROCm (https://rocmdocs.amd.com/)
```

## 🎯 Запуск

```bash
# Активация окружения
source venv/bin/activate

# Запуск обучения v63
python sentiment_trading_v63.py

# Мониторинг в отдельном терминале
watch -n 1 nvidia-smi  # или rocm-smi
```

## 🔗 Полезные ссылки

- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [AMD ROCm Installation Guide](https://rocmdocs.amd.com/en/latest/deploy/linux/quick_start.html)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [Stable-Baselines3 GPU Guide](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) 