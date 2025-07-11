#!/usr/bin/env python3
import subprocess
import time
import sys

def get_gpu_temp():
    """Получить текущую температуру GPU"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return None

def get_power_limit():
    """Получить текущий лимит мощности"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.limit', '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return None

def set_power_limit(watts):
    """Установить лимит мощности"""
    try:
        subprocess.run(['sudo', 'nvidia-smi', '-pl', str(watts)], check=True)
        print(f"✅ Лимит мощности установлен: {watts}W")
        return True
    except:
        print(f"❌ Ошибка установки лимита мощности: {watts}W")
        return False

def monitor_temp():
    """Основной цикл мониторинга"""
    print("🌡️  Запуск мониторинга температуры GPU...")
    print("Ctrl+C для остановки")
    
    current_power_limit = 200
    
    while True:
        try:
            temp = get_gpu_temp()
            if temp is None:
                print("❌ Не удалось получить температуру GPU")
                time.sleep(10)
                continue
            
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] 🌡️  Температура: {temp}°C", end="")
            
            # Проверка температуры и автоматическая коррекция
            if temp >= 89:  # Критическая температура
                new_limit = max(180, current_power_limit - 10)
                print(f" 🚨 КРИТИЧНО! Снижаю мощность до {new_limit}W")
                if set_power_limit(new_limit):
                    current_power_limit = new_limit
            elif temp >= 87:  # Высокая температура
                new_limit = max(180, current_power_limit - 5)
                print(f" ⚠️  ВЫСОКАЯ! Снижаю мощность до {new_limit}W")
                if set_power_limit(new_limit):
                    current_power_limit = new_limit
            elif temp <= 82 and current_power_limit < 200:  # Температура нормализовалась
                new_limit = min(200, current_power_limit + 10)
                print(f" ✅ НОРМА! Повышаю мощность до {new_limit}W")
                if set_power_limit(new_limit):
                    current_power_limit = new_limit
            else:
                print(" ✅ ОК")
            
            time.sleep(15)  # Проверка каждые 15 секунд
            
        except KeyboardInterrupt:
            print("\n👋 Мониторинг остановлен")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_temp() 