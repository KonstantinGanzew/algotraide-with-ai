#!/usr/bin/env python3
import subprocess
import time
import os

def monitor_process():
    pid = 177228
    print("🚀 МОНИТОРИНГ ОБУЧЕНИЯ RL АГЕНТА")
    print("=" * 50)
    
    while True:
        try:
            # Проверка что процесс еще жив
            if not os.path.exists(f'/proc/{pid}'):
                print("❌ Процесс завершен!")
                break
                
            # Статистика процесса
            with open(f'/proc/{pid}/stat', 'r') as f:
                stat_data = f.read().split()
                
            # Время работы в секундах
            uptime_ticks = int(stat_data[21])
            uptime_seconds = uptime_ticks // 100  # предполагая 100 Hz
            hours = uptime_seconds // 3600
            minutes = (uptime_seconds % 3600) // 60
            secs = uptime_seconds % 60
            
            # GPU статус
            gpu_result = subprocess.run([
                'nvidia-smi', '--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if gpu_result.returncode == 0:
                gpu_temp, gpu_power, gpu_util, gpu_mem = gpu_result.stdout.strip().split(', ')
            else:
                gpu_temp = gpu_power = gpu_util = gpu_mem = "N/A"
            
            # Память процесса
            mem_result = subprocess.run([
                'ps', '-p', str(pid), '-o', 'rss=', '--no-headers'
            ], capture_output=True, text=True)
            
            if mem_result.returncode == 0:
                memory_kb = int(mem_result.stdout.strip())
                memory_mb = memory_kb // 1024
            else:
                memory_mb = 0
                
            print(f"\r🕒 Время: {hours:02d}:{minutes:02d}:{secs:02d} | "
                  f"🌡️ {gpu_temp}°C | ⚡ {gpu_power}W | 🎯 {gpu_util}% | "
                  f"💾 {memory_mb}MB RAM | 🖥️ {gpu_mem}MB VRAM", end="", flush=True)
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n👋 Мониторинг остановлен")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_process()
