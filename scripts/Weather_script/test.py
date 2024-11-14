import subprocess

# Chạy file .sh bằng subprocess.Popen()
process = subprocess.Popen(['bash', 'X:\ATMEL\pythonProject\Autoformer\scripts\Weather_script\Autoformer.sh'])

# In ra thông báo tiến trình vẫn đang chạy
print("Script đang chạy...")

# Giữ tiến trình cho đến khi nó kết thúc
process.wait()

# Sau khi tiến trình kết thúc, cửa sổ console vẫn còn mở
print("Script đã hoàn thành nhưng cửa sổ vẫn mở.")
