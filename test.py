import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import pywt

# 讀取資料集
data = pd.read_csv('china_steel_stock_data_cleaned.csv', encoding="gbk")

# 確保日期欄位名稱正確
data['Price'] = pd.to_datetime(data['Price'])  # 將日期轉換為 datetime 格式
x_values = pd.to_numeric(data['Price'])  # 將日期轉為數值
date_values = data['Price']  # 保留原始日期作為橫軸
y_values = np.array(data['Close'])

# 繪製原始數據
plt.figure(figsize=(10, 4))
plt.plot(date_values, y_values, label='Original Signal')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Close Price vs Date')
plt.legend()
plt.show()

# 快速傅里叶變換（FFT）
yy = fft(y_values)
yf = abs(yy)  # 取模
yf1 = yf / (len(x_values) / 2)
yf2 = yf1[range(int(len(x_values) / 2))]
xf = np.arange(len(y_values))
xf2 = xf[range(int(len(x_values) / 2))]

plt.figure(figsize=(10, 4))
plt.plot(xf2, yf2, 'g')
plt.title('FFT of Close Price')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

# 小波轉換與重構
w = pywt.Wavelet('db4')  # 使用 Daubechies 4 小波基
maxlev = min(4, pywt.dwt_max_level(len(data), w))  # 限制分解層數最多為 4 層
coeffs = pywt.wavedec(y_values, w, mode='constant', level=maxlev)

# 調整濾波閾值
threshold = 0.1
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 降低閾值

datarec = pywt.waverec(coeffs, w)  # 小波重構

# 修正繪圖部分，對齊 date_values 和 datarec 的長度
min_length = min(len(date_values), len(datarec))  # 找到兩者的最小長度
date_values = date_values[:min_length]  # 截取到最小長度
y_values = y_values[:min_length]  # 截取到最小長度
datarec = datarec[:min_length]  # 截取到最小長度

# 提取噪音部分
extracted_noise = y_values - datarec

# 比較原始與重構訊號
plt.figure(figsize=(10, 6))
plt.plot(date_values, y_values, label='Original Signal', alpha=0.7)
plt.plot(date_values, datarec, label='Reconstructed Signal', linestyle='--', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Original vs Reconstructed Signal')
plt.legend()
plt.show()

# 繪製經過小波轉換後的信號（橫軸為日期）
plt.figure(figsize=(10, 4))
plt.plot(date_values, datarec, label='Wavelet Transformed Signal', color='purple', alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Wavelet Transformed Signal vs Date')
plt.legend()
plt.show()

# 繪製噪音部分
plt.figure(figsize=(10, 4))
plt.plot(date_values, extracted_noise, label='Extracted Noise', color='red', alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Noise Amplitude')
plt.title('Extracted Noise from Wavelet Transform')
plt.legend()
plt.show()

# 重構訊號的 FFT
yy1 = fft(datarec)
yf_1 = abs(yy1)
yf1_1 = yf_1 / (len(x_values) / 2)
yf2_1 = yf1_1[range(int(len(x_values) / 2))]
xf_1 = np.arange(len(datarec))
xf2_1 = xf_1[range(int(len(x_values) / 2))]

plt.figure(figsize=(10, 4))
plt.plot(xf2_1, yf2_1, 'b')
plt.title('FFT of Reconstructed Signal')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

# 儲存經過小波轉換處理後的數據為 CSV 檔案
output_data = pd.DataFrame({
    'Date': date_values, 
    'Original Close Price': y_values, 
    'Wavelet Transformed Price': datarec,
    'Extracted Noise': extracted_noise
})
output_data.to_csv('china_steel_wavelet_transformed_data.csv', index=False, encoding='utf-8-sig')

print("經過小波轉換處理的數據與噪音已成功儲存為 china_steel_wavelet_transformed_data.csv")
