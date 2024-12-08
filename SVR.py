import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas_ta as ta

# 讀取資料
# file_name = "./china_steel"
file_name = "./taiwan_mobile"
name = ["_stock_data_cleaned", "_wavelet_reconstructed_only"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for name_idx in range(len(name)):
    file_path = file_name + name[name_idx] + ".csv"
    data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    data.columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]

    # 計算技術指標
    data['MA_5'] = data['Close'].rolling(window=5).mean()  # 5日移動平均
    data['MA_10'] = data['Close'].rolling(window=10).mean()  # 10日移動平均
    data['RSI'] = ta.rsi(data['Close'], length=14)  # 14日RSI
    data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()  # 成交量移動平均
    data['Price_Range'] = data['High'] - data['Low']  # 日內價格波動幅度
    data['Daily_Return'] = ((data['Close'] - data['Open']) / data['Open']) * 100  # 日內價格變化率
    data['Volume_Change'] = data['Volume'].pct_change() * 100  # 交易量變化率

    # 移除空值
    data = data.dropna()
    # 僅保留數值列
    data = data.select_dtypes(include=[np.number])  

    # 選擇特徵與目標
    features = ['Close', 'MA_5', 'MA_10', 'RSI', 'Volume_MA_5', 'Price_Range', 'Daily_Return', 'Volume_Change']
    X = data[features].values
    y = data['Close'].values

    # 資料標準化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 分割資料集 (80% training, 10% validation, 10% testing)
    train_size = int(len(X_scaled) * 0.8)
    valid_size = int(len(X_scaled) * 0.9)

    X_train, y_train = X_scaled[:train_size], y[:train_size]
    X_valid, y_valid = X_scaled[train_size:valid_size], y[train_size:valid_size]
    X_test, y_test = X_scaled[valid_size:], y[valid_size:]

    # 訓練SVR模型
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model.fit(X_train, y_train)

    # 驗證模型
    y_valid_pred = model.predict(X_valid)
    valid_mse = mean_squared_error(y_valid, y_valid_pred)
    print(f"Validation MSE: {valid_mse:.4f}")

    # 測試模型
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Test MSE: {test_mse:.4f}")

    # 預測未來一周
    future_data = X_scaled[-1:].reshape(1, -1)  # 使用最後一筆資料作為起點
    future_predictions = []
    for _ in range(5):  # 預測未來5天
        next_pred = model.predict(future_data)
        future_predictions.append(next_pred[0])

        # 模擬將當前預測加入資料
        # 未來一天數據需要適配您的情境，這裡假設平移技術指標
        next_row = future_data[0].copy()
        future_data = np.roll(future_data, -1, axis=1)  # 滾動特徵
        future_data[0, -1] = next_row[-1]

    # 還原未來預測值
    future_predictions = np.array(future_predictions)

    # 未來日期
    future_dates = pd.date_range(start=data.index[-1], periods=6, freq='B')[1:]  # 未來一周的交易日
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})
    print("未來一周的預測結果：")
    print(future_df)
    print()

    # 繪圖
    axes[name_idx].plot(data.index[-len(y_test):], y_test, label='True Close')
    axes[name_idx].plot(data.index[-len(y_test):], y_test_pred, label='Predicted Close')
    axes[name_idx].set_title(f"Actual vs Predicted Close Prices {'(original)' if name_idx==0 else '(wavelet denoising)'}")
    axes[name_idx].set_xlabel("Date")
    axes[name_idx].set_ylabel("Close Price")    
    axes[name_idx].legend()
    axes[name_idx].tick_params(axis='x', rotation=45)  # 調整 x 軸標籤旋轉角度
    
plt.tight_layout()
plt.show()
