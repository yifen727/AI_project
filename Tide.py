import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TiDEModel
from darts.dataprocessing.transformers import Scaler
import pandas_ta as ta

# 讀取資料
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

    # 轉換為 DARTS TimeSeries
    series = TimeSeries.from_dataframe(
        data,
        value_cols='Close',
        fill_missing_dates=True,  # Fill missing dates
        freq='B'  # 'B' represents business days;
    )
    covariates = TimeSeries.from_dataframe(
        data,
        value_cols=['MA_5', 'MA_10', 'RSI', 'Volume_MA_5', 'Price_Range', 'Daily_Return', 'Volume_Change'],
        fill_missing_dates=True,
        freq='B'
    )


    # 資料標準化
    scaler = Scaler()
    series = scaler.fit_transform(series)
    covariates = scaler.fit_transform(covariates)

    # 分割資料集 (80% training, 10% validation, 10% testing)
    train_size = int(len(series) * 0.8)
    valid_size = int(len(series) * 0.9)

    train_covariates, remaining_covariates = covariates.split_before(train_size + model.input_chunk_length) # Include input_chunk_length
    val_covariates, test_covariates = remaining_covariates.split_before(valid_size - train_size) 

    # Adjust split for target series accordingly 
    train_series, remaining_series = series.split_before(train_size + model.input_chunk_length)
    val_series, test_series = remaining_series.split_before(valid_size - train_size)

    # 訓練Tide模型
    model = TiDEModel(
        input_chunk_length=10,
        output_chunk_length=3,
        dropout=0.1,
        batch_size=16,
        n_epochs=100,
        random_state=42,
    )

    model.fit(train_series, past_covariates=train_covariates, val_series=val_series, val_past_covariates=val_covariates)

    # 驗證模型
    y_valid_pred = model.predict(n=len(val_series), past_covariates=val_covariates)

    # Remove NaN values before calculating MSE
    # Convert to numpy arrays and remove NaNs
    y_valid_pred_np = y_valid_pred.values().flatten()
    val_series_np = val_series.values().flatten()

    # Find common indices without NaNs
    not_nan_indices = np.logical_and(np.isfinite(y_valid_pred_np), np.isfinite(val_series_np))

    # Check if not_nan_indices is empty
    if not_nan_indices.sum() == 0:
        print("Warning: No valid data points for validation MSE calculation. Skipping.")
        valid_mse = np.nan  # Assign NaN to valid_mse if no valid data points
    else:
        # Calculate MSE using only valid values
        valid_mse = mean_squared_error(val_series_np[not_nan_indices], y_valid_pred_np[not_nan_indices])

    print(f"Validation MSE: {valid_mse:.4f}")

    # 測試模型
    y_test_pred = model.predict(n=len(test_series), past_covariates=test_covariates)
    test_mse = mean_squared_error(test_series.values(), y_test_pred.values())
    print(f"Test MSE: {test_mse:.4f}")

    # 預測未來一周
    future_covariates = covariates[-30:]  # 使用最後30天的共變量
    future_predictions = model.predict(n=5, past_covariates=future_covariates)

    # 還原未來預測值
    future_predictions = scaler.inverse_transform(future_predictions)

    # 未來日期
    future_dates = pd.date_range(start=data.index[-1], periods=6, freq='B')[1:]  # 未來一周的交易日
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions.values().flatten()})
    print("未來一周的預測結果：")
    print(future_df)
    print()

    # 繪圖
    axes[name_idx].plot(series[-len(test_series):].time_index, test_series.values(), label='True Close')
    axes[name_idx].plot(series[-len(test_series):].time_index, y_test_pred.values(), label='Predicted Close')
    axes[name_idx].set_title(f"Actual vs Predicted Close Prices {'(original)' if name_idx==0 else '(wavelet denoising)'}")
    axes[name_idx].set_xlabel("Date")
    axes[name_idx].set_ylabel("Close Price")
    axes[name_idx].legend()
    axes[name_idx].tick_params(axis='x', rotation=45)  # 調整 x 軸標籤旋轉角度

plt.tight_layout()
plt.show()
