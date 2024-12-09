import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from copy import deepcopy as dc

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers,device='cuda'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df
def train_one_epoch(model, train_loader, loss_function, optimizer, device, epoch):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()

def validate_one_epoch(model, test_loader, loss_function, device):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()

def main():
    origin_data = pd.read_csv('taiwan_mobile_stock_data_cleaned.csv')
    # For wavlet datas
    # origin_data.rename(columns={'Close Reconstructed': 'Close'}, inplace=True)
    data = origin_data[['Date', 'Close']]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Transform the data with 7 days lookback (hyperparameter)
    lookback = 7
    shifted_df = prepare_dataframe_for_lstm(data, lookback)
    shifted_df_as_np = shifted_df.to_numpy()
    print(f'Shifted data shape with {lookback} days look back: {shifted_df_as_np.shape}')
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

    X = shifted_df_as_np[:, 1:]
    X = dc(np.flip(X, axis=1))
    y = shifted_df_as_np[:, 0]

    # Split the data into train and test
    # toggle the #* line as comment If want to predict future price, use all data as train data
    # split_index = int(len(X) * 0.95) #*
    # X_train = X[:split_index]
    # X_test = X[split_index:] #*

    # y_train = y[:split_index]
    # y_test = y[split_index:] #*

    # For predict future price
    X_train = X
    y_train = y
    
    X_train = X_train.reshape((-1, lookback, 1))
    # X_test = X_test.reshape((-1, lookback, 1)) #*
    y_train = y_train.reshape((-1, 1))
    # y_test = y_test.reshape((-1, 1)) #*

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    # X_test = torch.tensor(X_test).float() #*
    # y_test = torch.tensor(y_test).float() #*

    train_dataset = TimeSeriesDataset(X_train, y_train)
    # test_dataset = TimeSeriesDataset(X_test, y_test) #*
    
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #*

    model = LSTM(1,32,2).to(device)
    print(model)

    learning_rate = 0.001
    num_epochs = 50
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, loss_function, optimizer, device, epoch)
        # validate_one_epoch(model, test_loader, loss_function, device) #*
    
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()
    train_predictions = predicted.flatten()

    dummies = np.zeros((X_train.shape[0], lookback+1))
    dummies[:, 0] = train_predictions
    dummies = scaler.inverse_transform(dummies)

    train_predictions = dc(dummies[:, 0])


    dummies = np.zeros((X_train.shape[0], lookback+1))
    dummies[:, 0] = y_train.flatten()
    dummies = scaler.inverse_transform(dummies)

    new_y_train = dc(dummies[:, 0])

    plt.plot(new_y_train, label='Actual Close')
    plt.plot(train_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.savefig('result_LSTM/LSTM_train.png')

    # Get the last 7 days of data to use as input for future predictions
    last_7_days = X_train[-1:].to(device)  # Shape: [1, 7, 1]

    # Generate future dates (skip weekends)
    last_date = data['Date'].iloc[-1]  # This is Friday, 2024-12-06
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5, freq='B')
    # This will give us the next 5 business days: Mon, Tue, Wed, Thu, Fri

    # Predict next 7 business days
    future_predictions = []
    current_input = last_7_days.clone()

    with torch.no_grad():
        for _ in range(5):
            # Get prediction for next day
            next_pred = model(current_input)
            future_predictions.append(next_pred.item())
            
            # Update input sequence by removing oldest prediction and adding new one
            current_input = current_input.roll(-1, dims=1)
            current_input[0, -1, 0] = next_pred

    # Convert predictions back to original scale
    dummy_array = np.zeros((len(future_predictions), lookback+1))
    dummy_array[:, 0] = future_predictions
    future_predictions_unscaled = scaler.inverse_transform(dummy_array)[:, 0]

    # Create prediction results dataframe
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_predictions_unscaled
    })

    print("Future 5 business days predictions:")
    print(future_df)

    # Calculate and print MSE for training and test sets
    actual_future_values = [22.5, 22.5, 22.5, 22.5, 22.5]  # Actual future values for next 5 business days
    future_df['Actual_Close'] = actual_future_values

    train_mse = mean_squared_error(new_y_train, train_predictions)
    test_mse = mean_squared_error(future_df['Actual_Close'], future_df['Predicted_Close'])

    print("\nModel Performance Metrics:")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Plot historical + future predictions
    plt.figure(figsize=(9, 6))
    plt.plot(future_df['Date'], future_df['Predicted_Close'], 'r--', label='Future Predictions')
    plt.plot(future_df['Date'], future_df['Actual_Close'], label='Actual Close')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Price Prediction - Next 5 Business Days')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('result_LSTM/LSTM_future.png')

if __name__ == '__main__':
    main()
