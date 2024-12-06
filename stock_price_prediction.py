import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import joblib

def train_model():
    stock_data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
    
    stock_data['Next_Close'] = stock_data['Close'].shift(-1)
    
    stock_data = stock_data.dropna()

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = stock_data[features]
    y = stock_data['Next_Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'stock_model.h5')

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    plt.figure(figsize=(14, 7))
    plt.plot(y_test.values, label='Actual Prices', marker='o')
    plt.plot(y_pred, label='Predicted Prices', linestyle='--')
    plt.legend()
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Test Data Points')
    plt.ylabel('Stock Price')
    plt.show()

def stock_price(ticker):
    model = joblib.load('stock_model.h5')

    stock_data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    latest_data = stock_data[features].iloc[-1].values.reshape(1, -1)

    next_day_prediction = model.predict(latest_data)
    return next_day_prediction[0]

if __name__ == "__main__":
    train_model()

    next_price = stock_price('TATAMOTORS.NS')
    print(f"Next day's predicted price for Tata Motors: {next_price:.2f}")
