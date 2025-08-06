import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
import datetime
import warnings
import talib
from config import Config, is_market_open
from indicators import calculate_indicators

def get_stock_data(ticker, period="3y", interval="1d"):
    """獲取股票數據，使用yfinance，並進行緩存與指標計算。"""
    cache_file = f"{ticker}_data.pickle"
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            last_date = data.index[-1].date()
            today = datetime.date.today()
            if today > last_date:
                new_data = yf.Ticker(ticker).history(start=last_date + datetime.timedelta(days=1), interval=interval)
                if not new_data.empty:
                    data = pd.concat([data, new_data])
                    data = calculate_indicators(data)  # 已包含預處理
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                else:
                    warnings.warn(f"No new data for {ticker}, using cached.")
        else:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            if data.empty or len(data) < Config.MIN_HISTORY_DAYS:
                warnings.warn(f"Insufficient data for {ticker} ({len(data)} days)")
                return None
            data = calculate_indicators(data)  # 已包含預處理
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        
        # 原有數據驗證 (NaN比例檢查移至預處理後，無需改)
        nan_ratio = data.isnull().mean().mean()
        if nan_ratio > 0.2:
            warnings.warn(f"High NaN ratio ({nan_ratio:.2f}) in data for {ticker}, data may be unreliable.")
            return None
        
        return data
    except Exception as e:
        warnings.warn(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_market_condition():
    """判斷市場條件，使用S&P 500。
    輸出: market_condition (str), market_params (dict)。
    """
    try:
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(period="1y", interval="1d")
        if len(data) < 200:
            return "Unknown", {}
        close = data['Close'].values.astype('float64')
        sma_200 = talib.SMA(close, timeperiod=200)
        rsi_14 = talib.RSI(close, timeperiod=14)
        index = -1 if is_market_open() else -2
        if np.isnan(sma_200[index]) or np.isnan(rsi_14[index]):
            return "Unknown", {}
        if close[index] > sma_200[index] and rsi_14[index] > 50:
            return "Bullish", {"rsi_high": 70, "rsi_low": 40, "macd_relaxed": True, "cci_high": 150, "stoch_high": 85, "crash_rsi": 75}
        elif close[index] < sma_200[index] and rsi_14[index] < 50:
            return "Bearish", {"rsi_high": 60, "rsi_low": 40, "macd_relaxed": False, "cci_high": 100, "stoch_high": 80, "crash_rsi": 70}
        return "Neutral", {"rsi_high": 65, "rsi_low": 40, "macd_relaxed": False, "cci_high": 100, "stoch_high": 80, "crash_rsi": 70}
    except Exception as e:
        warnings.warn(f"Error determining market condition: {str(e)}")
        return "Unknown", {}