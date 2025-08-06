import pandas as pd
import numpy as np
import talib
from arch import arch_model
from config import Config
import warnings
from sklearn.impute import KNNImputer  # 新增 import
from sklearn.preprocessing import MinMaxScaler  # 已存在，但確保 import
from sklearn.decomposition import PCA  # 新增 import

def calculate_indicators(data):
    """Calculate all technical indicators"""
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Missing required data columns")
    
    data['SMA5'] = talib.SMA(data['Close'], timeperiod=Config.SMA_SHORT)
    data['SMA20'] = talib.SMA(data['Close'], timeperiod=Config.SMA_LONG)
    data['SMA50'] = talib.SMA(data['Close'], timeperiod=Config.SMA_LONG2)
    data['UpperBB'], data['MiddleBB'], data['LowerBB'] = talib.BBANDS(
        data['Close'], timeperiod=Config.BOLLINGER_PERIOD, 
        nbdevup=Config.BOLLINGER_STD, nbdevdn=Config.BOLLINGER_STD
    )
    data['MACD'], data['MACDSignal'], _ = talib.MACD(
        data['Close'], fastperiod=Config.MACD_FAST, slowperiod=Config.MACD_SLOW, signalperiod=Config.MACD_SIGNAL
    )
    data['OBV'] = talib.OBV(data['Close'], data['Volume'])
    data['OBV_SMA'] = talib.SMA(data['OBV'], timeperiod=Config.OBV_SMA_PERIOD)
    data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=Config.ADX_PERIOD)
    data['PLUS_DI'] = talib.PLUS_DI(data['High'], data['Low'], data['Close'], timeperiod=Config.ADX_PERIOD)
    data['MINUS_DI'] = talib.MINUS_DI(data['High'], data['Low'], data['Close'], timeperiod=Config.ADX_PERIOD)
    data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=Config.CCI_PERIOD)
    data['SlowK'], data['SlowD'] = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowd_period=3)
    data['MFI'] = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=Config.MFI_PERIOD)
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=Config.ATR_PERIOD)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['Volatility20'] = data['Close'].pct_change().rolling(window=Config.VOLATILITY_PERIOD).std() * np.sqrt(252)
    data['Volatility60'] = data['Close'].pct_change().rolling(window=Config.VOLATILITY_LONG_PERIOD).std() * np.sqrt(252)
    
    # GARCH Volatility with robust handling
    try:
        returns = data['Close'].pct_change().dropna() * 100
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=True)
        garch_fit = model.fit(disp='off')
        data['GARCH_Volatility'] = garch_fit.conditional_volatility.shift(-len(returns) + len(data))
    except Exception as e:
        warnings.warn(f"GARCH calculation failed: {str(e)}. Falling back to Volatility20.")
        data['GARCH_Volatility'] = data['Volatility20']  # Fallback to simple volatility
    
    # Chaikin Money Flow
    data['CMF'] = talib.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=3, slowperiod=10) / data['Volume'].rolling(Config.CMF_PERIOD).sum()
    
    # 修正：計算缺失的特徵
    data['Lag_Close_1'] = data['Close'].shift(1)
    data['Rolling_Mean_5'] = data['Close'].rolling(window=5).mean()
    
    # 新增：加強數據預處理
    # 步驟1: KNN插值 (取代原 interpolate)
    try:
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        nan_ratio = data[numerical_cols].isnull().mean().mean()
        if nan_ratio < 0.3:  # 條件：NaN比例<30%
            imputer = KNNImputer(n_neighbors=5)
            data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
        else:
            warnings.warn("High NaN ratio, falling back to linear interpolation")
            data = data.interpolate(method='linear').fillna(0)
    except Exception as e:
        warnings.warn(f"KNN imputation failed: {str(e)}. Falling back to original.")
        data = data.interpolate(method='linear').fillna(0)
    
    # 步驟2: 特徵標準化
    try:
        scaler = MinMaxScaler()
        features_to_scale = ['SMA5', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACDSignal', 'OBV', 'OBV_SMA',
                             'ADX', 'PLUS_DI', 'MINUS_DI', 'CCI', 'SlowK', 'SlowD', 'MFI', 'ATR',
                             'Volatility20', 'Volatility60', 'GARCH_Volatility', 'CMF']
        valid_features = [f for f in features_to_scale if f in data.columns and data[f].notna().sum() > len(data) * 0.8]
        if valid_features:
            data[valid_features] = scaler.fit_transform(data[valid_features])
    except Exception as e:
        warnings.warn(f"Feature scaling failed: {str(e)}. Skipping.")
    
    # 步驟3: PCA降維
    try:
        features_for_pca = ['SMA5', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACDSignal', 'OBV', 'OBV_SMA',
                            'ADX', 'PLUS_DI', 'MINUS_DI', 'CCI', 'SlowK', 'SlowD', 'MFI', 'ATR',
                            'Volatility20', 'Volatility60', 'GARCH_Volatility', 'CMF']
        if len(data) > 100 and len(features_for_pca) > 8:  # 條件：數據充足
            pca_features = data[features_for_pca].dropna(axis=1, how='all')
            if pca_features.shape[1] > 1:
                pca = PCA(n_components=0.95)
                pca_data = pca.fit_transform(pca_features.fillna(0))
                explained_var = pca.explained_variance_ratio_.sum()
                if explained_var > 0.9:  # 條件：變異>90%
                    for i in range(pca_data.shape[1]):
                        data[f'PCA{i+1}'] = pca_data[:, i]
                else:
                    warnings.warn("PCA explained variance <0.9, skipping PCA.")
    except Exception as e:
        warnings.warn(f"PCA failed: {str(e)}. Skipping.")
    
    return data

def calculate_target_stop_loss(data):
    """Calculate Target and Stop Loss using Money Flow Profile and ATR"""
    lookback_days = 200
    num_rows = 50
    if len(data) < lookback_days:
        return 0.0, 0.0  # 數據不足，返回預設值
    
    high = data['High'].values[-lookback_days:].astype('float64')
    low = data['Low'].values[-lookback_days:].astype('float64')
    close = data['Close'].values[-lookback_days:].astype('float64')
    volume = data['Volume'].values[-lookback_days:].astype('float64')
    open_price = data['Open'].values[-lookback_days:].astype('float64')
    atr = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, 14)[-1]
    
    price_range = max(high) - min(low)
    price_step = price_range / num_rows
    price_levels = [min(low) + i * price_step for i in range(num_rows)]
    
    flow_profile = np.zeros(num_rows)
    for i in range(len(high)):
        for j, level in enumerate(price_levels):
            if high[i] >= level and low[i] < level + price_step:
                vpor = min(1.0, max(0.0, (min(high[i], level + price_step) - max(low[i], level)) / (high[i] - low[i])))
                flow_profile[j] += volume[i] * vpor
    
    total_flow = np.sum(flow_profile)
    high_threshold = 0.53 * total_flow
    low_threshold = 0.37 * total_flow
    high_nodes = [i for i, v in enumerate(flow_profile) if v >= high_threshold]
    low_nodes = [i for i, v in enumerate(flow_profile) if v <= low_threshold]
    poc_idx = np.argmax(flow_profile)
    poc_price = price_levels[poc_idx] + price_step / 2
    
    target = price_levels[max(high_nodes)] + price_step / 2 if high_nodes else poc_price
    stop_loss = price_levels[min(low_nodes)] + price_step / 2 if low_nodes else close[-1] - atr * 1.5
    
    return target, stop_loss