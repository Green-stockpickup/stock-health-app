import pandas as pd
import numpy as np
import pandas_ta as ta
from arch import arch_model
from config import Config
import warnings  # 已加入

def calculate_indicators(data):
    """Calculate all technical indicators using pandas-ta"""
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Missing required data columns")
    
    data = data.copy()  # 避免修改原始數據
    
    data['SMA5'] = ta.sma(data['Close'], length=Config.SMA_SHORT)
    data['SMA20'] = ta.sma(data['Close'], length=Config.SMA_LONG)
    data['SMA50'] = ta.sma(data['Close'], length=Config.SMA_LONG2)
    
    bbands = ta.bbands(data['Close'], length=Config.BOLLINGER_PERIOD, std=Config.BOLLINGER_STD)
    data['UpperBB'] = bbands[f'BBU_{Config.BOLLINGER_PERIOD}_{Config.BOLLINGER_STD}']
    data['MiddleBB'] = bbands[f'BBM_{Config.BOLLINGER_PERIOD}_{Config.BOLLINGER_STD}']
    data['LowerBB'] = bbands[f'BBL_{Config.BOLLINGER_PERIOD}_{Config.BOLLINGER_STD}']
    
    macd = ta.macd(data['Close'], fast=Config.MACD_FAST, slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL)
    data['MACD'] = macd[f'MACD_{Config.MACD_FAST}_{Config.MACD_SLOW}_{Config.MACD_SIGNAL}']
    data['MACDSignal'] = macd[f'MACDs_{Config.MACD_FAST}_{Config.MACD_SLOW}_{Config.MACD_SIGNAL}']
    
    data['OBV'] = ta.obv(data['Close'], data['Volume'])
    data['OBV_SMA'] = ta.sma(data['OBV'], length=Config.OBV_SMA_PERIOD)
    
    adx = ta.adx(data['High'], data['Low'], data['Close'], length=Config.ADX_PERIOD)
    data['ADX'] = adx[f'ADX_{Config.ADX_PERIOD}']
    data['PLUS_DI'] = adx[f'DMP_{Config.ADX_PERIOD}']
    data['MINUS_DI'] = adx[f'DMN_{Config.ADX_PERIOD}']
    
    data['CCI'] = ta.cci(data['High'], data['Low'], data['Close'], length=Config.CCI_PERIOD)
    
    stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=Config.STOCH_SLOWK_PERIOD, d=Config.STOCH_SLOWD_PERIOD)
    data['SlowK'] = stoch[f'STOCHk_{Config.STOCH_SLOWK_PERIOD}_{Config.STOCH_SLOWD_PERIOD}_3']
    data['SlowD'] = stoch[f'STOCHd_{Config.STOCH_SLOWK_PERIOD}_{Config.STOCH_SLOWD_PERIOD}_3']
    
    data['MFI'] = ta.mfi(data['High'], data['Low'], data['Close'], data['Volume'], length=Config.MFI_PERIOD)
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=Config.ATR_PERIOD)
    data['RSI'] = ta.rsi(data['Close'], length=14)
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
        data['GARCH_Volatility'] = data['Volatility20']
    
    # Chaikin Money Flow
    data['CMF'] = ta.adosc(data['High'], data['Low'], data['Close'], data['Volume'], fast=3, slow=10) / data['Volume'].rolling(Config.CMF_PERIOD).sum()
    
    # 修正：計算缺失的特徵
    data['Lag_Close_1'] = data['Close'].shift(1)
    data['Rolling_Mean_5'] = data['Close'].rolling(window=5).mean()
    
    # 修正：填補NaN值 - 改用 forward fill，避免 0 扭曲 SMA
    data = data.ffill().bfill()  # ffill 前向填補，bfill 後向填補，符合現實 TA 數據處理
    if data.isnull().mean().mean() > 0:
        warnings.warn("Some NaNs remain after ffill, data may be incomplete.")
    
    return data

def calculate_target_stop_loss(data):
    """Calculate Target and Stop Loss using Money Flow Profile and ATR"""
    lookback_days = 200
    num_rows = 50
    if len(data) < lookback_days:
        return 0.0, 0.0  # 數據不足，返回預設值
    
    # 修改：使用 Series 輸入確保 pandas_ta.atr 正常計算（現實中 ATR 需序列數據處理 NaN）
    high_series = data['High'].tail(lookback_days)
    low_series = data['Low'].tail(lookback_days)
    close_series = data['Close'].tail(lookback_days)
    volume_series = data['Volume'].tail(lookback_days)
    open_series = data['Open'].tail(lookback_days)
    
    atr_series = ta.atr(high_series, low_series, close_series, length=14)
    if atr_series is None or atr_series.empty:
        warnings.warn("ATR calculation returned None or empty, falling back to default.")
        atr = 0.0  # 預設值，避免 NoneType 錯誤
    else:
        atr = atr_series.iloc[-1]  # 取最後值，符合現實 ATR 計算
    
    # 剩餘邏輯不變
    high = high_series.values.astype('float64')
    low = low_series.values.astype('float64')
    close = close_series.values.astype('float64')
    volume = volume_series.values.astype('float64')
    open_price = open_series.values.astype('float64')
    
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