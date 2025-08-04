import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import warnings
import datetime
import pytz
from colorama import Fore, Style, init
import re
import unicodedata
from arch import arch_model
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit

init(autoreset=True)
warnings.filterwarnings("ignore")

# 整合 config.py
class Config:
    CATEGORY_WEIGHTS = {
        "Trend": 0.3,
        "Momentum": 0.25,
        "Volatility": 0.15,
        "Volume/Flow": 0.2,
        "Risk": 0.1
    }
    BONUS_MAX = 25
    RISK_THRESHOLD = 25
    MIN_HISTORY_DAYS = 60
    SMA_SHORT = 5
    SMA_LONG = 20
    SMA_LONG2 = 50
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2.0
    VMA_PERIOD = 20
    ATR_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    OBV_SMA_PERIOD = 20
    ADX_PERIOD = 14
    CCI_PERIOD = 20
    STOCH_SLOWK_PERIOD = 14
    STOCH_SLOWD_PERIOD = 3
    MFI_PERIOD = 14
    VOLATILITY_PERIOD = 20
    VOLATILITY_LONG_PERIOD = 60
    GARCH_WINDOW = 100
    CMF_PERIOD = 20

def is_market_open(data=None):
    hkt = pytz.timezone("Asia/Hong_Kong")
    current_time = datetime.datetime.now(hkt)
    hour, minute = current_time.hour, current_time.minute
    time_open = (21 <= hour <= 23) or (0 <= hour < 4) or (hour == 4 and minute == 0)
    if data is not None and not data.empty:
        last_volume = data['Volume'].iloc[-1]
        if np.isnan(last_volume) or last_volume <= 0:
            return True
        return False
    return time_open

# 整合 indicators.py
def calculate_indicators(data):
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
    
    try:
        returns = data['Close'].pct_change().dropna() * 100
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=True)
        garch_fit = model.fit(disp='off')
        data['GARCH_Volatility'] = garch_fit.conditional_volatility.shift(-len(returns) + len(data))
    except Exception as e:
        warnings.warn(f"GARCH calculation failed: {str(e)}. Falling back to Volatility20.")
        data['GARCH_Volatility'] = data['Volatility20']
    
    data['CMF'] = talib.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=3, slowperiod=10) / data['Volume'].rolling(Config.CMF_PERIOD).sum()
    
    data['Lag_Close_1'] = data['Close'].shift(1)
    data['Rolling_Mean_5'] = data['Close'].rolling(window=5).mean()
    
    data = data.ffill().bfill()
    nan_ratio = data.isnull().mean().mean()
    if nan_ratio > 0.2:
        warnings.warn(f"High NaN ratio ({nan_ratio:.2f}) after fill, data may be unreliable.")
        return None
    if nan_ratio > 0:
        warnings.warn("Some NaNs remain after ffill, but proceeding.")
    
    return data

def calculate_target_stop_loss(data):
    lookback_days = 200
    num_rows = 50
    if len(data) < lookback_days:
        return 0.0, 0.0
    
    high = data['High'].values[-lookback_days:].astype('float64')
    low = data['Low'].values[-lookback_days:].astype('float64')
    close = data['Close'].values[-lookback_days:].astype('float64')
    volume = data['Volume'].values[-lookback_days:].astype('float64')
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

# 整合 conditions.py
def condition_a_trend_strength(data, market_params=None):
    try:
        market_params = market_params or {}
        index = -1 if not is_market_open(data) else -2
        close = data['Close'].iloc[index]
        sma5 = data['SMA5'].iloc[index]
        sma20 = data['SMA20'].iloc[index]
        sma50 = data['SMA50'].iloc[index]
        atr = data['ATR'].iloc[index]
        
        if np.isnan(sma5) or np.isnan(sma20) or np.isnan(sma50) or np.isnan(atr):
            return 0.0, "Invalid SMA or ATR data", None
        if close > sma20 and sma20 > sma50:
            return 1.0, "Fully met: Close above SMA20 and SMA20 above SMA50", None
        elif close > sma5 - atr * 0.5:
            return 0.75, "Partially met: Close above adjusted SMA5 with ATR buffer", None
        elif close < sma5 and close > sma20 and close > sma50:
            return 0.5, "Close below SMA5 but above SMA20 and SMA50", None
        elif close < sma5 and close < sma20 and close > sma50:
            return 0.3, "Close below SMA5 and SMA20 but above SMA50", None
        return 0.0, "Close below SMA5, SMA20, and SMA50", None
    except Exception as e:
        warnings.warn(f"Error in condition_a_trend_strength: {str(e)}")
        return 0.0, "Error occurred", None

def condition_b_sma_trend(data, market_params=None):
    try:
        market_params = market_params or {}
        index = -1 if not is_market_open(data) else -2
        sma5 = data['SMA5'].iloc[index]
        sma20 = data['SMA20'].iloc[index]
        sma50 = data['SMA50'].iloc[index]
        if np.isnan(sma5) or np.isnan(sma20) or np.isnan(sma50):
            return 0.0, "Invalid SMA data", None
        
        crossover = False
        if len(data) >= 3:
            sma5_prev = data['SMA5'].iloc[index - 1]
            sma20_prev = data['SMA20'].iloc[index - 1]
            if sma5_prev <= sma20_prev and sma5 > sma20:
                crossover = True
        
        if sma5 >= sma20 and sma20 > sma50:
            if crossover:
                return 1.0, "Fully met: SMA5 >= SMA20, SMA20 > SMA50 with golden cross", None
            return 1.0, "Fully met: SMA5 >= SMA20 and SMA20 > SMA50", None
        elif crossover:
            return 0.75, "Partially met: Golden cross detected", None
        elif sma5 >= sma50 and sma20 > sma50:
            return 0.5, "SMA5 >= SMA50 and SMA20 > SMA50", None
        elif sma5 < sma50 and sma20 < sma50:
            return 0.0, "SMA5 and SMA20 below SMA50", None
        return 0.0, "No specific condition met", None
    except Exception as e:
        warnings.warn(f"Error in condition_b_sma_trend: {str(e)}")
        return 0.0, "Error occurred", None

def condition_c_rsi(data, market_params=None):
    try:
        market_params = market_params or {}
        close = data['Close'].values.astype('float64')
        rsi = talib.RSI(close, 14)
        index = -1 if not is_market_open(data) else -2
        rsi_value = rsi[index]
        
        market_condition = market_params.get('market_condition', "Neutral")
        if market_condition == "Bullish":
            rsi_low, rsi_high = 50, 80
        elif market_condition == "Bearish":
            rsi_low, rsi_high = 30, 50
        else:
            rsi_low, rsi_high = 40, 65
        
        divergence = False
        if len(rsi) >= 3:
            rsi_prev = rsi[index - 1]
            close_prev = close[index - 1]
            if (rsi_value > rsi_prev and close[index] < close_prev) or (rsi_value < rsi_prev and close[index] > close_prev):
                divergence = True
        
        if rsi_low <= rsi_value <= rsi_high:
            if divergence:
                return 0.75, "Partially met: RSI in range but with divergence", None
            return 1.0, "Fully met: RSI in dynamic range", None
        elif (rsi_high < rsi_value <= rsi_high + 15) or (rsi_low - 10 <= rsi_value < rsi_low):
            return 0.5, "RSI in moderate overbought/oversold zone", None
        return 0.0, "RSI in extreme overbought/oversold zone", None
    except Exception as e:
        warnings.warn(f"Error in condition_c_rsi: {str(e)}")
        return 0.0, "Error occurred", None

def condition_d_bollinger_bands(data, market_params=None):
    try:
        market_params = market_params or {}
        close = data['Close'].values.astype('float64')
        upper_band, middle_band, lower_band = talib.BBANDS(
            close, timeperiod=Config.BOLLINGER_PERIOD, 
            nbdevup=Config.BOLLINGER_STD, nbdevdn=Config.BOLLINGER_STD, matype=0
        )
        index = -1 if not is_market_open(data) else -2
        close_price = close[index]
        upper, middle, lower = upper_band[index], middle_band[index], lower_band[index]
        
        if np.isnan(upper) or np.isnan(middle) or np.isnan(lower):
            return 0.0, "Invalid Bollinger Bands data", None
        
        band_width = (upper - lower) / middle
        
        if market_params.get('market_condition', "Neutral") == "Bullish" and close_price >= upper * 0.95:
            return 1.0, "Fully met: Close above upper band in Bullish", None
        elif close_price > middle and close_price < upper * 0.95:
            return 1.0, "Fully met", None
        elif band_width < 0.1 and close_price > middle:
            return 0.75, "Partially met with band contraction", None
        elif close_price >= upper * 0.95:
            return 0.5, "Close near/above upper band (overbought risk)", None
        elif lower < close_price <= middle:
            return 0.5, "Close between lower band and middle band (neutral)", None
        return 0.0, "Close at/below lower band (oversold or downtrend)", None
    except Exception as e:
        warnings.warn(f"Error in condition_d_bollinger_bands: {str(e)}")
        return 0.0, "Error occurred", None

def condition_e_volume(data, market_params=None):
    try:
        market_params = market_params or {}
        open_price = data['Open'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        volume = data['Volume'].values.astype('float64')
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        atr = talib.ATR(high, low, close, Config.ATR_PERIOD)
        vma = talib.SMA(volume, Config.VMA_PERIOD)
        index = -1 if not is_market_open(data) else -2
        open_today, close_today = open_price[index], close[index]
        volume_today, vma_today, atr_today = volume[index], vma[index], atr[index]
        k_line_change = close_today - open_today
        high_today, low_today = high[index], low[index]
        
        k_line_pattern = "Other"
        if k_line_change > atr_today * 0.5 and close_today >= high_today * 0.95:
            k_line_pattern = "Bullish Candle"
        elif abs(k_line_change) < atr_today * 0.3 and (high_today - low_today) > atr_today * 1.2:
            k_line_pattern = "Doji"
        
        if volume_today > 1.5 * vma_today and k_line_change > atr_today:
            score = 1.0
            explanation = "Fully met: Strong volume and large price gain"
        elif volume_today > 1.2 * vma_today and k_line_change > 0.3 * atr_today:
            score = 0.75
            explanation = "Good volume and moderate price gain"
        elif volume_today > 0.8 * vma_today and k_line_change >= 0:
            score = 0.5
            explanation = "Moderate volume with non-negative close"
        elif volume_today >= 0.3 * vma_today and k_line_change >= -atr_today * 0.3:
            score = 0.25
            explanation = "Average volume with positive close"
        else:
            score = 0.0
            explanation = "Price declined or low volume"
        
        return score, explanation, k_line_pattern
    except Exception as e:
        warnings.warn(f"Error in condition_e_volume: {str(e)}")
        return 0.0, "Error occurred", "Other"

def condition_f_macd(data, market_params=None):
    try:
        market_params = market_params or {}
        close = data['Close'].values.astype('float64')
        macd, signal, histogram = talib.MACD(close, fastperiod=Config.MACD_FAST, slowperiod=Config.MACD_SLOW, signalperiod=Config.MACD_SIGNAL)
        index = -1 if not is_market_open(data) else -2
        macd_value, signal_value, hist_value = macd[index], signal[index], histogram[index]
        atr = data['ATR'].iloc[index]
        market_condition = market_params.get('market_condition', "Neutral")
        
        if np.isnan(macd_value) or np.isnan(signal_value) or np.isnan(hist_value) or np.isnan(atr):
            return 0.5, "Partially met: Invalid MACD or histogram data", None
        
        macd_relaxed = market_params.get('macd_relaxed', False)
        
        score = 0.0
        explanation = "MACD below signal line"
        if macd_value > signal_value:
            score = 0.75
            explanation = "Partially met: MACD above signal line"
            if hist_value > 0:
                score = 1.0
                explanation = "Fully met: MACD above signal with positive histogram"
            if market_condition == "Bearish" and abs(hist_value) < 0.1 * atr:
                score = 0.5
                explanation = "Partially met: Histogram near zero in Bearish market"
        return score, explanation, None
    except Exception as e:
        warnings.warn(f"Error in condition_f_macd: {str(e)}")
        return 0.0, "Error occurred", None

def condition_g_obv(data, market_params=None):
    try:
        market_params = market_params or {}
        close = data['Close'].values.astype('float64')
        volume = data['Volume'].values.astype('float64')
        obv = talib.OBV(close, volume)
        obv_sma = talib.SMA(obv, Config.OBV_SMA_PERIOD)
        index = -1 if not is_market_open(data) else -2
        
        if len(obv) < Config.OBV_SMA_PERIOD or np.isnan(obv[index]) or np.isnan(obv_sma[index]):
            return 0.0, "Invalid OBV or OBV_SMA data", None

        volatility = data['GARCH_Volatility'].iloc[index] if 'GARCH_Volatility' in data.columns else 0.05
        decline_threshold = 0.90 if volatility > 0.05 else 0.95
        sma_threshold = 1.05 if volatility > 0.05 else 1.0

        lookback = 5
        if len(obv) >= lookback:
            x = np.arange(lookback)
            y = obv[-lookback:]
            slope, _ = np.polyfit(x, y, 1)
            obv_mean = np.mean(y)
            slope_percent = (slope / obv_mean) * 100 if obv_mean != 0 else 0.0
        else:
            slope_percent = 0.0
            warnings.warn("Insufficient data for OBV slope analysis")

        divergence = False
        if len(obv) >= 2:
            obv_prev = obv[index - 1]
            close_prev = close[index - 1]
            if (obv[index] > obv_prev and close[index] < close_prev) or (obv[index] < obv_prev and close[index] > close_prev):
                divergence = True
        
        score = 0.0
        explanation = "OBV declining significantly or negative slope"
        if (obv[index] > obv[index-1] and obv[index] > obv_sma[index] * sma_threshold and slope_percent > 0.1):
            score = 1.0
            explanation = "Fully met: OBV rising, above SMA (dynamic threshold), and positive slope"
        elif (obv[index] >= obv[index-1] * decline_threshold or slope_percent >= -0.05 or divergence):
            score = 0.5
            explanation = f"Partially met: OBV stable (threshold {decline_threshold:.2f}) or slightly negative slope" if not divergence else "Partially met: OBV with price divergence"
        return score, explanation, None
    except Exception as e:
        warnings.warn(f"Error in condition_g_obv: {str(e)}")
        return 0.0, "Error occurred", None

def condition_h_adx(data, market_params=None):
    try:
        market_params = market_params or {}
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        adx = talib.ADX(high, low, close, timeperiod=Config.ADX_PERIOD)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=Config.ADX_PERIOD)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=Config.ADX_PERIOD)
        index = -1 if not is_market_open(data) else -2
        adx_value, plus_di_value, minus_di_value = adx[index], plus_di[index], minus_di[index]
        
        if adx_value > 30 and plus_di_value > minus_di_value:
            return 1.0, "Fully met: Strong trend with +DI above -DI", None
        if 25 <= adx_value <= 30 and plus_di_value > minus_di_value:
            return 0.5, "Moderately strong trend with +DI above -DI", None
        if 20 <= adx_value <= 25 or (plus_di_value >= minus_di_value * 0.95 and plus_di_value <= minus_di_value * 1.05):
            return 0.25, "Moderate trend or +DI close to -DI", None
        return 0.0, "Weak trend or -DI above +DI", None
    except Exception as e:
        warnings.warn(f"Error in condition_h_adx: {str(e)}")
        return 0.0, "Error occurred", None

def condition_i_cci(data, market_params=None):
    try:
        market_params = market_params or {}
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        cci = talib.CCI(high, low, close, timeperiod=Config.CCI_PERIOD)
        index = -1 if not is_market_open(data) else -2
        cci_value = cci[index]
        market_condition = market_params.get('market_condition', "Neutral")
        cci_high = 150 if market_condition == "Bullish" else 100
        
        if np.isnan(cci_value):
            return 0.0, "Invalid CCI data", None
        
        divergence = False
        if len(close) >= 2 and cci_value > 100 and close[index] < close[index - 1]:
            divergence = True
        
        if 0 < cci_value < cci_high:
            return 1.0, "Fully met: CCI in bullish zone", None
        if cci_value >= cci_high or (-100 < cci_value <= 0):
            if divergence:
                return 0.75, "Partially met with divergence", None
            return 0.5, "CCI overbought or in neutral zone", None
        return 0.0, "CCI oversold", None
    except Exception as e:
        warnings.warn(f"Error in condition_i_cci: {str(e)}")
        return 0.0, "Error occurred", None

def condition_j_stoch(data, market_params=None):
    try:
        market_params = market_params or {}
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        index = -1 if not is_market_open(data) else -2
        slowk_value, slowd_value = slowk[index], slowd[index]
        
        if np.isnan(slowk_value) or np.isnan(slowd_value):
            return 0.0, "Invalid Stochastic data", None
        
        market_condition = market_params.get('market_condition', "Neutral")
        stoch_high = 70 if market_condition == "Bullish" else 60
        
        divergence = False
        if len(close) >= 2 and slowk_value > 80 and close[index] < close[index - 1]:
            divergence = True
        
        if slowk_value > slowd_value and slowk_value > stoch_high:
            return 1.0, "Fully met: %K above %D in bullish zone", None
        if slowk_value >= 20 and slowk_value <= 80:
            if divergence:
                return 0.75, "Partially met with divergence", None
            return 0.5, "Stochastic overbought or in neutral zone", None
        return 0.0, "Stochastic oversold", None
    except Exception as e:
        warnings.warn(f"Error in condition_j_stoch: {str(e)}")
        return 0.0, "Error occurred", None

def condition_k_atr(data, market_params=None):
    try:
        market_params = market_params or {}
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        open_price = data['Open'].values.astype('float64')
        atr = talib.ATR(high, low, close, Config.ATR_PERIOD)
        atr_sma = talib.SMA(atr, 20)
        index = -1 if not is_market_open(data) else -2
        atr_value, atr_sma_value = atr[index], atr_sma[index]
        k_line_change = close[index] - open_price[index]
        
        if atr_value <= atr_sma_value * 1.4 and k_line_change >= -atr_sma_value * 0.3:
            return 1.0, "Fully met: Low volatility with positive close", None
        if atr_value <= atr_sma_value * 1.6 and k_line_change >= -atr_sma_value * 0.5:
            return 0.5, "Moderate volatility with non-negative close", None
        return 0.0, "High volatility or negative close", None
    except Exception as e:
        warnings.warn(f"Error in condition_k_atr: {str(e)}")
        return 0.0, "Error occurred", None

def condition_m_mfi(data, market_params=None):
    try:
        market_params = market_params or {}
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        volume = data['Volume'].values.astype('float64')
        mfi = talib.MFI(high, low, close, volume, timeperiod=Config.MFI_PERIOD)
        index = -1 if not is_market_open(data) else -2
        mfi_value = mfi[index]
        
        if 20 < mfi_value < 80:
            return 1.0, "Fully met: MFI in healthy zone", None
        if mfi_value >= 80 or mfi_value <= 20:
            return 0.5, "MFI overbought or oversold", None
        return 0.0, "MFI oversold with negative close", None
    except Exception as e:
        warnings.warn(f"Error in condition_m_mfi: {str(e)}")
        return 0.0, "Error occurred", None

def condition_o_volatility(data, market_params=None):
    try:
        market_params = market_params or {}
        close = data['Close'].values.astype('float64')
        open_price = data['Open'].values.astype('float64')
        index = -1 if not is_market_open(data) else -2
        if len(close) < Config.VOLATILITY_LONG_PERIOD:
            return 0.0, "Insufficient data for volatility calculation", None
        volatility_20 = np.std(close[index-19:index+1]) / np.mean(close[index-19:index+1])
        volatility_60 = np.mean([np.std(close[i-19:i+1]) / np.mean(close[i-19:i+1]) for i in range(index-59, index+1)])
        k_line_change = close[index] - open_price[index]
        
        if volatility_20 <= volatility_60 * 1.7:
            return 1.0, "Fully met: Low volatility relative to historical average", None
        if volatility_20 <= volatility_60 * 1.9 and k_line_change >= -volatility_60 * 0.5:
            return 0.5, "Moderate volatility with non-negative close", None
        return 0.0, "High volatility or negative close", None
    except Exception as e:
        warnings.warn(f"Error in condition_o_volatility: {str(e)}")
        return 0.0, "Error occurred", None

def condition_p_cmf(data, market_params=None):
    try:
        market_params = market_params or {}
        index = -1 if not is_market_open(data) else -2
        cmf = data['CMF'].iloc[index]
        if np.isnan(cmf):
            return 0.0, "Invalid CMF data", None
        if cmf > 0.05:
            return 1.0, "Strong money inflow", None
        elif cmf > -0.05:
            return 0.5, "Neutral money flow", None
        return 0.0, "Strong money outflow", None
    except Exception as e:
        warnings.warn(f"Error in condition_p_cmf: {str(e)}")
        return 0.0, "Error occurred", None

def condition_r_var(data, market_params=None):
    try:
        market_params = market_params or {}
        returns = data['Close'].pct_change().dropna()[-252:]
        if len(returns) < 100:
            return 0.0, "Insufficient data for VaR", None
        var_95 = np.percentile(returns, 5)
        abs_var = abs(var_95)
        if abs_var < 0.02:
            return 1.0, "Low VaR risk", None
        elif abs_var < 0.05:
            return 0.5, "Moderate VaR risk", None
        return 0.0, "High VaR risk", None
    except Exception as e:
        warnings.warn(f"Error in condition_r_var: {str(e)}")
        return 0.0, "Error occurred", None

def condition_l_crash_risk(data, market_params, ticker, market_condition):
    try:
        score = 0
        crash_triggers = []
        close = data['Close'].values.astype('float64')
        volume = data['Volume'].values.astype('float64')
        macd = data['MACD'].values.astype('float64')
        index = -1 if not is_market_open(data) else -2
        sma200 = talib.SMA(close, 200)[index]
        bb_upper = data['UpperBB'].iloc[index]
        forward_pe = yf.Ticker(ticker).info.get('forwardPE', np.nan)
        
        if not np.isnan(forward_pe) and forward_pe > 50:
            score += 10
            crash_triggers.append("High forward P/E (+10)")
        
        if len(macd) >= 2 and macd[index] > macd[index-1] and close[index] < close[index-1]:
            score += 15
            crash_triggers.append("MACD bullish divergence but price down (+15)")
        
        vma = talib.SMA(volume, Config.VMA_PERIOD)[index]
        if volume[index] < 0.8 * vma:
            score += 10
            crash_triggers.append("Volume shrinkage below average (+10)")
        
        garch_vol = data['GARCH_Volatility'].iloc[-1]
        if not np.isnan(garch_vol) and garch_vol > 0.05:
            score += 10
            crash_triggers.append("High GARCH volatility forecast (+10)")

        var_score, var_explanation, _ = condition_r_var(data)
        if var_score == 0.0:
            score += 15
            crash_triggers.append(f"High VaR (+15): {var_explanation}")
        
        crash_index = score
        if score >= 50:
            explanation = "High risk (+" + str(score) + ")"
        elif score >= 25:
            explanation = "Moderate risk (+" + str(score) + ")"
        else:
            explanation = "Low risk (+" + str(score) + ")"
        
        if not crash_triggers:
            crash_triggers.append("No major triggers")
        
        return score, explanation, crash_index, crash_triggers, []
    except Exception as e:
        warnings.warn(f"Error in condition_l_crash_risk for {ticker}: {str(e)}")
        return 0, "Error occurred", 0, [], []

# 整合 health_score.py
def calculate_health_score(data, market_params=None, ticker="", market_condition="Unknown"):
    if data is None or data.empty:
        return 0.0, 0.0, {}, {}, [], [], 0.0, 0.0, "", 0.0, 0.0, {}, {}
    
    category_mapping = {
        "Trend": [
            ("K-line Trend (Close vs SMA20, SMA50 with ATR buffer)", condition_a_trend_strength),
            ("SMA Trend (SMA5 vs SMA20, SMA50)", condition_b_sma_trend),
            ("MACD (12,26,9)", condition_f_macd),
            ("ADX (14-day)", condition_h_adx)
        ],
        "Momentum": [
            ("RSI (14-day)", condition_c_rsi),
            ("Bollinger Bands (Close vs Upper/Middle/Lower)", condition_d_bollinger_bands),
            ("CCI (20-day)", condition_i_cci),
            ("Stochastic Oscillator (14,3,3)", condition_j_stoch)
        ],
        "Volatility": [
            ("ATR (14-day)", condition_k_atr),
            ("Historical Volatility (20-day vs 60-day)", condition_o_volatility),
            ("95% VaR Risk", condition_r_var)
        ],
        "Volume/Flow": [
            ("Volume and K-line", condition_e_volume),
            ("OBV (On-Balance Volume)", condition_g_obv),
            ("MFI (14-day)", condition_m_mfi),
            ("CMF (Chaikin Money Flow)", condition_p_cmf)
        ],
        "Risk": [
            ("Crash Risk (PE, GARCH, VaR)", condition_l_crash_risk)
        ]
    }
    
    category_scores = {}
    percentages = {}
    category_explanations = {}
    bonus_score = 0.0
    bonus_details = []
    warnings_list = []
    
    condition_scores = {}
    for category, cond_list in category_mapping.items():
        total_score = 0.0
        explanations = []
        for name, cond_func in cond_list:
            if name == "Crash Risk (PE, GARCH, VaR)":
                score, explanation, extra, crash_triggers, crash_warnings = cond_func(data, market_params, ticker, market_condition)
                extra = crash_triggers
            else:
                if name in ["Volume and K-line"]:
                    score, explanation, extra = cond_func(data, market_params)
                else:
                    score, explanation, extra = cond_func(data, market_params)
            condition_scores[name] = (score, explanation, extra)
            total_score += score
            explanations.append((name, score, explanation))
        avg_score = total_score / len(cond_list)
        if category == "Risk":
            avg_score = 1 - (avg_score / 50) if avg_score <= 50 else 0.0
        category_score = avg_score * 100 * Config.CATEGORY_WEIGHTS[category]
        category_scores[category] = round(category_score)
        percentages[category] = round((category_score / (Config.CATEGORY_WEIGHTS[category] * 100)) * 100)
        category_explanations[category] = explanations
    
    # Full bonus logic
    if condition_scores["Volume and K-line"][0] == 1.0 and condition_scores["Volume and K-line"][2] == "Bullish Candle":
        bonus_score += 0.5
        bonus_details.append("Bonus: Strong K-line and volume (+10)")
    
    if condition_scores["RSI (14-day)"][0] == 1.0 and condition_scores["MACD (12,26,9)"][0] == 1.0:
        bonus_score += 0.5
        bonus_details.append("Bonus: Healthy RSI and MACD (+10)")
    
    if condition_scores["Volume and K-line"][0] == 1.0 and condition_scores["Volume and K-line"][2] == "Doji":
        sma5 = talib.SMA(data['Close'].values, Config.SMA_SHORT)
        atr = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, Config.ATR_PERIOD)
        adx = talib.ADX(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=Config.ADX_PERIOD)
        index = -1 if is_market_open() else -2
        if len(sma5) >= 5:
            sma_diff = sma5[index] - sma5[index-4]
            atr_value = atr[index]
            adx_value = adx[index]
            warning_strength = "Strong" if adx_value > 30 else "Weak" if adx_value <= 20 else ""
            
            if sma_diff > 0 and adx_value > 30:
                bonus_score -= 0.25
                warnings_list.append(f"Warning: Doji with high volume: {warning_strength} Reversal risk")
            elif sma_diff < 0:
                bonus_score += 0.25
                bonus_details.append("Bonus: Doji with high volume in downtrend (+5)")
                warnings_list.append(f"Warning: Doji with high volume: {warning_strength} Rebound potential")
            elif abs(sma_diff) < 0.5 * atr_value:
                warnings_list.append(f"Warning: Doji with high volume: Market indecision")
    
    if condition_scores["Volume and K-line"][0] <= 0.25 and condition_scores["Volume and K-line"][2] == "Doji":
        bonus_score += 0.25
        bonus_details.append("Bonus: Doji with low volume (+5)")
        warnings_list.append("Warning: Doji with low volume: Weak momentum")
    
    if condition_scores["ADX (14-day)"][0] == 1.0 and condition_scores["OBV (On-Balance Volume)"][0] == 1.0:
        bonus_score += 0.25
        bonus_details.append("Bonus: High ADX and rising OBV (+5)")
    
    if condition_scores.get("CMF (Chaikin Money Flow)", (0, ""))[0] == 1.0 and condition_scores["OBV (On-Balance Volume)"][0] == 1.0:
        bonus_score += 0.25
        bonus_details.append("Bonus: Strong CMF and OBV (+10)")
    
    bonus_score = min(bonus_score * (Config.BONUS_MAX / 1.25), Config.BONUS_MAX)
    bonus_score = max(bonus_score, 0.0)
    
    total_health_score = min(sum(category_scores.values()), 100)
    
    tree_explanation = ""
    dt_adjustment = 0.0
    try:
        features = ['RSI', 'MACD', 'ADX', 'CMF', 'Volatility20', 'GARCH_Volatility']
        if all(f in data.columns for f in features) and len(data) >= 100:
            data_clean = data[features].dropna()
            data_clean['Target'] = np.where(data['Close'] > data['SMA50'], 2, np.where(data['Close'] > data['SMA20'], 1, 0))
            train_data = data_clean.iloc[:-1]
            X_train = train_data[features]
            y_train = train_data['Target']
            if len(X_train) >= 50:
                dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
                dt_model.fit(X_train, y_train)
                X_latest = data_clean[features].iloc[-1:].values
                predicted_class = dt_model.predict(X_latest)[0]
                if predicted_class == 2:
                    dt_adjustment = 5.0
                elif predicted_class == 0:
                    dt_adjustment = -5.0
                tree_explanation = export_text(dt_model, feature_names=features)
        else:
            warnings.warn("Insufficient data for decision tree optimization")
    except Exception as e:
        warnings.warn(f"Error in decision tree: {str(e)}")
    
    total_health_score = max(0, min(100, total_health_score + dt_adjustment))
    
    crash_score, crash_explanation, crash_index, crash_triggers, crash_warnings = condition_l_crash_risk(data, market_params, ticker, market_condition)
    if crash_score >= Config.RISK_THRESHOLD:
        warnings_list.append(f"High reversal risk: Based on high volatility forecast and potential indicator divergence, monitor closely.")
    if condition_scores.get("CMF (Chaikin Money Flow)", (0, ""))[0] < 0.5 and condition_scores["Volume and K-line"][0] < 0.5:
        warnings_list.append(f"Moderate outflow risk: Neutral volume and negative CMF, may indicate weakening trend.")
    if condition_scores["MACD (12,26,9)"][0] < 1.0 and condition_scores["RSI (14-day)"][0] < 1.0:
        warnings_list.append(f"Momentum weakening alert: Negative MACD and overbought/sold RSI, increased downside reversal probability.")
    
    trend_names = [name for name, _ in category_mapping["Trend"]]
    trend_avg = sum([condition_scores.get(name, (0, ""))[0] for name in trend_names]) / len(trend_names) if trend_names else 0.0
    if trend_avg > 0.5:
        trend_prob = (trend_avg * 2 - 1) * 100
    elif trend_avg < 0.5:
        trend_prob = (trend_avg * 2 - 1) * 100
    else:
        trend_prob = 0.0
    
    target, stop_loss = calculate_target_stop_loss(data)
    
    return total_health_score, bonus_score, category_scores, category_explanations, bonus_details, warnings_list, trend_prob, dt_adjustment, tree_explanation, target, stop_loss, percentages, condition_scores

# 整合 models.py
def predict_price_direction(data, ticker, total_health_score):
    try:
        features = [
            'SMA5', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACDSignal', 'OBV', 'OBV_SMA',
            'ADX', 'PLUS_DI', 'MINUS_DI', 'CCI', 'SlowK', 'SlowD', 'MFI', 'ATR',
            'Volatility20', 'Volatility60', 'GARCH_Volatility', 'CMF',
            'Lag_Close_1', 'Rolling_Mean_5'
        ]
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            warnings.warn(f"Missing features for XGBoost: {missing_features}")
            return None, "Missing features for prediction", []
        
        data = data.dropna(subset=features)
        if len(data) < 100:
            warnings.warn(f"Insufficient data for XGBoost: {len(data)} days")
            return None, "Insufficient data for prediction", []
        
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        train_data = data.iloc[-500:-1]
        X_train = train_data[features]
        y_train = train_data['Target']
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy', n_jobs=-1)
        avg_accuracy = np.mean(scores)
        if avg_accuracy < 0.6:
            warnings.warn(f"Low model accuracy for {ticker}: {avg_accuracy:.2f}")
        
        X_latest = data[features].iloc[-1:].dropna()
        if X_latest.empty:
            warnings.warn("Latest data contains NaNs for prediction")
            return None, "Invalid latest data", []
        
        pred = model.predict_proba(X_latest)[0]
        bullish_prob = pred[1] * 100
        direction = "Bullish" if bullish_prob >= 50 else "Bearish"
        confidence = max(bullish_prob, 100 - bullish_prob)
        
        if direction == "Bullish" and total_health_score > 70:
            suggestion = "繼續持有"
        elif direction == "Bearish" and total_health_score < 50:
            suggestion = "考慮沽空"
        else:
            suggestion = "觀望"
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_latest)
        shap_contributions = [(feat, shap_values[0][i] if direction == "Bullish" else -shap_values[0][i]) for i, feat in enumerate(features)]
        shap_contributions = sorted(shap_contributions, key=lambda x: abs(x[1]), reverse=True)[:3]
        shap_explanation = [f"{feat}: {val:.3f}" for feat, val in shap_contributions]
        
        explanation = f"{direction} (Confidence: {confidence:.2f}%), Suggestion: {suggestion}"
        return direction, explanation, shap_explanation
    except Exception as e:
        warnings.warn(f"Error in price direction prediction for {ticker}: {str(e)}")
        return None, "Error in prediction", []

# 整合 data_fetch.py
def get_stock_data(ticker, period="3y", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty or len(data) < Config.MIN_HISTORY_DAYS:
            warnings.warn(f"Insufficient data for {ticker} ({len(data)} days)")
            return None
        data = data.reset_index(drop=False).drop_duplicates(subset='Date').set_index('Date')
        data = calculate_indicators(data)
        if data is None:  # 若 NaN 過高，返回 None
            return None
        return data
    except Exception as e:
        warnings.warn(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_market_condition():
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

# Streamlit App
st.title("股票健康監測工具")
st.write("輸入股票代碼，查看完整健康報告")
ticker_input = st.text_input("輸入股票代碼 (e.g., AAPL 或 AAPL,NVDA):", "AAPL")
if st.button("分析股票健康"):
    tickers = [t.strip().upper() for t in ticker_input.split(',')]
    for ticker in tickers:
        data = get_stock_data(ticker)
        if data is None:
            st.error(f"無法獲取 {ticker} 數據")
            continue
        market_condition, market_params = get_market_condition()
        total_health_score, bonus_score, category_scores, category_explanations, bonus_details, warnings_list, trend_prob, dt_adjustment, tree_explanation, target, stop_loss, percentages, condition_scores = calculate_health_score(data, market_params, ticker, market_condition)
        price_direction, direction_explanation, shap_explanation = predict_price_direction(data, ticker, total_health_score)
        
        st.subheader(f"📊 {ticker} 健康報告 - 價格: ${data['Close'].iloc[-1]:.2f}")
        st.write(f"🔍 市場條件: {market_condition}")
        st.write(f"📈 價格方向預測: {direction_explanation}")
        st.write(f"💎 總健康分數: {int(round(total_health_score))} / 100 {'(Decision Tree Adjustment: {:.1f})'.format(dt_adjustment) if dt_adjustment else ''}")
        st.write(f"🚀 趨勢持續性: {trend_prob:+.2f}%")
        st.write(f"🎯 目標價: ${target:.2f}, 止損價: ${stop_loss:.2f}")
        if shap_explanation:
            st.write(f"🔧 主要影響因子: {', '.join(shap_explanation)}")
        
        # 類別分數
        st.write("📋 類別分數:")
        for category, score in percentages.items():
            st.write(f"  - {category}: {score}%")
        
        # 條件評估
        st.write("🔎 條件評估:")
        for category, explanations in category_explanations.items():
            if category == "Trend":
                st.write(f"  {category} 條件 (滿分要求強趨勢與動能正)")
                sorted_explanations = [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "ADX (14-day)"
                ] + [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "MACD (12,26,9)"
                ] + [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "SMA Trend (SMA5 vs SMA20, SMA50)"
                ] + [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "K-line Trend (Close vs SMA20, SMA50 with ATR buffer)"
                ]
                for name, score, explanation in sorted_explanations:
                    status = "✅" if score == 1.0 else "⚠️" if 0.25 <= score < 1.0 else "❌"
                    st.write(f"    - {name}: {status} ({explanation})")
            elif category == "Momentum":
                st.write(f"  {category} 條件 (滿分要求指標在看漲區無背離)")
                sorted_explanations = [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "RSI (14-day)"
                ] + [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "Stochastic Oscillator (14,3,3)"
                ] + [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "CCI (20-day)"
                ] + [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "Bollinger Bands (Close vs Upper/Middle/Lower)"
                ]
                for name, score, explanation in sorted_explanations:
                    status = "✅" if score == 1.0 else "⚠️" if 0.25 <= score < 1.0 else "❌"
                    st.write(f"    - {name}: {status} ({explanation})")
            elif category == "Volatility":
                st.write(f"  {category} 條件 (滿分要求低波動率與風險極低)")
                sorted_explanations = [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "95% VaR Risk"
                ] + [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "Historical Volatility (20-day vs 60-day)"
                ] + [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "ATR (14-day)"
                ]
                for name, score, explanation in sorted_explanations:
                    status = "✅" if score == 1.0 else "⚠️" if 0.25 <= score < 1.0 else "❌"
                    st.write(f"    - {name}: {status} ({explanation})")
            elif category == "Volume/Flow":
                st.write(f"  {category} 條件 (滿分要求強資金流入與高成交量支持)")
                sorted_explanations = [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "OBV (On-Balance Volume)"
                ] + [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "CMF (Chaikin Money Flow)"
                ] + [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "Volume and K-line"
                ] + [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "MFI (14-day)"
                ]
                for name, score, explanation in sorted_explanations:
                    status = "✅" if score == 1.0 else "⚠️" if 0.25 <= score < 1.0 else "❌"
                    st.write(f"    - {name}: {status} ({explanation})")
            elif category == "Risk":
                st.write(f"  {category} 條件 (滿分要求低崩盤風險)")
                sorted_explanations = [
                    (name, score, explanation) for name, score, explanation in explanations
                    if name == "Crash Risk (PE, GARCH, VaR)"
                ]
                for name, score, explanation in sorted_explanations:
                    status = "✅" if score == 1.0 else "⚠️" if 0.25 <= score < 1.0 else "❌"
                    st.write(f"    - {name}: {status} ({explanation})")
        
        # 警告
        st.write("‼️ 警告訊息:")
        crash_score, crash_explanation, crash_index, crash_triggers, crash_warnings = condition_l_crash_risk(data, market_params, ticker, market_condition)
        st.write(f"  - 崩盤風險: {crash_explanation}")
        st.write(f"  - 觸發因素: {', '.join(crash_triggers)}")
        if warnings_list or crash_warnings:
            for warn in warnings_list + crash_warnings:
                st.write(f"  - {warn}")
        else:
            st.write("  - 無")
        
        # 獎勵分數
        st.write("⭐ 獎勵分數:")
        st.write(f"  - {(' | '.join(bonus_details) if bonus_details else '無')}")
        
        st.write(f"📅 數據獲取日期: {data.index[-1].strftime('%Y-%m-%d')}")