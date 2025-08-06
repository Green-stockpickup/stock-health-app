import numpy as np
import talib
import warnings
import yfinance as yf  # Added for yf.Ticker in crash_risk
from config import Config, is_market_open

def condition_a_trend_strength(data, market_params=None):
    """評估趨勢強度基於 Close vs SMA。
    輸入: data (pd.DataFrame), market_params (dict)。
    輸出: (score (float), explanation (str), extra (None))。
    """
    try:
        market_params = market_params or {}
        close = data['Close'].iloc[-1]
        sma5 = data['SMA5'].iloc[-1]
        sma20 = data['SMA20'].iloc[-1]
        sma50 = data['SMA50'].iloc[-1]
        if np.isnan(sma5) or np.isnan(sma20) or np.isnan(sma50):
            return 0.0, "Invalid SMA data", None
        if close > sma5 and sma5 > sma20 and sma20 > sma50:
            return 1.0, "Fully met", None
        elif close < sma5 and close > sma20 and close > sma50:
            return 0.5, "Close below SMA5 but above SMA20 and SMA50", None
        elif close < sma5 and close < sma20 and close > sma50:
            return 0.3, "Close below SMA5 and SMA20 but above SMA50", None
        return 0.0, "Close below SMA5, SMA20, and SMA50", None
    except Exception as e:
        warnings.warn(f"Error in condition_a_trend_strength: {str(e)}")
        return 0.0, "Error occurred", None

def condition_b_sma_trend(data, market_params=None):
    """評估 SMA 趨勢。
    輸入: data (pd.DataFrame), market_params (dict)。
    輸出: (score (float), explanation (str), extra (None))。
    """
    try:
        market_params = market_params or {}
        sma5 = data['SMA5'].iloc[-1]
        sma20 = data['SMA20'].iloc[-1]
        sma50 = data['SMA50'].iloc[-1]
        if np.isnan(sma5) or np.isnan(sma20) or np.isnan(sma50):
            return 0.0, "Invalid SMA data", None
        if sma5 > sma20 and sma20 > sma50:
            return 1.0, "Fully met", None
        elif sma5 >= sma50 and sma20 > sma50:
            return 0.5, "SMA5 >= SMA50 and SMA20 > SMA50", None
        elif sma5 < sma50 and sma20 < sma50:
            return 0.0, "SMA5 and SMA20 below SMA50", None
        return 0.0, "No specific condition met", None
    except Exception as e:
        warnings.warn(f"Error in condition_b_sma_trend: {str(e)}")
        return 0.0, "Error occurred", None

def condition_c_rsi(data, market_params=None):
    """評估 RSI 指標。
    輸入: data (pd.DataFrame), market_params (dict)。
    輸出: (score (float), explanation (str), extra (None))。
    """
    try:
        market_params = market_params or {}
        close = data['Close'].values.astype('float64')
        rsi = talib.RSI(close, 14)
        index = -1 if is_market_open() else -2
        rsi_value = rsi[index]
        
        rsi_high = market_params.get('rsi_high', 65)
        rsi_low = market_params.get('rsi_low', 40)
        if rsi_low <= rsi_value <= rsi_high:
            return 1.0, "Fully met", None
        if (rsi_high < rsi_value <= rsi_high + 15) or (rsi_low - 10 <= rsi_value < rsi_low):
            return 0.5, "RSI in moderate overbought/oversold zone", None
        return 0.0, "RSI in extreme overbought/oversold zone", None
    except Exception as e:
        warnings.warn(f"Error in condition_c_rsi: {str(e)}")
        return 0.0, "Error occurred", None

def condition_d_bollinger_bands(data):
    """評估 Bollinger Bands。
    輸入: data (pd.DataFrame)。
    輸出: (score (float), explanation (str), extra (None))。
    """
    try:
        close = data['Close'].values.astype('float64')
        upper_band, middle_band, lower_band = talib.BBANDS(
            close, timeperiod=Config.BOLLINGER_PERIOD, 
            nbdevup=Config.BOLLINGER_STD, nbdevdn=Config.BOLLINGER_STD, matype=0
        )
        index = -1 if is_market_open() else -2
        close_price = close[index]
        upper, middle, lower = upper_band[index], middle_band[index], lower_band[index]
        
        if close_price > middle and close_price < upper * 0.95:
            return 1.0, "Fully met", None
        if close_price >= upper * 0.95:
            return 0.5, "Close near/above upper band (overbought risk)", None
        if lower < close_price <= middle:
            return 0.5, "Close between lower band and middle band (neutral)", None
        return 0.0, "Close at/below lower band (oversold or downtrend)", None
    except Exception as e:
        warnings.warn(f"Error in condition_d_bollinger_bands: {str(e)}")
        return 0.0, "Error occurred", None

def condition_e_volume(data, market_params=None):
    """評估成交量與K線。
    輸入: data (pd.DataFrame), market_params (dict)。
    輸出: (score (float), explanation (str), k_line_pattern (str))。
    """
    try:
        market_params = market_params or {}
        open_price = data['Open'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        volume = data['Volume'].values.astype('float64')
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        atr = talib.ATR(high, low, close, Config.ATR_PERIOD)
        vma = talib.SMA(volume, Config.VMA_PERIOD)
        index = -1 if is_market_open() else -2
        open_today, close_today = open_price[index], close[index]
        volume_today, vma_today, atr_today = volume[index], vma[index], atr[index]
        k_line_change = close_today - open_today
        high_today, low_today = high[index], low_today = high[index], low[index]
        
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
    """評估 MACD 指標。
    """
    try:
        market_params = market_params or {}
        close = data['Close'].values.astype('float64')
        macd, signal, _ = talib.MACD(close, fastperiod=Config.MACD_FAST, slowperiod=Config.MACD_SLOW, signalperiod=Config.MACD_SIGNAL)
        index = -1 if is_market_open() else -2
        macd_value, signal_value = macd[index], signal[index]
        
        macd_relaxed = market_params.get('macd_relaxed', False)
        if macd_value > signal_value:
            if macd_value > 0:
                return 1.0, "Fully met: MACD above signal line and positive", None
            elif macd_relaxed:
                return 0.75, "Partially met: MACD above signal but negative (relaxed in Bullish market)", None
            return 0.5, "Partially met: MACD above signal line but negative", None
        return 0.0, "MACD below signal line", None
    except Exception as e:
        warnings.warn(f"Error in condition_f_macd: {str(e)}")
        return 0.0, "Error occurred", None

def condition_g_obv(data, market_params=None):
    """評估 OBV 指標，加入動態閾值和斜率分析。
    輸入: data (pd.DataFrame), market_params (dict)。
    輸出: (score (float), explanation (str), extra (None))。
    """
    try:
        market_params = market_params or {}
        close = data['Close'].values.astype('float64')
        volume = data['Volume'].values.astype('float64')
        obv = talib.OBV(close, volume)
        obv_sma = talib.SMA(obv, Config.OBV_SMA_PERIOD)
        index = -1 if is_market_open() else -2
        
        if len(obv) < Config.OBV_SMA_PERIOD or np.isnan(obv[index]) or np.isnan(obv_sma[index]):
            return 0.0, "Invalid OBV or OBV_SMA data", None

        # 動態閾值：根據市場波動率調整
        volatility = data['GARCH_Volatility'].iloc[index] if 'GARCH_Volatility' in data.columns else 0.05
        decline_threshold = 0.90 if volatility > 0.05 else 0.95  # 高波動放寬至90%
        sma_threshold = 1.05 if volatility > 0.05 else 1.0  # 高波動要求OBV高於SMA*1.05

        # OBV 斜率分析：過去5日線性回歸斜率
        lookback = 5
        if len(obv) >= lookback:
            x = np.arange(lookback)
            y = obv[-lookback:]
            slope, _ = np.polyfit(x, y, 1)  # 計算斜率
            # 標準化斜率（相對OBV平均值）
            slope_normalized = slope / np.mean(np.abs(obv[-lookback:])) if np.mean(np.abs(obv[-lookback:])) != 0 else 0
        else:
            slope_normalized = 0.0
            warnings.warn("Insufficient data for OBV slope analysis")

        # 評分邏輯（保留3級制，後續討論5級制）
        if (obv[index] > obv[index-1] and obv[index] > obv_sma[index] * sma_threshold and 
            slope_normalized > 0.001):  # 斜率正且顯著
            return 1.0, "Fully met: OBV rising, above SMA (dynamic threshold), and positive slope", None
        elif (obv[index] >= obv[index-1] * decline_threshold or 
              slope_normalized >= -0.0005):  # 穩定或斜率微負
            return 0.5, f"Partially met: OBV stable (threshold {decline_threshold:.2f}) or slightly negative slope", None
        return 0.0, "OBV declining significantly or negative slope", None

    except Exception as e:
        warnings.warn(f"Error in condition_g_obv: {str(e)}")
        return 0.0, "Error occurred", None

def condition_h_adx(data):
    """評估 ADX 指標。
    """
    try:
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        adx = talib.ADX(high, low, close, timeperiod=Config.ADX_PERIOD)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=Config.ADX_PERIOD)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=Config.ADX_PERIOD)
        index = -1 if is_market_open() else -2
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
    """評估 CCI 指標。
    """
    try:
        market_params = market_params or {}
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        cci = talib.CCI(high, low, close, timeperiod=Config.CCI_PERIOD)
        index = -1 if is_market_open() else -2
        cci_value = cci[index]
        cci_high = market_params.get('cci_high', 100)
        
        if 0 < cci_value < cci_high:
            return 1.0, "Fully met: CCI in bullish zone", None
        if cci_value >= cci_high or (-100 < cci_value <= 0):
            return 0.5, "CCI overbought or in neutral zone", None
        return 0.0, "CCI oversold", None
    except Exception as e:
        warnings.warn(f"Error in condition_i_cci: {str(e)}")
        return 0.0, "Error occurred", None

def condition_j_stoch(data, market_params=None):
    """評估 Stochastic 指標。
    """
    try:
        market_params = market_params or {}
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        index = -1 if is_market_open() else -2
        slowk_value, slowd_value = slowk[index], slowd[index]
        
        if slowk_value > slowd_value and slowk_value > 60:
            return 1.0, "Fully met: %K above %D in bullish zone", None
        if slowk_value >= 20 and slowk_value <= 80:
            return 0.5, "Stochastic overbought or in neutral zone", None
        return 0.0, "Stochastic oversold", None
    except Exception as e:
        warnings.warn(f"Error in condition_j_stoch: {str(e)}")
        return 0.0, "Error occurred", None

def condition_k_atr(data, market_params=None):
    """評估 ATR 指標。
    """
    try:
        market_params = market_params or {}
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        open_price = data['Open'].values.astype('float64')
        atr = talib.ATR(high, low, close, Config.ATR_PERIOD)
        atr_sma = talib.SMA(atr, 20)
        index = -1 if is_market_open() else -2
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

def condition_m_mfi(data):
    """評估 MFI 指標。
    """
    try:
        high = data['High'].values.astype('float64')
        low = data['Low'].values.astype('float64')
        close = data['Close'].values.astype('float64')
        volume = data['Volume'].values.astype('float64')
        mfi = talib.MFI(high, low, close, volume, timeperiod=Config.MFI_PERIOD)
        index = -1 if is_market_open() else -2
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
    """評估歷史波動率。
    """
    try:
        market_params = market_params or {}
        close = data['Close'].values.astype('float64')
        open_price = data['Open'].values.astype('float64')
        index = -1 if is_market_open() else -2
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

def condition_p_cmf(data):
    """評估 CMF。
    """
    try:
        cmf = data['CMF'].iloc[-1]
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

def condition_r_var(data):
    """計算 95% VaR 風險。
    輸入: data (pd.DataFrame)。
    輸出: (score (float), explanation (str), extra (None))。
    """
    try:
        returns = data['Close'].pct_change().dropna()[-252:]  # 最近252天
        if len(returns) < 100:
            return 0.0, "Insufficient data for VaR", None
        var_95 = np.percentile(returns, 5)  # 5% 最壞情況
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
    """評估崩盤風險，整合 VaR。
    輸入: data (pd.DataFrame), market_params (dict), ticker (str), market_condition (str)。
    輸出: (score (int), explanation (str), crash_index (int), crash_triggers (list), crash_warnings (list))。
    """
    try:
        score = 0
        crash_triggers = []
        close = data['Close'].values.astype('float64')
        volume = data['Volume'].values.astype('float64')
        macd = data['MACD'].values.astype('float64')
        index = -1 if is_market_open() else -2
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

        # 整合VaR
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