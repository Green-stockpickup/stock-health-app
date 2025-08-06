import numpy as np
import pandas_ta as ta
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
        index = -1  # 修改：始終用最新交易日
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:  # 修改：只開盤不完整 fallback -2
            index = -2
        close = data['Close'].iloc[index]  # 修改：用 index 計算 close
        sma5 = data['SMA5'].iloc[index]
        sma20 = data['SMA20'].iloc[index]
        sma50 = data['SMA50'].iloc[index]
        atr = data['ATR'].iloc[index]  # 新增：使用 ATR 作為波動緩衝
        
        # 修改：移除 Debug print
        
        if np.isnan(sma5) or np.isnan(sma20) or np.isnan(sma50) or np.isnan(atr):  # 修改：加 ATR isnan 檢查
            return 0.0, "Invalid SMA or ATR data", None
        if close > sma20 and sma20 > sma50:  # 修改：放寬為 close > SMA20 > SMA50 得滿分
            return 1.0, "Fully met: Close above SMA20 and SMA20 above SMA50", None
        elif close > sma5 - atr * 0.5:  # 修改：加入 ATR 調整，給部分分數
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
    """評估 SMA 趨勢。
    輸入: data (pd.DataFrame), market_params (dict)。
    輸出: (score (float), explanation (str), extra (None))。
    """
    try:
        market_params = market_params or {}
        index = -1  # 修改：始終用最新
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        sma5 = data['SMA5'].iloc[index]
        sma20 = data['SMA20'].iloc[index]
        sma50 = data['SMA50'].iloc[index]
        if np.isnan(sma5) or np.isnan(sma20) or np.isnan(sma50):
            return 0.0, "Invalid SMA data", None
        
        # 新增：檢查最近 3 天 SMA5 是否穿越 SMA20 上方（金叉檢測）
        crossover = False
        if len(data) >= 3:
            sma5_prev = data['SMA5'].iloc[index - 1]
            sma20_prev = data['SMA20'].iloc[index - 1]
            if sma5_prev <= sma20_prev and sma5 > sma20:
                crossover = True
        
        if sma5 >= sma20 and sma20 > sma50:  # 修改：放寬為 SMA5 >= SMA20 且 SMA20 > SMA50 得滿分
            if crossover:
                return 1.0, "Fully met: SMA5 >= SMA20, SMA20 > SMA50 with golden cross", None
            return 1.0, "Fully met: SMA5 >= SMA20 and SMA20 > SMA50", None
        elif crossover:  # 修改：金叉檢測給部分分數，即使不完全排列
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
    """評估 RSI 指標。
    輸入: data (pd.DataFrame), market_params (dict)。
    輸出: (score (float), explanation (str), extra (None))。
    """
    try:
        market_params = market_params or {}
        rsi = ta.rsi(data['Close'], length=14)
        index = -1  # 修改：始終用最新
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        rsi_value = rsi.iloc[index]
        
        # 修改：從 market_params 取 market_condition，避免未定義錯誤
        market_condition = market_params.get('market_condition', "Neutral")  # fallback "Neutral" 若無
        if market_condition == "Bullish":
            rsi_high = market_params.get('rsi_high', 70)
            rsi_low = market_params.get('rsi_low', 40)
        elif market_condition == "Bearish":
            rsi_high = market_params.get('rsi_high', 60)
            rsi_low = market_params.get('rsi_low', 40)
        else:
            rsi_high = market_params.get('rsi_high', 65)
            rsi_low = market_params.get('rsi_low', 40)
        
        if rsi_value > rsi_high:
            return 0.0, "Overbought", None
        elif rsi_value < rsi_low:
            return 0.0, "Oversold", None
        else:
            return 1.0, "Neutral", None
    except Exception as e:
        warnings.warn(f"Error in condition_c_rsi: {str(e)}")
        return 0.0, "Error occurred", None

def condition_d_bollinger_bands(data, market_params=None):
    """評估 Bollinger Bands。
    """
    try:
        market_params = market_params or {}
        index = -1
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        close = data['Close'].iloc[index]
        upper_bb = data['UpperBB'].iloc[index]
        lower_bb = data['LowerBB'].iloc[index]
        middle_bb = data['MiddleBB'].iloc[index]
        if np.isnan(upper_bb) or np.isnan(lower_bb) or np.isnan(middle_bb):
            return 0.0, "Invalid BB data", None
        if close > upper_bb:
            return 0.0, "Above upper band", None
        elif close < lower_bb:
            return 0.0, "Below lower band", None
        elif close > middle_bb:
            return 1.0, "Above middle band", None
        return 0.5, "Below middle band", None
    except Exception as e:
        warnings.warn(f"Error in condition_d_bollinger_bands: {str(e)}")
        return 0.0, "Error occurred", None

def condition_e_volume(data, market_params=None):
    """評估成交量與 K線。
    輸出: score, explanation, k_line_pattern
    """
    try:
        market_params = market_params or {}
        index = -1
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        volume = data['Volume'].iloc[index]
        open_price = data['Open'].iloc[index]
        close = data['Close'].iloc[index]
        high = data['High'].iloc[index]
        low = data['Low'].iloc[index]
        vma = ta.sma(data['Volume'], length=Config.VMA_PERIOD).iloc[index]
        
        if np.isnan(volume) or np.isnan(vma):
            return 0.0, "Invalid volume data", "Unknown"
        
        k_line_pattern = "Unknown"
        if abs(open_price - close) < (high - low) * 0.1:
            k_line_pattern = "Doji"
        elif close > open_price and (high - low) > (close - open_price) * 3:
            k_line_pattern = "Bullish Candle"
        elif close < open_price and (high - low) > (open_price - close) * 3:
            k_line_pattern = "Bearish Candle"
        
        if volume > vma * 1.5:
            return 1.0, "High volume", k_line_pattern
        elif volume > vma:
            return 0.75, "Moderate volume", k_line_pattern
        elif volume < vma * 0.5:
            return 0.0, "Low volume", k_line_pattern
        return 0.5, "Average volume", k_line_pattern
    except Exception as e:
        warnings.warn(f"Error in condition_e_volume: {str(e)}")
        return 0.0, "Error occurred", "Unknown"

def condition_f_macd(data, market_params=None):
    """評估 MACD 指標。
    """
    try:
        market_params = market_params or {}
        index = -1
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        macd = data['MACD'].iloc[index]
        macd_signal = data['MACDSignal'].iloc[index]
        macd_prev = data['MACD'].iloc[index - 1]
        macd_signal_prev = data['MACDSignal'].iloc[index - 1]
        if np.isnan(macd) or np.isnan(macd_signal) or np.isnan(macd_prev) or np.isnan(macd_signal_prev):
            return 0.0, "Invalid MACD data", None
        
        crossover = macd_prev < macd_signal_prev and macd > macd_signal
        if macd > macd_signal and crossover:
            return 1.0, "Bullish crossover", None
        elif macd > macd_signal:
            return 0.75, "Bullish", None
        elif macd < macd_signal and macd_prev > macd_signal_prev:
            return 0.0, "Bearish crossover", None
        return 0.5, "Neutral", None
    except Exception as e:
        warnings.warn(f"Error in condition_f_macd: {str(e)}")
        return 0.0, "Error occurred", None

def condition_g_obv(data, market_params=None):
    """評估 OBV 指標。
    """
    try:
        market_params = market_params or {}
        index = -1
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        obv = data['OBV'].iloc[index]
        obv_sma = data['OBV_SMA'].iloc[index]
        if np.isnan(obv) or np.isnan(obv_sma):
            return 0.0, "Invalid OBV data", None
        if obv > obv_sma:
            return 1.0, "Rising OBV", None
        return 0.0, "Falling OBV", None
    except Exception as e:
        warnings.warn(f"Error in condition_g_obv: {str(e)}")
        return 0.0, "Error occurred", None

def condition_h_adx(data, market_params=None):
    """評估 ADX 指標。
    """
    try:
        market_params = market_params or {}
        index = -1
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        adx = data['ADX'].iloc[index]
        plus_di = data['PLUS_DI'].iloc[index]
        minus_di = data['MINUS_DI'].iloc[index]
        if np.isnan(adx) or np.isnan(plus_di) or np.isnan(minus_di):
            return 0.0, "Invalid ADX data", None
        if adx > 25 and plus_di > minus_di:
            return 1.0, "Strong trend", None
        elif adx > 25:
            return 0.5, "Strong trend but negative DI", None
        return 0.0, "Weak trend", None
    except Exception as e:
        warnings.warn(f"Error in condition_h_adx: {str(e)}")
        return 0.0, "Error occurred", None

def condition_i_cci(data, market_params=None):
    """評估 CCI 指標。
    """
    try:
        market_params = market_params or {}
        index = -1
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        cci = data['CCI'].iloc[index]
        if np.isnan(cci):
            return 0.0, "Invalid CCI data", None
        if cci > 100:
            return 0.0, "Overbought", None
        elif cci < -100:
            return 0.0, "Oversold", None
        return 1.0, "Neutral", None
    except Exception as e:
        warnings.warn(f"Error in condition_i_cci: {str(e)}")
        return 0.0, "Error occurred", None

def condition_j_stoch(data, market_params=None):
    """評估 Stochastic Oscillator。
    """
    try:
        market_params = market_params or {}
        index = -1
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        slowk = data['SlowK'].iloc[index]
        slowd = data['SlowD'].iloc[index]
        if np.isnan(slowk) or np.isnan(slowd):
            return 0.0, "Invalid Stochastic data", None
        if slowk > slowd and slowk < 20:
            return 1.0, "Oversold crossover", None
        elif slowk < slowd and slowk > 80:
            return 0.0, "Overbought crossover", None
        return 0.5, "Neutral", None
    except Exception as e:
        warnings.warn(f"Error in condition_j_stoch: {str(e)}")
        return 0.0, "Error occurred", None

def condition_k_atr(data, market_params=None):
    """評估 ATR。
    """
    try:
        market_params = market_params or {}
        index = -1
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        atr = data['ATR'].iloc[index]
        close = data['Close'].iloc[index]
        if np.isnan(atr):
            return 0.0, "Invalid ATR data", None
        if atr < close * 0.01:
            return 1.0, "Low ATR volatility", None
        elif atr < close * 0.02:
            return 0.5, "Moderate ATR volatility", None
        return 0.0, "High ATR volatility", None
    except Exception as e:
        warnings.warn(f"Error in condition_k_atr: {str(e)}")
        return 0.0, "Error occurred", None

def condition_m_mfi(data, market_params=None):
    """評估 MFI。
    """
    try:
        market_params = market_params or {}
        index = -1
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        mfi = data['MFI'].iloc[index]
        if np.isnan(mfi):
            return 0.0, "Invalid MFI data", None
        if mfi > 80:
            return 0.0, "Overbought", None
        elif mfi < 20:
            return 0.0, "Oversold", None
        return 1.0, "Neutral", None
    except Exception as e:
        warnings.warn(f"Error in condition_m_mfi: {str(e)}")
        return 0.0, "Error occurred", None

def condition_o_volatility(data, market_params=None):
    """評估波動率。
    """
    try:
        market_params = market_params or {}
        index = -1
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        volatility20 = data['Volatility20'].iloc[index]
        volatility60 = data['Volatility60'].iloc[index]
        close = data['Close'].iloc[index]
        if np.isnan(volatility20) or np.isnan(volatility60):
            return 0.0, "Invalid volatility data", None
        if volatility20 < volatility60 and close > 0:
            return 1.0, "Low volatility", None
        elif volatility20 < volatility60:
            return 0.5, "Moderate volatility with non-negative close", None
        return 0.0, "High volatility or negative close", None
    except Exception as e:
        warnings.warn(f"Error in condition_o_volatility: {str(e)}")
        return 0.0, "Error occurred", None

def condition_p_cmf(data, market_params=None):  # 修改：加 market_params=None
    """評估 CMF。
    """
    try:
        market_params = market_params or {}
        index = -1  # 修改：始終用最新
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
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

def condition_r_var(data, market_params=None):  # 修改：加 market_params=None
    """計算 95% VaR 風險。
    輸入: data (pd.DataFrame)。
    輸出: (score (float), explanation (str), extra (None))。
    """
    try:
        market_params = market_params or {}
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
        index = -1  # 修改：始終用最新
        if is_market_open(data) and data['Volume'].iloc[-1] == 0:
            index = -2
        sma200 = ta.sma(data['Close'], length=200).iloc[index]
        bb_upper = data['UpperBB'].iloc[index]
        forward_pe = yf.Ticker(ticker).info.get('forwardPE', np.nan)
        
        if not np.isnan(forward_pe) and forward_pe > 50:
            score += 10
            crash_triggers.append("High forward P/E (+10)")
        
        if len(macd) >= 2 and macd[index] > macd[index-1] and close[index] < close[index-1]:
            score += 15
            crash_triggers.append("MACD bullish divergence but price down (+15)")
        
        vma = ta.sma(data['Volume'], length=Config.VMA_PERIOD).iloc[index]
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