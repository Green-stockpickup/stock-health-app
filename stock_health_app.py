import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import warnings
import datetime
import pytz
import os
import pickle
import unicodedata
from arch import arch_model
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from health_score import calculate_health_score
from models import predict_price_direction
from conditions import condition_l_crash_risk
from config import Config, is_market_open

warnings.filterwarnings("ignore")

def get_score_color(score, max_score):
    """Get color code for score display (返回空字串以移除顏色)."""
    return ""

def get_trend_prob_color(prob):
    """Get color code for trend probability display (返回空字串以移除顏色)."""
    return ""

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

def calculate_indicators(data):
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Missing required data columns")
    
    data = data.copy()
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
    
    try:
        returns = data['Close'].pct_change().dropna() * 100
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=True)
        garch_fit = model.fit(disp='off')
        data['GARCH_Volatility'] = garch_fit.conditional_volatility.shift(-len(returns) + len(data))
    except Exception as e:
        warnings.warn(f"GARCH calculation failed: {str(e)}. Falling back to Volatility20.")
        data['GARCH_Volatility'] = data['Volatility20']
    
    data['CMF'] = ta.adosc(data['High'], data['Low'], data['Close'], data['Volume'], fast=3, slow=10) / data['Volume'].rolling(Config.CMF_PERIOD).sum()
    
    data['Lag_Close_1'] = data['Close'].shift(1)
    data['Rolling_Mean_5'] = data['Close'].rolling(window=5).mean()
    
    data = data.ffill().bfill()
    return data

def get_stock_data(ticker, period="3y", interval="1d"):
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
                    data = data.reset_index(drop=False)
                    data = data.drop_duplicates(subset='Date')
                    data = data.set_index('Date')
                    data = calculate_indicators(data)
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
            data = data.reset_index(drop=False)
            data = data.drop_duplicates(subset='Date')
            data = data.set_index('Date')
            data = calculate_indicators(data)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        
        nan_ratio = data.isnull().mean().mean()
        if nan_ratio > 0.2:
            warnings.warn(f"High NaN ratio ({nan_ratio:.2f}) in data for {ticker}, data may be unreliable.")
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
        close = data['Close']
        sma_200 = ta.sma(close, length=200)
        rsi_14 = ta.rsi(close, length=14)
        index = -1 if is_market_open() else -2
        if np.isnan(sma_200.iloc[index]) or np.isnan(rsi_14.iloc[index]):
            return "Unknown", {}
        if close.iloc[index] > sma_200.iloc[index] and rsi_14.iloc[index] > 50:
            return "Bullish", {"rsi_high": 70, "rsi_low": 40, "macd_relaxed": True, "cci_high": 150, "stoch_high": 85, "crash_rsi": 75}
        elif close.iloc[index] < sma_200.iloc[index] and rsi_14.iloc[index] < 50:
            return "Bearish", {"rsi_high": 60, "rsi_low": 40, "macd_relaxed": False, "cci_high": 100, "stoch_high": 80, "crash_rsi": 70}
        return "Neutral", {"rsi_high": 65, "rsi_low": 40, "macd_relaxed": False, "cci_high": 100, "stoch_high": 80, "crash_rsi": 70}
    except Exception as e:
        warnings.warn(f"Error determining market condition: {str(e)}")
        return "Unknown", {}

def main():
    st.set_page_config(page_title="Stock Health Monitor", layout="wide")
    st.title("🚀 Stock Health Monitor v3.9 (With Enhanced Price Prediction)")
    st.write("📝 請輸入美國股票代碼（例如：AAPL,NVDA）或單一股票代碼")
    st.write("📌 按 Enter 或輸入 'quit' 退出")
    
    market_condition, market_params = get_market_condition()
    
    ticker_input = st.text_input("輸入股票代碼（多個代碼用逗號分隔）：", placeholder="例如：AAPL,NVDA")
    
    if ticker_input:
        tickers = [t.strip().upper() for t in ticker_input.split(',')]
        if len(tickers) > 10:
            st.warning("⚠️ 每次最多處理 10 個股票代碼，將處理前 10 個")
            tickers = tickers[:10]
        
        for ticker in tickers:
            data = get_stock_data(ticker)
            if data is None:
                st.error(f"❌ 無法處理 {ticker}，請檢查代碼或數據")
                continue
            
            total_health_score, bonus_score, category_scores, category_explanations, bonus_details, warnings_list, trend_prob, dt_adjustment, tree_explanation, target, stop_loss, percentages, condition_scores = calculate_health_score(data, market_params, ticker, market_condition)
            
            price_direction, direction_explanation, shap_explanation = predict_price_direction(data, ticker, total_health_score)
            
            st.header(f"📊 股票健康報告 ({ticker} - ${data['Close'].iloc[-1]:.2f})")
            st.write(f"🔍 市場狀況: {market_condition}")
            
            st.subheader("📈 價格方向預測")
            if price_direction:
                st.write(f"- {direction_explanation}")
                if shap_explanation:
                    st.write(f"- 主要影響因子: {', '.join(shap_explanation)}")
            else:
                st.write(f"- 無法預測: {direction_explanation}")
            
            st.write(f"💎 總健康分數: {int(round(total_health_score))} / 100")
            if dt_adjustment != 0:
                st.write(f"  (決策樹調整: {dt_adjustment:+.1f})")
            st.write(f"⭐ 獎勵分數: {int(round(bonus_score))} / {int(Config.BONUS_MAX)}")
            st.write(f"🚀 目前趨勢持續性: {trend_prob:+.2f}%")
            
            st.write(f"🎯 目標價: ${target:.2f}")
            st.write(f"🛑 止損價: ${stop_loss:.2f}")
            
            st.subheader("Category Scores")
            for category in percentages:
                st.write(f"  {category}: {percentages[category]}%")
            
            st.subheader("🔍 評估條件")
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

if __name__ == "__main__":
    main()