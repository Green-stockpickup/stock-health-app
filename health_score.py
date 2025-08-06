from colorama import Fore, Style
from config import Config
from conditions import *  # Import all conditions
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np
import warnings
import talib
from indicators import calculate_target_stop_loss

def calculate_health_score(data, market_params=None, ticker="", market_condition="Unknown"):
    """Calculate stock health score, integrating all conditions and bonus logic.
    Input: data (pd.DataFrame) - stock historical data; market_params (dict) - market parameters; ticker (str) - stock ticker; market_condition (str) - market condition.
    Output: total_health_score (float) - total health score; bonus_score (float) - bonus score; category_scores (dict) - category scores; 
          category_explanations (dict) - category explanations; bonus_details (list) - bonus details; warnings_list (list) - warnings list; 
          trend_prob (float) - trend continuity probability (signed: + upward, - downward); dt_adjustment (float) - decision tree adjustment; tree_explanation (str) - decision tree explanation; 
          target (float) - target price; stop_loss (float) - stop loss price; percentages (dict) - category percentages; condition_scores (dict) - sub-condition scores.
    """
    if data is None or data.empty:
        return 0.0, 0.0, {}, {}, [], [], 0.0, 0.0, "", 0.0, 0.0, {}, {}  # 新增 condition_scores 預設 {}
    
    # 修改：重新分配為 5 類 (Trend/Momentum/Volatility/Volume/Flow/Risk), 子標題全英文
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
    percentages = {}  # 新增：用於儲存每個類別的百分比分數
    category_explanations = {}
    bonus_score = 0.0
    bonus_details = []
    warnings_list = []
    
    condition_scores = {}
    for category, cond_list in category_mapping.items():
        total_score = 0.0
        explanations = []
        for name, cond_func in cond_list:
            # 修改：特殊處理 Crash Risk 條件, 加 ticker/market_condition 引數
            if name == "Crash Risk (PE, GARCH, VaR)":
                score, explanation, extra, crash_triggers, crash_warnings = cond_func(data, market_params, ticker, market_condition)  # 修改：unpack 5 個值 (score, explanation, crash_index, crash_triggers, crash_warnings)
                extra = crash_triggers  # 調整 extra 以相容原 3 值 unpack (可依需求調整)
            else:
                if name in ["Volume and K-line"]:  # e_volume 特殊 3 值
                    score, explanation, extra = cond_func(data, market_params)
                else:
                    score, explanation, extra = cond_func(data, market_params)
            condition_scores[name] = (score, explanation, extra)
            total_score += score
            explanations.append((name, score, explanation))
        avg_score = total_score / len(cond_list)
        # 修改：對 Risk 類別反轉 avg_score (低風險滿分1.0, 高風險0.0, 以50為 max_score)
        if category == "Risk":
            avg_score = 1 - (avg_score / 50) if avg_score <= 50 else 0.0  # 修改：反轉邏輯, 並 cap <=50 避免負值脫離
        category_score = avg_score * 100 * Config.CATEGORY_WEIGHTS[category]  # 修改：用新權重計算
        category_scores[category] = round(category_score)
        percentages[category] = round((category_score / (Config.CATEGORY_WEIGHTS[category] * 100)) * 100)  # 新增：計算百分比並 round 到整數
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
    
    # Decision Tree
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
    
    # Enhanced Warnings
    crash_score, crash_explanation, crash_index, crash_triggers, crash_warnings = condition_l_crash_risk(data, market_params, ticker, market_condition)  # 修改：直接 unpack 5 個值
    if crash_score >= Config.RISK_THRESHOLD:
        warnings_list.append(f"High reversal risk: Based on high volatility forecast and potential indicator divergence, monitor closely.")
    if condition_scores.get("CMF (Chaikin Money Flow)", (0, ""))[0] < 0.5 and condition_scores["Volume and K-line"][0] < 0.5:
        warnings_list.append(f"Moderate outflow risk: Neutral volume and negative CMF, may indicate weakening trend.")
    if condition_scores["MACD (12,26,9)"][0] < 1.0 and condition_scores["RSI (14-day)"][0] < 1.0:
        warnings_list.append(f"Momentum weakening alert: Negative MACD and overbought/sold RSI, increased downside reversal probability.")
    
    # 修改：趨勢持續概率 - 先識別方向，再計算帶符號百分比
    trend_names = [name for name, _ in category_mapping["Trend"]]  # 修改：匹配新類別
    trend_avg = sum([condition_scores.get(name, (0, ""))[0] for name in trend_names]) / len(trend_names) if trend_names else 0.0
    if trend_avg > 0.5:
        trend_prob = (trend_avg * 2 - 1) * 100
    elif trend_avg < 0.5:
        trend_prob = (trend_avg * 2 - 1) * 100
    else:
        trend_prob = 0.0
    
    # 原有 target/stop_loss 不動...
    target, stop_loss = calculate_target_stop_loss(data)
    return total_health_score, bonus_score, category_scores, category_explanations, bonus_details, warnings_list, trend_prob, dt_adjustment, tree_explanation, target, stop_loss, percentages, condition_scores  # 新增 return condition_scores

def get_score_color(score, max_score):
    """Get color code based on score for output display.
    Input: score (float) - score; max_score (float) - maximum score.
    Output: color (str) - Fore color code.
    """
    percentage = score / max_score * 100
    if max_score == 100:
        if percentage >= 80:
            return Fore.GREEN
        elif percentage >= 60:
            return Fore.YELLOW
        return Fore.RED
    elif max_score == 25:
        if percentage >= 60:
            return Fore.GREEN
        elif percentage >= 20:
            return Fore.YELLOW
        return Fore.RED
    else:
        if percentage >= 80:
            return Fore.GREEN
        elif percentage >= 50:
            return Fore.YELLOW
        return Fore.RED