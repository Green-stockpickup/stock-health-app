from colorama import Fore, Style
from config import Config
from conditions import *  # Import all conditions
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np
import warnings
import talib
from indicators import calculate_target_stop_loss  # 新增匯入

def calculate_health_score(data, market_params=None, ticker="", market_condition="Unknown"):
    """計算股票健康分數，整合所有條件與獎勵邏輯。
    輸入: data (pd.DataFrame) - 股票歷史數據; market_params (dict) - 市場參數; ticker (str) - 股票代碼; market_condition (str) - 市場條件。
    輸出: total_health_score (float) - 總健康分數; bonus_score (float) - 獎勵分數; category_scores (dict) - 類別分數; 
          category_explanations (dict) - 類別解釋; bonus_details (list) - 獎勵細節; warnings_list (list) - 警告列表; 
          trend_prob (float) - 趨勢持續概率 (帶符號: +向上, -向下); dt_adjustment (float) - 決策樹調整; tree_explanation (str) - 決策樹解釋; 
          target (float) - 目標價; stop_loss (float) - 止損價。
    """
    if data is None or data.empty:
        return 0.0, 0.0, {}, {}, [], [], 0.0, 0.0, "", 0.0, 0.0  # 新增Target及Stop Loss預設值
    
    category_mapping = {
        "Trend": [
            ("K-line Trend (Close vs SMA5, SMA20, SMA50)", condition_a_trend_strength),
            ("SMA Trend (SMA5 vs SMA20, SMA50)", condition_b_sma_trend),
            ("MACD (12,26,9)", condition_f_macd),
            ("ADX (14-day)", condition_h_adx)
        ],
        "Volume": [
            ("Volume and K-line", condition_e_volume),
            ("OBV (On-Balance Volume)", condition_g_obv),
            ("ATR (14-day)", condition_k_atr),
            ("MFI (14-day)", condition_m_mfi),
            ("Historical Volatility (20-day vs 60-day)", condition_o_volatility),
            ("CMF (Chaikin Money Flow)", condition_p_cmf)
        ],
        "Momentum": [
            ("RSI (14-day)", condition_c_rsi),
            ("Bollinger Bands (Close vs Upper/Middle/Lower)", condition_d_bollinger_bands),
            ("CCI (20-day)", condition_i_cci),
            ("Stochastic Oscillator (14,3,3)", condition_j_stoch)
        ]
    }
    
    category_scores = {}
    category_explanations = {}
    bonus_score = 0.0
    bonus_details = []
    warnings_list = []
    
    condition_scores = {}
    for category, cond_list in category_mapping.items():
        total_score = 0.0
        explanations = []
        for name, cond_func in cond_list:
            # 統一 unpack 到3個值，無論是否特殊條件
            if name in ["RSI (14-day)", "MACD (12,26,9)", "CCI (20-day)", "Stochastic Oscillator (14,3,3)"]:
                score, explanation, extra = cond_func(data, market_params)
            else:
                score, explanation, extra = cond_func(data)
            condition_scores[name] = (score, explanation, extra)
            total_score += score
            explanations.append((name, score, explanation))
        avg_score = total_score / len(cond_list)
        category_score = avg_score * 100 * Config.CATEGORY_WEIGHTS[category]
        category_scores[category] = round(category_score)
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
        bonus_score += 0.5
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
    crash_score, _, _, _, _ = condition_l_crash_risk(data, market_params, ticker, market_condition)
    if crash_score >= Config.RISK_THRESHOLD:
        warnings_list.append(f"高轉勢風險：基於高波動預測及潛在指標背離，建議密切監測。")
    if condition_scores.get("CMF (Chaikin Money Flow)", (0, ""))[0] < 0.5 and condition_scores["Volume and K-line"][0] < 0.5:
        warnings_list.append(f"中度資金流出風險：成交量中性且CMF負向，可能預示趨勢轉弱。")
    if condition_scores["MACD (12,26,9)"][0] < 1.0 and condition_scores["RSI (14-day)"][0] < 1.0:
        warnings_list.append(f"動能弱化警示：MACD負向及RSI過買/賣，轉勢下跌概率增加。")
    
    # 修改：趨勢持續概率 - 先識別方向，再計算帶符號百分比
    trend_names = [name for name, _ in category_mapping["Trend"]]
    trend_avg = sum([condition_scores.get(name, (0, ""))[0] for name in trend_names]) / len(trend_names) if trend_names else 0.0
    if trend_avg > 0.5:
        # 向上趨勢：計算持續機會 (+0% 到 +100%)
        trend_prob = (trend_avg * 2 - 1) * 100  # e.g., avg=0.75 → (1.5-1)*100 = +50%
    elif trend_avg < 0.5:
        # 向下趨勢：計算持續機會 (-0% 到 -100%)
        trend_prob = (trend_avg * 2 - 1) * 100  # e.g., avg=0.25 → (0.5-1)*100 = -50%
    else:
        trend_prob = 0.0  # 不明確/轉勢
    
    # 新增：計算Target及Stop Loss
    target, stop_loss = calculate_target_stop_loss(data)
    
    return total_health_score, bonus_score, category_scores, category_explanations, bonus_details, warnings_list, trend_prob, dt_adjustment, tree_explanation, target, stop_loss

def get_score_color(score, max_score):
    """根據分數返回顏色代碼，用於輸出顯示。
    輸入: score (float) - 分數; max_score (float) - 最大分數。
    輸出: color (str) - Fore顏色代碼。
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