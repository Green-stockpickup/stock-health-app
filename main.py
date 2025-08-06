from colorama import init, Fore, Style
import warnings
import pandas as pd
from data_fetch import get_stock_data, get_market_condition
from health_score import calculate_health_score, get_score_color
from models import predict_price_direction
from conditions import condition_l_crash_risk, condition_e_volume
from config import Config, is_market_open
import re
import unicodedata  # 用於計算中文字顯示寬度

from backtest import backtest_strategy

init()
warnings.filterwarnings("ignore")

def main():
    print("🚀 Stock Health Monitor v3.9 (With Enhanced Price Prediction and Target/Stop Loss)")
    print("📝 Please enter valid US stock tickers (e.g., AAPL,NVDA) or single ticker")
    print("📌 Press Enter or type 'quit' to exit")
    print("📌 To run backtest, prefix with 'backtest' e.g., 'backtest AAPL'")
    
    market_condition, market_params = get_market_condition()
    
    while True:
        input_str = input("\nPlease enter the stock ticker(s) (comma-separated, or 'quit' to exit): ").strip().upper()
        if not input_str or input_str.lower() == 'quit':
            print(f"{Fore.CYAN}👋 Exiting Stock Health Monitor. Goodbye!{Style.RESET_ALL}")
            break
        
        lower_input = input_str.lower()
        do_backtest = lower_input.startswith('backtest')
        if do_backtest:
            lower_input = lower_input.replace('backtest', '').strip()
            tickers = [t.strip().upper() for t in lower_input.split(',')]
        else:
            tickers = [t.strip() for t in input_str.split(',')]
        
        if len(tickers) > 10:
            print(f"{Fore.YELLOW}⚠️ Batch limited to 10 tickers. Processing first 10.{Style.RESET_ALL}")
            tickers = tickers[:10]
        
        results = []
        for ticker in tickers:
            data = get_stock_data(ticker)
            if data is None:
                print(f"{Fore.RED}❌ Unable to process {ticker}. Skipping.{Style.RESET_ALL}")
                continue
            
            if do_backtest:
                backtest_report = backtest_strategy(data, ticker, days=90, health_score_min=80, trend_prob_min=75, confidence_min=80)
                print(f"\n🔍 Backtest Report: {backtest_report}")
            
            total_health_score, bonus_score, category_scores, category_explanations, bonus_details, warnings_list, trend_prob, dt_adjustment, tree_explanation, target, stop_loss, percentages, condition_scores = calculate_health_score(data, market_params, ticker, market_condition)
            
            price_direction, direction_explanation, shap_explanation = predict_price_direction(data, ticker, total_health_score)
            
            price = data['Close'].iloc[-1]
            atr = data['ATR'].iloc[-1]
            if price_direction == "Bullish":
                target = price + atr * 2
                stop_loss = price - atr * 1
                if price > target:
                    second_target = price + atr * 3
                    target = second_target
            elif price_direction == "Bearish":
                target = price - atr * 2
                stop_loss = price + atr * 1
            else:
                target = price
                stop_loss = price
            
            print(f"\n📊 Stock Health Report ({ticker} - ${price:.2f})")
            print(f"🔍 Market Condition: {market_condition}")
            print(f"📈 Price Direction Prediction:")
            if price_direction:
                print(f"  - {direction_explanation}")
                if shap_explanation:
                    print(f"  - Top Features: {', '.join(shap_explanation)}")
            else:
                print(f"  - Unable to predict: {direction_explanation}")
            print(f"{get_score_color(total_health_score, 100)}💎 Total Health Score: {int(round(total_health_score))} / 100{Style.RESET_ALL}")
            if dt_adjustment != 0:
                print(f"  (Decision Tree Adjustment: {dt_adjustment:+.1f})")
            print(f"{get_score_color(bonus_score, Config.BONUS_MAX)}⭐ Bonus Score: {int(round(bonus_score))} / {int(Config.BONUS_MAX)}{Style.RESET_ALL}")
            print(f"🚀 目前趨勢持續性: {get_trend_prob_color(trend_prob)}{trend_prob:+.2f}%{Style.RESET_ALL}")
            
            # 新增反轉信號輸出，縮進為子項目 (4 個字符)
            print(f"    - K線反轉: {get_k_line_reversal(data, condition_e_volume)}")
            print(f"    - 背離: {get_divergence(data, condition_scores)}")
            print(f"    - 突破: {get_breakout(data, condition_scores)}")
            print(f"    - 波動率變化: {get_volatility_change(data, condition_scores)}")
            print(f"    - 成交量/資金流變化: {get_volume_flow_change(data, condition_scores)}")
            print(f"    - 趨勢強度衰退: {get_trend_strength_decay(data, condition_scores)}")
            print()  # 與 Category Scores 隔一行
            
            print(f"Category Scores:")
            for category in percentages:
                print(f"  {get_score_color(percentages[category], 100)} {category}: {percentages[category]}%{Style.RESET_ALL}")
            print()  # 添加空行以分隔
            
            print(f"\n🔍 Conditions Evaluated:")
            for category, explanations in category_explanations.items():
                if category == "Trend":
                    print(f"  {category} Conditions:")
                    print(f"  (滿分要求強趨勢與動能正)")
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
                        status = f"{Fore.GREEN}✅{Style.RESET_ALL}" if score == 1.0 else f"{Fore.YELLOW}⚠️{Style.RESET_ALL}" if 0.25 <= score < 1.0 else f"{Fore.RED}❌{Style.RESET_ALL}"
                        print(f"    - {name}: {status} ({explanation})")
                    print()  # 空行分隔
                elif category == "Momentum":
                    print(f"  {category} Conditions:")
                    print(f"  (滿分要求指標在看漲區無背離)")
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
                        status = f"{Fore.GREEN}✅{Style.RESET_ALL}" if score == 1.0 else f"{Fore.YELLOW}⚠️{Style.RESET_ALL}" if 0.25 <= score < 1.0 else f"{Fore.RED}❌{Style.RESET_ALL}"
                        print(f"    - {name}: {status} ({explanation})")
                    print()  # 空行分隔
                elif category == "Volatility":
                    print(f"  {category} Conditions:")
                    print(f"  (滿分要求低波動率與風險極低)")
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
                        status = f"{Fore.GREEN}✅{Style.RESET_ALL}" if score == 1.0 else f"{Fore.YELLOW}⚠️{Style.RESET_ALL}" if 0.25 <= score < 1.0 else f"{Fore.RED}❌{Style.RESET_ALL}"
                        print(f"    - {name}: {status} ({explanation})")
                    print()  # 空行分隔
                elif category == "Volume/Flow":
                    print(f"  {category} Conditions:")
                    print(f"  (滿分要求強資金流入與高成交量支持)")
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
                        status = f"{Fore.GREEN}✅{Style.RESET_ALL}" if score == 1.0 else f"{Fore.YELLOW}⚠️{Style.RESET_ALL}" if 0.25 <= score < 1.0 else f"{Fore.RED}❌{Style.RESET_ALL}"
                        print(f"    - {name}: {status} ({explanation})")
                    print()  # 空行分隔
                elif category == "Risk":
                    print(f"  {category} Conditions:")
                    print(f"  (滿分要求低崩盤風險)")
                    sorted_explanations = [
                        (name, score, explanation) for name, score, explanation in explanations
                        if name == "Crash Risk (PE, GARCH, VaR)"
                    ]
                    for name, score, explanation in sorted_explanations:
                        status = f"{Fore.GREEN}✅{Style.RESET_ALL}" if score == 1.0 else f"{Fore.YELLOW}⚠️{Style.RESET_ALL}" if 0.25 <= score < 1.0 else f"{Fore.RED}❌{Style.RESET_ALL}"
                        print(f"    - {name}: {status} ({explanation})")
                    print()  # 空行分隔
            
            print(f"\n💥 Crash Risk Warning:")
            crash_score, crash_explanation, crash_index, crash_triggers, crash_warnings = condition_l_crash_risk(data, market_params, ticker, market_condition)
            print(f"  - Crash Risk: {crash_explanation}")
            print("  - Triggered by: " + ", ".join(crash_triggers))
            
            print(f"\n‼️ Warnings:")
            all_warnings = warnings_list + crash_warnings
            if all_warnings:
                for warning in all_warnings:
                    print(f"  - {Fore.YELLOW}{warning}{Style.RESET_ALL}")
            else:
                print("None")
            
            print(f"\n⭐ Bonus Scores:")
            print("  - " + " | ".join(bonus_details) if bonus_details else "No bonuses")
            
            print(f"\n🎯 Target: {target:.2f}")
            print(f"🛑 Stop Loss: {stop_loss:.2f}")
            
            print(f"Data Retrieved Up To: {data.index[-1].strftime('%Y-%m-%d')}")
            
            direction_en = direction_explanation.replace("Bearish", "Bearish").replace("Bullish", "Bullish").replace("Confidence", "").replace("信心度: ", "").replace(" (: ", " (").replace(":", "").replace("Suggestion: ", "").replace(", 建議: ", "").replace("建議", "").strip()
            confidence_match = re.search(r'\((\d+\.\d+)%\)', direction_en)
            if confidence_match:
                confidence = round(float(confidence_match.group(1).replace('%', '')))
                direction_en = re.sub(r'\(\d+\.\d+%\)', f'({confidence}%)', direction_en)
            direction_en = re.sub(r',.*', '', direction_en).strip()
            
            if "Bearish" in direction_en:
                direction_colored = Fore.RED + direction_en + Style.RESET_ALL
            elif "Bullish" in direction_en:
                direction_colored = Fore.GREEN + direction_en + Style.RESET_ALL
            else:
                direction_colored = Fore.YELLOW + direction_en + Style.RESET_ALL
            
            health_score_colored = get_score_color(total_health_score, 100) + str(int(round(total_health_score))) + Style.RESET_ALL
            trend_prob_colored = get_trend_prob_color(trend_prob) + f"{trend_prob:+.2f}%" + Style.RESET_ALL
            
            warnings_status = Fore.YELLOW + "Yes" + Style.RESET_ALL if all_warnings else ""
            
            suggestion = direction_explanation.split("Suggestion: ")[1].strip() if "Suggestion: " in direction_explanation else "Neutral"
            
            if "Hold" in suggestion:
                suggestion_colored = Fore.GREEN + suggestion + Style.RESET_ALL
            elif "Sell Short" in suggestion:
                suggestion_colored = Fore.RED + suggestion + Style.RESET_ALL
            else:
                suggestion_colored = Fore.YELLOW + suggestion + Style.RESET_ALL
            
            volume_score, volume_explanation, k_line_pattern = condition_e_volume(data)
            volume_output = 'Bullish' if k_line_pattern == "Bullish Candle" and volume_score >= 0.75 else ''
            if volume_output:
                volume_colored = Fore.GREEN + volume_output + Style.RESET_ALL
            else:
                volume_colored = volume_output
            
            results.append({
                'Ticker': ticker,
                'Price': f"{price:.2f}",
                'Direction': direction_colored if price_direction else "N/A",
                'Attention': volume_colored,
                'Suggestion': suggestion_colored,
                'Warnings': warnings_status,
                'Up Trend': get_score_color(percentages['Trend'], 100) + f"{percentages['Trend']}%" + Style.RESET_ALL,
                'Momentum': get_score_color(percentages['Momentum'], 100) + f"{percentages['Momentum']}%" + Style.RESET_ALL,
                'Vol Risk': get_score_color(percentages['Volatility'], 100) + f"{percentages['Volatility']}%" + Style.RESET_ALL,
                'Flow In': get_score_color(percentages['Volume/Flow'], 100) + f"{percentages['Volume/Flow']}%" + Style.RESET_ALL,
                'Crash Risk': get_score_color(percentages['Risk'], 100) + f"{percentages['Risk']}%" + Style.RESET_ALL
            })
        
        if len(results) > 1:
            print("\n📊 Batch Comparison:")
            def calculate_display_width(text):
                return sum(2 if unicodedata.east_asian_width(c) in ('F', 'W', 'A') else 1 for c in text)
            
            headers = [
                ("Ticker", calculate_display_width("Ticker") + 4, "center"),
                ("Price", calculate_display_width("Price") + 4, "center"),
                ("Direction", calculate_display_width("Direction") + 4, "center"),
                ("Attention", calculate_display_width("Attention") + 4, "center"),
                ("Suggestion", calculate_display_width("Suggestion") + 4, "center"),
                ("Warnings", calculate_display_width("Warnings") + 4, "center"),
                ("Up Trend", calculate_display_width("Up Trend") + 4, "center"),
                ("Momentum", calculate_display_width("Momentum") + 4, "center"),
                ("Vol Risk", calculate_display_width("Vol Risk") + 4, "center"),
                ("Flow In", calculate_display_width("Flow In") + 4, "center"),
                ("Crash Risk", calculate_display_width("Crash Risk") + 4, "center")
            ]
            
            max_data_widths = {}
            for result in results:
                for key in result:
                    value = str(result[key])
                    clean_value = re.sub(r'\x1b\[[0-9;]*m', '', value)
                    data_width = calculate_display_width(clean_value) + 2
                    if key not in max_data_widths or data_width > max_data_widths[key]:
                        max_data_widths[key] = data_width
            
            header_widths = []
            for name, base_width, align in headers:
                key = name
                final_width = max(base_width, max_data_widths.get(key, 0))
                header_widths.append((name, final_width, align))
            
            header_line = ""
            for name, width, align in header_widths:
                header_line += f"{name:^{width}}"
            print(header_line)
            print("-" * sum(width for _, width, _ in header_widths))
            
            for result in results:
                line = ""
                for (name, width, align), key in zip(header_widths, result.keys()):
                    value = str(result[key])
                    clean_value = re.sub(r'\x1b\[[0-9;]*m', '', value)
                    display_width = calculate_display_width(clean_value)
                    padding_adjust = len(value) - display_width
                    line += f"{value:^{width + padding_adjust}}"
                print(line)

def get_k_line_reversal(data, condition_e_volume):
    _, _, k_line_pattern = condition_e_volume(data, None)
    if k_line_pattern == "Doji":
        return "Doji"
    else:
        return "無"

def get_divergence(data, condition_scores):
    rsi_score, rsi_explanation, _ = condition_scores.get("RSI (14-day)", (0.0, "無", None))
    macd_score, macd_explanation, _ = condition_scores.get("MACD (12,26,9)", (0.0, "無", None))
    cci_score, cci_explanation, _ = condition_scores.get("CCI (20-day)", (0.0, "無", None))
    index = -1 if not is_market_open(data) else -2
    close = data['Close'].iloc[index]
    close_prev = data['Close'].iloc[index - 1]
    if close > close_prev and (rsi_score < 1.0 or macd_score < 1.0 or cci_score < 1.0):
        return "價格創新高但指標低，頂部反轉"
    elif close < close_prev and (rsi_score > 0.0 or macd_score > 0.0 or cci_score > 0.0):
        return "價格創新低但指標高，底部反轉"
    return "無"

def get_breakout(data, condition_scores):
    sma_score, sma_explanation, _ = condition_scores.get("SMA Trend (SMA5 vs SMA20, SMA50)", (0.0, "無", None))
    k_line_score, k_line_explanation, _ = condition_scores.get("K-line Trend (Close vs SMA20, SMA50 with ATR buffer)", (0.0, "無", None))
    if sma_score == 0.0:
        return "死叉頂部反轉"
    elif sma_score == 1.0:
        return "金叉為底部反轉"
    elif k_line_score == 1.0:
        return "價格突破趨勢線"
    return "無"

def get_volatility_change(data, condition_scores):
    bollinger_score, bollinger_explanation, _ = condition_scores.get("Bollinger Bands (Close vs Upper/Middle/Lower)", (0.0, "無", None))
    atr_score, atr_explanation, _ = condition_scores.get("ATR (14-day)", (0.0, "無", None))
    if bollinger_score == 1.0 and "upper band" in bollinger_explanation:
        return "Bollinger Bands 收斂後突破上軌底部反轉"
    elif bollinger_score == 0.0 and "lower band" in bollinger_explanation:
        return "Bollinger Bands 收斂後突破下軌底部反轉或頂部調整結束"
    elif atr_score == 0.0:
        return "ATR 激增不穩"
    return "無"

def get_volume_flow_change(data, condition_scores):
    volume_score, volume_explanation, _ = condition_scores.get("Volume and K-line", (0.0, "無", None))
    obv_score, obv_explanation, _ = condition_scores.get("OBV (On-Balance Volume)", (0.0, "無", None))
    cmf_score, cmf_explanation, _ = condition_scores.get("CMF (Chaikin Money Flow)", (0.0, "無", None))
    if volume_score == 1.0 and "low volume" in volume_explanation:
        return "Volume 萎縮後放大反轉"
    elif obv_score < 1.0 or cmf_score < 1.0:
        return "OBV/CMF 背離價格轉折"
    return "無"

def get_trend_strength_decay(data, condition_scores):
    adx_score, adx_explanation, _ = condition_scores.get("ADX (14-day)", (0.0, "無", None))
    cci_score, cci_explanation, _ = condition_scores.get("CCI (20-day)", (0.0, "無", None))
    stoch_score, stoch_explanation, _ = condition_scores.get("Stochastic Oscillator (14,3,3)", (0.0, "無", None))
    if adx_score == 0.0:
        return "ADX 從高降低趨勢結束"
    elif cci_score == 0.5 or stoch_score == 0.5:
        return "CCI/Stochastic 從極端返回反轉"
    return "無"

def get_trend_prob_color(prob):
    if prob >= 50:
        return Fore.GREEN
    elif prob <= -50:
        return Fore.RED
    else:
        return Fore.YELLOW

if __name__ == "__main__":
    main()