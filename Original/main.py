from colorama import init, Fore, Style
import warnings
import pandas as pd
from data_fetch import get_stock_data, get_market_condition
from health_score import calculate_health_score, get_score_color
from models import predict_price_direction
from conditions import condition_l_crash_risk
from config import Config
import re
import unicodedata  # 新增：用於計算中文字顯示寬度

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
        
        # 檢查是否啟動回測：先轉小寫處理
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
                print(f"\n🔍 回測報告: {backtest_report}")
            
            total_health_score, bonus_score, category_scores, category_explanations, bonus_details, warnings_list, trend_prob, dt_adjustment, tree_explanation, target, stop_loss = calculate_health_score(data, market_params, ticker, market_condition)
            
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
            
            print(f"\n📊 Stock Health Report ({ticker})")
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
            print(f"🚀 Trend Continuity Probability: {trend_prob:+.2f}%")
            print(f"Category Scores:")
            print(f"  {get_score_color(category_scores['Trend'], 40)}🚀 Trend ({int(round(category_scores['Trend']))} / 40){Style.RESET_ALL}")
            print(f"  {get_score_color(category_scores['Volume'], 30)}💰 Volume ({int(round(category_scores['Volume']))} / 30){Style.RESET_ALL}")
            print(f"  {get_score_color(category_scores['Momentum'], 30)}🚵 Momentum ({int(round(category_scores['Momentum']))} / 30){Style.RESET_ALL}")
            print(f"\n🔍 Conditions Evaluated:")
            for category, explanations in category_explanations.items():
                print(f"  {category} Conditions:")
                for name, score, explanation in explanations:
                    status = f"{Fore.GREEN}✅{Style.RESET_ALL}" if score == 1.0 else f"{Fore.YELLOW}⚠️{Style.RESET_ALL}" if 0.25 <= score < 1.0 else f"{Fore.RED}❌{Style.RESET_ALL}"
                    print(f"    - {name}: {status} ({explanation})")
            
            print(f"\n⭐ Bonus Scores:")
            print("  - " + " | ".join(bonus_details) if bonus_details else "No bonuses")
            
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
            
            print(f"\n🎯 Target (止盈點): {target:.2f}")
            print(f"🛑 Stop Loss (止損點): {stop_loss:.2f}")
            
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
            
            suggestion = direction_explanation.split("Suggestion: ")[1].strip() if "Suggestion: " in direction_explanation else "觀望"
            
            if "繼續持有" in suggestion:
                suggestion_colored = Fore.GREEN + suggestion + Style.RESET_ALL
            elif "考慮沽空" in suggestion:
                suggestion_colored = Fore.RED + suggestion + Style.RESET_ALL
            else:
                suggestion_colored = Fore.YELLOW + suggestion + Style.RESET_ALL
            
            # 修改：調整字典鍵順序，以匹配新子標題順序
            results.append({
                'Ticker': ticker,
                'Health Score': health_score_colored,
                'Trend Prob (%)': trend_prob_colored,
                'Crash Risk': crash_explanation,
                'Price': f"{price:.2f}",
                'Target': f"{target:.2f}",
                'Stop Loss': f"{stop_loss:.2f}",
                'Direction': direction_colored if price_direction else "N/A",
                'Suggestion': suggestion_colored,
                'Warnings': warnings_status
            })
        
        if len(results) > 1:
            print("\n📊 Batch Comparison:")
            # 修改：定義欄位寬度和對齊方式，增加至少4個空白鍵間距，新順序
            headers = [
                ("Ticker    ", 10 + 4, "left"),
                ("Health Score    ", 12 + 4, "center"),
                ("Trend Prob(%)    ", 15 + 4, "center"),
                ("Crash Risk    ", 20 + 4, "left"),
                ("Price    ", 10 + 4, "center"),
                ("Target    ", 10 + 4, "center"),
                ("Stop Loss    ", 10 + 4, "center"),
                ("Direction    ", 15 + 4, "left"),
                ("Suggestion    ", 15 + 4, "left"),
                ("Warnings    ", 10 + 4, "center")
            ]
            # 輸出標題
            header_line = ""
            for name, width, align in headers:
                if align == "left":
                    header_line += f"{name:<{width}}"
                elif align == "center":
                    header_line += f"{name:^{width}}"
                else:  # right
                    header_line += f"{name:>{width}}"
            print(header_line)
            print("-" * sum(width for _, width, _ in headers))  # 分隔線
            
            # 輸出每行數據
            for result in results:
                line = ""
                for (name, width, align), key in zip(headers, result.keys()):
                    value = str(result[key])
                    # 移除 ANSI 顏色代碼以計算真實長度
                    clean_value = re.sub(r'\x1b\[[0-9;]*m', '', value)
                    # 修改：計算顯示寬度（中文字算2，英文算1）
                    display_width = 0
                    for char in clean_value:
                        if unicodedata.east_asian_width(char) in ('F', 'W', 'A'):
                            display_width += 2
                        else:
                            display_width += 1
                    # 調整 padding：寬度 + (顏色代碼長度) - 顯示寬度
                    padding_adjust = len(value) - display_width
                    if align == "left":
                        line += f"{value:<{width + padding_adjust}}"
                    elif align == "center":
                        line += f"{value:^{width + padding_adjust}}"
                    else:  # right
                        line += f"{value:>{width + padding_adjust}}"
                print(line)

def get_trend_prob_color(prob):
    if prob >= 50:
        return Fore.GREEN
    elif prob <= -50:
        return Fore.RED
    else:
        return Fore.YELLOW

if __name__ == "__main__":
    main()