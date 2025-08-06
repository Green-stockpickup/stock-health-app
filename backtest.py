# 修改後的 backtest.py 文件
import numpy as np
import pandas as pd
import warnings
from models import predict_price_direction
from health_score import calculate_health_score
from conditions import condition_l_crash_risk
from data_fetch import get_market_condition

def backtest_strategy(data, ticker, days=90, health_score_min=80, trend_prob_min=75, confidence_min=80):
    """回測指定策略的歷史表現，基於健康分數、趨勢概率、方向預測。
    輸入:
        data (pd.DataFrame): 股票歷史數據
        ticker (str): 股票代碼
        days (int): 回測天數（預設90天）
        health_score_min (float): 最低健康分數（預設80）
        trend_prob_min (float): 最低趨勢概率（預設75）
        confidence_min (float): 最低Bullish信心度（預設80）
    輸出:
        report (str): 回測報告，包括交易次數、勝率、平均報酬等
    """
    try:
        if len(data) < days + 1:
            warnings.warn(f"Insufficient data for backtest: {len(data)} days, need {days + 1}")
            return f"Error: Insufficient data for {ticker} (only {len(data)} days)"

        market_condition, market_params = get_market_condition()
        trades = []
        holding = False
        entry_price = 0.0
        entry_date = None
        target = 0.0
        stop_loss = 0.0

        # 回測最近 days 天（跳過最後一天以確保有下一交易日數據）
        for i in range(len(data) - days - 1, len(data) - 1):
            historical_data = data.iloc[:i + 1]
            date = historical_data.index[-1]

            # 計算健康分數和其他條件
            total_health_score, _, _, _, _, warnings_list, trend_prob, _, _, _, _ = calculate_health_score(
                historical_data, market_params, ticker, market_condition)
            direction, direction_explanation, _ = predict_price_direction(historical_data, ticker, total_health_score)

            # 解析信心度
            confidence = 0.0
            if direction:
                import re
                match = re.search(r'(\d+\.\d+)%', direction_explanation)
                if match:
                    confidence = float(match.group(1))

            # 買入條件
            buy_conditions = (
                total_health_score >= health_score_min and
                trend_prob >= trend_prob_min and
                direction == "Bullish" and
                confidence >= confidence_min
            )

            # 當日收盤價
            current_price = historical_data['Close'].iloc[-1]

            if holding:
                # 檢查賣出條件
                if current_price >= target:
                    # 達到Target賣出
                    holding = False
                    exit_price = current_price
                    profit = (exit_price - entry_price) / entry_price
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_reason': 'Reached Target'
                    })
                    entry_price = 0.0
                    entry_date = None
                    target = 0.0
                    stop_loss = 0.0
                elif current_price <= stop_loss:
                    # 達到Stop Loss賣出
                    holding = False
                    exit_price = current_price
                    profit = (exit_price - entry_price) / entry_price
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_reason': 'Hit Stop Loss'
                    })
                    entry_price = 0.0
                    entry_date = None
                    target = 0.0
                    stop_loss = 0.0
                elif not buy_conditions:
                    # 趨勢不穩賣出
                    holding = False
                    exit_price = current_price
                    profit = (exit_price - entry_price) / entry_price
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_reason': 'Trend Unstable'
                    })
                    entry_price = 0.0
                    entry_date = None
                    target = 0.0
                    stop_loss = 0.0

            if buy_conditions and not holding:
                # 買入並計算Target和Stop Loss
                atr = historical_data['ATR'].iloc[-1]
                holding = True
                entry_price = current_price
                entry_date = date
                target = current_price + atr * 2  # Target = entry + ATR * 2
                stop_loss = current_price - atr * 1  # Stop Loss = entry - ATR * 1

        # 若最後仍持有，模擬以最後一天收盤價賣出
        if holding:
            exit_price = data['Close'].iloc[-2]
            profit = (exit_price - entry_price) / entry_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': data.index[-2],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit': profit,
                'exit_reason': 'End of Period'
            })

        if not trades:
            return f"No trades executed for {ticker} in the last {days} days"

        # 計算回測指標
        df_trades = pd.DataFrame(trades)
        win_rate = len(df_trades[df_trades['profit'] > 0]) / len(df_trades) * 100
        avg_profit = df_trades['profit'].mean() * 100
        total_profit = ((1 + df_trades['profit']).prod() - 1) * 100
        max_drawdown = df_trades['profit'].min() * 100 if not df_trades.empty else 0.0

        # 格式化報告
        report = f"回測報告 ({ticker}, {days} 天):\n"
        report += f"總交易次數: {len(df_trades)}\n"
        report += f"勝率: {win_rate:.2f}%\n"
        report += f"平均報酬: {avg_profit:.2f}%\n"
        report += f"總報酬: {total_profit:.2f}%\n"
        report += f"最大回撤: {max_drawdown:.2f}%\n"
        report += "\n交易詳情:\n"
        for _, trade in df_trades.iterrows():
            report += f"買入日期: {trade['entry_date'].strftime('%Y-%m-%d')}, "
            report += f"賣出日期: {trade['exit_date'].strftime('%Y-%m-%d')}, "
            report += f"買入價: {trade['entry_price']:.2f}, "
            report += f"賣出價: {trade['exit_price']:.2f}, "
            report += f"報酬: {trade['profit']*100:.2f}%, "
            report += f"賣出原因: {trade['exit_reason']}\n"

        return report

    except Exception as e:
        warnings.warn(f"Error in backtest for {ticker}: {str(e)}")
        return f"Error in backtest for {ticker}: {str(e)}"

# 原有 backtest_model 保持不變（但main.py將使用backtest_strategy）
def backtest_model(data, ticker):
    try:
        if len(data) < 500:
            warnings.warn(f"Insufficient data for backtest: {len(data)} days")
            return "Insufficient data for backtest"
        
        results = []
        for i in range(len(data) - 60, len(data) - 1):
            historical_data = data.iloc[:i]
            direction, _, _ = predict_price_direction(historical_data, ticker, 0)
            if direction is None:
                continue
            actual_change = data['Close'].iloc[i] > data['Close'].iloc[i-1]
            is_correct = (direction == "Bullish" and actual_change) or (direction == "Bearish" and not actual_change)
            results.append(is_correct)
        
        if not results:
            return "No valid predictions in backtest"
        
        accuracy = sum(results) / len(results) * 100
        return f"回測報告 ({ticker}): 準確率 {accuracy:.2f}% (基於最近 {len(results)} 天模擬)"
    except Exception as e:
        warnings.warn(f"Error in backtest for {ticker}: {str(e)}")
        return "Error in backtest"