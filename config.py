import datetime
import pytz
import numpy as np  # 新增：用於 np.isnan 檢查，確保 NaN 處理穩定

class Config:
    """Configuration for stock health monitoring parameters"""
    # 修改：重新設定為 5 類權重 (總1.0, 趨勢/動量主導, 風險獨立低權)
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
    CMF_PERIOD = 20  # New for Chaikin Money Flow

def is_market_open(data=None):
    """Check if US stock market is open (HKT) and ensure the last bar is complete.
    If data is provided, check if the last bar has volume >0 (complete); else fallback to time check.
    """
    hkt = pytz.timezone("Asia/Hong_Kong")
    current_time = datetime.datetime.now(hkt)
    hour, minute = current_time.hour, current_time.minute
    time_open = (21 <= hour <= 23) or (0 <= hour < 4) or (hour == 4 and minute == 0)
    
    if data is not None and not data.empty:  # 新增：若有數據，檢查最後 bar 是否完整
        last_volume = data['Volume'].iloc[-1]
        if np.isnan(last_volume) or last_volume <= 0:  # 不完整 bar (Volume=0 or NaN)
            return True  # 視為開盤中，使用 -2
        return False  # 完整 bar，使用 -1
    
    return time_open  # 無數據時，fallback 到時間檢查