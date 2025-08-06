import datetime
import pytz

class Config:
    """Configuration for stock health monitoring parameters"""
    DAYS_BACK = 300  # Increased for more history
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
    BONUS_MAX = 25.0
    GARCH_WINDOW = 100
    CMF_PERIOD = 20  # New for Chaikin Money Flow
    CATEGORY_WEIGHTS = {"Trend": 0.4, "Volume": 0.3, "Momentum": 0.3}  # Quantifiable weights
    RISK_THRESHOLD = 25  # Crash risk threshold for warnings

def is_market_open():
    """Check if US stock market is open (HKT)"""
    hkt = pytz.timezone("Asia/Hong_Kong")
    current_time = datetime.datetime.now(hkt)
    hour, minute = current_time.hour, current_time.minute
    return (21 <= hour <= 23) or (0 <= hour < 4) or (hour == 4 and minute == 0)