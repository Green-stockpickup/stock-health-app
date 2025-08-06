from xgboost import XGBClassifier
import shap
import warnings
from config import is_market_open
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit  # 新增TimeSeriesSplit

def predict_price_direction(data, ticker, total_health_score):
    """預測下一個交易日價格方向，使用XGBoost與SHAP，並提供投資建議。
    輸入: data (pd.DataFrame), ticker (str), total_health_score (float)。
    輸出: direction (str), explanation (str), shap_explanation (list)。
    """
    try:
        features = [
            'SMA5', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACDSignal', 'OBV', 'OBV_SMA',
            'ADX', 'PLUS_DI', 'MINUS_DI', 'CCI', 'SlowK', 'SlowD', 'MFI', 'ATR',
            'Volatility20', 'Volatility60', 'GARCH_Volatility', 'CMF',
            'Lag_Close_1', 'Rolling_Mean_5'
        ]
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            warnings.warn(f"Missing features for XGBoost: {missing_features}")
            return None, "Missing features for prediction", []
        
        data = data.dropna(subset=features)
        if len(data) < 100:
            warnings.warn(f"Insufficient data for XGBoost: {len(data)} days")
            return None, "Insufficient data for prediction", []
        
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        train_data = data.iloc[-500:-1]  # 擴大到500天
        X_train = train_data[features]
        y_train = train_data['Target']
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        # 修改：添加 random_state=42 固定種子
        grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        
        # 使用TimeSeriesSplit交叉驗證，並固定 random_state
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy', n_jobs=-1)
        avg_accuracy = np.mean(scores)
        if avg_accuracy < 0.6:
            warnings.warn(f"Low model accuracy for {ticker}: {avg_accuracy:.2f}")
        
        X_latest = data[features].iloc[-1:].dropna()
        if X_latest.empty:
            warnings.warn("Latest data contains NaNs for prediction")
            return None, "Invalid latest data", []
        
        pred = model.predict_proba(X_latest)[0]
        bullish_prob = pred[1] * 100
        direction = "Bullish" if bullish_prob >= 50 else "Bearish"
        confidence = max(bullish_prob, 100 - bullish_prob)
        
        # Investment suggestion based on direction and health score
        if direction == "Bullish" and total_health_score > 70:
            suggestion = "繼續持有"
        elif direction == "Bearish" and total_health_score < 50:
            suggestion = "考慮沽空"
        else:
            suggestion = "觀望"
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_latest)
        shap_contributions = [(feat, shap_values[0][i] if direction == "Bullish" else -shap_values[0][i]) for i, feat in enumerate(features)]
        shap_contributions = sorted(shap_contributions, key=lambda x: abs(x[1]), reverse=True)[:3]
        shap_explanation = [f"{feat}: {val:.3f}" for feat, val in shap_contributions]
        
        explanation = f"{direction} (Confidence: {confidence:.2f}%), Suggestion: {suggestion}"
        return direction, explanation, shap_explanation
    except Exception as e:
        warnings.warn(f"Error in price direction prediction for {ticker}: {str(e)}")
        return None, "Error in prediction", []