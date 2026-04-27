# train_price_with_trends.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("🚀 Training Price Prediction with Trend Analysis...")

def load_and_prepare_data():
    """Load and prepare price data with trend features"""
    
    df = pd.read_csv('price_prediction_cleaned.csv')
    print(f"📊 Dataset shape: {df.shape}")
    
    # Sort by date
    df = df.sort_values(['District', 'Commodity', 'Arrival_Date'])
    
    # Create trend features for each commodity-district pair
    df['Price_MA7'] = df.groupby(['District', 'Commodity'])['Modal_Price'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    df['Price_MA30'] = df.groupby(['District', 'Commodity'])['Modal_Price'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    
    # Price momentum (7-day vs 30-day)
    df['Price_Momentum'] = df['Price_MA7'] / df['Price_MA30']
    
    # Weekly price change
    df['Weekly_Change'] = df.groupby(['District', 'Commodity'])['Modal_Price'].transform(
        lambda x: x.pct_change(periods=7)
    )
    
    # Volatility (standard deviation over last 30 days)
    df['Volatility_30d'] = df.groupby(['District', 'Commodity'])['Modal_Price'].transform(
        lambda x: x.rolling(window=30, min_periods=1).std()
    )
    
    # Price position (current price relative to 90-day range)
    df['Price_Min_90d'] = df.groupby(['District', 'Commodity'])['Modal_Price'].transform(
        lambda x: x.rolling(window=90, min_periods=1).min()
    )
    df['Price_Max_90d'] = df.groupby(['District', 'Commodity'])['Modal_Price'].transform(
        lambda x: x.rolling(window=90, min_periods=1).max()
    )
    df['Price_Position'] = (df['Modal_Price'] - df['Price_Min_90d']) / (df['Price_Max_90d'] - df['Price_Min_90d'] + 1)
    
    # Drop NaN from rolling calculations
    df = df.dropna()
    
    # Feature columns
    feature_cols = [
        'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter',
        'District_Encoded', 'Market_Encoded', 'Commodity_Encoded',
        'Variety_Encoded', 'Grade_Encoded', 'Season_Encoded',
        'Price_MA7', 'Price_MA30', 'Price_Momentum',
        'Weekly_Change', 'Volatility_30d', 'Price_Position'
    ]
    
    # Ensure all features exist
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y = df['Modal_Price']
    
    print(f"✅ Using {len(available_features)} features")
    
    return X, y, available_features, df

def train_with_trend_analysis():
    """Train model with trend analysis for sell/hold recommendations"""
    
    X, y, feature_cols, df = load_and_prepare_data()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-based split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
    
    # Train model
    model = XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import r2_score, mean_absolute_error
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n✅ Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: ₹{mae:.2f}")
    
    # Save model
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'r2_score': r2,
        'mae': mae,
        'training_date': datetime.now().isoformat()
    }
    
    joblib.dump(model_package, 'models/price_trend_model.pkl')
    print(f"\n💾 Model saved to: models/price_trend_model.pkl")
    
    return model_package

if __name__ == "__main__":
    train_with_trend_analysis()