# retrain_price_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def retrain_price_model():
    """Retrain the price prediction model"""
    
    print("="*60)
    print("🔄 RETAINING PRICE PREDICTION MODEL")
    print("="*60)
    
    # Load the data
    df = pd.read_csv('maharashtra_districts_complete.csv')
    print(f"\n📊 Loaded data: {df.shape}")
    
    # Data preprocessing
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')
    df = df.dropna(subset=['Arrival_Date'])
    
    # Ensure price columns are numeric
    for col in ['Min_Price', 'Max_Price', 'Modal_Price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Modal_Price'])
    df = df[df['Modal_Price'] > 0]
    
    print(f"📊 After cleaning: {df.shape}")
    
    # Feature engineering
    df['Year'] = df['Arrival_Date'].dt.year
    df['Month'] = df['Arrival_Date'].dt.month
    df['Day'] = df['Arrival_Date'].dt.day
    df['DayOfWeek'] = df['Arrival_Date'].dt.dayofweek
    df['Quarter'] = df['Arrival_Date'].dt.quarter
    
    # Season feature
    def get_season(month):
        if month in [6, 7, 8, 9]:
            return 'Kharif'
        elif month in [10, 11, 12, 1]:
            return 'Rabi'
        else:
            return 'Summer'
    
    df['Season'] = df['Month'].apply(get_season)
    
    # Price features
    df['Price_Range'] = df['Max_Price'] - df['Min_Price']
    df['Price_Volatility'] = df['Price_Range'] / (df['Modal_Price'] + 1)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['District', 'Market', 'Commodity', 'Variety', 'Grade', 'Season']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Features for training
    feature_cols = [
        'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter',
        'District_Encoded', 'Market_Encoded', 'Commodity_Encoded',
        'Variety_Encoded', 'Grade_Encoded', 'Season_Encoded',
        'Price_Range', 'Price_Volatility'
    ]
    
    X = df[feature_cols]
    y = df['Modal_Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n🚀 Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   MAE: ₹{mae:.2f}")
    print(f"   R² Score: {r2:.4f}")
    
    # Save model and encoders
    os.makedirs('models', exist_ok=True)
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'label_encoders': label_encoders,
        'metrics': {
            'MAE': mae,
            'R2': r2
        },
        'model_name': 'Random Forest Regressor',
        'training_date': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    joblib.dump(model_package, 'models/price_prediction_model.pkl')
    print(f"\n✅ Model saved to: models/price_prediction_model.pkl")
    
    return model_package

if __name__ == "__main__":
    retrain_price_model()