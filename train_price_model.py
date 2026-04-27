import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 Training Price Prediction Models (Fixed for Data Leakage)...")

def load_and_prepare_data():
    """Load and prepare the price prediction data WITHOUT data leakage"""
    
    df = pd.read_csv('price_prediction_cleaned.csv')
    print(f"\n📊 Dataset shape: {df.shape}")
    
    # Sort by date to maintain temporal order
    df = df.sort_values(['District', 'Commodity', 'Year', 'Month', 'Day'])
    
    # === FIXED: Remove features that cause data leakage ===
    # These features use future information
    leakage_features = [
        'Prev_Modal_Price',      # Uses future data when shifted incorrectly
        'Price_Change',           # Calculated from Prev_Modal_Price
        'Price_Change_Pct',       # Calculated from Prev_Modal_Price
        'Modal_Price_MA3',        # Uses future data if not properly lagged
        'Price_Momentum'          # Uses future data
    ]
    
    # Keep only features available at prediction time
    feature_cols = [
        # Temporal features (always available)
        'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter',
        
        # Categorical encodings (known in advance)
        'District_Encoded', 'Market_Encoded', 'Commodity_Encoded', 
        'Variety_Encoded', 'Grade_Encoded', 'Season_Encoded',
        
        # Price-derived features (calculated from same time period only)
        'Price_Range',           # Min-Max range for same day - OK
        'Price_Volatility',       # Range/Modal for same day - OK
    ]
    
    # Target variable
    target_col = 'Modal_Price'
    
    # Check which features exist
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"\n✅ Using {len(available_features)} features: {available_features}")
    
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Handle any remaining missing values
    X = X.fillna(X.mean())
    
    return X, y, available_features, df

def create_time_based_splits(df, X, y):
    """Create time-based train/test splits to prevent future data leakage"""
    
    # Sort by date
    df_sorted = df.sort_values(['Year', 'Month', 'Day'])
    
    # Use 80% oldest data for training, 20% newest for testing
    split_idx = int(len(df_sorted) * 0.8)
    
    train_indices = df_sorted.index[:split_idx]
    test_indices = df_sorted.index[split_idx:]
    
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    
    print(f"\n📊 Time-based split:")
    print(f"   Training: {X_train.shape[0]} rows (older data)")
    print(f"   Testing: {X_test.shape[0]} rows (newer data)")
    print(f"   Date range - Train: {df_sorted['Year'].iloc[0]}-{df_sorted['Month'].iloc[split_idx-1]}")
    print(f"   Date range - Test: {df_sorted['Year'].iloc[split_idx]}-{df_sorted['Month'].iloc[-1]}")
    
    return X_train, X_test, y_train, y_test

def train_price_models():
    """Train multiple models for price prediction WITHOUT data leakage"""
    
    X, y, feature_cols, df = load_and_prepare_data()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    
    # Use time-based split instead of random split
    X_train, X_test, y_train, y_test = create_time_based_splits(df, X_scaled, y)
    
    # Define models with regularization to prevent overfitting
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,           # Reduced
            max_depth=15,                # Limited depth
            min_samples_split=10,        # Higher = less overfitting
            min_samples_leaf=5,           # Higher = less overfitting
            max_features='sqrt',          # Limit features per tree
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,                  # Shallower trees
            learning_rate=0.05,
            subsample=0.8,                 # Use 80% of data per iteration
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,          # Feature sampling
            reg_alpha=0.1,                  # L1 regularization
            reg_lambda=1.0,                  # L2 regularization
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    print("\n" + "="*60)
    print("TRAINING INDIVIDUAL MODELS (WITH REGULARIZATION)")
    print("="*60)
    
    # Use time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict on test (future data)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Time series cross-validation
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model_cv = model.__class__(**model.get_params())
            model_cv.fit(X_cv_train, y_cv_train)
            y_cv_pred = model_cv.predict(X_cv_val)
            cv_scores.append(r2_score(y_cv_val, y_cv_pred))
        
        cv_scores = np.array(cv_scores)
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_R2_mean': cv_scores.mean(),
            'CV_R2_std': cv_scores.std()
        }
        
        trained_models[name] = model
        
        print(f"   ✅ Test R² Score: {r2:.4f}")
        print(f"   📊 Test MAE: ₹{mae:.2f}")
        print(f"   📊 Test RMSE: ₹{rmse:.2f}")
        print(f"   📊 CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Check for overfitting
        train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        print(f"   📊 Train R²: {train_r2:.4f} (Overfitting gap: {train_r2 - r2:.4f})")
    
    # Select best model based on test R²
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_model = trained_models[best_model_name]
    
    print("\n" + "="*60)
    print("BEST MODEL")
    print("="*60)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test R² Score: {results[best_model_name]['R2']:.4f}")
    print(f"   Test MAE: ₹{results[best_model_name]['MAE']:.2f}")
    print(f"   Test RMSE: ₹{results[best_model_name]['RMSE']:.2f}")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Feature Importance:")
        print(importance)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    
    model_package = {
        'model': best_model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'model_name': best_model_name,
        'metrics': results[best_model_name],
        'all_results': results,
        'training_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'feature_importance': importance.to_dict() if hasattr(best_model, 'feature_importances_') else None
    }
    
    joblib.dump(model_package, 'models/price_prediction_model.pkl')
    print(f"\n💾 Model saved to: models/price_prediction_model.pkl")
    
    return model_package

if __name__ == "__main__":
    model_package = train_price_models()