# run_enhanced_training.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def train_enhanced_model():
    """Train enhanced crop recommendation model"""
    
    print("="*60)
    print("🚀 ENHANCED CROP RECOMMENDATION MODEL TRAINING")
    print("="*60)
    
    # Check if enhanced data exists, if not create it
    if not os.path.exists('enhanced_crop_data.csv'):
        print("📝 Creating enhanced crop data first...")
        exec(open('enhanced_crop_data.py').read())
    
    # Load enhanced data
    df = pd.read_csv('enhanced_crop_data.csv')
    print(f"\n📊 Loaded enhanced data: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Features for prediction
    feature_cols = ['temperature', 'humidity', 'rainfall', 'N', 'P', 'K', 'ph']
    X = df[feature_cols].copy()
    
    # Train crop classifier
    label_encoder = LabelEncoder()
    y_crop = label_encoder.fit_transform(df['label'])
    
    print(f"\n📈 Classes: {label_encoder.classes_.tolist()}")
    print(f"   Total crops: {len(label_encoder.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_crop, test_size=0.2, random_state=42, stratify=y_crop
    )
    
    print(f"\n📊 Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    print("\n🚀 Training Random Forest Classifier...")
    classifier = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1
    )
    classifier.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy:.2%}")
    
    # Create crop metrics database
    crop_metrics = {}
    for crop in label_encoder.classes_:
        crop_data = df[df['label'] == crop]
        if len(crop_data) > 0:
            crop_metrics[crop] = {
                'avg_suitability': float(crop_data['suitability_score'].mean()),
                'avg_yield': float(crop_data['expected_yield'].mean()),
                'avg_profit': float(crop_data['estimated_profit'].mean()),
                'profit_margin': float(crop_data['profit_margin'].iloc[0]) if 'profit_margin' in crop_data.columns else 0.3,
                'typical_risk': crop_data['risk_level'].mode()[0] if 'risk_level' in crop_data.columns else 'Medium',
                'market_demand': crop_data['market_demand'].iloc[0] if 'market_demand' in crop_data.columns else 'Medium',
                'growing_days': int(crop_data['growing_days'].iloc[0]) if 'growing_days' in crop_data.columns else 120,
                'soil_requirement': crop_data['soil_requirement'].iloc[0] if 'soil_requirement' in crop_data.columns else 'Well-drained soil',
                'temp_min': float(crop_data['temperature'].min()),
                'temp_max': float(crop_data['temperature'].max()),
                'rain_min': float(crop_data['rainfall'].min()),
                'rain_max': float(crop_data['rainfall'].max()),
                'ph_min': float(crop_data['ph'].min()),
                'ph_max': float(crop_data['ph'].max())
            }
    
    # Save everything
    os.makedirs('models', exist_ok=True)
    
    model_package = {
        'classifier': classifier,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': feature_cols,
        'crop_metrics': crop_metrics,
        'accuracy': accuracy,
        'all_crops': label_encoder.classes_.tolist()
    }
    
    # Save as enhanced model
    joblib.dump(model_package, 'models/enhanced_crop_model.pkl')
    print("\n💾 Enhanced model saved to: models/enhanced_crop_model.pkl")
    
    # Also save as the main model for the app
    simple_package = {
        'model': classifier,
        'label_encoder': label_encoder,
        'feature_columns': feature_cols,
        'crops': label_encoder.classes_.tolist(),
        'accuracy': accuracy,
        'model_name': 'Enhanced Random Forest'
    }
    joblib.dump(simple_package, 'models/crop_model.pkl')
    print("💾 Model also saved to: models/crop_model.pkl (for app use)")
    
    # Display crop metrics summary
    print("\n📊 Crop Metrics Summary:")
    print("-" * 60)
    for crop, metrics in list(crop_metrics.items())[:10]:
        print(f"   {crop:<12} | Yield: {metrics['avg_yield']:4.1f} tons/ha | Profit: ₹{metrics['avg_profit']:7,.0f} | Risk: {metrics['typical_risk']}")
    
    return model_package

if __name__ == "__main__":
    model = train_enhanced_model()
    print("\n✅ Training completed successfully!")
    print("   You can now run: python app.py")