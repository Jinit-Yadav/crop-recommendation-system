import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def train_enhanced_model():
    """Train enhanced crop recommendation model"""
    
    # Load enhanced data
    df = pd.read_csv('enhanced_crop_data.csv')
    print(f"📊 Loaded enhanced data: {df.shape}")
    
    # Features for prediction
    feature_cols = ['temperature', 'humidity', 'rainfall', 'N', 'P', 'K', 'ph']
    X = df[feature_cols].copy()
    
    # Train crop classifier
    label_encoder = LabelEncoder()
    y_crop = label_encoder.fit_transform(df['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_crop, test_size=0.2, random_state=42, stratify=y_crop
    )
    
    # Train classifier
    classifier = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        random_state=42, 
        n_jobs=-1
    )
    classifier.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    print(f"✅ Crop Classifier Accuracy: {accuracy:.2%}")
    
    # Create crop metrics database
    crop_metrics = {}
    
    for crop in label_encoder.classes_:
        crop_data = df[df['label'] == crop]
        crop_metrics[crop] = {
            'avg_suitability': float(crop_data['suitability_score'].mean()),
            'avg_yield': float(crop_data['expected_yield'].mean()),
            'avg_profit': float(crop_data['estimated_profit'].mean()),
            'profit_margin': float(crop_data['profit_margin'].iloc[0]),
            'typical_risk': crop_data['risk_level'].mode()[0] if len(crop_data) > 0 else 'Medium',
            'market_demand': crop_data['market_demand'].iloc[0],
            'growing_days': int(crop_data['growing_days'].iloc[0]),
            'soil_requirement': crop_data['soil_requirement'].iloc[0],
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
        'label_encoder': label_encoder,
        'feature_columns': feature_cols,
        'crop_metrics': crop_metrics,
        'accuracy': accuracy,
        'all_crops': label_encoder.classes_.tolist()
    }
    
    joblib.dump(model_package, 'models/enhanced_crop_model.pkl')
    print("\n💾 Enhanced model saved to: models/enhanced_crop_model.pkl")
    
    # Display crop metrics summary
    print("\n📊 Crop Metrics Summary:")
    for crop, metrics in list(crop_metrics.items())[:5]:
        print(f"   {crop}: Yield={metrics['avg_yield']} tons/ha, Profit=₹{metrics['avg_profit']:,.0f}")
    
    return model_package

if __name__ == "__main__":
    model = train_enhanced_model()