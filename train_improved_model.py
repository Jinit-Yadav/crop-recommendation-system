import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def train_improved_model():
    """Train improved crop recommendation model with better features"""
    
    print("="*60)
    print("🚀 IMPROVED CROP RECOMMENDATION MODEL TRAINING")
    print("="*60)
    
    # Load enhanced data
    df = pd.read_csv('enhanced_crop_data.csv')
    print(f"\n📊 Loaded enhanced data: {df.shape}")
    
    # Create more relevant features that match the data generation logic
    print("\n🔧 Creating enhanced features...")
    
    # Temperature features
    df['temp_squared'] = df['temperature'] ** 2
    df['temp_category_binned'] = pd.cut(df['temperature'], 
                                        bins=[0, 15, 20, 25, 30, 35, 50],
                                        labels=[0, 1, 2, 3, 4, 5])
    
    # Rainfall features
    df['rainfall_squared'] = df['rainfall'] ** 2
    df['rainfall_category_binned'] = pd.cut(df['rainfall'],
                                            bins=[0, 50, 80, 100, 150, 200, 500],
                                            labels=[0, 1, 2, 3, 4, 5])
    
    # Soil nutrient ratios (important for crop selection)
    df['NP_ratio'] = df['N'] / (df['P'] + 1)
    df['PK_ratio'] = df['P'] / (df['K'] + 1)
    df['NK_ratio'] = df['N'] / (df['K'] + 1)
    df['NPK_sum'] = df['N'] + df['P'] + df['K']
    df['NPK_product'] = df['N'] * df['P'] * df['K']
    
    # pH features
    df['ph_from_neutral'] = abs(df['ph'] - 6.5)
    df['ph_category_binned'] = pd.cut(df['ph'],
                                      bins=[0, 5.5, 6.0, 6.5, 7.0, 7.5, 14],
                                      labels=[0, 1, 2, 3, 4, 5])
    
    # Interaction features
    df['temp_rain_interaction'] = df['temperature'] * df['rainfall'] / 100
    df['temp_ph_interaction'] = df['temperature'] * df['ph']
    df['rain_ph_interaction'] = df['rainfall'] * df['ph']
    
    # Humidity influence (since humidity wasn't in the original generation logic)
    df['humidity_normalized'] = df['humidity'] / 100
    
    # Feature selection - use features that actually determine crop selection
    feature_cols = [
        'temperature', 'temp_squared',
        'rainfall', 'rainfall_squared',
        'ph', 'ph_from_neutral',
        'N', 'P', 'K',
        'NP_ratio', 'PK_ratio', 'NK_ratio',
        'NPK_sum', 'NPK_product',
        'temp_rain_interaction', 'temp_ph_interaction',
        'humidity_normalized',
        'temp_category_binned', 'rainfall_category_binned', 'ph_category_binned'
    ]
    
    X = df[feature_cols].copy()
    y = df['label']
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    # Try different models
    print("\n🚀 Testing multiple models...")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"   ✅ Accuracy: {acc:.2%}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best Model: {best_name} with {best_accuracy:.2%} accuracy")
    
    # Create crop metrics database from actual data
    crop_metrics = {}
    for crop in label_encoder.classes_:
        crop_data = df[df['label'] == crop]
        if len(crop_data) > 0:
            # Calculate optimal ranges from data
            crop_metrics[crop] = {
                'avg_suitability': float(crop_data['suitability_score'].mean()),
                'avg_yield': float(crop_data['expected_yield'].mean()),
                'avg_profit': float(crop_data['estimated_profit'].mean()),
                'profit_margin': float(crop_data['profit_margin'].iloc[0]),
                'typical_risk': crop_data['risk_level'].mode()[0] if len(crop_data) > 0 else 'Medium',
                'market_demand': crop_data['market_demand'].iloc[0],
                'growing_days': int(crop_data['growing_days'].iloc[0]),
                'soil_requirement': crop_data['soil_requirement'].iloc[0],
                'temp_min': float(crop_data['temperature'].quantile(0.1)),
                'temp_max': float(crop_data['temperature'].quantile(0.9)),
                'temp_optimal': float(crop_data['temperature'].mean()),
                'rain_min': float(crop_data['rainfall'].quantile(0.1)),
                'rain_max': float(crop_data['rainfall'].quantile(0.9)),
                'rain_optimal': float(crop_data['rainfall'].mean()),
                'ph_min': float(crop_data['ph'].quantile(0.1)),
                'ph_max': float(crop_data['ph'].quantile(0.9)),
                'ph_optimal': float(crop_data['ph'].mean())
            }
    
    # Save everything
    os.makedirs('models', exist_ok=True)
    
    model_package = {
        'classifier': best_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': feature_cols,
        'crop_metrics': crop_metrics,
        'accuracy': best_accuracy,
        'all_crops': label_encoder.classes_.tolist(),
        'model_name': best_name,
        'training_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save as the main model
    joblib.dump(model_package, 'models/enhanced_crop_model.pkl')
    print(f"\n💾 Model saved to: models/enhanced_crop_model.pkl")
    
    # Also create the simple format expected by app.py
    simple_package = {
        'model': best_model,
        'label_encoder': label_encoder,
        'feature_columns': feature_cols,
        'crops': label_encoder.classes_.tolist(),
        'accuracy': best_accuracy,
        'model_name': best_name,
        'crop_metrics': crop_metrics
    }
    joblib.dump(simple_package, 'models/crop_model.pkl')
    print(f"💾 Model also saved to: models/crop_model.pkl")
    
    # Display results
    print("\n" + "="*60)
    print("📊 CROP METRICS SUMMARY")
    print("="*60)
    for crop, metrics in list(crop_metrics.items())[:10]:
        print(f"\n{crop}:")
        print(f"   Yield: {metrics['avg_yield']:.1f} tons/ha")
        print(f"   Profit: ₹{metrics['avg_profit']:,.0f}/ha")
        print(f"   Risk: {metrics['typical_risk']}")
        print(f"   Market: {metrics['market_demand']}")
    
    return model_package

if __name__ == "__main__":
    model = train_improved_model()
    
    # Test with sample predictions
    print("\n" + "="*60)
    print("🔮 TESTING MODEL WITH SAMPLE INPUTS")
    print("="*60)
    
    test_cases = [
        {'temp': 25, 'rainfall': 150, 'ph': 6.5, 'N': 70, 'P': 60, 'K': 65},
        {'temp': 32, 'rainfall': 80, 'ph': 7.0, 'N': 50, 'P': 40, 'K': 45},
        {'temp': 18, 'rainfall': 200, 'ph': 6.0, 'N': 90, 'P': 80, 'K': 85},
    ]
    
    scaler = model['scaler']
    classifier = model['classifier']
    label_encoder = model['label_encoder']
    feature_cols = model['feature_columns']
    
    for i, test in enumerate(test_cases):
        # Create input features
        input_dict = {
            'temperature': test['temp'],
            'temp_squared': test['temp'] ** 2,
            'rainfall': test['rainfall'],
            'rainfall_squared': test['rainfall'] ** 2,
            'ph': test['ph'],
            'ph_from_neutral': abs(test['ph'] - 6.5),
            'N': test['N'],
            'P': test['P'],
            'K': test['K'],
            'NP_ratio': test['N'] / (test['P'] + 1),
            'PK_ratio': test['P'] / (test['K'] + 1),
            'NK_ratio': test['N'] / (test['K'] + 1),
            'NPK_sum': test['N'] + test['P'] + test['K'],
            'NPK_product': test['N'] * test['P'] * test['K'],
            'temp_rain_interaction': test['temp'] * test['rainfall'] / 100,
            'temp_ph_interaction': test['temp'] * test['ph'],
            'humidity_normalized': 0.65,  # default moderate humidity
        }
        
        # Add binned categories
        if test['temp'] < 15:
            input_dict['temp_category_binned'] = 0
        elif test['temp'] < 20:
            input_dict['temp_category_binned'] = 1
        elif test['temp'] < 25:
            input_dict['temp_category_binned'] = 2
        elif test['temp'] < 30:
            input_dict['temp_category_binned'] = 3
        elif test['temp'] < 35:
            input_dict['temp_category_binned'] = 4
        else:
            input_dict['temp_category_binned'] = 5
        
        if test['rainfall'] < 50:
            input_dict['rainfall_category_binned'] = 0
        elif test['rainfall'] < 80:
            input_dict['rainfall_category_binned'] = 1
        elif test['rainfall'] < 100:
            input_dict['rainfall_category_binned'] = 2
        elif test['rainfall'] < 150:
            input_dict['rainfall_category_binned'] = 3
        elif test['rainfall'] < 200:
            input_dict['rainfall_category_binned'] = 4
        else:
            input_dict['rainfall_category_binned'] = 5
        
        if test['ph'] < 5.5:
            input_dict['ph_category_binned'] = 0
        elif test['ph'] < 6.0:
            input_dict['ph_category_binned'] = 1
        elif test['ph'] < 6.5:
            input_dict['ph_category_binned'] = 2
        elif test['ph'] < 7.0:
            input_dict['ph_category_binned'] = 3
        elif test['ph'] < 7.5:
            input_dict['ph_category_binned'] = 4
        else:
            input_dict['ph_category_binned'] = 5
        
        # Create DataFrame
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_cols]
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        pred = classifier.predict(input_scaled)[0]
        crop = label_encoder.inverse_transform([pred])[0]
        
        # Get probability
        if hasattr(classifier, 'predict_proba'):
            probs = classifier.predict_proba(input_scaled)[0]
            confidence = probs[pred] * 100
        else:
            confidence = 85.0
        
        print(f"\n🔮 Test {i+1}: Temp={test['temp']}°C, Rain={test['rainfall']}mm, pH={test['ph']}")
        print(f"   ✅ Recommended: {crop} (Confidence: {confidence:.1f}%)")
    
    print("\n✅ Training complete! Run 'python app.py' to start the server.")