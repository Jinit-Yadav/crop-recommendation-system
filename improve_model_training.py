# improve_model_training.py - Fixed Version
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print("="*70)
print("🌾 IMPROVING MODEL WITH BETTER FEATURE ENGINEERING")
print("="*70)

# Load original crop recommendation dataset
try:
    df = pd.read_csv('crop_recommendation.csv')
    print(f"\n✅ Loaded dataset with {len(df)} samples")
    print(f"   Crops: {df['label'].nunique()} different crops")
    
    # Create enhanced features
    print("\n🔧 Creating enhanced features...")
    
    # Normalize features to 0-100 scale
    df['temp_normalized'] = (df['temperature'] + 10) / 50 * 100
    df['temp_normalized'] = df['temp_normalized'].clip(0, 100)
    
    df['rainfall_normalized'] = df['rainfall'] / 300 * 100
    df['rainfall_normalized'] = df['rainfall_normalized'].clip(0, 100)
    
    df['humidity_normalized'] = df['humidity']
    df['ph_normalized'] = (df['ph'] / 14) * 100
    
    # Create interaction features
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    df['rain_fertility_interaction'] = df['rainfall'] * (df['N'] + df['P'] + df['K']) / 1000
    
    # Create suitability score (combined metric)
    df['suitability_score'] = (
        df['temp_normalized'] * 0.3 +
        df['rainfall_normalized'] * 0.3 +
        df['humidity_normalized'] * 0.2 +
        df['ph_normalized'] * 0.2
    )
    
    # Select features for training
    feature_cols = [
        'temperature', 'humidity', 'rainfall', 'ph',
        'N', 'P', 'K',
        'temp_normalized', 'rainfall_normalized', 
        'temp_humidity_interaction', 'rain_fertility_interaction',
        'suitability_score'
    ]
    
    X = df[feature_cols]
    y = df['label']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\n📊 Training with {len(feature_cols)} features")
    print(f"   Features: {', '.join(feature_cols[:5])}...")
    
    # Train improved model
    print("\n🚀 Training improved Random Forest model...")
    improved_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    improved_model.fit(X, y_encoded)
    
    # Evaluate
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(improved_model, X, y_encoded, cv=5)
    
    print(f"\n✅ Improved Model Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Save improved model
    model_package = {
        'model': improved_model,
        'label_encoder': le,
        'feature_columns': feature_cols,
        'crops': le.classes_.tolist(),
        'accuracy': cv_scores.mean(),
        'model_name': 'Improved_Random_Forest'
    }
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model_package, 'models/crop_model_improved.pkl')
    print(f"\n💾 Improved model saved to: models/crop_model_improved.pkl")
    
    # Test predictions - FIXED VERSION
    print("\n" + "="*70)
    print("🧪 TESTING IMPROVED MODEL")
    print("="*70)
    
    test_cases = [
        {"name": "Hot & Arid", "temp": 38, "humidity": 30, "rainfall": 50, "ph": 7.8, "N": 50, "P": 40, "K": 45},
        {"name": "Warm & Humid", "temp": 28, "humidity": 85, "rainfall": 250, "ph": 6.2, "N": 80, "P": 70, "K": 75},
        {"name": "Cool & Moist", "temp": 15, "humidity": 70, "rainfall": 150, "ph": 6.5, "N": 60, "P": 55, "K": 50},
    ]
    
    for test in test_cases:
        # Create dataframe with proper column names
        test_df = pd.DataFrame([{
            'temperature': test['temp'],
            'humidity': test['humidity'],
            'rainfall': test['rainfall'],
            'ph': test['ph'],
            'N': test['N'],
            'P': test['P'],
            'K': test['K']
        }])
        
        # Add engineered features
        test_df['temp_normalized'] = (test_df['temperature'] + 10) / 50 * 100
        test_df['temp_normalized'] = test_df['temp_normalized'].clip(0, 100)
        test_df['rainfall_normalized'] = test_df['rainfall'] / 300 * 100
        test_df['rainfall_normalized'] = test_df['rainfall_normalized'].clip(0, 100)
        test_df['temp_humidity_interaction'] = test_df['temperature'] * test_df['humidity'] / 100
        test_df['rain_fertility_interaction'] = test_df['rainfall'] * (test_df['N'] + test_df['P'] + test_df['K']) / 1000
        test_df['suitability_score'] = (
            test_df['temp_normalized'] * 0.3 +
            test_df['rainfall_normalized'] * 0.3 +
            test_df['humidity'] * 0.2 +
            (test_df['ph'] / 14 * 100) * 0.2
        )
        
        # Make prediction
        pred = improved_model.predict(test_df[feature_cols])[0]
        crop = le.inverse_transform([pred])[0]
        
        # Get probability
        probs = improved_model.predict_proba(test_df[feature_cols])[0]
        confidence = max(probs) * 100
        
        print(f"\n📋 {test['name']}:")
        print(f"   Temperature: {test['temp']}°C, Rainfall: {test['rainfall']}mm, Humidity: {test['humidity']}%")
        print(f"   🌾 Predicted: {crop}")
        print(f"   📊 Confidence: {confidence:.1f}%")
    
    print("\n" + "="*70)
    print("✅ Model improvement complete!")
    print("="*70)
    
except FileNotFoundError:
    print("\n❌ crop_recommendation.csv not found!")
    print("   Please ensure the dataset is available")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()