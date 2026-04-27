# train_complete_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🌾 TRAINING COMPLETE CROP RECOMMENDATION MODEL")
print("="*70)

# Load your dataset
df = pd.read_csv('crop_recommendation.csv')
print(f"\n📊 Dataset shape: {df.shape}")
print(f"📈 Number of crops: {df['label'].nunique()}")
print(f"🌾 Crops: {', '.join(df['label'].unique()[:10])}...")

# Feature engineering
print("\n🔧 Creating enhanced features...")

# Normalize features
df['temp_normalized'] = (df['temperature'] + 10) / 50 * 100
df['temp_normalized'] = df['temp_normalized'].clip(0, 100)

df['rainfall_normalized'] = df['rainfall'] / 300 * 100
df['rainfall_normalized'] = df['rainfall_normalized'].clip(0, 100)

df['humidity_normalized'] = df['humidity']

# Create NPK ratios
df['N_P_ratio'] = df['N'] / (df['P'] + 1)
df['K_P_ratio'] = df['K'] / (df['P'] + 1)
df['NPK_total'] = df['N'] + df['P'] + df['K']

# Create interaction features
df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
df['rain_fertility'] = df['rainfall'] * df['NPK_total'] / 1000

# Create suitability score based on ideal conditions
def calculate_suitability(row):
    score = 0
    # Temperature (ideal range depends on crop - will be learned by model)
    if 20 <= row['temperature'] <= 30:
        score += 25
    elif 15 <= row['temperature'] <= 35:
        score += 15
    else:
        score += 5
    
    # Rainfall
    if 100 <= row['rainfall'] <= 200:
        score += 25
    elif 50 <= row['rainfall'] <= 250:
        score += 15
    else:
        score += 5
    
    # NPK balance
    if row['NPK_total'] > 200:
        score += 25
    elif row['NPK_total'] > 100:
        score += 15
    else:
        score += 5
    
    # pH
    if 6.0 <= row['ph'] <= 7.5:
        score += 25
    elif 5.5 <= row['ph'] <= 8.0:
        score += 15
    else:
        score += 5
    
    return score

df['suitability_score'] = df.apply(calculate_suitability, axis=1)

# Select features for training
feature_cols = [
    'temperature', 'humidity', 'rainfall', 'ph',
    'N', 'P', 'K',
    'temp_normalized', 'rainfall_normalized',
    'N_P_ratio', 'K_P_ratio', 'NPK_total',
    'temp_humidity', 'rain_fertility',
    'suitability_score'
]

X = df[feature_cols]
y = df['label']

print(f"\n📊 Training features: {len(feature_cols)}")
print(f"   Features: {', '.join(feature_cols[:8])}...")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Train model
print("\n🚀 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\n✅ Training accuracy: {train_score:.4f}")
print(f"✅ Test accuracy: {test_score:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y_encoded, cv=5)
print(f"📊 Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Feature importance
importances = model.feature_importances_
feature_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

print("\n📈 Top 10 most important features:")
for i, (feature, importance) in enumerate(feature_importance[:10], 1):
    print(f"   {i}. {feature}: {importance:.4f}")

# Save model
model_package = {
    'model': model,
    'label_encoder': label_encoder,
    'feature_columns': feature_cols,
    'crops': label_encoder.classes_.tolist(),
    'accuracy': test_score,
    'cv_accuracy': cv_scores.mean(),
    'model_name': 'RandomForest'
}

import os
os.makedirs('models', exist_ok=True)
joblib.dump(model_package, 'models/crop_model_complete.pkl')
print(f"\n💾 Model saved to: models/crop_model_complete.pkl")

# Test predictions
print("\n" + "="*70)
print("🧪 TESTING MODEL PREDICTIONS")
print("="*70)

test_cases = [
    {"name": "Hot & Dry", "temp": 38, "humidity": 35, "rainfall": 60, "ph": 7.8, "N": 30, "P": 25, "K": 35},
    {"name": "Warm & Humid", "temp": 28, "humidity": 85, "rainfall": 250, "ph": 6.2, "N": 80, "P": 70, "K": 75},
    {"name": "Cool & Moist", "temp": 18, "humidity": 75, "rainfall": 150, "ph": 6.5, "N": 60, "P": 55, "K": 50},
]

for test in test_cases:
    # Prepare features
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
    test_df['N_P_ratio'] = test_df['N'] / (test_df['P'] + 1)
    test_df['K_P_ratio'] = test_df['K'] / (test_df['P'] + 1)
    test_df['NPK_total'] = test_df['N'] + test_df['P'] + test_df['K']
    test_df['temp_humidity'] = test_df['temperature'] * test_df['humidity'] / 100
    test_df['rain_fertility'] = test_df['rainfall'] * test_df['NPK_total'] / 1000
    test_df['suitability_score'] = test_df.apply(calculate_suitability, axis=1)
    
    # Predict
    pred = model.predict(test_df[feature_cols])[0]
    crop = label_encoder.inverse_transform([pred])[0]
    
    # Get probability
    probs = model.predict_proba(test_df[feature_cols])[0]
    confidence = max(probs) * 100
    
    print(f"\n📋 {test['name']}:")
    print(f"   Temperature: {test['temp']}°C, Rainfall: {test['rainfall']}mm")
    print(f"   🌾 Predicted: {crop}")
    print(f"   📊 Confidence: {confidence:.1f}%")

print("\n" + "="*70)
print("✅ Model training complete!")
print("="*70)