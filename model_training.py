# model_training_ensemble.py
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("✅ Training ENSEMBLE model with multiple algorithms...")

def prepare_enhanced_data():
    """Prepare data with better feature engineering"""
    
    df = pd.read_csv("crop_recommendation.csv")
    
    # === FEATURE ENGINEERING - Create more informative features ===
    
    # 1. Temperature with more granular categories
    df['temp_category'] = pd.cut(df['temperature'], 
                                 bins=[0, 15, 25, 35, 50],
                                 labels=['Cool', 'Moderate', 'Warm', 'Hot'])
    
    # 2. Humidity with more granular categories
    df['humidity_category'] = pd.cut(df['humidity'],
                                     bins=[0, 50, 70, 85, 100],
                                     labels=['Dry', 'Moderate', 'Humid', 'Very Humid'])
    
    # 3. Rainfall with more granular categories
    df['rainfall_category'] = pd.cut(df['rainfall'],
                                     bins=[0, 80, 150, 250, 400],
                                     labels=['Very Low', 'Low', 'Medium', 'High'])
    
    # 4. Soil fertility score (continuous)
    df['fertility_score'] = (df['N']/140 + df['P']/145 + df['K']/205) / 3 * 100
    
    # 5. NPK ratios (important for crop selection)
    df['N_P_ratio'] = df['N'] / (df['P'] + 1)
    df['K_P_ratio'] = df['K'] / (df['P'] + 1)
    df['NPK_total'] = df['N'] + df['P'] + df['K']
    
    # 6. pH category
    df['pH_category'] = pd.cut(df['ph'],
                               bins=[0, 5.5, 6.5, 7.5, 8.5, 14],
                               labels=['Very Acidic', 'Acidic', 'Neutral', 'Alkaline', 'Very Alkaline'])
    
    # Select features
    feature_cols = [
        'temperature', 'fertility_score', 'N_P_ratio', 'K_P_ratio', 'NPK_total',
        'temp_category', 'humidity_category', 'rainfall_category', 'pH_category'
    ]
    
    X = df[feature_cols].copy()
    y = df['label']
    
    # One-hot encode categoricals
    categorical_cols = ['temp_category', 'humidity_category', 'rainfall_category', 'pH_category']
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    # Convert boolean to int (0/1) for better compatibility
    bool_cols = X_encoded.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        X_encoded[col] = X_encoded[col].astype(int)
    
    print(f"\n📊 Total features created: {X_encoded.shape[1]}")
    print(f"   Feature names: {X_encoded.columns.tolist()}")
    
    return X_encoded, y, X_encoded.columns.tolist()

def train_ensemble_model():
    """Train multiple models and combine them"""
    
    X, y, feature_columns = prepare_enhanced_data()
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 Training set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    
    # === INDIVIDUAL MODELS ===
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,  # Reduced for faster training
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
    }
    
    # Train individual models
    individual_accuracies = {}
    trained_models = {}
    
    print("\n" + "="*60)
    print("TRAINING INDIVIDUAL MODELS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        individual_accuracies[name] = acc
        trained_models[name] = model
        print(f"   ✅ Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=3)  # Reduced folds for speed
        print(f"   📊 CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # === SIMPLE ENSEMBLE: Just use the best individual model ===
    best_individual = max(individual_accuracies, key=individual_accuracies.get)
    best_accuracy = individual_accuracies[best_individual]
    
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)
    print(f"\n🏆 Best Individual Model: {best_individual} with {best_accuracy*100:.1f}% accuracy")
    
    # Create a simple voting ensemble (without CatBoost to avoid issues)
    if len(trained_models) >= 2:
        try:
            # Simple Voting Classifier
            estimators = [(name, model) for name, model in trained_models.items()]
            voting_clf = VotingClassifier(estimators=estimators, voting='soft')
            voting_clf.fit(X_train, y_train)
            y_pred_voting = voting_clf.predict(X_test)
            voting_acc = accuracy_score(y_test, y_pred_voting)
            print(f"\n🚀 Simple Voting Ensemble:")
            print(f"   ✅ Accuracy: {voting_acc:.4f} ({voting_acc*100:.1f}%)")
            
            if voting_acc > best_accuracy:
                best_accuracy = voting_acc
                best_individual = "Voting Ensemble"
                final_model = voting_clf
            else:
                final_model = trained_models[best_individual]
        except Exception as e:
            print(f"   ⚠️ Voting ensemble failed: {e}")
            final_model = trained_models[best_individual]
    else:
        final_model = trained_models[best_individual]
    
    print(f"\n🏆 FINAL MODEL: {best_individual} with {best_accuracy*100:.1f}% accuracy")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    
    model_package = {
        'model': final_model,
        'label_encoder': label_encoder,
        'feature_columns': feature_columns,
        'crops': label_encoder.classes_.tolist(),
        'accuracy': best_accuracy,
        'model_name': best_individual,
        'individual_accuracies': individual_accuracies,
        'training_date': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Save the model
    joblib.dump(model_package, 'models/crop_model_ensemble.pkl')
    print(f"\n💾 Model saved to: models/crop_model_ensemble.pkl")
    
    # Also save a simplified version for the app
    simple_package = {
        'model': final_model,
        'label_encoder': label_encoder,
        'feature_columns': feature_columns,
        'crops': label_encoder.classes_.tolist(),
        'accuracy': best_accuracy,
        'model_name': best_individual
    }
    joblib.dump(simple_package, 'models/crop_model.pkl')
    print(f"💾 Model also saved to: models/crop_model.pkl (for app use)")
    
    return model_package

if __name__ == "__main__":
    model_package = train_ensemble_model()
    
    # Test with sample predictions
    print("\n" + "="*60)
    print("TESTING WITH SIMPLE INPUTS")
    print("="*60)
    
    # Simple test cases that match your UI
    test_scenarios = [
        {'temp': 25, 'soil': 'Medium Fertility', 'rain': 'Medium'},
        {'temp': 30, 'soil': 'High Fertility', 'rain': 'Low'},
        {'temp': 20, 'soil': 'Low Fertility', 'rain': 'High'}
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n🔮 Test {i+1}: Temp={scenario['temp']}°C, Soil={scenario['soil']}, Rain={scenario['rain']}")
        
        # Derive features from simple inputs
        soil_map = {'Low Fertility': 30, 'Medium Fertility': 60, 'High Fertility': 90}
        rain_map = {'Low': 50, 'Medium': 150, 'High': 300}
        
        fertility = soil_map[scenario['soil']]
        rainfall = rain_map[scenario['rain']]
        
        # Determine categories
        if scenario['temp'] < 15:
            temp_cat = 'Cool'
        elif scenario['temp'] < 25:
            temp_cat = 'Moderate'
        elif scenario['temp'] < 35:
            temp_cat = 'Warm'
        else:
            temp_cat = 'Hot'
            
        if rainfall < 80:
            rain_cat = 'Very Low'
        elif rainfall < 150:
            rain_cat = 'Low'
        elif rainfall < 250:
            rain_cat = 'Medium'
        else:
            rain_cat = 'High'
        
        # Create input dictionary
        input_dict = {
            'temperature': scenario['temp'],
            'fertility_score': fertility,
            'N_P_ratio': 1.0,
            'K_P_ratio': 1.0,
            'NPK_total': fertility * 2
        }
        
        # Add category columns
        for col in model_package['feature_columns']:
            if col.startswith('temp_category_'):
                input_dict[col] = 1 if col == f'temp_category_{temp_cat}' else 0
            elif col.startswith('humidity_category_'):
                input_dict[col] = 1 if col == 'humidity_category_Moderate' else 0
            elif col.startswith('rainfall_category_'):
                input_dict[col] = 1 if col == f'rainfall_category_{rain_cat}' else 0
            elif col.startswith('pH_category_'):
                input_dict[col] = 1 if col == 'pH_category_Neutral' else 0
        
        # Create DataFrame
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[model_package['feature_columns']]
        
        # Predict
        pred = model_package['model'].predict(input_df)[0]
        crop = model_package['label_encoder'].inverse_transform([pred])[0]
        
        # Get probabilities if available
        if hasattr(model_package['model'], 'predict_proba'):
            probs = model_package['model'].predict_proba(input_df)[0]
            confidence = probs[pred] * 100
            print(f"   ✅ Predicted: {crop} (Confidence: {confidence:.1f}%)")
        else:
            print(f"   ✅ Predicted: {crop}")