# diagnose_model.py
import joblib
import pandas as pd

print("🔍 Diagnosing Your ML Model")
print("="*60)

try:
    # Load your model
    model_package = joblib.load('models/crop_model_ensemble.pkl')
    
    print("\n✅ Model loaded successfully!")
    print(f"\n📊 Model Type: {model_package.get('model_name', 'Unknown')}")
    print(f"📈 Model Accuracy: {model_package.get('accuracy', 'N/A')}")
    
    # Check what features the model expects
    if 'feature_columns' in model_package:
        print(f"\n🔧 Number of features expected: {len(model_package['feature_columns'])}")
        print("\n📋 Feature columns the model expects:")
        for i, col in enumerate(model_package['feature_columns'][:20]):  # Show first 20
            print(f"   {i+1}. {col}")
        if len(model_package['feature_columns']) > 20:
            print(f"   ... and {len(model_package['feature_columns']) - 20} more features")
    else:
        print("\n⚠️ No feature_columns found in model package")
        
    # Check available crops
    if 'crops' in model_package:
        print(f"\n🌾 Crops the model can predict ({len(model_package['crops'])} crops):")
        for i, crop in enumerate(model_package['crops'][:10]):
            print(f"   {i+1}. {crop}")
    
    # Check the model object
    if 'model' in model_package:
        model = model_package['model']
        print(f"\n🤖 Model type: {type(model).__name__}")
        
        # If it's a sklearn model, we can get feature importances
        if hasattr(model, 'feature_importances_'):
            print(f"\n📊 Model has feature importances available")
            
except FileNotFoundError:
    print("\n❌ Model file not found at 'models/crop_model_ensemble.pkl'")
    print("   Please train your model first using: python model_training_ensemble.py")
except Exception as e:
    print(f"\n❌ Error loading model: {e}")