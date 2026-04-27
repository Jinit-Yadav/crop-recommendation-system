import subprocess
import os
import sys

print("="*60)
print("🚀 KrishiVerse - Complete Model Training Pipeline")
print("="*60)

# Step 1: Prepare data
print("\n📊 STEP 1: Preparing and Cleaning Data...")
print("-"*40)
subprocess.run([sys.executable, "data_preparation.py"])

# Step 2: Train price prediction model
print("\n💰 STEP 2: Training Price Prediction Model...")
print("-"*40)
subprocess.run([sys.executable, "train_price_model.py"])

# Step 3: Train crop recommendation model (using your existing code)
print("\n🌾 STEP 3: Training Crop Recommendation Model...")
print("-"*40)
subprocess.run([sys.executable, "model_training_ensemble.py"])

print("\n" + "="*60)
print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
print("="*60)
print("\nModels saved in 'models/' directory:")
print("  - crop_recommendation_model.pkl")
print("  - price_prediction_model.pkl")
print("\nStart the application:")
print("  python app.py")
print("="*60)