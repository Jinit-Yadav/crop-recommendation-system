#!/usr/bin/env python3
"""
Enhanced Crop Recommendation System with Multi-Output Predictions
This script trains models that predict: crop + yield + profit + risk level
"""

import os
import sys

def main():
    print("🚀 Enhanced Crop Recommendation System Setup")
    print("=" * 60)

    # Step 1: Train the multi-output model
    print("\n📚 Step 1: Training Multi-Output Model")
    print("-" * 40)

    try:
        from enhanced_multi_output_model import train_multi_output_model
        model = train_multi_output_model()
        print("✅ Multi-output model trained successfully!")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return

    # Step 2: Update the app to use the new model
    print("\n📱 Step 2: Updating Application")
    print("-" * 40)

    # Copy the enhanced app
    try:
        import shutil
        shutil.copy('enhanced_app.py', 'app_enhanced.py')
        print("✅ Enhanced app created as 'app_enhanced.py'")
    except Exception as e:
        print(f"❌ Error copying app: {e}")

    print("\n🎯 Enhanced Features Added:")
    print("- ✅ Risk Level Assessment (High/Medium/Low)")
    print("- ✅ Profitability Prediction (₹/hectare)")
    print("- ✅ Expected Yield (tons/hectare)")
    print("- ✅ Suitability Percentage")
    print("- ✅ Detailed Recommendation Reasons")
    print("- ✅ Market Demand Analysis")
    print("- ✅ Growing Period Information")

    print("\n📊 Model Performance:")
    print(f"- Crop Classification: {model.get('crop_accuracy', 0):.1%}")
    print(f"- Yield Prediction MAE: {model.get('yield_mae', 0):.2f} tons/ha")
    print(f"- Profit Prediction MAE: ₹{model.get('profit_mae', 0):.0f}")
    print(f"- Risk Assessment: {model.get('risk_accuracy', 0):.1%}")

    print("\n🚀 To run the enhanced system:")
    print("1. python app_enhanced.py")
    print("2. Open http://localhost:5000")
    print("3. Test the enhanced crop recommendations!")

    print("\n🔑 API Endpoints now return:")
    print("- best_crop: Recommended crop")
    print("- risk_level: High/Medium/Low")
    print("- profitability: High/Medium/Low")
    print("- expected_yield: tons/hectare")
    print("- profit_estimate: ₹/hectare")
    print("- suitability_percentage: %")
    print("- reasons: List of detailed explanations")

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\Lenovo\OneDrive\Desktop\crop_recommendation_system\crop_recommendation_system\run_enhanced_system.py