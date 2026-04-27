# run_all.py
import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print('='*60)
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(result.stdout[-500:])  # Print last 500 chars
    else:
        print(f"❌ Error in {description}")
        print(result.stderr)
        return False
    return True

def main():
    print("="*60)
    print("🚀 CROP RECOMMENDATION SYSTEM - COMPLETE SETUP")
    print("="*60)
    
    # Step 1: Create enhanced crop data
    if not os.path.exists('enhanced_crop_data.csv'):
        if not run_command("python enhanced_crop_data.py", "Creating enhanced crop dataset"):
            print("\n⚠️ Could not create enhanced data. Using existing data...")
    else:
        print("\n✅ Enhanced crop data already exists")
    
    # Step 2: Train enhanced model
    if not run_command("python run_enhanced_training.py", "Training enhanced crop model"):
        print("\n❌ Model training failed!")
        return False
    
    # Step 3: Retrain price model
    if not run_command("python retrain_price_model.py", "Training price prediction model"):
        print("\n⚠️ Price model training had issues, but app may still work with fallback")
    
    print("\n" + "="*60)
    print("🎉 ALL SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Next steps:")
    print("   1. Run: python app.py")
    print("   2. Open browser to: http://localhost:5000")
    print("   3. Navigate to Crop Recommendation page")
    print("\n💡 Tips:")
    print("   - Make sure all CSV files are in the same directory")
    print("   - The app will use enhanced model with risk and profitability metrics")
    print("   - API endpoints are available at /api/v1/ with API key authentication")
    
    return True

if __name__ == "__main__":
    main()