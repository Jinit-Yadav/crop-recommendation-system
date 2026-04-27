# test_improved_api.py
import requests
import json

API_KEY = "agri_test__pxv3bwZ9UsuYJcevVOoSjKHjcV_0_3cCcBlWjxbyDg"
BASE_URL = "http://127.0.0.1:5000"

def test_varied_conditions():
    print("="*70)
    print("🌾 TESTING ML MODEL WITH DIFFERENT CONDITIONS")
    print("="*70)
    
    test_scenarios = [
        {
            "name": "🌡️ Hot & Arid (Summer in Rajasthan)",
            "conditions": {
                "temperature": 38,
                "rainfall_category": "Very Low",
                "soil_fertility": "Low Fertility",
                "ph": 7.8
            }
        },
        {
            "name": "🌧️ Warm & Humid (Monsoon in Kerala)",
            "conditions": {
                "temperature": 28,
                "rainfall_category": "High",
                "soil_fertility": "High Fertility",
                "ph": 6.2
            }
        },
        {
            "name": "❄️ Cool & Moist (Hills of Himachal)",
            "conditions": {
                "temperature": 15,
                "rainfall_category": "Medium",
                "soil_fertility": "Medium Fertility",
                "ph": 6.5
            }
        },
        {
            "name": "🌾 Temperate Plains (Punjab)",
            "conditions": {
                "temperature": 22,
                "rainfall_category": "Medium",
                "soil_fertility": "High Fertility",
                "ph": 7.0
            }
        },
        {
            "name": "🏜️ Dry & Warm (Deccan Plateau)",
            "conditions": {
                "temperature": 32,
                "rainfall_category": "Low",
                "soil_fertility": "Medium Fertility",
                "ph": 7.2
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*70}")
        print(f"{scenario['name']}")
        print(f"{'='*70}")
        print(f"Input: {json.dumps(scenario['conditions'], indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/api/v1/predict-crop-enhanced",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": API_KEY
            },
            json=scenario['conditions']
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ ML Prediction:")
            print(f"   🌾 Top Crop: {result['top_recommendation']['crop']}")
            print(f"   📊 Confidence: {result['top_recommendation']['confidence']}%")
            
            print(f"\n   🏆 Top 3 Recommendations:")
            for i, pred in enumerate(result['predictions'], 1):
                print(f"      {i}. {pred['crop']} ({pred['confidence']}%)")
            
            print(f"\n   📈 Confidence Level: {result['interpretation']['confidence_level']}")
            print(f"   💡 Suggestion: {result['interpretation']['suggestion']}")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"   {response.text}")

if __name__ == "__main__":
    test_varied_conditions()