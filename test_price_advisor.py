# test_price_advisor.py
import requests
import json

# First, let's create/get an API key
import os
import secrets
from datetime import datetime, timedelta

# Create API key if not exists
keys_file = 'production_keys.json'
if not os.path.exists(keys_file):
    api_key = f"test_{secrets.token_urlsafe(32)}"
    keys = {
        api_key: {
            'name': 'test_user',
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=365)).isoformat(),
            'active': True
        }
    }
    with open(keys_file, 'w') as f:
        json.dump(keys, f, indent=2)
else:
    with open(keys_file, 'r') as f:
        keys = json.load(f)
        api_key = list(keys.keys())[0]

BASE_URL = "http://127.0.0.1:5000"

print("="*70)
print("💰 PRICE ADVISOR API TEST")
print("="*70)
print(f"\n🔑 Using API Key: {api_key[:30]}...")

# Test the test endpoint
print("\n1. Testing API connection...")
response = requests.get(
    f"{BASE_URL}/api/v1/test",
    headers={"X-API-Key": api_key}
)
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    print("   ✅ API is working!")

# Test price advisor for different commodities
test_cases = [
    {
        "name": "Rice in Pune",
        "data": {
            "district": "Pune",
            "market": "Pune",
            "commodity": "Rice",
            "variety": "Local",
            "grade": "A"
        }
    },
    {
        "name": "Cotton in Nagpur", 
        "data": {
            "district": "Nagpur",
            "market": "Nagpur",
            "commodity": "Cotton",
            "variety": "Local",
            "grade": "A"
        }
    },
    {
        "name": "Wheat in Solapur",
        "data": {
            "district": "Solapur",
            "market": "Solapur",
            "commodity": "Wheat",
            "variety": "Local",
            "grade": "A"
        }
    }
]

print("\n" + "="*70)
print("📊 PRICE TREND ANALYSIS")
print("="*70)

for test in test_cases:
    print(f"\n{'='*50}")
    print(f"📋 {test['name']}")
    print(f"{'='*50}")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/price-advisor",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key
        },
        json=test['data']
    )
    
    if response.status_code == 200:
        result = response.json()
        data = result['data']
        
        print(f"\n💰 Current Price: ₹{data['current_price']:,.2f}")
        print(f"\n📈 RECOMMENDATION:")
        print(f"   Action: {data['recommendation']['action_text']}")
        print(f"   Reason: {data['recommendation']['reason']}")
        print(f"   Urgency: {data['recommendation']['urgency']}")
        print(f"   Confidence: {data['recommendation']['confidence']}%")
        
        print(f"\n📊 Price Forecast:")
        for forecast in data['price_forecast']:
            change_symbol = "+" if forecast['change_percent'] > 0 else ""
            print(f"   {forecast['weeks_ahead']} weeks: ₹{forecast['predicted_price']:,.2f} ({change_symbol}{forecast['change_percent']}%)")
        
        print(f"\n📉 Volatility: {data['volatility']}")
        print(f"💡 Best Holding Period: {data['best_holding_period']}")
        print(f"💰 Expected Gain: {data['expected_gain_percent']:+.1f}%")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}")

print("\n" + "="*70)
print("✅ Testing Complete!")
print("="*70)