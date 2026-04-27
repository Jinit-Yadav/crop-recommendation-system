# generate_api_key.py
import secrets
import json
import os
from datetime import datetime, timedelta
import sys

KEYS_FILE = 'production_keys.json'

def generate_api_key(name, expiry_days=365):
    """Generate a new API key for a user"""
    
    # Generate a secure random key
    api_key = f"agri_{secrets.token_urlsafe(32)}"
    
    # Load existing keys
    keys = {}
    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, 'r') as f:
            keys = json.load(f)
    
    # Store the key
    keys[api_key] = {
        'name': name,
        'created_at': datetime.now().isoformat(),
        'expires_at': (datetime.now() + timedelta(days=expiry_days)).isoformat(),
        'active': True
    }
    
    # Save back to file
    with open(KEYS_FILE, 'w') as f:
        json.dump(keys, f, indent=2)
    
    return api_key

def list_api_keys():
    """List all active API keys"""
    if not os.path.exists(KEYS_FILE):
        print("No API keys found.")
        return
    
    with open(KEYS_FILE, 'r') as f:
        keys = json.load(f)
    
    print("\n" + "="*70)
    print("📋 ACTIVE API KEYS")
    print("="*70)
    
    for key, info in keys.items():
        print(f"\n🔑 User: {info['name']}")
        print(f"   Key: {key[:30]}...")
        print(f"   Created: {info['created_at'][:10]}")
        print(f"   Expires: {info['expires_at'][:10]}")
        print(f"   Active: {info['active']}")

def revoke_api_key(key_prefix):
    """Revoke an API key"""
    if not os.path.exists(KEYS_FILE):
        print("No API keys found.")
        return
    
    with open(KEYS_FILE, 'r') as f:
        keys = json.load(f)
    
    found = False
    for key, info in keys.items():
        if key.startswith(key_prefix) or info['name'].lower() == key_prefix.lower():
            keys[key]['active'] = False
            found = True
            print(f"✅ Revoked key for: {info['name']}")
    
    if found:
        with open(KEYS_FILE, 'w') as f:
            json.dump(keys, f, indent=2)
    else:
        print(f"❌ No key found matching: {key_prefix}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔑 API KEY MANAGEMENT SYSTEM")
    print("="*70)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python generate_api_key.py generate <name> [days]")
        print("  python generate_api_key.py list")
        print("  python generate_api_key.py revoke <name_or_key_prefix>")
        print("\nExamples:")
        print("  python generate_api_key.py generate Vaibhav 365")
        print("  python generate_api_key.py generate John 30")
        print("  python generate_api_key.py list")
        print("  python generate_api_key.py revoke Vaibhav")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "generate":
        if len(sys.argv) < 3:
            print("❌ Please provide a name")
            sys.exit(1)
        
        name = sys.argv[2]
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 365
        
        api_key = generate_api_key(name, days)
        
        # Get server IP
        import socket
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
        except:
            local_ip = "your-server-ip"
        
        print("\n" + "="*70)
        print(f"🔑 API KEY FOR {name.upper()}")
        print("="*70)
        print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Expires: {(datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')}")
        print(f"\nAPI Key: {api_key}")
        print("\n" + "="*70)
        print("📌 ENDPOINTS")
        print("="*70)
        print(f"\n🌾 Crop Recommendation:")
        print(f"   POST http://{local_ip}:5000/api/v1/predict")
        print(f"\n💰 Price Advisor:")
        print(f"   POST http://{local_ip}:5000/api/v1/price-advisor")
        print(f"\n🧪 Test API:")
        print(f"   GET http://{local_ip}:5000/api/v1/test")
        print("\n" + "="*70)
        print("📋 Node.js Usage Example:")
        print("-"*70)
        print(f"""
const axios = require('axios');

const API_KEY = '{api_key}';
const BASE_URL = 'http://{local_ip}:5000';

// Get crop recommendation
async function getCropRecommendation() {{
    const response = await axios.post(
        `${{BASE_URL}}/api/v1/predict`,
        {{
            temperature: 28,
            rainfall_category: "High",
            soil_fertility: "High Fertility",
            ph: 6.2
        }},
        {{
            headers: {{ 'X-API-Key': API_KEY }}
        }}
    );
    console.log('Best crop:', response.data.best_crop);
}}

// Get price advice
async function getPriceAdvice() {{
    const response = await axios.post(
        `${{BASE_URL}}/api/v1/price-advisor`,
        {{
            district: "Pune",
            market: "Pune",
            commodity: "Rice",
            variety: "Local",
            grade: "A"
        }},
        {{
            headers: {{ 'X-API-Key': API_KEY }}
        }}
    );
    console.log('Recommendation:', response.data.data.recommendation.action_text);
}}

getCropRecommendation();
        """)
        print("="*70)
        
        # Save to file for sharing
        filename = f"api_key_{name.lower().replace(' ', '_')}.txt"
        with open(filename, 'w') as f:
            f.write(f"API Key for {name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Expires: {(datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')}\n")
            f.write(f"\nAPI Key: {api_key}\n\n")
            f.write("ENDPOINTS:\n")
            f.write(f"- Crop: POST http://{local_ip}:5000/api/v1/predict\n")
            f.write(f"- Price: POST http://{local_ip}:5000/api/v1/price-advisor\n")
            f.write(f"- Test: GET http://{local_ip}:5000/api/v1/test\n\n")
            f.write("Node.js Example:\n")
            f.write("```javascript\n")
            f.write(f"const API_KEY = '{api_key}';\n")
            f.write(f"const BASE_URL = 'http://{local_ip}:5000';\n\n")
            f.write("const response = await axios.post(`${BASE_URL}/api/v1/predict`, {\n")
            f.write("    temperature: 28,\n")
            f.write("    rainfall_category: 'High',\n")
            f.write("    soil_fertility: 'High Fertility',\n")
            f.write("    ph: 6.2\n")
            f.write("}, { headers: { 'X-API-Key': API_KEY } });\n")
            f.write("```\n")
        
        print(f"\n📄 Key also saved to: {filename}")
        print("Send this file to the user!\n")
        
    elif command == "list":
        list_api_keys()
        
    elif command == "revoke":
        if len(sys.argv) < 3:
            print("❌ Please provide name or key prefix to revoke")
            sys.exit(1)
        revoke_api_key(sys.argv[2])
        
    else:
        print(f"❌ Unknown command: {command}")
        print("Available: generate, list, revoke")