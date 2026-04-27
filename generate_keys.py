#!/usr/bin/env python3
"""
Simple script to generate API keys for your production users
Run this whenever you need to create a new API key for someone
"""

from api_key_manager import SimpleAPIKeyManager
import sys

def main():
    manager = SimpleAPIKeyManager()
    
    print("\n🔑 API Key Generator")
    print("-" * 50)
    
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == "list":
            keys = manager.list_keys()
            print("\n📋 Existing API Keys:")
            for k in keys:
                status = "✅ Active" if k['is_active'] else "❌ Inactive"
                print(f"\nKey: {k['key']}")
                print(f"  Name: {k['name']}")
                print(f"  Email: {k['email']}")
                print(f"  Status: {status}")
                print(f"  Created: {k['created_at'][:10]}")
                print(f"  Expires: {k['expires_at'][:10]}")
                print(f"  Usage: {k['usage_count']} requests")
        
        elif sys.argv[1] == "deactivate" and len(sys.argv) > 2:
            if manager.deactivate_key(sys.argv[2]):
                print(f"✅ Key deactivated successfully")
            else:
                print(f"❌ Key not found")
        
        return
    
    # Interactive mode
    print("\nThis will generate a new API key for production use")
    print("The key will be shown only once - save it securely!\n")
    
    name = input("Enter user/company name: ").strip()
    email = input("Enter email (optional): ").strip()
    
    print("\nExpiry options:")
    print("1. 30 days")
    print("2. 90 days")
    print("3. 1 year")
    print("4. Never (not recommended)")
    
    expiry_choice = input("Choose expiry (1-4) [3]: ").strip() or "3"
    
    expiry_days = {
        "1": 30,
        "2": 90,
        "3": 365,
        "4": 3650  # 10 years
    }.get(expiry_choice, 365)
    
    # Generate the key
    api_key = manager.generate_key(
        name=name,
        email=email if email else None,
        expiry_days=expiry_days
    )
    
    print("\n" + "="*60)
    print("✅ NEW API KEY GENERATED")
    print("="*60)
    print(f"\n🔐 API Key: {api_key}")
    print(f"\n📋 Details:")
    print(f"   Name: {name}")
    print(f"   Email: {email or 'Not provided'}")
    print(f"   Expires: {(datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')}")
    print(f"\n⚠️  IMPORTANT: Save this key now! It won't be shown again.")
    print("="*60)
    
    # Save to a text file for sharing
    filename = f"apikey_{name.lower().replace(' ', '_')}.txt"
    with open(filename, 'w') as f:
        f.write(f"API Key for {name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Expires: {(datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')}\n")
        f.write(f"\nKey: {api_key}\n")
        f.write("\n--- Usage Instructions ---\n")
        f.write("Include this key in your requests:\n")
        f.write("Header: X-API-Key: your-key-here\n")
        f.write("Or as query parameter: ?api_key=your-key-here\n")
    
    print(f"\n📄 Key also saved to: {filename}")

if __name__ == "__main__":
    from datetime import datetime, timedelta
    main()