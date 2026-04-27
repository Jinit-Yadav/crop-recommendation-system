// test_final.js - Use this with your actual IP
const API_URL = 'http://192.168.0.172:5000';  // Your actual IP from ipconfig
const API_KEY = 'agri_8BzZERoE4ozpReGohEnS2t1JCFy98m52dgtGamXKF_g';

async function testAPI() {
  console.log('🔑 Testing with API Key:', API_KEY);
  console.log('📡 Connecting to:', API_URL);
  console.log('=' . repeat(50));
  
  try {
    // Test connection
    console.log('\n1️⃣ Testing connection...');
    const testRes = await fetch(`${API_URL}/api/v1/test`, {
      method: 'GET',
      headers: { 
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
      }
    });
    
    if (!testRes.ok) {
      throw new Error(`HTTP error! status: ${testRes.status}`);
    }
    
    const testData = await testRes.json();
    console.log('✅ Connected!', testData);
    
    // Test crop recommendation
    console.log('\n2️⃣ Testing crop recommendation...');
    const cropRes = await fetch(`${API_URL}/api/v1/predict-crop`, {
      method: 'POST',
      headers: {
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        temperature: 28,
        soil_type: 'Medium Fertility',
        rainfall_category: 'Medium'
      })
    });
    
    const cropData = await cropRes.json();
    console.log('✅ Crop recommendation successful!');
    console.log('🌱 Best crop:', cropData.data?.best_crop);
    console.log('📊 Confidence:', cropData.data?.confidence_percentage);
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    console.log('\n💡 Troubleshooting:');
    console.log('1. Is Flask running? Run: python app.py');
    console.log('2. Did you set host="0.0.0.0" in app.py?');
    console.log('3. Try opening in browser: http://192.168.0.172:5000');
    console.log('4. Check Windows Firewall - might be blocking port 5000');
  }
}

testAPI();