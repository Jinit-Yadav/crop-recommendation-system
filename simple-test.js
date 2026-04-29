// simple-test.js - Minimal test to verify API connectivity
const API_KEY = 'agri_ipam8n8l9r_s-2GA9_e4EL9Gb_nkrgCgwVBQ09Ktug8';
const BASE_URL = 'https://crop-api-tljn.onrender.com';

async function quickTest() {
    console.log('🚀 Quick API Test\n');
    
    // Test 1: Basic connectivity
    console.log('1. Testing basic connectivity...');
    try {
        const response = await fetch(`${BASE_URL}/`);
        const data = await response.json();
        console.log('   ✅ API is reachable');
        console.log('   Version:', data.version);
        console.log('   Status:', data.status);
    } catch (error) {
        console.error('   ❌ Cannot reach API:', error.message);
        return;
    }
    
    // Test 2: Authentication
    console.log('\n2. Testing authentication...');
    try {
        const response = await fetch(`${BASE_URL}/api/v1/test`, {
            method: 'GET',
            headers: { 'X-API-Key': API_KEY }
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('   ✅ Authentication successful');
            console.log('   Message:', data.message);
        } else {
            console.log('   ❌ Authentication failed:', response.status);
        }
    } catch (error) {
        console.error('   ❌ Authentication error:', error.message);
    }
    
    // Test 3: Crop prediction
    console.log('\n3. Testing crop prediction...');
    try {
        const response = await fetch(`${BASE_URL}/api/v1/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY
            },
            body: JSON.stringify({
                temperature: 28,
                rainfall_category: 'High',
                soil_fertility: 'High Fertility',
                ph: 6.5
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('   ✅ Prediction successful');
            console.log('   Best Crop:', data.best_crop);
            console.log('   Confidence:', data.best_crop_confidence + '%');
            console.log('   Recommendations:', data.recommendations.length);
        } else {
            const error = await response.json();
            console.log('   ❌ Prediction failed:', error.error);
        }
    } catch (error) {
        console.error('   ❌ Prediction error:', error.message);
    }
    
    console.log('\n✨ Test complete');
}

quickTest();