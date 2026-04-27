const fetch = require('node-fetch');

class CropRecommenderClient {
    constructor(apiKey, baseUrl = 'https://crop-api-tljn.onrender.com/api/v1') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
    }
    
    async testConnection() {
        const response = await fetch(`${this.baseUrl}/test`, {
            method: 'GET',
            headers: {
                'X-API-Key': this.apiKey
            }
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Test failed');
        return data;
    }
    
    async getCropRecommendations(params) {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': this.apiKey
            },
            body: JSON.stringify({
                temperature: params.temperature,
                rainfall_category: params.rainfall_category,
                soil_fertility: params.soil_fertility,
                ph: params.ph || 6.5
            })
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'API request failed');
        return data;
    }
    
    async getPriceAdvice(params) {
        const response = await fetch(`${this.baseUrl}/price-advisor`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': this.apiKey
            },
            body: JSON.stringify({
                district: params.district,
                market: params.market,
                commodity: params.commodity,
                variety: params.variety || 'Any',
                grade: params.grade || 'A'
            })
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'API request failed');
        return data;
    }
}

// Usage example
async function main() {
    const client = new CropRecommenderClient('agri_ipam8n8l9r_s-2GA9_e4EL9Gb_nkrgCgwVBQ09Ktug8');
    
    try {
        // Test connection
        console.log('🔍 Testing connection...');
        const testResult = await client.testConnection();
        console.log('✅ API Connected:', testResult);
        
        // Get crop recommendations
        console.log('\n🌾 Getting crop recommendations...');
        const recommendations = await client.getCropRecommendations({
            temperature: 28,
            rainfall_category: 'High',
            soil_fertility: 'High Fertility',
            ph: 6.2
        });
        
        console.log('✅ Best Crop:', recommendations.best_crop);
        console.log('📊 Confidence:', recommendations.best_crop_confidence, '%');
        console.log('\n📋 Top 5 Recommendations:');
        recommendations.recommendations.forEach((crop, i) => {
            console.log(`${i+1}. ${crop.crop} - ${crop.confidence}% confidence`);
            console.log(`   Yield: ${crop.expected_yield_tons_per_acre} tons/acre`);
            console.log(`   Profit: ₹${crop.profit_per_hectare}/hectare`);
            console.log(`   Risk: ${crop.risk_level}`);
        });
        
        // Get price advice (example)
        console.log('\n💰 Getting price advice...');
        const priceAdvice = await client.getPriceAdvice({
            district: 'Pune',
            market: 'Gultekdi',
            commodity: 'Rice',
            variety: 'Basmati',
            grade: 'A'
        });
        
        console.log('✅ Current Price:', priceAdvice.data.current_price);
        console.log('💡 Recommendation:', priceAdvice.data.recommendation.action_text);
        console.log('📈 Expected Gain:', priceAdvice.data.expected_gain_percent, '%');
        
    } catch (error) {
        console.error('❌ Error:', error.message);
    }
}

// Run the example
main();

module.exports = CropRecommenderClient;