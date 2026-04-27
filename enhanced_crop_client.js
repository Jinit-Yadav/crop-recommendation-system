// enhanced_crop_client.js
const fetch = require('node-fetch');

class EnhancedCropRecommenderClient {
    constructor(apiKey, baseUrl = 'http://localhost:5000/api/v1') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
    }
    
    async getEnhancedRecommendations(params) {
        const response = await fetch(`${this.baseUrl}/predict-crop-enhanced`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': this.apiKey
            },
            body: JSON.stringify({
                temperature: params.temperature,
                rainfall_category: params.rainfall_category,
                soil_fertility: params.soil_fertility,
                ph: params.ph || 6.5,
                district: params.district,
                season: params.season
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'API request failed');
        }
        
        return data;
    }
    
    async compareCrops(crops, conditions) {
        const response = await fetch(`${this.baseUrl}/compare-crops`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': this.apiKey
            },
            body: JSON.stringify({
                crops: crops,
                conditions: conditions
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'API request failed');
        }
        
        return data;
    }
    
    async getCropDetails(cropName) {
        const response = await fetch(`${this.baseUrl}/crop-details/${cropName}`, {
            headers: {
                'X-API-Key': this.apiKey
            }
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'API request failed');
        }
        
        return data;
    }
}

// Usage example
async function main() {
    const client = new EnhancedCropRecommenderClient('YOUR_API_KEY');
    
    // Get recommendations
    const recommendations = await client.getEnhancedRecommendations({
        temperature: 26,
        rainfall_category: 'Medium',
        soil_fertility: 'Medium Fertility',
        ph: 6.8,
        district: 'Pune'
    });
    
    console.log('🌟 Top Recommendation:', recommendations.recommendations[0]);
    console.log('📊 Suitability:', recommendations.recommendations[0].suitability.overall, '%');
    console.log('💰 Expected Profit:', recommendations.recommendations[0].profitability.profit_per_hectare);
    console.log('⚠️ Risk Level:', recommendations.recommendations[0].risk_assessment.level);
    
    // Compare specific crops
    const comparison = await client.compareCrops(
        ['Rice', 'Wheat', 'Maize'],
        {
            temperature: 26,
            rainfall_category: 'Medium',
            soil_fertility: 'Medium Fertility'
        }
    );
    
    console.log('\n📊 Crop Comparison:');
    comparison.comparison.forEach(crop => {
        console.log(`${crop.crop}: ${crop.suitability}% suitability, ${crop.expected_yield_tons} tons/hectare`);
    });
}

// Export for use in other modules
module.exports = EnhancedCropRecommenderClient;