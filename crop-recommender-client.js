// crop-recommender-client.js - Complete Working Client
const axios = require('axios');

class CropRecommenderClient {
    constructor(apiKey, baseUrl = 'https://crop-api-tljn.onrender.com') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        
        // Create axios instance with default config
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 30000,
            headers: {
                'X-API-Key': apiKey,
                'Content-Type': 'application/json'
            }
        });
        
        // Add response interceptor for better error handling
        this.client.interceptors.response.use(
            response => response,
            error => {
                if (error.response) {
                    console.error('API Error:', {
                        status: error.response.status,
                        data: error.response.data,
                        headers: error.response.headers
                    });
                } else if (error.request) {
                    console.error('Network Error:', error.message);
                } else {
                    console.error('Request Error:', error.message);
                }
                return Promise.reject(error);
            }
        );
    }
    
    async testConnection() {
        try {
            const response = await this.client.get('/api/v1/test');
            return response.data;
        } catch (error) {
            throw new Error(`Connection test failed: ${error.message}`);
        }
    }
    
    async getCropRecommendations(params) {
        try {
            // Validate required parameters
            if (!params.temperature) throw new Error('Temperature is required');
            if (!params.rainfall_category) throw new Error('Rainfall category is required');
            if (!params.soil_fertility) throw new Error('Soil fertility is required');
            
            const response = await this.client.post('/api/v1/predict', {
                temperature: parseFloat(params.temperature),
                rainfall_category: params.rainfall_category,
                soil_fertility: params.soil_fertility,
                ph: params.ph || 6.5
            });
            
            if (response.data.status === 'success') {
                return response.data;
            } else {
                throw new Error(response.data.error || 'Unknown error occurred');
            }
        } catch (error) {
            if (error.response) {
                throw new Error(error.response.data.error || `API Error: ${error.response.status}`);
            }
            throw error;
        }
    }
    
    async getPriceAdvice(params) {
        try {
            // Validate required parameters
            if (!params.district) throw new Error('District is required');
            if (!params.market) throw new Error('Market is required');
            if (!params.commodity) throw new Error('Commodity is required');
            
            const response = await this.client.post('/api/v1/price-advisor', {
                district: params.district,
                market: params.market,
                commodity: params.commodity,
                variety: params.variety || 'Any',
                grade: params.grade || 'A'
            });
            
            if (response.data.success) {
                return response.data;
            } else {
                throw new Error(response.data.error || 'Unknown error occurred');
            }
        } catch (error) {
            if (error.response) {
                throw new Error(error.response.data.error || `API Error: ${error.response.status}`);
            }
            throw error;
        }
    }
    
    async compareCrops(params) {
        try {
            if (!params.crops || params.crops.length === 0) {
                throw new Error('At least one crop is required for comparison');
            }
            
            if (!params.conditions) {
                throw new Error('Growing conditions are required');
            }
            
            const response = await this.client.post('/api/v1/compare-crops', {
                crops: params.crops,
                conditions: params.conditions
            });
            
            return response.data;
        } catch (error) {
            if (error.response) {
                throw new Error(error.response.data.error || `API Error: ${error.response.status}`);
            }
            throw error;
        }
    }
    
    async getCropDetails(cropName) {
        try {
            const response = await this.client.get(`/api/v1/crop-details/${encodeURIComponent(cropName)}`);
            return response.data;
        } catch (error) {
            if (error.response) {
                throw new Error(error.response.data.error || `API Error: ${error.response.status}`);
            }
            throw error;
        }
    }
    
    async getAPIInfo() {
        try {
            const response = await this.client.get('/');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get API info: ${error.message}`);
        }
    }
}

// Example usage with proper error handling
async function example() {
    const API_KEY = 'agri_ipam8n8l9r_s-2GA9_e4EL9Gb_nkrgCgwVBQ09Ktug8';
    const client = new CropRecommenderClient(API_KEY);
    
    try {
        // 1. Get API info
        console.log('📡 Getting API information...');
        const apiInfo = await client.getAPIInfo();
        console.log('✅ API Version:', apiInfo.version);
        console.log('✅ Status:', apiInfo.status);
        
        // 2. Test connection
        console.log('\n🔍 Testing connection...');
        const testResult = await client.testConnection();
        console.log('✅ Connection successful:', testResult.message);
        
        // 3. Get crop recommendations
        console.log('\n🌾 Getting crop recommendations...');
        const recommendations = await client.getCropRecommendations({
            temperature: 28,
            rainfall_category: 'High',
            soil_fertility: 'High Fertility',
            ph: 6.5
        });
        
        console.log('✅ Best Crop:', recommendations.best_crop);
        console.log('📊 Confidence:', recommendations.best_crop_confidence, '%');
        console.log('\n📋 Top 5 Recommendations:');
        
        recommendations.recommendations.forEach((crop, index) => {
            console.log(`${index + 1}. ${crop.crop} - ${crop.confidence}% confidence`);
            console.log(`   Yield: ${crop.expected_yield_tons_per_acre} tons/acre`);
            console.log(`   Profit: ₹${crop.profit_per_hectare.toLocaleString()}/hectare`);
            console.log(`   Risk: ${crop.risk_level}`);
            console.log(`   Suitability: ${crop.suitability_percentage}%`);
            console.log('');
        });
        
        // 4. Get price advice
        console.log('💰 Getting price advice...');
        const priceAdvice = await client.getPriceAdvice({
            district: 'Pune',
            market: 'Gultekdi',
            commodity: 'Rice',
            variety: 'Basmati',
            grade: 'A'
        });
        
        console.log('✅ Current Price: ₹', priceAdvice.data.current_price);
        console.log('💡 Recommendation:', priceAdvice.data.recommendation.action_text);
        console.log('📈 Expected Gain:', priceAdvice.data.expected_gain_percent, '%');
        console.log('🎯 Confidence:', priceAdvice.data.recommendation.confidence, '%');
        
        // 5. Compare crops
        console.log('\n🔄 Comparing crops...');
        const comparison = await client.compareCrops({
            crops: ['rice', 'cotton', 'wheat'],
            conditions: {
                temperature: 28,
                rainfall_category: 'Medium',
                soil_fertility: 'High Fertility',
                ph: 6.5
            }
        });
        
        console.log('✅ Best Crop for Comparison:', comparison.best_crop);
        comparison.comparison.forEach(crop => {
            console.log(`   ${crop.crop}: ${crop.suitability}% suitability`);
        });
        
        // 6. Get specific crop details
        console.log('\n📚 Getting crop details...');
        const cropDetails = await client.getCropDetails('rice');
        console.log('✅ Rice Details:', cropDetails.details);
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        if (error.response) {
            console.error('Response data:', error.response.data);
        }
    }
}

// Export for use in other files
module.exports = CropRecommenderClient;

// Run example if executed directly
if (require.main === module) {
    example();
}