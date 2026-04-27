import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_enhanced_crop_data():
    """Create enhanced dataset with yield, profitability, and risk data"""
    
    # Define comprehensive crop database with all metrics
    crop_database = {
        'Rice': {
            'temp_min': 20, 'temp_max': 35,
            'rainfall_min': 100, 'rainfall_max': 250,
            'ph_min': 5.5, 'ph_max': 6.5,
            'base_yield': 2.8,  # tons per hectare
            'profit_per_hectare': 35000,  # INR
            'profit_margin': 0.28,
            'risk_factors': {'weather': 0.3, 'pest': 0.4, 'market': 0.2},
            'market_demand': 'High',
            'growing_days': 120,
            'soil_requirement': 'Clay loam with good water retention'
        },
        'Wheat': {
            'temp_min': 12, 'temp_max': 28,
            'rainfall_min': 50, 'rainfall_max': 120,
            'ph_min': 6.0, 'ph_max': 7.5,
            'base_yield': 3.2,
            'profit_per_hectare': 40000,
            'profit_margin': 0.32,
            'risk_factors': {'weather': 0.35, 'pest': 0.2, 'market': 0.15},
            'market_demand': 'Very High',
            'growing_days': 110,
            'soil_requirement': 'Well-drained loamy soil'
        },
        'Cotton': {
            'temp_min': 21, 'temp_max': 37,
            'rainfall_min': 60, 'rainfall_max': 100,
            'ph_min': 6.0, 'ph_max': 7.5,
            'base_yield': 2.2,
            'profit_per_hectare': 55000,
            'profit_margin': 0.35,
            'risk_factors': {'weather': 0.4, 'pest': 0.6, 'market': 0.3},
            'market_demand': 'High',
            'growing_days': 150,
            'soil_requirement': 'Black cotton soil or deep loamy soil'
        },
        'Maize': {
            'temp_min': 18, 'temp_max': 32,
            'rainfall_min': 60, 'rainfall_max': 150,
            'ph_min': 5.5, 'ph_max': 7.0,
            'base_yield': 4.5,
            'profit_per_hectare': 32000,
            'profit_margin': 0.26,
            'risk_factors': {'weather': 0.25, 'pest': 0.3, 'market': 0.25},
            'market_demand': 'High',
            'growing_days': 100,
            'soil_requirement': 'Well-drained fertile soil'
        },
        'Sugarcane': {
            'temp_min': 20, 'temp_max': 35,
            'rainfall_min': 100, 'rainfall_max': 200,
            'ph_min': 6.0, 'ph_max': 7.5,
            'base_yield': 75,
            'profit_per_hectare': 80000,
            'profit_margin': 0.22,
            'risk_factors': {'weather': 0.3, 'pest': 0.25, 'market': 0.35},
            'market_demand': 'Medium',
            'growing_days': 365,
            'soil_requirement': 'Deep rich loamy soil'
        },
        'Groundnut': {
            'temp_min': 20, 'temp_max': 30,
            'rainfall_min': 50, 'rainfall_max': 80,
            'ph_min': 6.0, 'ph_max': 7.0,
            'base_yield': 1.8,
            'profit_per_hectare': 45000,
            'profit_margin': 0.34,
            'risk_factors': {'weather': 0.3, 'pest': 0.25, 'market': 0.2},
            'market_demand': 'Medium',
            'growing_days': 105,
            'soil_requirement': 'Light sandy loam'
        },
        'Soybean': {
            'temp_min': 20, 'temp_max': 32,
            'rainfall_min': 60, 'rainfall_max': 100,
            'ph_min': 6.0, 'ph_max': 7.0,
            'base_yield': 2.5,
            'profit_per_hectare': 38000,
            'profit_margin': 0.30,
            'risk_factors': {'weather': 0.28, 'pest': 0.3, 'market': 0.22},
            'market_demand': 'High',
            'growing_days': 90,
            'soil_requirement': 'Well-drained loamy soil'
        },
        'Bajra': {
            'temp_min': 25, 'temp_max': 40,
            'rainfall_min': 40, 'rainfall_max': 80,
            'ph_min': 7.0, 'ph_max': 8.5,
            'base_yield': 2.0,
            'profit_per_hectare': 28000,
            'profit_margin': 0.30,
            'risk_factors': {'weather': 0.2, 'pest': 0.25, 'market': 0.25},
            'market_demand': 'Medium',
            'growing_days': 80,
            'soil_requirement': 'Sandy to loamy soil'
        },
        'Jowar': {
            'temp_min': 25, 'temp_max': 35,
            'rainfall_min': 45, 'rainfall_max': 100,
            'ph_min': 6.0, 'ph_max': 7.5,
            'base_yield': 2.2,
            'profit_per_hectare': 30000,
            'profit_margin': 0.28,
            'risk_factors': {'weather': 0.22, 'pest': 0.28, 'market': 0.25},
            'market_demand': 'Medium',
            'growing_days': 100,
            'soil_requirement': 'Well-drained black soil'
        },
        'Turmeric': {
            'temp_min': 20, 'temp_max': 35,
            'rainfall_min': 100, 'rainfall_max': 200,
            'ph_min': 5.5, 'ph_max': 7.0,
            'base_yield': 8.0,
            'profit_per_hectare': 120000,
            'profit_margin': 0.45,
            'risk_factors': {'weather': 0.35, 'pest': 0.4, 'market': 0.3},
            'market_demand': 'High',
            'growing_days': 210,
            'soil_requirement': 'Rich loamy soil with good drainage'
        }
    }
    
    np.random.seed(42)
    n_samples = 10000
    data = []
    
    for _ in range(n_samples):
        crop = np.random.choice(list(crop_database.keys()))
        crop_info = crop_database[crop]
        
        # Generate realistic parameters within optimal ranges
        temp = np.random.uniform(crop_info['temp_min'] - 5, crop_info['temp_max'] + 5)
        rainfall = np.random.uniform(crop_info['rainfall_min'] - 30, crop_info['rainfall_max'] + 50)
        ph = np.random.uniform(crop_info['ph_min'] - 0.5, crop_info['ph_max'] + 0.5)
        
        # Soil nutrients (N, P, K)
        n = np.random.uniform(30, 140)
        p = np.random.uniform(20, 145)
        k = np.random.uniform(30, 205)
        
        # Calculate suitability scores
        temp_optimal = (crop_info['temp_min'] + crop_info['temp_max']) / 2
        temp_score = max(0, 100 - abs(temp - temp_optimal) / 10 * 20)
        
        rain_optimal = (crop_info['rainfall_min'] + crop_info['rainfall_max']) / 2
        rain_score = max(0, 100 - abs(rainfall - rain_optimal) / 50 * 20)
        
        ph_score = max(0, 100 - abs(ph - 6.5) / 2 * 30)
        
        # Overall suitability
        overall_suitability = (temp_score * 0.4 + rain_score * 0.35 + ph_score * 0.25)
        
        # Calculate expected yield based on conditions
        yield_factor = overall_suitability / 100
        expected_yield = crop_info['base_yield'] * yield_factor * np.random.uniform(0.9, 1.1)
        
        # Calculate profit based on yield and market conditions
        estimated_profit = crop_info['profit_per_hectare'] * yield_factor * np.random.uniform(0.85, 1.15)
        
        # Determine risk level
        weather_risk = crop_info['risk_factors']['weather'] * (1 - temp_score/100)
        pest_risk = crop_info['risk_factors']['pest'] * np.random.uniform(0.5, 1)
        market_risk = crop_info['risk_factors']['market'] * np.random.uniform(0.3, 1)
        overall_risk = (weather_risk + pest_risk + market_risk) / 3
        
        if overall_risk >= 0.4:
            risk_level = 'High'
        elif overall_risk >= 0.25:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Determine profitability level
        if estimated_profit > 60000:
            profitability = 'High'
        elif estimated_profit > 35000:
            profitability = 'Medium'
        else:
            profitability = 'Low'
        
        data.append({
            'temperature': round(temp, 1),
            'humidity': round(np.random.uniform(40, 85), 1),
            'rainfall': round(rainfall, 1),
            'N': round(n, 1),
            'P': round(p, 1),
            'K': round(k, 1),
            'ph': round(ph, 1),
            'label': crop,
            'suitability_score': round(overall_suitability, 1),
            'expected_yield': round(expected_yield, 2),
            'estimated_profit': round(estimated_profit, 0),
            'profit_margin': crop_info['profit_margin'],
            'profitability': profitability,
            'risk_level': risk_level,
            'risk_score': round(overall_risk * 100, 1),
            'market_demand': crop_info['market_demand'],
            'growing_days': crop_info['growing_days'],
            'soil_requirement': crop_info['soil_requirement']
        })
    
    df = pd.DataFrame(data)
    df.to_csv('enhanced_crop_data.csv', index=False)
    print(f"✅ Created enhanced dataset with {len(df)} samples")
    print(f"\n📊 Dataset columns: {df.columns.tolist()}")
    print(f"\n📈 Sample data:")
    print(df.head())
    return df

if __name__ == "__main__":
    create_enhanced_crop_data()