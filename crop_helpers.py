import numpy as np

def calculate_actual_suitability(crop_name, temperature, rainfall, soil_fertility, crop_metrics):
    """Calculate actual suitability percentage based on current conditions"""
    
    if crop_name not in crop_metrics:
        return 70, "Average suitability based on general conditions"
    
    metrics = crop_metrics[crop_name]
    
    # Temperature suitability
    temp_optimal = (metrics['temp_min'] + metrics['temp_max']) / 2
    temp_range = metrics['temp_max'] - metrics['temp_min']
    
    if temperature < metrics['temp_min']:
        temp_score = max(0, 100 - (metrics['temp_min'] - temperature) / 10 * 30)
    elif temperature > metrics['temp_max']:
        temp_score = max(0, 100 - (temperature - metrics['temp_max']) / 10 * 30)
    else:
        temp_score = 100 - abs(temperature - temp_optimal) / temp_range * 20
    
    # Rainfall suitability
    if rainfall < metrics['rain_min']:
        rain_score = max(0, 100 - (metrics['rain_min'] - rainfall) / 20 * 40)
    elif rainfall > metrics['rain_max']:
        rain_score = max(0, 100 - (rainfall - metrics['rain_max']) / 30 * 40)
    else:
        rain_score = 100
    
    # Soil fertility suitability
    fertility_score = min(100, (soil_fertility / 70) * 100)
    
    # Weighted average
    total_score = (temp_score * 0.4 + rain_score * 0.35 + fertility_score * 0.25)
    total_score = min(100, max(0, total_score))
    
    # Generate reason
    reasons = []
    if temp_score < 70:
        reasons.append(f"Temperature {temperature}°C is {'below' if temperature < metrics['temp_min'] else 'above'} optimal range ({metrics['temp_min']}-{metrics['temp_max']}°C)")
    if rain_score < 70:
        reasons.append(f"Rainfall {rainfall}mm is {'below' if rainfall < metrics['rain_min'] else 'above'} optimal requirement ({metrics['rain_min']}-{metrics['rain_max']}mm)")
    if fertility_score < 70:
        reasons.append(f"Soil fertility needs improvement for optimal {crop_name} growth")
    
    if len(reasons) == 0:
        reason = f"Excellent conditions! Temperature, rainfall, and soil are perfect for {crop_name}"
    elif len(reasons) == 1:
        reason = reasons[0] + ". Other conditions are favorable."
    else:
        reason = "; ".join(reasons[:2]) + ". Consider addressing these factors."
    
    return round(total_score, 1), reason

def calculate_expected_yield(crop_name, suitability_score, crop_metrics):
    """Calculate expected yield based on suitability"""
    
    metrics = crop_metrics.get(crop_name, {'avg_yield': 2.5})
    base_yield = metrics['avg_yield']
    
    # Adjust yield based on suitability
    adjusted_yield = base_yield * (suitability_score / 100)
    
    # Add realistic variation
    variation = np.random.uniform(0.9, 1.1)
    final_yield = round(adjusted_yield * variation, 1)
    
    return final_yield

def calculate_expected_profit(crop_name, suitability_score, crop_metrics):
    """Calculate expected profit based on suitability"""
    
    metrics = crop_metrics.get(crop_name, {'avg_profit': 35000})
    base_profit = metrics['avg_profit']
    
    # Adjust profit based on suitability
    adjusted_profit = base_profit * (suitability_score / 100)
    
    # Add realistic variation
    variation = np.random.uniform(0.9, 1.1)
    final_profit = round(adjusted_profit * variation, 0)
    
    return final_profit

def assess_risk_level(crop_name, suitability_score, market_demand):
    """Assess overall risk level for the crop"""
    
    # Base risk score from suitability (lower suitability = higher risk)
    suitability_risk = max(0, (100 - suitability_score) / 100 * 50)
    
    # Market demand risk
    market_risk_map = {'Very High': 5, 'High': 15, 'Medium': 25, 'Low': 40}
    market_risk = market_risk_map.get(market_demand, 20)
    
    # Total risk score
    risk_score = suitability_risk + market_risk
    
    if risk_score >= 50:
        risk_level = 'High'
    elif risk_score >= 30:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'
    
    return risk_level, round(risk_score, 1)

def generate_recommendation_reasons(crop_name, temperature, rainfall, soil_type, suitability_score):
    """Generate detailed reasons for recommending this crop"""
    
    reasons = []
    
    # Temperature-related reason
    if temperature >= 20 and temperature <= 35:
        reasons.append(f"Current temperature of {temperature}°C is ideal for {crop_name} growth")
    elif temperature > 35:
        reasons.append(f"Warm climate ({temperature}°C) supports good {crop_name} cultivation")
    else:
        reasons.append(f"Moderate temperature ({temperature}°C) is suitable for {crop_name}")
    
    # Soil-related reason
    soil_reasons = {
        'Low Fertility': f"{crop_name} can tolerate moderate soil conditions",
        'Medium Fertility': f"Current soil fertility level is adequate for {crop_name}",
        'High Fertility': f"Rich soil conditions will maximize {crop_name} yield potential"
    }
    reasons.append(soil_reasons.get(soil_type, f"{crop_name} grows well in this soil type"))
    
    # Rainfall-related reason
    if rainfall >= 100:
        reasons.append(f"Adequate rainfall supports healthy {crop_name} development")
    else:
        reasons.append(f"{crop_name} requires less water, suitable for current rainfall pattern")
    
    # Market-related reason
    reasons.append(f"Strong market demand for {crop_name} ensures good price realization")
    
    # Profitability reason if suitability is high
    if suitability_score >= 75:
        reasons.append(f"High suitability ({suitability_score}%) indicates excellent profit potential")
    elif suitability_score >= 60:
        reasons.append(f"Good growing conditions suggest moderate to good returns")
    
    return reasons

def get_profitability_level(profit_amount):
    """Get profitability level based on estimated profit"""
    
    if profit_amount >= 60000:
        return 'High', 'fa-chart-line'
    elif profit_amount >= 35000:
        return 'Medium', 'fa-chart-simple'
    else:
        return 'Low', 'fa-chart-line'

def get_profitability_color(profitability):
    """Get color for profitability display"""
    
    colors = {
        'High': '#2ecc71',
        'Medium': '#f39c12',
        'Low': '#e74c3c'
    }
    return colors.get(profitability, '#666')

def get_risk_color(risk_level):
    """Get color for risk level display"""
    
    colors = {
        'High': '#e74c3c',
        'Medium': '#f39c12',
        'Low': '#2ecc71'
    }
    return colors.get(risk_level, '#666')   