# enhanced_crop_recommendation.py - COMPLETE FIXED VERSION
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedCropRecommender:
    """Enhanced crop recommendation with risk, profitability, and yield predictions"""
    
    def __init__(self):
        self.crop_profiles = {}
        self.market_data = {}
        self.soil_conditions = {}
        self.load_crop_database()
    
    def load_crop_database(self):
        """Create comprehensive crop database with yield, risk, and profitability metrics"""
        
        self.crop_profiles = {
            'Rice': {
                'ideal_temp': (22, 32),
                'ideal_rainfall': (150, 250),
                'ideal_humidity': (70, 85),
                'ideal_ph': (5.5, 7.0),
                'growing_days': 120,
                'base_yield_kg_per_hectare': (3500, 4500),
                'risk_level': 'Medium',
                'risk_factors': ['Water intensive', 'Pest susceptible', 'Price volatility'],
                'profit_margin': 'Medium',
                'input_cost_per_hectare': 35000,
                'expected_price_per_kg': 28,
                'suitable_regions': ['Coastal', 'River deltas', 'High rainfall areas'],
                'season': ['Kharif', 'Rabi'],
                'water_requirement': 'High',
                'drought_resistance': 'Low',
                'market_demand': 'High',
                'storage_life_days': 365,
                'value_addition_options': ['Parboiled', 'Brown rice', 'Flour']
            },
            'Wheat': {
                'ideal_temp': (15, 25),
                'ideal_rainfall': (50, 150),
                'ideal_humidity': (50, 65),
                'ideal_ph': (6.0, 7.5),
                'growing_days': 110,
                'base_yield_kg_per_hectare': (2500, 3500),
                'risk_level': 'Low',
                'risk_factors': ['Rust disease', 'Termite attack'],
                'profit_margin': 'Medium-High',
                'input_cost_per_hectare': 28000,
                'expected_price_per_kg': 25,
                'suitable_regions': ['Plains', 'Temperate zones'],
                'season': ['Rabi'],
                'water_requirement': 'Medium',
                'drought_resistance': 'Medium',
                'market_demand': 'Very High',
                'storage_life_days': 730,
                'value_addition_options': ['Flour', 'Bran', 'Germ']
            },
            'Cotton': {
                'ideal_temp': (25, 35),
                'ideal_rainfall': (50, 100),
                'ideal_humidity': (45, 60),
                'ideal_ph': (6.0, 8.0),
                'growing_days': 160,
                'base_yield_kg_per_hectare': (400, 600),
                'risk_level': 'High',
                'risk_factors': ['Pest attack (bollworm)', 'Frost sensitive', 'Price fluctuation'],
                'profit_margin': 'Variable',
                'input_cost_per_hectare': 45000,
                'expected_price_per_kg': 60,
                'suitable_regions': ['Black soil regions', 'Dry areas'],
                'season': ['Kharif'],
                'water_requirement': 'Low-Medium',
                'drought_resistance': 'High',
                'market_demand': 'Medium',
                'storage_life_days': 365,
                'value_addition_options': ['Ginned cotton', 'Cotton seed oil']
            },
            'Sugarcane': {
                'ideal_temp': (20, 32),
                'ideal_rainfall': (100, 150),
                'ideal_humidity': (60, 75),
                'ideal_ph': (6.5, 8.0),
                'growing_days': 365,
                'base_yield_kg_per_hectare': (65000, 85000),
                'risk_level': 'Medium',
                'risk_factors': ['Water logging', 'Red rot disease'],
                'profit_margin': 'Low-Medium',
                'input_cost_per_hectare': 55000,
                'expected_price_per_kg': 3.5,
                'suitable_regions': ['Tropical regions', 'River basins'],
                'season': ['Annual'],
                'water_requirement': 'High',
                'drought_resistance': 'Low',
                'market_demand': 'Stable',
                'storage_life_days': 45,
                'value_addition_options': ['Jaggery', 'Molasses', 'Biofuel']
            },
            'Maize': {
                'ideal_temp': (20, 30),
                'ideal_rainfall': (75, 150),
                'ideal_humidity': (55, 70),
                'ideal_ph': (5.5, 7.0),
                'growing_days': 90,
                'base_yield_kg_per_hectare': (2500, 3500),
                'risk_level': 'Low-Medium',
                'risk_factors': ['Stem borer', 'Drought sensitive'],
                'profit_margin': 'Medium',
                'input_cost_per_hectare': 30000,
                'expected_price_per_kg': 20,
                'suitable_regions': ['Versatile, wide range'],
                'season': ['Kharif', 'Rabi', 'Summer'],
                'water_requirement': 'Medium',
                'drought_resistance': 'Medium',
                'market_demand': 'High',
                'storage_life_days': 365,
                'value_addition_options': ['Corn flour', 'Animal feed', 'Popcorn']
            },
            'Soybean': {
                'ideal_temp': (20, 30),
                'ideal_rainfall': (60, 125),
                'ideal_humidity': (55, 70),
                'ideal_ph': (6.0, 7.5),
                'growing_days': 100,
                'base_yield_kg_per_hectare': (1200, 2000),
                'risk_level': 'Medium',
                'risk_factors': ['Leaf rust', 'Root rot'],
                'profit_margin': 'High',
                'input_cost_per_hectare': 32000,
                'expected_price_per_kg': 45,
                'suitable_regions': ['Temperate regions', 'Well-drained soil'],
                'season': ['Kharif'],
                'water_requirement': 'Low-Medium',
                'drought_resistance': 'Medium',
                'market_demand': 'Very High',
                'storage_life_days': 365,
                'value_addition_options': ['Oil', 'Soy milk', 'Tofu', 'Animal feed']
            },
            'Groundnut': {
                'ideal_temp': (25, 35),
                'ideal_rainfall': (50, 125),
                'ideal_humidity': (50, 70),
                'ideal_ph': (6.0, 7.0),
                'growing_days': 130,
                'base_yield_kg_per_hectare': (1500, 2500),
                'risk_level': 'Medium',
                'risk_factors': ['Tikka disease', 'Aflatoxin'],
                'profit_margin': 'High',
                'input_cost_per_hectare': 35000,
                'expected_price_per_kg': 55,
                'suitable_regions': ['Sandy loam soils', 'Dry regions'],
                'season': ['Kharif', 'Summer'],
                'water_requirement': 'Low',
                'drought_resistance': 'High',
                'market_demand': 'High',
                'storage_life_days': 180,
                'value_addition_options': ['Oil', 'Butter', 'Snacks']
            },
            'Potato': {
                'ideal_temp': (15, 22),
                'ideal_rainfall': (70, 120),
                'ideal_humidity': (70, 85),
                'ideal_ph': (5.5, 6.5),
                'growing_days': 90,
                'base_yield_kg_per_hectare': (18000, 25000),
                'risk_level': 'Medium-High',
                'risk_factors': ['Late blight', 'Tuber moth', 'Storage losses'],
                'profit_margin': 'Medium',
                'input_cost_per_hectare': 40000,
                'expected_price_per_kg': 20,
                'suitable_regions': ['Cool temperate', 'High altitude'],
                'season': ['Rabi'],
                'water_requirement': 'Medium',
                'drought_resistance': 'Low',
                'market_demand': 'High',
                'storage_life_days': 150,
                'value_addition_options': ['Chips', 'Flakes', 'Starch']
            }
        }
        
        # Market dynamics data
        self.market_data = {
            'Rice': {'demand_trend': 'Stable', 'price_volatility': 'Low', 'export_potential': 'High'},
            'Wheat': {'demand_trend': 'Increasing', 'price_volatility': 'Low', 'export_potential': 'Medium'},
            'Cotton': {'demand_trend': 'Stable', 'price_volatility': 'High', 'export_potential': 'High'},
            'Sugarcane': {'demand_trend': 'Stable', 'price_volatility': 'Low', 'export_potential': 'Medium'},
            'Maize': {'demand_trend': 'Increasing', 'price_volatility': 'Medium', 'export_potential': 'High'},
            'Soybean': {'demand_trend': 'Increasing', 'price_volatility': 'Medium', 'export_potential': 'High'},
            'Groundnut': {'demand_trend': 'Stable', 'price_volatility': 'Medium', 'export_potential': 'Medium'},
            'Potato': {'demand_trend': 'Increasing', 'price_volatility': 'Low', 'export_potential': 'Low'}
        }
    
    def calculate_suitability(self, crop, conditions):
        """Calculate suitability percentage based on environmental conditions"""
        
        profile = self.crop_profiles[crop]
        scores = []
        
        # Temperature suitability
        temp = conditions['temperature']
        if profile['ideal_temp'][0] <= temp <= profile['ideal_temp'][1]:
            temp_score = 100
        elif temp < profile['ideal_temp'][0]:
            temp_score = max(0, 100 - (profile['ideal_temp'][0] - temp) * 10)
        else:
            temp_score = max(0, 100 - (temp - profile['ideal_temp'][1]) * 10)
        scores.append(('Temperature', temp_score))
        
        # Rainfall suitability
        rainfall = conditions.get('rainfall', conditions.get('rainfall_category', 150))
        if isinstance(rainfall, str):
            rainfall_map = {'Very Low': 50, 'Low': 100, 'Medium': 150, 'High': 250}
            rainfall = rainfall_map.get(rainfall, 150)
        
        if profile['ideal_rainfall'][0] <= rainfall <= profile['ideal_rainfall'][1]:
            rain_score = 100
        elif rainfall < profile['ideal_rainfall'][0]:
            rain_score = max(0, 100 - (profile['ideal_rainfall'][0] - rainfall) * 0.5)
        else:
            rain_score = max(0, 100 - (rainfall - profile['ideal_rainfall'][1]) * 0.5)
        scores.append(('Rainfall', rain_score))
        
        # Soil fertility score
        fertility = conditions.get('fertility_score', conditions.get('soil_fertility', 60))
        if isinstance(fertility, str):
            soil_map = {'Low Fertility': 30, 'Medium Fertility': 60, 'High Fertility': 90}
            fertility = soil_map.get(fertility, 60)
        
        fert_score = min(100, fertility)
        scores.append(('Soil Fertility', fert_score))
        
        # pH suitability
        if 'ph' in conditions:
            ph = conditions['ph']
            if profile['ideal_ph'][0] <= ph <= profile['ideal_ph'][1]:
                ph_score = 100
            else:
                ph_score = max(0, 100 - min(abs(ph - profile['ideal_ph'][0]), abs(ph - profile['ideal_ph'][1])) * 20)
            scores.append(('pH Level', ph_score))
        
        # Calculate overall suitability (weighted average)
        weights = {'Temperature': 0.35, 'Rainfall': 0.30, 'Soil Fertility': 0.25, 'pH Level': 0.10}
        total_weight = 0
        weighted_score = 0
        
        for factor, score in scores:
            weight = weights.get(factor, 1)
            weighted_score += score * weight
            total_weight += weight
        
        overall_suitability = weighted_score / total_weight if total_weight > 0 else 0
        
        return {
            'overall': round(overall_suitability, 1),
            'factors': dict(scores),
            'recommendation': self._get_recommendation_level(overall_suitability)
        }
    
    def predict_yield(self, crop, conditions):
        """Predict expected yield in tons/hectare based on conditions"""
        
        profile = self.crop_profiles[crop]
        base_yield_low, base_yield_high = profile['base_yield_kg_per_hectare']
        base_yield = (base_yield_low + base_yield_high) / 2
        
        # Adjust yield based on suitability
        suitability = self.calculate_suitability(crop, conditions)['overall']
        
        # Yield adjustment factor based on suitability
        if suitability >= 80:
            yield_factor = 1.2
        elif suitability >= 60:
            yield_factor = 1.0
        elif suitability >= 40:
            yield_factor = 0.8
        elif suitability >= 25:
            yield_factor = 0.6
        else:
            yield_factor = 0.4
        
        # Add soil fertility effect
        fertility = conditions.get('fertility_score', conditions.get('soil_fertility', 60))
        if isinstance(fertility, str):
            soil_fertility_map = {'Low Fertility': 30, 'Medium Fertility': 60, 'High Fertility': 90}
            fertility = soil_fertility_map.get(fertility, 60)
        
        fertility_factor = 0.8 + (fertility / 100) * 0.4
        
        # Calculate predicted yield
        predicted_yield_kg = base_yield * yield_factor * fertility_factor
        
        # Calculate range
        min_yield = predicted_yield_kg * 0.85
        max_yield = predicted_yield_kg * 1.15
        
        return {
            'expected_tons_per_hectare': round(predicted_yield_kg / 1000, 2),
            'range_tons_per_hectare': f"{round(min_yield/1000, 2)} - {round(max_yield/1000, 2)}",
            'kg_per_hectare': round(predicted_yield_kg, 0),
            'yield_quality': self._get_yield_quality(predicted_yield_kg, base_yield)
        }
    
    def calculate_risk_level(self, crop, conditions):
        """Calculate risk level with detailed risk factors"""
        
        profile = self.crop_profiles[crop]
        base_risk = profile['risk_level']
        
        # Adjust risk based on conditions
        suitability = self.calculate_suitability(crop, conditions)['overall']
        
        if suitability >= 80:
            risk_adjustment = 'reduced'
        elif suitability >= 60:
            risk_adjustment = 'normal'
        elif suitability >= 40:
            risk_adjustment = 'elevated'
        else:
            risk_adjustment = 'high'
        
        # Determine final risk level
        risk_levels = {'Low': 1, 'Low-Medium': 2, 'Medium': 3, 'Medium-High': 4, 'High': 5}
        base_risk_score = risk_levels.get(base_risk, 3)
        
        if risk_adjustment == 'reduced' and base_risk_score > 1:
            risk_score = base_risk_score - 1
        elif risk_adjustment == 'elevated' and base_risk_score < 5:
            risk_score = base_risk_score + 1
        elif risk_adjustment == 'high' and base_risk_score < 4:
            risk_score = base_risk_score + 2
        else:
            risk_score = base_risk_score
        
        # Convert back to risk level
        risk_level_map = {1: 'Low', 2: 'Low-Medium', 3: 'Medium', 4: 'Medium-High', 5: 'High'}
        final_risk = risk_level_map.get(risk_score, 'Medium')
        
        # Calculate risk percentage
        risk_percentage = (risk_score / 5) * 100
        
        # Add specific risk alerts
        risk_alerts = []
        if suitability < 50:
            risk_alerts.append("Environmental conditions are suboptimal")
        if conditions.get('rainfall_category') in ['Very Low', 'Low'] and profile['water_requirement'] == 'High':
            risk_alerts.append(f"Water scarcity risk - crop requires {profile['water_requirement']} water")
        
        return {
            'level': final_risk,
            'percentage': round(risk_percentage, 1),
            'factors': profile['risk_factors'],
            'alerts': risk_alerts,
            'mitigation_strategies': self._get_mitigation_strategies(crop, risk_alerts)
        }
    
    def calculate_profitability(self, crop, conditions):
        """Calculate expected profitability"""
        
        profile = self.crop_profiles[crop]
        yield_pred = self.predict_yield(crop, conditions)
        
        # Calculate revenue
        expected_yield_kg = yield_pred['kg_per_hectare']
        price_per_kg = profile['expected_price_per_kg']
        revenue = expected_yield_kg * price_per_kg
        
        # Calculate costs
        input_cost = profile['input_cost_per_hectare']
        labor_cost = input_cost * 0.3
        misc_cost = input_cost * 0.1
        total_cost = input_cost + labor_cost + misc_cost
        
        # Calculate profit
        profit = revenue - total_cost
        profit_margin = (profit / revenue) * 100 if revenue > 0 else 0
        
        # Determine profitability level
        if profit_margin >= 40:
            profit_level = 'High'
        elif profit_margin >= 25:
            profit_level = 'Medium-High'
        elif profit_margin >= 15:
            profit_level = 'Medium'
        elif profit_margin >= 5:
            profit_level = 'Low-Medium'
        else:
            profit_level = 'Low'
        
        # ROI percentage
        roi = (profit / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            'level': profit_level,
            'margin_percentage': round(profit_margin, 1),
            'roi_percentage': round(roi, 1),
            'revenue_per_hectare': round(revenue, 0),
            'cost_per_hectare': round(total_cost, 0),
            'profit_per_hectare': round(profit, 0),
            'break_even_yield_kg': round(total_cost / price_per_kg, 0) if price_per_kg > 0 else 0,
            'payback_period_months': round(profile['growing_days'] / 30, 1)
        }
    
    def get_reasoning(self, crop, conditions):
        """Generate reasoning for why this crop is recommended"""
        
        profile = self.crop_profiles[crop]
        suitability = self.calculate_suitability(crop, conditions)
        reasons = []
        
        # Temperature reasoning
        temp = conditions['temperature']
        if profile['ideal_temp'][0] <= temp <= profile['ideal_temp'][1]:
            reasons.append(f"✓ Temperature of {temp}°C is ideal for {crop} (range: {profile['ideal_temp'][0]}-{profile['ideal_temp'][1]}°C)")
        elif temp < profile['ideal_temp'][0]:
            reasons.append(f"⚠ Temperature of {temp}°C is slightly below ideal range ({profile['ideal_temp'][0]}-{profile['ideal_temp'][1]}°C)")
        else:
            reasons.append(f"⚠ Temperature of {temp}°C is above ideal range ({profile['ideal_temp'][0]}-{profile['ideal_temp'][1]}°C)")
        
        # Rainfall reasoning
        rainfall = conditions.get('rainfall', conditions.get('rainfall_category', 150))
        if isinstance(rainfall, str):
            reasons.append(f"✓ {rainfall} rainfall conditions match {crop}'s requirements")
        else:
            if profile['ideal_rainfall'][0] <= rainfall <= profile['ideal_rainfall'][1]:
                reasons.append(f"✓ Rainfall of {rainfall}mm is well-suited for {crop}")
            else:
                reasons.append(f"⚠ Rainfall of {rainfall}mm {'exceeds' if rainfall > profile['ideal_rainfall'][1] else 'is below'} optimal range")
        
        # Soil reasoning
        soil_fertility_map = {'Low Fertility': 30, 'Medium Fertility': 60, 'High Fertility': 90}
        fertility = conditions.get('fertility_score', conditions.get('soil_fertility', 60))
        if isinstance(fertility, str):
            fertility = soil_fertility_map.get(fertility, 60)
        
        if fertility >= 70:
            reasons.append(f"✓ Rich soil fertility ({fertility}%) will boost {crop} yields")
        elif fertility >= 50:
            reasons.append(f"✓ Moderate soil fertility is adequate for {crop} production")
        else:
            reasons.append(f"⚠ Low soil fertility may require additional fertilization for {crop}")
        
        # Market demand reasoning
        market = self.market_data.get(crop, {})
        if market.get('demand_trend') == 'Increasing':
            reasons.append(f"📈 {crop} has INCREASING market demand with good price prospects")
        elif market.get('demand_trend') == 'Very High':
            reasons.append(f"📈 {crop} is in VERY HIGH demand in current market")
        elif market.get('demand_trend') == 'High':
            reasons.append(f"📈 {crop} is in HIGH demand in current market")
        
        # Season matching
        if 'season' in profile:
            reasons.append(f"🌱 {crop} can be cultivated in {', '.join(profile['season'])} season")
        
        # Additional benefits
        if profile.get('drought_resistance') == 'High':
            reasons.append(f"💧 {crop} has HIGH drought resistance - good for water-scarce regions")
        if len(profile.get('value_addition_options', [])) > 2:
            reasons.append(f"➕ Multiple value-addition options available: {', '.join(profile['value_addition_options'][:3])}")
        
        # Add profitability note
        if suitability['overall'] >= 70:
            reasons.append(f"💰 High expected profitability with {suitability['overall']:.0f}% suitability")
        
        return reasons
    
    def get_top_recommendations(self, conditions, top_n=5):
        """Get top N crop recommendations with all metrics - FIXED VERSION with normalized scores"""
        
        recommendations = []
        
        for crop in self.crop_profiles.keys():
            # Calculate all metrics
            suitability = self.calculate_suitability(crop, conditions)
            yield_pred = self.predict_yield(crop, conditions)
            risk = self.calculate_risk_level(crop, conditions)
            profitability = self.calculate_profitability(crop, conditions)
            reasons = self.get_reasoning(crop, conditions)
            
            # Calculate overall score (NORMALIZED to 0-100)
            # Each component is already on a 0-100 scale
            suitability_score = suitability['overall']  # 0-100
            profitability_score = min(100, profitability['roi_percentage'] / 3)  # Normalize ROI
            risk_score = 100 - risk['percentage']  # Inverse relationship
            
            # Weighted average
            score = (
                suitability_score * 0.5 +      # 50% weight on suitability
                profitability_score * 0.3 +     # 30% weight on profitability
                risk_score * 0.2                # 20% weight on risk
            )
            
            # Cap at 100
            final_score = round(min(100, score), 1)
            
            recommendations.append({
                'crop': crop,
                'score': final_score,
                'suitability': suitability,
                'expected_yield': yield_pred,
                'risk': risk,
                'profitability': profitability,
                'reasons': reasons,
                'market_demand': self.market_data.get(crop, {}).get('demand_trend', 'Stable')
            })
        
        # Sort by overall score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:top_n]
    
    def _get_recommendation_level(self, suitability):
        """Get recommendation level based on suitability score"""
        if suitability >= 80:
            return "Excellent - Highly Recommended"
        elif suitability >= 60:
            return "Good - Recommended"
        elif suitability >= 40:
            return "Moderate - Consider with improvements"
        elif suitability >= 25:
            return "Marginal - Not recommended without interventions"
        else:
            return "Poor - Not recommended"
    
    def _get_yield_quality(self, predicted_yield, base_yield):
        """Get yield quality assessment"""
        ratio = predicted_yield / base_yield if base_yield > 0 else 0
        if ratio >= 1.2:
            return "Excellent (Above average)"
        elif ratio >= 1.0:
            return "Good (Average)"
        elif ratio >= 0.8:
            return "Fair (Below average)"
        else:
            return "Poor (Significantly below average)"
    
    def _get_mitigation_strategies(self, crop, alerts):
        """Get risk mitigation strategies"""
        strategies = []
        
        if any("suboptimal" in alert for alert in alerts):
            strategies.append("Consider investing in greenhouse or controlled environment")
        
        if any("Water scarcity" in alert for alert in alerts):
            strategies.append("Implement drip irrigation and mulching techniques")
            strategies.append("Consider rainwater harvesting systems")
        
        strategies.append(f"Use recommended fertilizers for {crop}")
        strategies.append("Implement integrated pest management (IPM)")
        
        return strategies[:3]


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("🌾 ENHANCED CROP RECOMMENDATION SYSTEM")
    print("="*70)
    
    # Initialize the recommender
    recommender = EnhancedCropRecommender()
    
    # Test conditions
    test_conditions = {
        'temperature': 26,
        'rainfall_category': 'Medium',
        'soil_fertility': 'Medium Fertility',
        'ph': 6.8
    }
    
    print("\n📋 Input Conditions:")
    for key, value in test_conditions.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*70)
    print("CROP RECOMMENDATIONS WITH DETAILED METRICS")
    print("="*70)
    
    # Get recommendations
    recommendations = recommender.get_top_recommendations(test_conditions, top_n=3)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{'='*70}")
        print(f"🏆 RECOMMENDATION #{i}: {rec['crop']}")
        print(f"{'='*70}")
        print(f"\n📊 OVERALL SCORE: {rec['score']}/100")
        
        # Suitability
        print(f"\n🌱 SUITABILITY: {rec['suitability']['overall']}%")
        print(f"   Status: {rec['suitability']['recommendation']}")
        print("   Factors:")
        for factor, score in rec['suitability']['factors'].items():
            print(f"     • {factor}: {score:.1f}%")
        
        # Yield prediction
        print(f"\n📈 YIELD PREDICTION:")
        print(f"   Expected: {rec['expected_yield']['expected_tons_per_hectare']} tons/hectare")
        print(f"   Range: {rec['expected_yield']['range_tons_per_hectare']}")
        print(f"   Quality: {rec['expected_yield']['yield_quality']}")
        
        # Risk assessment
        print(f"\n⚠️ RISK ASSESSMENT:")
        print(f"   Level: {rec['risk']['level']}")
        print(f"   Risk Score: {rec['risk']['percentage']}%")
        print("   Risk Factors:")
        for factor in rec['risk']['factors'][:2]:
            print(f"     • {factor}")
        if rec['risk']['alerts']:
            print("   ⚠️ Alerts:")
            for alert in rec['risk']['alerts']:
                print(f"     • {alert}")
        
        # Profitability
        print(f"\n💰 PROFITABILITY ANALYSIS:")
        print(f"   Level: {rec['profitability']['level']}")
        print(f"   Margin: {rec['profitability']['margin_percentage']}%")
        print(f"   ROI: {rec['profitability']['roi_percentage']}%")
        print(f"   Expected Revenue: ₹{rec['profitability']['revenue_per_hectare']:,.0f}/hectare")
        print(f"   Expected Profit: ₹{rec['profitability']['profit_per_hectare']:,.0f}/hectare")
        print(f"   Break-even Yield: {rec['profitability']['break_even_yield_kg']:,.0f} kg/hectare")
        
        # Reasoning
        print(f"\n💡 WHY {rec['crop']}?")
        for i, reason in enumerate(rec['reasons'][:4], 1):
            print(f"   {i}. {reason}")
        
        # Market demand
        print(f"\n📊 MARKET OUTLOOK:")
        print(f"   Demand Trend: {rec['market_demand']}")
        
        # Mitigation strategies
        if rec['risk']['mitigation_strategies']:
            print(f"\n🛡️ RISK MITIGATION STRATEGIES:")
            for strategy in rec['risk']['mitigation_strategies']:
                print(f"   • {strategy}")
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)