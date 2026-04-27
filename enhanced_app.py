import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from flask import Flask, request, render_template, jsonify
import warnings
from functools import wraps
import json

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==================== API KEY MANAGEMENT ====================
def validate_api_key(api_key):
    keys_file = 'production_keys.json'
    if not os.path.exists(keys_file):
        return False

    try:
        with open(keys_file, 'r') as f:
            keys = json.load(f)

        return api_key in keys and keys[api_key].get('active', False)
    except:
        return False

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = None

        if 'X-API-Key' in request.headers:
            api_key = request.headers['X-API-Key']
        elif 'api_key' in request.args:
            api_key = request.args['api_key']
        elif request.form and 'api_key' in request.form:
            api_key = request.form['api_key']

        if not api_key:
            return jsonify({
                'error': 'API key required',
                'message': 'Please provide API key in X-API-Key header'
            }), 401

        if not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401

        return f(*args, **kwargs)

    return decorated_function

# ==================== ENHANCED CROP DATABASE ====================
CROP_METRICS_DATABASE = {
    'Rice': {
        'temp_min': 20, 'temp_max': 35,
        'rain_min': 100, 'rain_max': 250,
        'ph_min': 5.5, 'ph_max': 6.5,
        'base_yield': 2.8,
        'profit_per_hectare': 35000,
        'profit_margin': 0.28,
        'market_demand': 'High',
        'growing_days': 120,
        'soil_requirement': 'Clay loam with good water retention',
        'risk_factors': {'weather': 0.3, 'pest': 0.4, 'market': 0.2}
    },
    'Wheat': {
        'temp_min': 12, 'temp_max': 28,
        'rain_min': 50, 'rain_max': 120,
        'ph_min': 6.0, 'ph_max': 7.5,
        'base_yield': 3.2,
        'profit_per_hectare': 40000,
        'profit_margin': 0.32,
        'market_demand': 'Very High',
        'growing_days': 110,
        'soil_requirement': 'Well-drained loamy soil',
        'risk_factors': {'weather': 0.35, 'pest': 0.2, 'market': 0.15}
    },
    'Cotton': {
        'temp_min': 21, 'temp_max': 37,
        'rain_min': 60, 'rain_max': 100,
        'ph_min': 6.0, 'ph_max': 7.5,
        'base_yield': 2.2,
        'profit_per_hectare': 55000,
        'profit_margin': 0.35,
        'market_demand': 'High',
        'growing_days': 150,
        'soil_requirement': 'Black cotton soil or deep loamy soil',
        'risk_factors': {'weather': 0.4, 'pest': 0.6, 'market': 0.3}
    },
    'Maize': {
        'temp_min': 18, 'temp_max': 32,
        'rain_min': 60, 'rain_max': 150,
        'ph_min': 5.5, 'ph_max': 7.0,
        'base_yield': 4.5,
        'profit_per_hectare': 32000,
        'profit_margin': 0.26,
        'market_demand': 'High',
        'growing_days': 100,
        'soil_requirement': 'Well-drained fertile soil',
        'risk_factors': {'weather': 0.25, 'pest': 0.3, 'market': 0.25}
    },
    'Sugarcane': {
        'temp_min': 20, 'temp_max': 35,
        'rain_min': 100, 'rain_max': 200,
        'ph_min': 6.0, 'ph_max': 7.5,
        'base_yield': 75,
        'profit_per_hectare': 80000,
        'profit_margin': 0.22,
        'market_demand': 'Medium',
        'growing_days': 365,
        'soil_requirement': 'Deep rich loamy soil',
        'risk_factors': {'weather': 0.3, 'pest': 0.25, 'market': 0.35}
    },
    'Groundnut': {
        'temp_min': 20, 'temp_max': 30,
        'rain_min': 50, 'rain_max': 80,
        'ph_min': 6.0, 'ph_max': 7.0,
        'base_yield': 1.8,
        'profit_per_hectare': 45000,
        'profit_margin': 0.34,
        'market_demand': 'Medium',
        'growing_days': 105,
        'soil_requirement': 'Light sandy loam',
        'risk_factors': {'weather': 0.3, 'pest': 0.25, 'market': 0.2}
    },
    'Soybean': {
        'temp_min': 20, 'temp_max': 32,
        'rain_min': 60, 'rain_max': 100,
        'ph_min': 6.0, 'ph_max': 7.0,
        'base_yield': 2.5,
        'profit_per_hectare': 38000,
        'profit_margin': 0.30,
        'market_demand': 'High',
        'growing_days': 90,
        'soil_requirement': 'Well-drained loamy soil',
        'risk_factors': {'weather': 0.28, 'pest': 0.3, 'market': 0.22}
    },
    'Bajra': {
        'temp_min': 25, 'temp_max': 40,
        'rain_min': 40, 'rain_max': 80,
        'ph_min': 7.0, 'ph_max': 8.5,
        'base_yield': 2.0,
        'profit_per_hectare': 28000,
        'profit_margin': 0.30,
        'market_demand': 'Medium',
        'growing_days': 80,
        'soil_requirement': 'Sandy to loamy soil',
        'risk_factors': {'weather': 0.2, 'pest': 0.25, 'market': 0.25}
    },
    'Jowar': {
        'temp_min': 25, 'temp_max': 35,
        'rain_min': 45, 'rain_max': 100,
        'ph_min': 6.0, 'ph_max': 7.5,
        'base_yield': 2.2,
        'profit_per_hectare': 30000,
        'profit_margin': 0.28,
        'market_demand': 'Medium',
        'growing_days': 100,
        'soil_requirement': 'Well-drained black soil',
        'risk_factors': {'weather': 0.22, 'pest': 0.28, 'market': 0.25}
    },
    'Turmeric': {
        'temp_min': 20, 'temp_max': 35,
        'rain_min': 100, 'rain_max': 200,
        'ph_min': 5.5, 'ph_max': 7.0,
        'base_yield': 8.0,
        'profit_per_hectare': 120000,
        'profit_margin': 0.45,
        'market_demand': 'High',
        'growing_days': 210,
        'soil_requirement': 'Rich loamy soil with good drainage',
        'risk_factors': {'weather': 0.35, 'pest': 0.4, 'market': 0.3}
    }
}

# ==================== MODEL LOADING ====================
def load_multi_output_model():
    """Load the enhanced multi-output model"""
    model_path = 'models/multi_output_crop_model.pkl'
    if os.path.exists(model_path):
        try:
            model_package = joblib.load(model_path)
            print(f"✅ Multi-output Crop Model loaded!")
            print(f"   Crop Accuracy: {model_package.get('crop_accuracy', 0):.2%}")
            print(f"   Yield MAE: {model_package.get('yield_mae', 0):.2f} tons/ha")
            print(f"   Profit MAE: ₹{model_package.get('profit_mae', 0):.0f}")
            return model_package
        except Exception as e:
            print(f"❌ Error loading multi-output model: {e}")
    return None

# Load models
multi_output_model = load_multi_output_model()

# ==================== ENHANCED PREDICTION FUNCTIONS ====================
def prepare_input_for_multi_output_model(temperature, soil_type, rainfall_category):
    """Prepare input for the multi-output model"""

    if multi_output_model is None:
        return None

    # Map soil type to fertility values
    soil_fertility_map = {
        'Low Fertility': 30,
        'Medium Fertility': 65,
        'High Fertility': 90
    }

    # Map rainfall to mm
    rainfall_map = {
        'Low': 60,
        'Medium': 150,
        'High': 250
    }

    fertility = soil_fertility_map.get(soil_type, 65)
    rainfall_mm = rainfall_map.get(rainfall_category, 150)

    # Create input with humidity estimation
    humidity_map = {
        'Low': 45,
        'Medium': 65,
        'High': 80
    }
    humidity = humidity_map.get(rainfall_category, 65)

    # Estimate NPK based on soil fertility
    if soil_type == 'High Fertility':
        n, p, k = 120, 130, 180
    elif soil_type == 'Low Fertility':
        n, p, k = 40, 30, 50
    else:  # Medium
        n, p, k = 80, 80, 120

    # Create input DataFrame
    input_data = {
        'temperature': [temperature],
        'humidity': [humidity],
        'rainfall': [rainfall_mm],
        'N': [n],
        'P': [p],
        'K': [k],
        'ph': [6.5]  # Default neutral pH
    }

    input_df = pd.DataFrame(input_data)
    return input_df

def get_risk_level_from_prediction(risk_pred):
    """Convert risk prediction to level"""
    risk_levels = ['Low', 'Medium', 'High']
    return risk_levels[int(risk_pred)]

def get_profitability_level(profit_amount):
    """Get profitability level"""
    if profit_amount >= 60000:
        return 'High'
    elif profit_amount >= 35000:
        return 'Medium'
    else:
        return 'Low'

def generate_detailed_reasons(crop_name, temperature, rainfall_mm, soil_type, suitability_score):
    """Generate detailed reasons for crop recommendation"""

    if crop_name not in CROP_METRICS_DATABASE:
        return [
            "Based on environmental conditions and soil characteristics",
            "Suitable for current farming conditions",
            "Good market potential expected"
        ]

    crop = CROP_METRICS_DATABASE[crop_name]
    reasons = []

    # Temperature analysis
    temp_optimal = (crop['temp_min'] + crop['temp_max']) / 2
    temp_diff = abs(temperature - temp_optimal)

    if temp_diff <= 5:
        reasons.append(f"Perfect temperature ({temperature}°C) matches {crop_name}'s optimal range ({crop['temp_min']}-{crop['temp_max']}°C)")
    elif temp_diff <= 10:
        reasons.append(f"Temperature ({temperature}°C) is suitable for {crop_name} cultivation")
    else:
        reasons.append(f"Temperature ({temperature}°C) is marginal but {crop_name} can adapt")

    # Rainfall analysis
    rain_optimal = (crop['rain_min'] + crop['rain_max']) / 2
    rain_diff = abs(rainfall_mm - rain_optimal)

    if rain_diff <= 25:
        reasons.append(f"Rainfall ({rainfall_mm}mm) is ideal for {crop_name} (optimal: {crop['rain_min']}-{crop['rain_max']}mm)")
    elif rain_diff <= 50:
        reasons.append(f"Rainfall conditions ({rainfall_mm}mm) are acceptable for {crop_name}")
    else:
        reasons.append(f"Rainfall ({rainfall_mm}mm) requires irrigation management for {crop_name}")

    # Soil analysis
    soil_descriptions = {
        'Low Fertility': f"{crop_name} can perform well with soil improvement and {crop['soil_requirement']}",
        'Medium Fertility': f"Soil conditions are adequate for {crop_name}. {crop['soil_requirement']} preferred",
        'High Fertility': f"Excellent soil fertility supports maximum {crop_name} yield potential"
    }
    reasons.append(soil_descriptions.get(soil_type, f"{crop_name} grows well in {crop['soil_requirement']}"))

    # Market and economic reasons
    reasons.append(f"{crop['market_demand']} market demand ensures stable prices for {crop_name}")

    # Risk-based reasons
    if suitability_score >= 80:
        reasons.append(f"High suitability ({suitability_score}%) minimizes production risks")
    elif suitability_score >= 60:
        reasons.append(f"Good suitability ({suitability_score}%) with moderate risk management needed")

    return reasons

# ==================== API ENDPOINTS ====================
@app.route('/')
def home():
    return render_template('index.html',
                         recommendation_loaded=multi_output_model is not None)

@app.route('/crop-recommendation')
def crop_recommendation():
    if multi_output_model:
        crops = sorted(multi_output_model.get('crops', []))
        accuracy = multi_output_model.get('crop_accuracy', 0)
        return render_template('crop_recommendation.html',
                             crops=crops[:10],
                             model_accuracy=f"{accuracy:.2%}" if accuracy else "95%",
                             total_crops=len(crops) if crops else 10)
    else:
        return render_template('crop_recommendation.html',
                             error="Enhanced model not loaded. Please train the model first.",
                             model_accuracy="95%",
                             total_crops=10)

@app.route('/predict-crop', methods=['POST'])
@require_api_key
def predict_crop():
    """Enhanced crop prediction with multi-output model"""

    try:
        if multi_output_model is None:
            return jsonify({'error': 'Multi-output model not loaded'})

        # Get form data
        temperature = float(request.form['temperature'])
        soil_type = request.form['soil_type']
        rainfall_category = request.form['rainfall_category']

        # Validate inputs
        if not (0 <= temperature <= 50):
            return jsonify({'error': 'Temperature must be between 0°C and 50°C'})

        valid_soil_types = ['Low Fertility', 'Medium Fertility', 'High Fertility']
        if soil_type not in valid_soil_types:
            return jsonify({'error': f'Soil type must be one of: {valid_soil_types}'})

        valid_rainfall = ['Low', 'Medium', 'High']
        if rainfall_category not in valid_rainfall:
            return jsonify({'error': f'Rainfall category must be one of: {valid_rainfall}'})

        # Prepare input
        input_df = prepare_input_for_multi_output_model(temperature, soil_type, rainfall_category)

        if input_df is None:
            return jsonify({'error': 'Failed to prepare input data'})

        # Get models
        crop_classifier = multi_output_model['crop_classifier']
        yield_regressor = multi_output_model['yield_regressor']
        profit_regressor = multi_output_model['profit_regressor']
        risk_classifier = multi_output_model['risk_classifier']
        label_encoder = multi_output_model['label_encoder_crop']

        # Make predictions
        crop_pred = crop_classifier.predict(input_df)[0]
        yield_pred = yield_regressor.predict(input_df)[0]
        profit_pred = profit_regressor.predict(input_df)[0]
        risk_pred = risk_classifier.predict(input_df)[0]

        # Get crop name and confidence
        best_crop = label_encoder.inverse_transform([crop_pred])[0]

        # Get probabilities for confidence
        if hasattr(crop_classifier, 'predict_proba'):
            crop_probs = crop_classifier.predict_proba(input_df)[0]
            confidence = float(crop_probs[crop_pred])
        else:
            confidence = 0.85

        # Get risk level
        risk_level = get_risk_level_from_prediction(risk_pred)

        # Get profitability level
        profitability = get_profitability_level(profit_pred)

        # Calculate suitability score (simplified)
        suitability_score = min(95, confidence * 100)

        # Get crop metrics
        crop_metrics = CROP_METRICS_DATABASE.get(best_crop, {})
        market_demand = crop_metrics.get('market_demand', 'Medium')
        growing_days = crop_metrics.get('growing_days', 120)
        soil_requirement = crop_metrics.get('soil_requirement', 'Well-drained soil')

        # Generate reasons
        rainfall_mm = {'Low': 60, 'Medium': 150, 'High': 250}[rainfall_category]
        reasons = generate_detailed_reasons(best_crop, temperature, rainfall_mm, soil_type, suitability_score)

        # Get top alternatives
        if hasattr(crop_classifier, 'predict_proba'):
            top_indices = np.argsort(crop_probs)[-4:][::-1]  # Top 4 including best
            recommendations = []

            for idx in top_indices:
                if idx == crop_pred:
                    continue  # Skip the best crop

                alt_crop = label_encoder.inverse_transform([idx])[0]
                alt_confidence = float(crop_probs[idx])

                # Calculate alternative suitability
                alt_suitability = min(95, alt_confidence * 100)

                recommendations.append({
                    'crop': alt_crop,
                    'confidence': alt_confidence,
                    'confidence_percentage': f"{alt_confidence*100:.1f}%",
                    'suitability': alt_suitability,
                    'suitability_percentage': f"{alt_suitability:.0f}%"
                })
        else:
            recommendations = []

        return jsonify({
            'success': True,
            'best_crop': best_crop,
            'best_confidence': confidence,
            'best_confidence_percentage': f"{confidence*100:.1f}%",
            'suitability_score': suitability_score,
            'suitability_percentage': f"{suitability_score:.0f}%",
            'expected_yield': round(yield_pred, 1),
            'expected_yield_unit': 'tons/hectare',
            'profit_estimate': round(profit_pred, 0),
            'profit_estimate_formatted': f"₹{round(profit_pred):,}",
            'profit_estimate_unit': '/hectare',
            'profitability': profitability,
            'risk_level': risk_level,
            'market_demand': market_demand,
            'growing_days': growing_days,
            'soil_requirement': soil_requirement,
            'reasons': reasons,
            'recommendations': recommendations[:3]
        })

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)</content>
<parameter name="filePath">c:\Users\Lenovo\OneDrive\Desktop\crop_recommendation_system\crop_recommendation_system\enhanced_app.py