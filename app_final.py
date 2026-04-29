# app_fixed.py - Complete Fixed Version with All Corrections
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from functools import wraps
import warnings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==================== FIXED CORS CONFIGURATION ====================
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["X-API-Key", "Content-Type", "Authorization"],
        "expose_headers": ["X-API-Key"],
        "supports_credentials": True,
        "max_age": 3600
    },
    r"/*": {
        "origins": "*",
        "methods": ["GET", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ==================== OPTIONS HANDLER FOR PREFLIGHT REQUESTS ====================
@app.route('/', methods=['OPTIONS'])
@app.route('/api/<path:path>', methods=['OPTIONS'])
@app.route('/api/v1/<path:path>', methods=['OPTIONS'])
def handle_options(path=None):
    """Handle CORS preflight requests"""
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'X-API-Key, Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response, 200

# ==================== REQUEST LOGGING MIDDLEWARE ====================
@app.before_request
def log_request_info():
    """Log all incoming requests for debugging"""
    if request.method != 'OPTIONS':  # Don't log OPTIONS requests
        logger.info(f"{request.method} {request.path}")
        logger.info(f"Headers: {dict(request.headers)}")
        if request.is_json:
            logger.info(f"Body: {request.json}")
        elif request.method == 'POST':
            logger.info(f"Raw data: {request.get_data(as_text=True)[:200]}")

# ==================== API KEY DECORATOR ====================

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
    
def load_api_keys():
    """Load API keys from file"""
    try:
        if os.path.exists('production_keys.json'):
            with open('production_keys.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading API keys: {e}")
    
    # Default API key for testing (in production, remove this)
    return {
        'agri_ipam8n8l9r_s-2GA9_e4EL9Gb_nkrgCgwVBQ09Ktug8': {
            'active': True,
            'name': 'Vaibhav',
            'created': '2026-04-27'
        }
    }

def verify_api_key(api_key):
    """Verify if API key is valid"""
    keys = load_api_keys()
    if api_key in keys:
        return keys[api_key].get('active', True)
    return False

def require_api_key(f):
    """Decorator to require API key for endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = None
        
        # Check header first
        if 'X-API-Key' in request.headers:
            api_key = request.headers['X-API-Key']
        # Then check query parameter
        elif 'api_key' in request.args:
            api_key = request.args['api_key']
        # Then check JSON body
        elif request.is_json and request.json and 'api_key' in request.json:
            api_key = request.json['api_key']
        
        if not api_key:
            logger.warning("No API key provided")
            return jsonify({
                'error': 'API key required',
                'message': 'Please provide API key in X-API-Key header'
            }), 401
        
        if not verify_api_key(api_key):
            logger.warning(f"Invalid API key attempted: {api_key[:20]}...")
            return jsonify({'error': 'Invalid API key'}), 401
        
        logger.info(f"API key verified successfully")
        return f(*args, **kwargs)
    
    return decorated_function

# ==================== LOAD MODEL ====================
print("\n" + "="*60)
print("🤖 Loading Crop Recommendation Model...")
print("="*60)

model = None
label_encoder = None
feature_cols = None
model_accuracy = 0

try:
    # Try different paths for the model
    model_paths = [
        'models/crop_model_complete.pkl',
        'crop_model_complete.pkl',
        '../models/crop_model_complete.pkl'
    ]
    
    model_package = None
    for path in model_paths:
        if os.path.exists(path):
            model_package = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            break
    
    if model_package:
        model = model_package['model']
        label_encoder = model_package['label_encoder']
        feature_cols = model_package['feature_columns']
        model_accuracy = model_package.get('accuracy', 0.85)
        
        print(f"✅ Model loaded!")
        print(f"   Accuracy: {model_accuracy*100:.1f}%")
        print(f"   Supports: {len(label_encoder.classes_)} crops")
    else:
        print(f"⚠️ Model file not found, using fallback mode")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    logger.error(f"Model loading error: {e}")

# Crop database with detailed information
CROP_DETAILS = {
    'rice': {
        'yield_tons_per_acre': (2.5, 4.0),
        'base_risk': 'Medium',
        'base_profitability': 'High',
        'water_needs': 'High',
        'growing_days': 120,
        'price_per_ton': 28000,
        'cost_per_acre': 35000,
        'reasons': [
            'Requires high rainfall or irrigation',
            'Warm and humid conditions ideal',
            'Fertile soil with good water retention',
            'High market demand throughout year'
        ]
    },
    'cotton': {
        'yield_tons_per_acre': (1.5, 2.5),
        'base_risk': 'High',
        'base_profitability': 'Medium-High',
        'water_needs': 'Medium',
        'growing_days': 160,
        'price_per_ton': 60000,
        'cost_per_acre': 45000,
        'reasons': [
            'Thrives in hot and dry conditions',
            'Well-drained black soil preferred',
            'Drought-tolerant once established',
            'Good for regions with moderate rainfall'
        ]
    },
    'coffee': {
        'yield_tons_per_acre': (0.8, 1.5),
        'base_risk': 'Medium',
        'base_profitability': 'High',
        'water_needs': 'Medium',
        'growing_days': 365,
        'price_per_ton': 150000,
        'cost_per_acre': 80000,
        'reasons': [
            'Cool and moist climate ideal',
            'Well-drained soil with organic matter',
            'Shaded cultivation beneficial',
            'Premium price in market'
        ]
    },
    'mango': {
        'yield_tons_per_acre': (5.0, 8.0),
        'base_risk': 'Low-Medium',
        'base_profitability': 'High',
        'water_needs': 'Low-Medium',
        'growing_days': 365,
        'price_per_ton': 35000,
        'cost_per_acre': 45000,
        'reasons': [
            'Hot and dry climate ideal for flowering',
            'Well-drained soil required',
            'Drought-tolerant once mature',
            'Excellent market demand'
        ]
    },
    'banana': {
        'yield_tons_per_acre': (25.0, 40.0),
        'base_risk': 'Medium',
        'base_profitability': 'Very High',
        'water_needs': 'High',
        'growing_days': 270,
        'price_per_ton': 25000,
        'cost_per_acre': 80000,
        'reasons': [
            'Warm and humid conditions essential',
            'Rich soil with good drainage',
            'Regular water supply required',
            'High yield potential'
        ]
    },
    'grapes': {
        'yield_tons_per_acre': (8.0, 15.0),
        'base_risk': 'Medium-High',
        'base_profitability': 'High',
        'water_needs': 'Medium',
        'growing_days': 180,
        'price_per_ton': 50000,
        'cost_per_acre': 120000,
        'reasons': [
            'Mediterranean climate ideal',
            'Well-drained soil essential',
            'Requires careful pruning and management',
            'High value crop'
        ]
    },
    'potato': {
        'yield_tons_per_acre': (8.0, 12.0),
        'base_risk': 'Medium',
        'base_profitability': 'Medium-High',
        'water_needs': 'Medium',
        'growing_days': 90,
        'price_per_ton': 20000,
        'cost_per_acre': 50000,
        'reasons': [
            'Cool temperatures ideal',
            'Well-drained loamy soil',
            'Regular irrigation needed',
            'Short growing cycle'
        ]
    },
    'orange': {
        'yield_tons_per_acre': (10.0, 20.0),
        'base_risk': 'Medium',
        'base_profitability': 'High',
        'water_needs': 'Medium',
        'growing_days': 365,
        'price_per_ton': 30000,
        'cost_per_acre': 60000,
        'reasons': [
            'Subtropical climate ideal',
            'Well-drained soil required',
            'Protection from frost needed',
            'Good market demand'
        ]
    }
}

# ==================== PRICE ADVISOR CLASS ====================
class PriceAdvisor:
    """Price prediction and sell/hold recommendation engine"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.historical_df = None
        self.load_price_model()
        self.load_historical_data()
    
    def load_price_model(self):
        """Load the price prediction model"""
        try:
            model_paths = [
                'models/price_trend_model.pkl',
                'price_trend_model.pkl',
                '../models/price_trend_model.pkl'
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    model_package = joblib.load(path)
                    self.model = model_package['model']
                    self.scaler = model_package['scaler']
                    self.feature_cols = model_package['feature_columns']
                    self.r2_score = model_package.get('r2_score', 0.85)
                    logger.info(f"✅ Price trend model loaded from {path}")
                    return
            
            logger.warning("⚠️ Price model not found")
            self.model = None
        except Exception as e:
            self.model = None
            logger.warning(f"⚠️ Price model not found: {e}")
    
    def load_historical_data(self):
        """Load historical price data for trend analysis"""
        try:
            data_paths = [
                'price_prediction_cleaned.csv',
                'data/price_prediction_cleaned.csv',
                '../price_prediction_cleaned.csv'
            ]
            
            for path in data_paths:
                if os.path.exists(path):
                    self.historical_df = pd.read_csv(path)
                    logger.info(f"✅ Historical data loaded from {path}: {len(self.historical_df)} records")
                    return
            
            self.historical_df = None
            logger.warning("⚠️ Historical data not found")
        except Exception as e:
            self.historical_df = None
            logger.warning(f"⚠️ Historical data not found: {e}")
    
    def predict_price(self, district, market, commodity, variety, grade, target_month):
        """Predict price for given parameters"""
        if self.model is None or self.scaler is None:
            return None
        
        current_date = datetime.now()
        
        # Create features
        features = {
            'Year': target_month // 12 + current_date.year if target_month > current_date.month else current_date.year,
            'Month': target_month % 12 if target_month % 12 != 0 else 12,
            'Day': 15,
            'DayOfWeek': 2,
            'Quarter': ((target_month % 12 if target_month % 12 != 0 else 12) - 1) // 3 + 1,
            'District_Encoded': abs(hash(district)) % 100,
            'Market_Encoded': abs(hash(market)) % 100,
            'Commodity_Encoded': abs(hash(commodity)) % 100,
            'Variety_Encoded': abs(hash(variety)) % 100,
            'Grade_Encoded': abs(hash(grade)) % 10,
            'Season_Encoded': self._get_season_encoding(target_month),
            'Price_MA7': 0,
            'Price_MA30': 0,
            'Price_Momentum': 1.0,
            'Weekly_Change': 0,
            'Volatility_30d': 0,
            'Price_Position': 0.5
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([{col: features.get(col, 0) for col in self.feature_cols}])
        
        # Scale and predict
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)[0]
        
        return prediction
    
    def _get_season_encoding(self, month):
        """Get season encoding for a given month"""
        if month in [6, 7, 8, 9]:
            return 0  # Kharif
        elif month in [10, 11, 12, 1]:
            return 1  # Rabi
        else:
            return 2  # Summer
    
    def get_price_trend(self, district, market, commodity, variety, grade):
        """Analyze price trend and provide sell/hold recommendations"""
        
        current_month = datetime.now().month
        current_price = self._get_current_price(district, market, commodity, variety, grade)
        
        if current_price is None:
            return None
        
        # Convert to Python float
        current_price = float(current_price)
        
        # Predict prices for next 3 months
        predictions = []
        for months_ahead in [1, 2, 3]:
            target_month = current_month + months_ahead
            if target_month > 12:
                target_month -= 12
            
            predicted_price = self.predict_price(
                district, market, commodity, variety, grade, target_month
            )
            if predicted_price:
                predicted_price = float(predicted_price)
                change_percent = round(((predicted_price - current_price) / current_price) * 100, 1)
                predictions.append({
                    'months_ahead': months_ahead,
                    'weeks_ahead': months_ahead * 4,
                    'predicted_price': predicted_price,
                    'change_percent': change_percent
                })
        
        if not predictions:
            return None
        
        # Determine best action based on price trends
        best_prediction = max(predictions, key=lambda x: x['predicted_price'])
        
        if best_prediction['change_percent'] > 5:
            action = 'HOLD'
            action_text = f"Hold for {best_prediction['weeks_ahead']} weeks"
            reason = f"Price expected to increase by {best_prediction['change_percent']:.1f}% in {best_prediction['weeks_ahead']} weeks"
            urgency = 'Low'
            confidence = min(85, 60 + best_prediction['change_percent'])
        elif best_prediction['change_percent'] > 0:
            action = 'CONSIDER_HOLD'
            action_text = f"Consider holding for {best_prediction['weeks_ahead']} weeks"
            reason = f"Modest price increase of {best_prediction['change_percent']:.1f}% expected"
            urgency = 'Medium'
            confidence = 65
        elif best_prediction['change_percent'] > -5:
            action = 'SELL_SOON'
            action_text = "Sell within 1-2 weeks"
            reason = f"Prices expected to stabilize or decline slightly ({best_prediction['change_percent']:.1f}%)"
            urgency = 'Medium-High'
            confidence = 70
        else:
            action = 'SELL_NOW'
            action_text = "Sell immediately"
            reason = f"Prices expected to drop by {abs(best_prediction['change_percent']):.1f}% in coming weeks"
            urgency = 'High'
            confidence = 75
        
        # Calculate risk level
        volatility = self._calculate_volatility(district, market, commodity)
        
        # Calculate expected gain percent
        positive_changes = [p['change_percent'] for p in predictions if p['change_percent'] > 0]
        if positive_changes:
            expected_gain = max(positive_changes)
        else:
            expected_gain = min(p['change_percent'] for p in predictions)
        
        return {
            'current_price': current_price,
            'action': action,
            'action_text': action_text,
            'reason': reason,
            'urgency': urgency,
            'confidence': int(confidence),
            'predictions': predictions,
            'volatility': volatility,
            'best_holding_period': f"{best_prediction['weeks_ahead']} weeks" if best_prediction['change_percent'] > 0 else "None",
            'expected_gain_percent': float(expected_gain)
        }

    def _get_current_price(self, district, market, commodity, variety, grade):
        """Get current market price"""
        if self.historical_df is None:
            # Return fallback price based on commodity
            fallback_prices = {
                'rice': 2500,
                'wheat': 2200,
                'cotton': 5500,
                'sugarcane': 3500,
                'maize': 1800
            }
            return fallback_prices.get(commodity.lower(), 2500)
        
        try:
            # Filter for latest price
            latest = self.historical_df[
                (self.historical_df['District'].str.contains(district, case=False, na=False)) &
                (self.historical_df['Market'].str.contains(market, case=False, na=False)) &
                (self.historical_df['Commodity'].str.contains(commodity, case=False, na=False))
            ].sort_values('Arrival_Date', ascending=False)
            
            if len(latest) > 0:
                return latest.iloc[0]['Modal_Price']
            else:
                # Fallback to average for commodity
                avg_price = self.historical_df[self.historical_df['Commodity'].str.contains(commodity, case=False, na=False)]['Modal_Price'].mean()
                return avg_price if not pd.isna(avg_price) else 2500
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return 2500
    
    def _calculate_volatility(self, district, market, commodity):
        """Calculate price volatility for recommendation"""
        if self.historical_df is None:
            return 'Medium'
        
        try:
            prices = self.historical_df[
                (self.historical_df['District'].str.contains(district, case=False, na=False)) &
                (self.historical_df['Market'].str.contains(market, case=False, na=False)) &
                (self.historical_df['Commodity'].str.contains(commodity, case=False, na=False))
            ]['Modal_Price'].tail(30)
            
            if len(prices) > 5:
                std = prices.std()
                mean = prices.mean()
                cv = std / mean if mean > 0 else 0
                
                if cv < 0.1:
                    return 'Low'
                elif cv < 0.2:
                    return 'Medium'
                else:
                    return 'High'
            return 'Medium'
        except:
            return 'Medium'

# ==================== HELPER FUNCTIONS ====================
def calculate_suitability_score(crop, conditions):
    """Calculate suitability percentage based on conditions"""
    crop_lower = crop.lower()
    if crop_lower not in CROP_DETAILS:
        return 75.0
    
    temp = conditions.get('temperature', 25)
    rainfall = conditions.get('rainfall', 150)
    
    score = 70
    
    if crop_lower == 'cotton' and 25 <= temp <= 35:
        score += 15
    elif crop_lower == 'coffee' and 15 <= temp <= 22:
        score += 15
    elif crop_lower == 'rice' and 22 <= temp <= 32:
        score += 15
    elif temp > 35:
        score -= 10
    
    if crop_lower == 'rice' and rainfall > 200:
        score += 10
    elif crop_lower == 'cotton' and 50 <= rainfall <= 100:
        score += 10
    elif rainfall < 50:
        score -= 15
    
    return min(98, max(40, score))

def prepare_features(temperature, humidity, rainfall, ph, N, P, K):
    """Prepare features for model prediction"""
    if feature_cols is None:
        # Return mock features if model not loaded
        return pd.DataFrame([{
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
            'ph': ph,
            'N': N,
            'P': P,
            'K': K
        }])
    
    df = pd.DataFrame([{
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall,
        'ph': ph,
        'N': N,
        'P': P,
        'K': K
    }])
    
    df['temp_normalized'] = (df['temperature'] + 10) / 50 * 100
    df['temp_normalized'] = df['temp_normalized'].clip(0, 100)
    df['rainfall_normalized'] = df['rainfall'] / 300 * 100
    df['rainfall_normalized'] = df['rainfall_normalized'].clip(0, 100)
    df['N_P_ratio'] = df['N'] / (df['P'] + 1)
    df['K_P_ratio'] = df['K'] / (df['P'] + 1)
    df['NPK_total'] = df['N'] + df['P'] + df['K']
    df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
    df['rain_fertility'] = df['rainfall'] * df['NPK_total'] / 1000
    
    def calc_suit(row):
        score = 0
        if 20 <= row['temperature'] <= 30:
            score += 25
        elif 15 <= row['temperature'] <= 35:
            score += 15
        else:
            score += 5
        if 100 <= row['rainfall'] <= 200:
            score += 25
        elif 50 <= row['rainfall'] <= 250:
            score += 15
        else:
            score += 5
        if row['NPK_total'] > 200:
            score += 25
        elif row['NPK_total'] > 100:
            score += 15
        else:
            score += 5
        if 6.0 <= row['ph'] <= 7.5:
            score += 25
        elif 5.5 <= row['ph'] <= 8.0:
            score += 15
        else:
            score += 5
        return score
    
    df['suitability_score'] = df.apply(calc_suit, axis=1)
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_cols]

def get_crop_details(crop_name, confidence, conditions):
    """Get comprehensive details for a crop"""
    crop_lower = crop_name.lower()
    
    if crop_lower in CROP_DETAILS:
        details = CROP_DETAILS[crop_lower]
        yield_min, yield_max = details['yield_tons_per_acre']
        expected_yield = (yield_min + yield_max) / 2
        
        revenue = expected_yield * details['price_per_ton']
        profit = revenue - details['cost_per_acre']
        profit_margin = (profit / revenue) * 100 if revenue > 0 else 0
        
        risk = details['base_risk']
        if confidence < 50:
            risk = f"{risk}+ (Higher due to suboptimal conditions)"
        
        suitability = calculate_suitability_score(crop_name, conditions)
        
        return {
            'expected_yield_tons_per_acre': round(expected_yield, 1),
            'yield_range': f"{yield_min} - {yield_max}",
            'risk_level': risk,
            'profitability_level': details['base_profitability'],
            'profit_per_hectare': round(profit * 2.47, 0),
            'profit_margin': round(profit_margin, 1),
            'suitability_percentage': suitability,
            'water_requirement': details['water_needs'],
            'growing_days': details['growing_days'],
            'reasons': details['reasons']
        }
    else:
        return {
            'expected_yield_tons_per_acre': round(confidence / 10, 1),
            'yield_range': f"{round(confidence/12, 1)} - {round(confidence/8, 1)}",
            'risk_level': 'Medium',
            'profitability_level': 'Medium',
            'profit_per_hectare': round(confidence * 1000, 0),
            'profit_margin': round(confidence / 2, 1),
            'suitability_percentage': round(confidence, 1),
            'water_requirement': 'Moderate',
            'growing_days': 120,
            'reasons': [
                f"{crop_name} is suitable for current conditions",
                "Moderate soil fertility supports growth",
                "Seasonal weather patterns favorable"
            ]
        }

def get_top_predictions(conditions, top_n=5):
    """Get top N crop predictions with probabilities"""
    if model is None:
        # Return fallback predictions if model not loaded
        fallback_crops = ['rice', 'cotton', 'maize', 'wheat', 'sugarcane']
        predictions = []
        for i, crop in enumerate(fallback_crops[:top_n]):
            confidence = 85 - (i * 10)
            details = get_crop_details(crop, confidence, conditions)
            predictions.append({
                'crop': crop,
                'confidence': max(40, confidence),
                **details
            })
        return predictions
    
    try:
        fertility = conditions.get('soil_fertility', 'Medium Fertility')
        soil_map = {'Low Fertility': (30, 25, 35), 'Medium Fertility': (60, 55, 65), 'High Fertility': (90, 85, 95)}
        N, P, K = soil_map.get(fertility, (60, 55, 65))
        
        rainfall_map = {'Very Low': 50, 'Low': 100, 'Medium': 150, 'High': 250}
        rainfall = rainfall_map.get(conditions.get('rainfall_category', 'Medium'), 150)
        
        if rainfall > 200:
            humidity = 85
        elif rainfall > 100:
            humidity = 65
        else:
            humidity = 45
        
        features = prepare_features(
            conditions['temperature'],
            humidity,
            rainfall,
            conditions.get('ph', 6.5),
            N, P, K
        )
        
        probabilities = model.predict_proba(features)[0]
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        
        predictions = []
        for idx in top_indices:
            crop = label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx] * 100
            details = get_crop_details(crop, confidence, conditions)
            
            predictions.append({
                'crop': crop,
                'confidence': round(confidence, 1),
                **details
            })
        
        return predictions
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Return fallback predictions
        return get_top_predictions(conditions, top_n)  # This will use fallback

# Initialize price advisor
price_advisor = PriceAdvisor()

# ==================== ROUTES ====================
@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Complete Crop Recommendation API',
        'version': '3.0',
        'status': 'running',
        'model_accuracy': f"{model_accuracy*100:.1f}%" if model_accuracy else "Model not loaded",
        'crops_supported': len(label_encoder.classes_) if label_encoder else len(CROP_DETAILS),
        'features': [
            'ML-based crop prediction',
            'Yield estimates (tons/acre)',
            'Risk assessment',
            'Profitability analysis',
            'Suitability percentage',
            'Detailed reasoning',
            'Price prediction with sell/hold recommendations'
        ],
        'endpoints': {
            'test': '/api/v1/test',
            'predict': '/api/v1/predict',
            'price_advisor': '/api/v1/price-advisor',
            'compare_crops': '/api/v1/compare-crops',
            'crop_details': '/api/v1/crop-details/{crop_name}'
        }
    })

@app.route('/api/v1/test', methods=['GET', 'OPTIONS'])
@require_api_key
def test_api():
    """Test endpoint to verify API is working"""
    return jsonify({
        'status': 'success',
        'message': 'API is working!',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/api/v1/predict', methods=['POST', 'OPTIONS'])
@require_api_key
def predict():
    """Get comprehensive crop recommendations"""
    
    try:
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        conditions = {
            'temperature': data.get('temperature'),
            'rainfall_category': data.get('rainfall_category'),
            'soil_fertility': data.get('soil_fertility'),
            'ph': data.get('ph', 6.5)
        }
        
        required = ['temperature', 'rainfall_category', 'soil_fertility']
        missing = [field for field in required if conditions.get(field) is None]
        
        if missing:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing)}',
                'required_fields': required,
                'example': {
                    'temperature': 28,
                    'rainfall_category': 'High',
                    'soil_fertility': 'High Fertility',
                    'ph': 6.5
                }
            }), 400
        
        # Validate temperature range
        if not (0 <= conditions['temperature'] <= 50):
            return jsonify({'error': 'Temperature must be between 0 and 50°C'}), 400
        
        # Validate rainfall category
        valid_rainfall = ['Very Low', 'Low', 'Medium', 'High']
        if conditions['rainfall_category'] not in valid_rainfall:
            return jsonify({'error': f'Rainfall category must be one of: {valid_rainfall}'}), 400
        
        # Validate soil fertility
        valid_fertility = ['Low Fertility', 'Medium Fertility', 'High Fertility']
        if conditions['soil_fertility'] not in valid_fertility:
            return jsonify({'error': f'Soil fertility must be one of: {valid_fertility}'}), 400
        
        # Validate pH
        if not (0 <= conditions['ph'] <= 14):
            return jsonify({'error': 'pH must be between 0 and 14'}), 400
        
        # Get predictions
        predictions = get_top_predictions(conditions, top_n=5)
        
        if not predictions:
            return jsonify({'error': 'Prediction failed. Please try again.'}), 500
        
        # Ensure all numpy types are converted
        predictions = convert_numpy_types(predictions)
        
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'input_conditions': conditions,
            'model_info': {
                'accuracy': f"{model_accuracy*100:.1f}%" if model_accuracy else "Fallback mode",
                'crops_analyzed': len(label_encoder.classes_) if label_encoder else len(CROP_DETAILS),
                'model_loaded': model is not None
            },
            'recommendations': predictions,
            'best_crop': predictions[0]['crop'],
            'best_crop_confidence': predictions[0]['confidence']
        }
        
        logger.info(f"Prediction successful for conditions: {conditions}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/v1/price-advisor', methods=['POST', 'OPTIONS'])
@require_api_key
def price_advisor_api():
    """Get price prediction with sell/hold recommendations"""
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        district = data.get('district')
        market = data.get('market')
        commodity = data.get('commodity')
        variety = data.get('variety', 'Any')
        grade = data.get('grade', 'A')
        
        required = ['district', 'market', 'commodity']
        missing = [f for f in required if not data.get(f)]
        
        if missing:
            return jsonify({
                'error': 'Missing parameters',
                'required': missing,
                'example': {
                    'district': 'Pune',
                    'market': 'Gultekdi',
                    'commodity': 'Rice',
                    'variety': 'Basmati',
                    'grade': 'A'
                }
            }), 400
        
        recommendation = price_advisor.get_price_trend(
            district, market, commodity, variety, grade
        )
        
        if recommendation is None:
            return jsonify({'error': 'Could not generate recommendation. Please try different inputs.'}), 500
        
        action_colors = {
            'SELL_NOW': 'red',
            'SELL_SOON': 'orange',
            'CONSIDER_HOLD': 'yellow',
            'HOLD': 'green'
        }
        
        # Process predictions to convert numpy types
        serializable_predictions = []
        for pred in recommendation['predictions']:
            serializable_predictions.append({
                'months_ahead': int(pred['months_ahead']),
                'weeks_ahead': int(pred['weeks_ahead']),
                'predicted_price': float(pred['predicted_price']),
                'change_percent': float(pred['change_percent'])
            })
        
        response = {
            'success': True,
            'data': {
                'commodity': commodity,
                'market': market,
                'district': district,
                'current_price': float(recommendation['current_price']),
                'recommendation': {
                    'action': recommendation['action'],
                    'action_text': recommendation['action_text'],
                    'reason': recommendation['reason'],
                    'urgency': recommendation['urgency'],
                    'confidence': int(recommendation['confidence']),
                    'color': action_colors.get(recommendation['action'], 'gray')
                },
                'price_forecast': serializable_predictions,
                'volatility': recommendation['volatility'],
                'best_holding_period': recommendation['best_holding_period'],
                'expected_gain_percent': float(recommendation['expected_gain_percent'])
            }
        }
        
        logger.info(f"Price advice generated for {commodity} in {district}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in price advisor: {e}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/v1/compare-crops', methods=['POST', 'OPTIONS'])
@require_api_key
def compare_crops():
    """Compare multiple crops"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        crops = data.get('crops', [])
        conditions = data.get('conditions', {})
        
        if not crops:
            return jsonify({'error': 'Please provide list of crops to compare'}), 400
        
        if not conditions.get('temperature') or not conditions.get('rainfall_category') or not conditions.get('soil_fertility'):
            return jsonify({'error': 'Please provide temperature, rainfall_category, and soil_fertility in conditions'}), 400
        
        # Get predictions for each crop
        all_predictions = get_top_predictions(conditions, top_n=len(CROP_DETAILS))
        
        comparisons = []
        for crop in crops:
            crop_data = next((p for p in all_predictions if p['crop'].lower() == crop.lower()), None)
            
            if crop_data:
                comparisons.append({
                    'crop': crop,
                    'suitability': crop_data.get('suitability_percentage', 0),
                    'expected_yield_tons': crop_data.get('expected_yield_tons_per_acre', 0),
                    'profit_margin': crop_data.get('profit_margin', 0),
                    'risk_level': crop_data.get('risk_level', 'Medium'),
                    'confidence': crop_data.get('confidence', 0)
                })
        
        if not comparisons:
            return jsonify({'error': 'No matching crops found'}), 404
        
        # Sort by suitability
        comparisons.sort(key=lambda x: x['suitability'], reverse=True)
        
        return jsonify({
            'success': True,
            'comparison': comparisons,
            'best_crop': comparisons[0]['crop'] if comparisons else None,
            'conditions': conditions
        }), 200
        
    except Exception as e:
        logger.error(f"Error in compare crops: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/crop-details/<crop_name>', methods=['GET', 'OPTIONS'])
@require_api_key
def crop_details_endpoint(crop_name):
    """Get detailed information about a specific crop"""
    try:
        crop_lower = crop_name.lower()
        
        if crop_lower in CROP_DETAILS:
            details = CROP_DETAILS[crop_lower]
            return jsonify({
                'crop': crop_name,
                'details': details,
                'success': True
            }), 200
        else:
            return jsonify({
                'error': f'Crop {crop_name} not found',
                'available_crops': list(CROP_DETAILS.keys())
            }), 404
            
    except Exception as e:
        logger.error(f"Error in crop details: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation for available endpoints'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An error occurred on the server. Please try again later.'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 COMPLETE CROP RECOMMENDATION API")
    print("="*60)
    print(f"\n📊 Model Status: {'Loaded' if model else 'Using Fallback Mode'}")
    if model:
        print(f"   Accuracy: {model_accuracy*100:.1f}%")
        print(f"   Supports: {len(label_encoder.classes_)} crops")
    else:
        print(f"   Using fallback crop database with {len(CROP_DETAILS)} crops")
    print("\n📋 Features available:")
    print("   • Crop recommendations with yield estimates")
    print("   • Risk assessment and profitability analysis")
    print("   • Suitability percentage and detailed reasoning")
    print("   • Price predictions with sell/hold recommendations")
    print("\n🌐 Endpoints:")
    print("   GET  / - API information")
    print("   GET  /api/v1/test - Test API")
    print("   POST /api/v1/predict - Crop recommendations")
    print("   POST /api/v1/price-advisor - Price trend analysis")
    print("   POST /api/v1/compare-crops - Compare multiple crops")
    print("   GET  /api/v1/crop-details/<crop> - Crop details")
    print("\n🔑 API Key Required for all /api/v1/* endpoints")
    print("="*60)
    
    # Get port from environment variable for Render.com
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production