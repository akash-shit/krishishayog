# All imports are grouped at the top for clarity and to prevent NameErrors
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import sqlite3
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import logging
from datetime import datetime
import requests
import random
import time
from threading import Thread
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2
import json
from googletrans import Translator

import joblib # New Import
import pandas as pd # New Import

from config import Config
from utils import (ImageProcessor, DatabaseUtils, WeatherUtils, 
                    CropDataAnalyzer, TranslationUtils)

# Load .env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)

# Check for API key at startup
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logging.error("GOOGLE_API_KEY not set in .env")
    raise ValueError("API key not found")

# The decorator function MUST be defined before it is used by any route
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# NUCLEAR OPTION - COMPLETELY DELETE AND RECREATE DATABASE
def force_database_recreation():
    db_path = 'agricultural_data.db'
    
    # Delete main database file
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"🗑️ DELETED old database: {db_path}")
        except Exception as e:
            print(f"Error deleting database: {e}")
    
    # Delete any journal files
    journal_path = db_path + '-journal'
    if os.path.exists(journal_path):
        try:
            os.remove(journal_path)
            print(f"🗑️ DELETED journal file: {journal_path}")
        except Exception as e:
            print(f"Error deleting journal: {e}")
    
    # Delete any other potential SQLite files
    for ext in ['-shm', '-wal']:
        temp_path = db_path + ext
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🗑️ DELETED temp file: {temp_path}")
            except Exception as e:
                print(f"Error deleting temp file: {e}")

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = 'your_super_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize translator
translator = Translator()

# Initialize database
DatabaseUtils.create_tables()

# Global sensor data
sensor_data = {
    'soil_ph': 0,
    'soil_moisture': 0,
    'soil_temperature': 0,
    'nitrogen': 0,
    'phosphorus': 0,
    'potassium': 0,
    'last_updated': datetime.now().isoformat()
}

# Add Fertilizer Prediction Model Loading
# Replace 'path/to/your/fertilizer_model.pkl' with the actual path to your file
try:
    fertilizer_model = joblib.load(os.path.join(Config.MODEL_DIR, 'fertilizer_model.pkl'))
    print("✓ Fertilizer recommendation model loaded successfully.")
except FileNotFoundError:
    print("⚠ Fertilizer model file not found. Using mock predictions.")
    fertilizer_model = None

# Plant Disease Detection Model
class EnhancedPlantDiseaseDetector:
    # ... (the entire class content, unchanged) ...
    def __init__(self):
        self.models = {}  # Dictionary to store multiple models
        self.classes = {
            'wheat': ['HealthyLeaf', 'BlackPoint', 'LeafBlight', 'FusariumFootRot','WheatBlast'],
            'tomato': ['healthy', 'bacterial_spot', 'early_blight', 'late_blight', 'leaf_mold', 'septoria_leaf_spot', 'spider_mites', 'target_spot', 'mosaic_virus', 'yellow_leaf_curl'],
            'potato': ['Potato___healthy', 'Potato___Early_blight', 'Potato___late_blight'],
            'rice': ['healthy', 'bacterial_blight', 'brown_spot', 'leaf_smut']
        }
        
        self.treatments = {
            # Wheat treatments
            'HealthyLeaf': 'No pesticide needed',
            'BlackPoint': 'Fungicides like Mancozeb, Azoxystrobin, or Chlorothalonil',
            'LeafBlight': 'Fungicides like Copper-based fungicides, Mancozeb, or Chlorothalonil',
            'FusariumFootRot': 'Fungicides like Prothioconazole or Tebuconazole',
            'WheatBlast': 'Fungicides like Tricyclazole or Azoxystrobin',
            
            # Tomato treatments
            'tomato_healthy': 'Continue preventive care and regular monitoring.',
            'tomato_bacterial_spot': 'Apply copper-based bactericides. Use resistant varieties.',
            'tomato_early_blight': 'Use chlorothalonil fungicide. Remove affected leaves.',
            'tomato_late_blight': 'Apply metalaxyl fungicides. Improve ventilation.',
            'tomato_leaf_mold': 'Increase ventilation. Apply fungicides.',
            'tomato_septoria_leaf_spot': 'Use chlorothalonil or copper fungicides.',
            'tomato_spider_mites': 'Apply miticides or neem oil. Increase humidity.',
            'tomato_target_spot': 'Use fungicides containing chlorothalonil.',
            'tomato_mosaic_virus': 'Remove infected plants. Sanitize tools and hands.',
            'tomato_yellow_leaf_curl': 'Control whiteflies. Remove infected plants.',
            
            # Potato treatments
            'Potato___healthy': 'Continue regular monitoring and care.',
            'Potato___Early_blight': 'Apply chlorothalonil or mancozeb. Improve air circulation.',
            'Potato___late_blight': 'Use metalaxyl-based fungicides. Avoid overhead irrigation.',
            
            # Rice treatments
            'rice_healthy': 'Maintain proper water management and nutrition.',
            'rice_bacterial_blight': 'Use copper oxychloride spray. Improve drainage.',
            'rice_brown_spot': 'Apply potash fertilizer. Use propiconazole or tricyclazole.',
            'rice_leaf_smut': 'Apply tricyclazole fungicide. Use resistant varieties.'
        }
        
        self.severity_levels = {
            'low': (0, 30),
            'medium': (30, 70),
            'high': (70, 100)
        }
    
    def load_models(self):
        """Load all plant disease models with robust error handling"""
        models_dir = 'models'
        model_files = {
            'wheat': 'Wheat_best_final_model.keras',
            'tomato': 'tomato_model.keras', 
            'potato': 'potato_best_final_model.keras',
            'rice': 'rice_best_final_model_fixed.keras'
        }
        
        loaded_count = 0
        for plant_type, model_file in model_files.items():
            model_path = os.path.join(models_dir, model_file)
            try:
                if os.path.exists(model_path):
                    # First try to load normally
                    try:
                        self.models[plant_type] = tf.keras.models.load_model(model_path)
                        print(f"✓ {plant_type.capitalize()} model loaded successfully")
                    except Exception as e:
                        print(f"Standard load failed for {plant_type}, trying compile=False: {e}")
                        # If that fails, try loading without compilation
                        self.models[plant_type] = tf.keras.models.load_model(
                            model_path, 
                            compile=False
                        )
                        print(f"✓ {plant_type.capitalize()} model loaded without compilation")
                    
                    loaded_count += 1
                else:
                    print(f"⚠ {plant_type.capitalize()} model file not found at {model_path}")
            except Exception as e:
                print(f"Error loading {plant_type} model: {e}")
                import traceback
                traceback.print_exc()
        
        return loaded_count > 0  # Return True if at least one model loaded
    
    def predict_disease(self, image_path, plant_type=None):
        """Predict disease from plant image"""
        print(f"Predicting disease for image: {image_path}, plant_type: {plant_type}")
        
        # If no specific plant type provided, try to detect it
        if plant_type is None:
            plant_type = self._detect_plant_type(image_path)
            print(f"Auto-detected plant type: {plant_type}")
        
        # Check if we have a model for this plant type
        if plant_type not in self.models:
            print(f"No model found for plant type: {plant_type}, using mock prediction")
            return self._get_mock_prediction(plant_type)
        
        processed_image = ImageProcessor.preprocess_for_ml(image_path)
        if processed_image is None:
            print(f"Image preprocessing failed for: {image_path}")
            return self._get_mock_prediction(plant_type)
        
        try:
            print(f"Making prediction with {plant_type} model...")
            predictions = self.models[plant_type].predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            predicted_class = self.classes[plant_type][predicted_class_idx]
            disease_key = f"{plant_type}_{predicted_class}"
            treatment = self.treatments.get(disease_key, "Consult agricultural expert")
            severity = self._determine_severity(confidence)
            
            result = {
                'plant_type': plant_type.capitalize(),
                'disease': predicted_class.replace('_', ' ').title(),
                'confidence': confidence * 100,
                'treatment': treatment,
                'severity': severity,
                'recommendations': self._get_detailed_recommendations(disease_key, severity)
            }
            
            print(f"Prediction successful: {result}")
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._get_mock_prediction(plant_type)
    
    def _detect_plant_type(self, image_path):
        """Simple plant type detection based on image features"""
        # This is a simplified approach - you might want to implement a proper plant classifier
        # For now, we'll use a mock detection based on file name or random selection
        filename = os.path.basename(image_path).lower()
        
        if 'wheat' in filename:
            return 'wheat'
        elif 'tomato' in filename:
            return 'tomato'
        elif 'potato' in filename:
            return 'potato'
        elif 'rice' in filename:
            return 'rice'
        else:
            # Random selection for demo purposes
            return random.choice(['wheat', 'tomato', 'potato', 'rice'])
    
    def _get_mock_prediction(self, plant_type):
        """Generate mock prediction for demo purposes"""
        if plant_type not in self.classes:
            plant_type = 'tomato'  # Default fallback
        
        disease = random.choice(self.classes[plant_type])
        confidence = random.uniform(75, 95)
        disease_key = f"{plant_type}_{disease}"
        
        return {
            'plant_type': plant_type.capitalize(),
            'disease': disease.replace('_', ' ').title(),
            'confidence': confidence,
            'treatment': self.treatments.get(disease_key, "Continue monitoring"),
            'severity': self._determine_severity(confidence/100),
            'recommendations': self._get_detailed_recommendations(disease_key, self._determine_severity(confidence/100))
        }
    
    def _determine_severity(self, confidence):
        if confidence < 0.3:
            return 'low'
        elif confidence < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def _get_detailed_recommendations(self, disease, severity):
        base_recommendations = [
            "Monitor plant regularly for symptoms",
            "Maintain proper field hygiene",
            "Use disease-resistant varieties when possible"
        ]
        
        if 'healthy' not in disease.lower():
            if severity == 'high':
                base_recommendations.extend([
                    "Immediate treatment required",
                    "Consider professional consultation",
                    "Isolate affected plants if possible"
                ])
            elif severity == 'medium':
                base_recommendations.extend([
                    "Apply preventive measures",
                    "Monitor closely for spread"
                ])
        
        return base_recommendations

# Initialize plant detector
plant_detector = EnhancedPlantDiseaseDetector()

# Enhanced Weather Service
class RealTimeWeatherService:
    @staticmethod
    def get_comprehensive_weather(lat=None, lon=None):
        if lat is None:
            lat = Config.DEFAULT_LAT
        if lon is None:
            lon = Config.DEFAULT_LON
            
        try:
            # Try real API first
            if Config.WEATHER_API_KEY and Config.WEATHER_API_KEY != 'your_openweather_api_key_here':
                return RealTimeWeatherService._get_real_weather(lat, lon)
            else:
                return RealTimeWeatherService._get_enhanced_mock_weather()
        except Exception as e:
            print(f"Weather service error: {e}")
            return RealTimeWeatherService._get_enhanced_mock_weather()
    
    @staticmethod
    def _get_real_weather(lat, lon):
        """Get real weather from OpenWeatherMap"""
        url = f"{Config.WEATHER_BASE_URL}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': Config.WEATHER_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Get forecast data
            forecast_url = f"{Config.WEATHER_BASE_URL}/forecast"
            forecast_response = requests.get(forecast_url, params=params, timeout=10)
            forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None
            
            weather_result = {
                'temperature': round(data['main']['temp'], 1),
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'].title(),
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'pressure': data['main']['pressure'],
                'location': data.get('name', Config.DEFAULT_LOCATION),
                'icon': data['weather'][0]['icon'],
                'visibility': data.get('visibility', 10000) / 1000,
                'uv_index': random.randint(1, 8),  # OpenWeather UV requires separate API call
                'rainfall': data.get('rain', {}).get('1h', 0),
                'alerts': RealTimeWeatherService._generate_agricultural_alerts(data)
            }
            
            if forecast_data:
                weather_result['forecast'] = RealTimeWeatherService._process_forecast(forecast_data)
            
            return weather_result
        else:
            return RealTimeWeatherService._get_enhanced_mock_weather()
    
    @staticmethod
    def _get_enhanced_mock_weather():
        """Enhanced mock weather with realistic West Bengal data"""
        month = datetime.now().month
        season_data = RealTimeWeatherService._get_seasonal_data(month)
        
        base_temp = season_data['base_temp'] + random.uniform(-3, 3)
        humidity = max(40, min(95, season_data['base_humidity'] + random.randint(-15, 15)))
        
        return {
            'temperature': round(base_temp, 1),
            'humidity': int(humidity),
            'description': random.choice(season_data['descriptions']),
            'wind_speed': round(random.uniform(2, 8), 1),
            'pressure': 1013 + random.randint(-10, 10),
            'location': Config.DEFAULT_LOCATION,
            'icon': random.choice(['01d', '02d', '03d', '04d', '09d', '10d', '11d']),
            'visibility': round(random.uniform(8, 15), 1),
            'uv_index': random.randint(3, 9),
            'rainfall': round(random.uniform(0, season_data['max_rainfall']), 1),
            'alerts': RealTimeWeatherService._generate_mock_alerts(season_data),
            'forecast': RealTimeWeatherService._generate_mock_forecast()
        }
    
    @staticmethod
    def _get_seasonal_data(month):
        if month in [12, 1, 2]:  # Winter
            return {
                'base_temp': 20,
                'base_humidity': 65,
                'descriptions': ['Clear Sky', 'Sunny', 'Partly Cloudy', 'Cool'],
                'max_rainfall': 2,
                'season': 'Winter'
            }
        elif month in [3, 4, 5]:  # Summer
            return {
                'base_temp': 32,
                'base_humidity': 70,
                'descriptions': ['Hot', 'Sunny', 'Partly Cloudy', 'Warm'],
                'max_rainfall': 5,
                'season': 'Summer'
            }
        elif month in [6, 7, 8, 9]:  # Monsoon
            return {
                'base_temp': 28,
                'base_humidity': 85,
                'descriptions': ['Heavy Rain', 'Moderate Rain', 'Light Rain', 'Cloudy', 'Overcast'],
                'max_rainfall': 25,
                'season': 'Monsoon'
            }
        else:  # Post-monsoon
            return {
                'base_temp': 26,
                'base_humidity': 75,
                'descriptions': ['Pleasant', 'Partly Cloudy', 'Clear Sky', 'Mild'],
                'max_rainfall': 8,
                'season': 'Post-Monsoon'
            }
    
    @staticmethod
    def _generate_agricultural_alerts(weather_data):
        alerts = []
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        
        if temp > 35:
            alerts.append("High temperature alert - Provide shade to crops")
        if temp < 10:
            alerts.append("Low temperature alert - Protect sensitive crops")
        if humidity > 85:
            alerts.append("High humidity - Monitor for fungal diseases")
        if humidity < 40:
            alerts.append("Low humidity - Increase irrigation")
            
        return alerts
    
    @staticmethod
    def _generate_mock_alerts(season_data):
        alerts = []
        if season_data['season'] == 'Monsoon':
            alerts.append("Heavy rainfall expected - Ensure proper drainage")
            alerts.append("High humidity - Monitor crops for disease")
        elif season_data['season'] == 'Summer':
            alerts.append("High temperature - Increase irrigation frequency")
        return alerts
    
    @staticmethod
    def _generate_mock_forecast():
        """Generate 5-day forecast"""
        forecast = []
        for i in range(5):
            forecast.append({
                'date': (datetime.now().date().strftime('%Y-%m-%d')),
                'temp_max': round(random.uniform(25, 35), 1),
                'temp_min': round(random.uniform(18, 25), 1),
                'humidity': random.randint(60, 90),
                'description': random.choice(['Sunny', 'Cloudy', 'Rainy', 'Partly Cloudy'])
            })
        return forecast
    
    @staticmethod
    def _process_forecast(forecast_data):
        """Process real forecast data"""
        processed = []
        for item in forecast_data['list'][:5]:
            processed.append({
                'date': item['dt_txt'][:10],
                'temp_max': round(item['main']['temp_max'], 1),
                'temp_min': round(item['main']['temp_min'], 1),
                'humidity': item['main']['humidity'],
                'description': item['weather'][0]['description'].title()
            })
        return processed

# Enhanced Market Price Service
class RealTimeMarketService:
    @staticmethod
    def get_comprehensive_market_data():
        """Get comprehensive market data with trends"""
        try:
            # Try to get real data from government APIs
            real_data = RealTimeMarketService._get_government_market_data()
            if real_data:
                return real_data
        except Exception as e:
            print(f"Real market data error: {e}")
        
        # Fallback to enhanced mock data
        return RealTimeMarketService._get_enhanced_mock_data()
    
    @staticmethod
    def _get_government_market_data():
        """Attempt to get data from Indian government APIs"""
        # This would require actual API keys and endpoints
        # For demo, return None to use mock data
        return None
    
    @staticmethod
    def _get_enhanced_mock_data():
        """Enhanced mock market data with trends and quality grades"""
        base_prices = {
            'rice': {'base': 28, 'trend': 'stable'},
            'wheat': {'base': 32, 'trend': 'up'},
            'potato': {'base': 18, 'trend': 'down'},
            'onion': {'base': 15, 'trend': 'up'},
            'tomato': {'base': 25, 'trend': 'stable'},
            'corn': {'base': 22, 'trend': 'up'},
            'soybean': {'base': 45, 'trend': 'stable'}
        }
        
        market_data = {}
        for crop, info in base_prices.items():
            base_price = info['base']
            trend = info['trend']
            
            # Apply trend-based variation
            if trend == 'up':
                variation = random.uniform(0.02, 0.08)
            elif trend == 'down':
                variation = random.uniform(-0.08, -0.02)
            else:  # stable
                variation = random.uniform(-0.03, 0.03)
            
            current_price = round(base_price * (1 + variation), 2)
            
            market_data[crop] = {
                'price': current_price,
                'trend': trend,
                'change_percent': round(variation * 100, 1),
                'quality_grades': {
                    'A': current_price,
                    'B': round(current_price * 0.85, 2),
                    'C': round(current_price * 0.7, 2)
                },
                'market_locations': [
                    'Kolkata Market',
                    'Howrah Market',
                    'Durgapur Market'
                ]
            }
        
        return market_data

# IoT Sensor Simulator
class AdvancedIoTSimulator:
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.sensor_locations = ['Field A', 'Field B', 'Greenhouse 1']
    
    def start_simulation(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._simulate_sensors)
            self.thread.daemon = True
            self.thread.start()
            print("IoT Sensor simulation started")
    
    def stop_simulation(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _simulate_sensors(self):
        global sensor_data
        while self.running:
            try:
                # Simulate realistic sensor variations
                sensor_data['soil_ph'] += random.uniform(-0.05, 0.05)
                sensor_data['soil_ph'] = max(5.0, min(8.5, sensor_data['soil_ph']))
                
                sensor_data['soil_moisture'] += random.randint(-2, 2)
                sensor_data['soil_moisture'] = max(15, min(95, sensor_data['soil_moisture']))
                
                sensor_data['soil_temperature'] += random.uniform(-0.5, 0.5)
                sensor_data['soil_temperature'] = max(10, min(40, sensor_data['soil_temperature']))
                
                # Simulate NPK levels
                sensor_data['nitrogen'] += random.randint(-1, 1)
                sensor_data['nitrogen'] = max(20, min(80, sensor_data['nitrogen']))
                
                sensor_data['phosphorus'] += random.randint(-1, 1)
                sensor_data['phosphorus'] = max(15, min(60, sensor_data['phosphorus']))
                
                sensor_data['potassium'] += random.randint(-1, 1)
                sensor_data['potassium'] = max(25, min(70, sensor_data['potassium']))
                
                sensor_data['last_updated'] = datetime.now().isoformat()
                
                # Store in database
                conn = DatabaseUtils.get_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sensor_readings 
                    (soil_ph, soil_moisture, soil_temperature, nitrogen, phosphorus, potassium)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    sensor_data['soil_ph'],
                    sensor_data['soil_moisture'], 
                    sensor_data['soil_temperature'],
                    sensor_data['nitrogen'],
                    sensor_data['phosphorus'],
                    sensor_data['potassium']
                ))
                conn.commit()
                conn.close()
                
                # Emit real-time data via WebSocket
                socketio.emit('sensor_update', sensor_data)
                
                time.sleep(Config.IOT_UPDATE_INTERVAL)
                
            except Exception as e:
                print(f"IoT Simulation error: {e}")
                time.sleep(5)

# Initialize IoT simulator
iot_simulator = AdvancedIoTSimulator()

# ROUTES
@app.route('/')
def home():
    if 'logged_in' in session:
        return redirect(url_for('index'))
    else:
        return redirect(url_for('landing'))

@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/diagnosis')
@login_required
def diagnosis_page():
    return render_template('diagnosis.html')

@app.route('/soil')
@login_required
def soil_page():
    return render_template('soil.html')

@app.route('/features')
@login_required
def features():
    return render_template('features.html')

@app.route('/advisory')
@login_required
def advisory():
    return render_template('advisory.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

# Routes for authentication
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required.'})
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        conn = DatabaseUtils.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, email, password) VALUES (?, ?, ?)
            ''', (username, email, hashed_password))
            conn.commit()
            return jsonify({'success': True, 'message': 'Registration successful! Please log in.'})
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'message': 'Username or email already exists.'})
        finally:
            conn.close()
    
    return render_template('register.html')

# Corrected login route in app.py
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        conn = DatabaseUtils.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['logged_in'] = True
            session['user_id'] = user[0]
            session['username'] = user[1]
            return jsonify({'success': True, 'message': 'Login successful!'})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password.'})
    
    return render_template('login.html')

# Corrected logout route in app.py
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('landing'))

# Gemini API URL for standard API (not Live API)
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

@app.route('/chatbot')
def chatbot():
    return render_template('Chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    voice_support = data.get('voice_support', False)
    lang_code = data.get('lang', 'en-IN')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    lang_map = {
        'en-IN': 'English',
        'hi-IN': 'Hindi',
        'bn-IN': 'Bengali'
    }
    
    # Standard payload for Gemini API
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"You are Krishi Sahyog, a friendly and knowledgeable Indian agricultural assistant. You are an expert on farming, crops, weather, and government schemes for Indian farmers. Always respond in {lang_map.get(lang_code, 'English')}. Your tone is helpful and empathetic.\n\nUser: {user_message}"
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        # Make a direct API call instead of streaming for now
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the text from the response
        if 'candidates' in result and len(result['candidates']) > 0:
            candidate = result['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                text_response = candidate['content']['parts'][0].get('text', 'Sorry, I could not generate a response.')
            else:
                text_response = 'Sorry, I could not generate a response.'
        else:
            text_response = 'Sorry, I could not generate a response.'
        
        return jsonify({
            "text": text_response,
            "audio": ""  # No audio for now
        })
        
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return jsonify({"error": "Failed to get response from AI"}), 500
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

# API ENDPOINTS
# ... (rest of the API endpoints and helper functions) ...
def get_chatbot_response(message, language):
    """Enhanced chatbot with context-aware responses"""
    
    responses = {
        'en': {
            'weather': f"Current weather: {get_current_weather_summary()}. Perfect conditions for most crops!",
            'price': f"Latest market prices: {get_market_summary()}. Prices are generally stable.",
            'sensor': f"Current soil conditions: {get_sensor_summary()}. Your soil health looks good!",
            'fertilizer': "For optimal growth, use NPK fertilizer (10:26:26) for flowering crops, or urea for leafy vegetables. Always test soil first.",
            'disease': "Please upload a clear image of the affected plant leaves for accurate disease diagnosis. Include the whole leaf in the photo.",
            'irrigation': "Water early morning or evening. Check soil moisture at 2-3 inch depth. Most crops need 1-2 inches of water per week.",
            'pest': "Common pests in West Bengal: aphids, thrips, bollworms. Use neem oil spray or integrated pest management techniques.",
            'harvest': "Harvest timing depends on crop type. Look for visual cues: color change, firmness, size. I can provide specific guidance for your crop.",
            'storage': "Proper storage prevents 30-40% post-harvest losses. Keep produce cool, dry, and well-ventilated.",
            'organic': "Organic farming tips: Use compost, crop rotation, companion planting, beneficial insects, and organic fertilizers like vermicompost.",
            'default': "I can help with crop advice, weather updates, market prices, disease diagnosis, and farming best practices. What specific information do you need?"
        },
        'hi': {
            'weather': f"वर्तमान मौसम: {get_current_weather_summary()}। अधिकांश फसलों के लिए उपयुक्त स्थितियां!",
            'price': f"नवीनतम बाजार भाव: {get_market_summary()}। कीमतें आम तौर पर स्थिर हैं।",
            'sensor': f"वर्तमान मिट्टी की स्थिति: {get_sensor_summary()}। आपकी मिट्टी का स्वास्थ्य अच्छा है!",
            'fertilizer': "अच्छी वृद्धि के लिए, फूल वाली फसलों के लिए NPK उर्वरक (10:26:26) या पत्तेदार सब्जियों के लिए यूरिया का उपयोग करें। हमेशा पहले मिट्टी की जांच करें।",
            'disease': "सटीक रोग निदान के लिए प्रभावित पौधे की पत्तियों की स्पष्ट छवि अपलोड करें।",
            'irrigation': "सुबह जल्दी या शाम को पानी दें। 2-3 इंच की गहराई पर मिट्टी की नमी जांचें।",
            'pest': "पश्चिम बंगाल में आम कीट: एफिड्स, थ्रिप्स, बॉलवर्म। नीम तेल स्प्रे का उपयोग करें।",
            'harvest': "कटाई का समय फसल के प्रकार पर निर्भर करता है। रंग, कठोरता, आकार जैसे संकेतों को देखें।",
            'storage': "उचित भंडारण से 30-40% फसल के बाद के नुकसान को रोका जा सकता है।",
            'organic': "जैविक खेती: कंपोस्ट, फसल चक्र, साथी रोपण, और वर्मीकम्पोस्ट का उपयोग करें।",
            'default': "मैं फसल सलाह, मौसम अपडेट, बाजार भाव, रोग निदान में मदद कर सकता हूं। आपको कैसी जानकारी चाहिए?"
        },
        'bn': {
            'weather': f"বর্তমান আবহাওয়া: {get_current_weather_summary()}। বেশিরভাগ ফসলের জন্য উপযুক্ত অবস্থা!",
            'price': f"সর্বশেষ বাজার দর: {get_market_summary()}। দাম সাধারণত স্থিতিশীল।",
            'sensor': f"বর্তমান মাটির অবস্থা: {get_sensor_summary()}। আপনার মাটির স্বাস্থ্য ভালো!",
            'fertilizer': "ভালো বৃদ্ধির জন্য, ফুল ফসলের জন্য NPK সার (10:26:26) বা পাতা সবজির জন্য ইউরিয়া ব্যবহার করুন।",
            'disease': "সঠিক রোগ নির্ণয়ের জন্য আক্রান্ত গাছের পাতার স্পষ্ট ছবি আপলোড করুন।",
            'irrigation': "ভোরে বা সন্ধ্যায় পানি দিন। ২-৩ ইঞ্চি গভীরতায় মাটির আর্দ্রতা পরীক্ষা করুন।",
            'pest': "পশ্চিমবঙ্গের সাধারণ পোকা: এফিড, থ্রিপস, বলওয়ার্ম। নিম তেল স্প্রে ব্যবহার করুন।",
            'harvest': "ফসল কাটার সময় ফসলের ধরনের উপর নির্ভর করে। রং, কঠিনতা দেখুন।",
            'storage': "সঠিক সংরক্ষণ ৩০-৪০% ফসল পরবর্তী ক্ষতি প্রতিরোধ করে।",
            'organic': "জৈব চাষ: কম্পোস্ট, ফসল আবর্তন, সহচর রোপণ, কেঁচো কম্পোস্ট ব্যবহার করুন।",
            'default': "আমি ফসল পরামর্শ, আবহাওয়া আপডেট, বাজার দর, রোগ নির্ণয়ে সাহায্য করতে পারি। কী তথ্য দরকার?"
        }
    }
    
    # Determine response category
    response_key = 'default'
    keywords = {
        'weather': ['weather', 'मौसम', 'আবহাওয়া', 'rain', 'बारिश', 'বৃষ্টি'],
        'price': ['price', 'market', 'भाव', 'বাজার', 'cost', 'कीमत', 'দাম'],
        'sensor': ['soil', 'ph', 'moisture', 'मिट्टी', 'মাটি'],
        'fertilizer': ['fertilizer', 'उर्वरक', 'সার', 'npk', 'urea'],
        'disease': ['disease', 'sick', 'रोग', 'রোগ', 'problem'],
        'irrigation': ['water', 'irrigation', 'पानी', 'পানি'],
        'pest': ['pest', 'insect', 'कीट', 'পোকা'],
        'harvest': ['harvest', 'कटाई', 'ফসল কাটা'],
        'storage': ['storage', 'store', 'ভন্ডারন', 'সংরক্ষণ'],
        'organic': ['organic', 'जैविक', 'জৈব']
    }
    
    for key, words in keywords.items():
        if any(word in message for word in words):
            response_key = key
            break
    
    return responses.get(language, responses['en']).get(response_key, responses['en']['default'])

def get_current_weather_summary():
    """Get brief weather summary"""
    try:
        weather = RealTimeWeatherService.get_comprehensive_weather()
        return f"{weather['temperature']}°C, {weather['description']}"
    except:
        return "25°C, Pleasant"

def get_market_summary():
    """Get brief market summary"""
    try:
        market = RealTimeMarketService.get_comprehensive_market_data()
        rice_price = market.get('rice', {}).get('price', 25)
        return f"Rice ₹{rice_price}/kg"
    except:
        return "Rice ₹25/kg"

def get_sensor_summary():
    """Get brief sensor summary"""
    return f"pH {sensor_data['soil_ph']:.1f}, Moisture {sensor_data['soil_moisture']}%"

# WEBSOCKET EVENTS
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('sensor_update', sensor_data)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_data')
def handle_data_request(data):
    data_type = data.get('type', 'sensor')
    if data_type == 'sensor':
        emit('sensor_update', sensor_data)
    elif data_type == 'weather':
        weather = RealTimeWeatherService.get_comprehensive_weather()
        emit('weather_update', weather)
    elif data_type == 'market':
        market = RealTimeMarketService.get_comprehensive_market_data()
        emit('market_update', market)

# API ENDPOINTS
@app.route('/api/upload-image', methods=['POST'])
def upload_image_for_diagnosis():
    # Get the image file and plant type from the form data
    image_file = request.files.get('image')
    plant_type = request.form.get('plant_type')

    # Check if an image was provided
    if not image_file or image_file.filename == '':
        return jsonify({'success': False, 'error': 'No image file provided.'}), 400

    # Save the file temporarily
    filename = secure_filename(image_file.filename)
    upload_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    image_file.save(upload_path)

    # Call the ML model for prediction
    # Ensure 'plant_detector' is initialized as a global object in app.py
    result = plant_detector.predict_disease(upload_path, plant_type)

    # Clean up the temporary file after prediction
    os.remove(upload_path)

    if result:
        return jsonify({'success': True, 'result': result})
    else:
        return jsonify({'success': False, 'error': 'Analysis failed. Please try again.'}), 500

# ERROR HANDLERS
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# ADDITIONAL UTILS for CropDataAnalyzer
def add_detailed_crop_advice(soil_ph, moisture, temperature):
    """Add this method to CropDataAnalyzer class"""
    advice = {
        'immediate_actions': [],
        'crop_suitability': [],
        'fertilizer_recommendations': [],
        'irrigation_advice': []
    }
    
    # pH-based advice
    if soil_ph < 6.0:
        advice['immediate_actions'].append('Add lime to raise pH')
        advice['fertilizer_recommendations'].append('Use alkaline fertilizers')
    elif soil_ph > 7.5:
        advice['immediate_actions'].append('Add organic matter to lower pH')
        advice['fertilizer_recommendations'].append('Use acidic fertilizers')
    
    # Moisture-based advice
    if moisture < 40:
        advice['irrigation_advice'].append('Increase watering frequency')
        advice['crop_suitability'].append('Consider drought-tolerant crops')
    elif moisture > 80:
        advice['irrigation_advice'].append('Reduce watering, improve drainage')
        advice['crop_suitability'].append('Good for rice cultivation')
    
    # Temperature-based advice
    if temperature < 20:
        advice['crop_suitability'].extend(['Wheat', 'Barley', 'Peas'])
    elif temperature > 30:
        advice['crop_suitability'].extend(['Rice', 'Cotton', 'Sugarcane'])
    else:
        advice['crop_suitability'].extend(['Tomato', 'Potato', 'Corn'])
    
    return advice

# Add to CropDataAnalyzer class
CropDataAnalyzer.get_detailed_crop_advice = staticmethod(add_detailed_crop_advice)

# ... (rest of your app.py code) ...

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # ✅ FIX: Removed the force_database_recreation() call to ensure data persistence.
    
    # Initialize database with a persistent schema.
    # This will create tables only if they don't exist.
    DatabaseUtils.create_tables()
    print("✓ Database initialized. Existing data preserved.")
    
    # Try to load ML models
    if plant_detector.load_models():
        print("✓ Plant disease detection models loaded successfully")
    else:
        print("⚠ Models not found - using mock predictions for demo")
    
    # Start IoT simulation
    iot_simulator.start_simulation()
    
    print("🌱 Krishi Shayog Agricultural Advisory System Starting...")
    print("📊 Real-time sensor simulation: ACTIVE")
    print("🤖 AI Plant Disease Detection: READY")
    print("🌤️ Weather Integration: ACTIVE")
    print("💰 Market Price Tracking: ACTIVE")
    print("🗣️ Multi-language Support: English, Hindi, Bengali")
    
    # Run the application
    socketio.run(
        app, 
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        allow_unsafe_werkzeug=True
    )