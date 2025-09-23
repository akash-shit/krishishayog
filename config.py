import os
from datetime import timedelta
from pathlib import Path

class Config:
    # Secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'krishimitra-secret-key-2024'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///agricultural_data.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # API Keys (Get these from respective services)
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY') or 'your_openweather_api_key_here'
    MARKET_API_KEY = os.environ.get('MARKET_API_KEY') or 'your_market_api_key_here'
    
    # Supported Crop Types for Disease Detection
    # Updated to include all four models you have: wheat, tomato, potato, and rice
    CROP_TYPES = ['wheat', 'tomato', 'potato', 'rice']
    
    # IoT Settings
    IOT_UPDATE_INTERVAL = 30  # seconds
    
    # Language settings
    LANGUAGES = {
        'en': 'English',
        'hi': 'Hindi',
        'bn': 'Bengali'
    }
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Real-time data sources
    WEATHER_BASE_URL = 'http://api.openweathermap.org/data/2.5'
    # Note: Accessing real-time market data from data.gov.in requires a specific dataset resource ID.
    # The current implementation uses mock data as a fallback.
    MARKET_BASE_URL = 'https://api.data.gov.in/resource' 
    
    # Default location (West Bengal)
    DEFAULT_LAT = 22.1667
    DEFAULT_LON = 88.1833
    DEFAULT_LOCATION = 'Kolkata, West Bengal'
    
    # Model paths
    MODEL_DIR = os.path.join(os.getcwd(), 'models')
    WHEAT_MODEL_PATH = os.path.join(MODEL_DIR, 'wheat_model.keras')
    TOMATO_MODEL_PATH = os.path.join(MODEL_DIR, 'tomato_model.keras')
    POTATO_MODEL_PATH = os.path.join(MODEL_DIR, 'potato_model.keras')
    RICE_MODEL_PATH = os.path.join(MODEL_DIR, 'rice_model.keras')
    
    # Plant disease model path (for backward compatibility)
    PLANT_DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, 'tomato_model.keras')  # Default to tomato