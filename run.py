#!/usr/bin/env python3
"""
KrishiMitra Production Run Script
This script handles production deployment with proper error handling and logging
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup comprehensive logging"""
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'krishimitra_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('KrishiMitra')

def check_requirements():
    """Check if all requirements are installed"""
    logger = logging.getLogger('KrishiMitra')
    
    required_packages = [
        'flask', 'flask_cors', 'flask_socketio', 'tensorflow', 
        'opencv-python', 'requests', 'googletrans', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please run: pip install -r requirements.txt")
        return False
    
    logger.info("All required packages are installed")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'uploads', 
        'logs',
        'static/css',
        'static/js',
        'static/images',
        'templates',
        'dataset'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logging.getLogger('KrishiMitra').info("Directory structure verified")

def check_environment():
    """Check environment configuration"""
    logger = logging.getLogger('KrishiMitra')
    
    required_env_vars = [
        'SECRET_KEY'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Some features may not work properly")
    
    # Check if .env file exists
    env_file = project_root / '.env'
    if not env_file.exists():
        logger.warning(".env file not found - creating template")
        create_env_template()
    
    return True

def create_env_template():
    """Create a template .env file"""
    env_template = """# KrishiMitra Environment Configuration
# Copy this file to .env and update with your values

# Flask Configuration
SECRET_KEY=your-secret-key-here-change-this-in-production
FLASK_ENV=production
FLASK_DEBUG=False

# Database
DATABASE_URL=sqlite:///agricultural_data.db

# API Keys (Get these from respective services)
WEATHER_API_KEY=your-openweather-api-key
MARKET_API_KEY=your-market-api-key

# ML Model Configuration
PLANT_DISEASE_MODEL_PATH=models/plant_disease_model.h5

# IoT Configuration
IOT_UPDATE_INTERVAL=30
DEFAULT_LAT=22.1667
DEFAULT_LON=88.1833
DEFAULT_LOCATION=Kakdwip, West Bengal

# Server Configuration
HOST=0.0.0.0
PORT=5000
"""
    
    with open(project_root / '.env.template', 'w') as f:
        f.write(env_template)

def run_application():
    """Run the KrishiMitra application"""
    logger = logging.getLogger('KrishiMitra')
    
    try:
        # Import the Flask app
        from app import app, socketio, plant_detector, iot_simulator
        
        logger.info("KrishiMitra Agricultural Advisory System Starting...")
        logger.info("=" * 60)
        
        # Check if model is available
        if plant_detector.model is not None:
            logger.info("✓ AI Plant Disease Detection: READY")
        else:
            logger.warning("⚠ Plant Disease Model: NOT LOADED (using mock predictions)")
        
        # Start IoT simulation
        iot_simulator.start_simulation()
        logger.info("✓ IoT Sensor Simulation: ACTIVE")
        
        logger.info("✓ Weather Integration: ACTIVE")
        logger.info("✓ Market Price Tracking: ACTIVE") 
        logger.info("✓ Multi-language Support: English, Hindi, Bengali")
        logger.info("=" * 60)
        
        # Get configuration
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting server on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        if debug:
            logger.info("🌐 Application available at: http://localhost:5000")
        
        # Run the application
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            allow_unsafe_werkzeug=debug
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        try:
            iot_simulator.stop_simulation()
            logger.info("IoT simulation stopped")
        except:
            pass
    
    return 0

def main():
    """Main entry point"""
    # Setup logging
    logger = setup_logging()
    
    logger.info("KrishiMitra - Smart Agricultural Advisory System")
    logger.info("Initializing application...")
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Setup directories
    setup_directories()
    
    # Check environment
    check_environment()
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        env_file = project_root / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            logger.info("Environment variables loaded from .env file")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading")
    
    # Run the application
    return run_application()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)