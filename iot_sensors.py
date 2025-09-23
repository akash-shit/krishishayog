"""
KrishiMitra IoT Sensor Device Code
This code runs on IoT devices (Raspberry Pi/Arduino) to collect soil data
"""

import time
import json
import requests
import random
from datetime import datetime
import logging

# For Raspberry Pi GPIO
try:
    import RPi.GPIO as GPIO
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    HAS_GPIO = True
except ImportError:
    print("GPIO libraries not available - running in simulation mode")
    HAS_GPIO = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sensor_logs.log'),
        logging.StreamHandler()
    ]
)

class SensorConfig:
    """Configuration for sensors and server connection"""
    # Server settings
    SERVER_URL = "http://localhost:5000"  # Your KrishiMitra server URL
    API_ENDPOINT = "/api/sensor-data-upload"
    
    # Device settings
    DEVICE_ID = "KRISHIMITRA_IOT_001"
    LOCATION = "West Bengal Farm A"
    
    # Sensor pins (for Raspberry Pi)
    PH_SENSOR_PIN = 0      # ADC channel for pH sensor
    MOISTURE_SENSOR_PIN = 1 # ADC channel for soil moisture
    TEMP_SENSOR_PIN = 2     # ADC channel for temperature
    
    # NPK sensor I2C addresses (if using NPK sensor module)
    NPK_SENSOR_ADDRESS = 0x4D
    
    # Data collection interval (seconds)
    COLLECTION_INTERVAL = 30
    
    # Calibration values (adjust based on your sensors)
    PH_CALIBRATION = {
        'slope': -0.1,
        'intercept': 7.0
    }
    
    MOISTURE_CALIBRATION = {
        'dry_value': 1023,    # ADC reading when completely dry
        'wet_value': 300      # ADC reading when completely wet
    }
    
    TEMP_CALIBRATION = {
        'slope': 0.48828125,  # For LM35 temperature sensor
        'offset': 0
    }

class KrishiMitraIoTSensor:
    """Main IoT sensor class for collecting and transmitting agricultural data"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_hardware()
        
    def setup_hardware(self):
        """Initialize hardware components"""
        if HAS_GPIO:
            try:
                # Initialize I2C bus
                self.i2c = busio.I2C(board.SCL, board.SDA)
                
                # Initialize ADC for analog sensors
                self.ads = ADS.ADS1115(self.i2c)
                
                # Setup analog input channels
                self.ph_channel = AnalogIn(self.ads, ADS.P0)
                self.moisture_channel = AnalogIn(self.ads, ADS.P1)
                self.temp_channel = AnalogIn(self.ads, ADS.P2)
                
                self.logger.info("Hardware initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Hardware initialization failed: {e}")
                self.logger.info("Switching to simulation mode")
                self.setup_simulation()
        else:
            self.setup_simulation()
    
    def setup_simulation(self):
        """Setup simulation mode for testing without hardware"""
        self.simulation_mode = True
        self.logger.info("Running in simulation mode")
    
    def read_ph_sensor(self):
        """Read pH sensor value"""
        try:
            if HAS_GPIO and hasattr(self, 'ph_channel'):
                # Read raw ADC value
                raw_value = self.ph_channel.voltage
                
                # Convert to pH using calibration
                ph_value = (raw_value * self.config.PH_CALIBRATION['slope']) + self.config.PH_CALIBRATION['intercept']
                
                # Clamp to valid pH range
                ph_value = max(0, min(14, ph_value))
                
                return round(ph_value, 2)
            else:
                # Simulation mode - generate realistic pH values
                return round(random.uniform(5.5, 8.0), 2)
                
        except Exception as e:
            self.logger.error(f"pH sensor read error: {e}")
            return None
    
    def read_moisture_sensor(self):
        """Read soil moisture sensor value"""
        try:
            if HAS_GPIO and hasattr(self, 'moisture_channel'):
                # Read raw ADC value
                raw_value = self.moisture_channel.value
                
                # Convert to percentage using calibration
                dry_val = self.config.MOISTURE_CALIBRATION['dry_value']
                wet_val = self.config.MOISTURE_CALIBRATION['wet_value']
                
                # Calculate moisture percentage (inverted because lower resistance = more moisture)
                moisture_percent = 100 - ((raw_value - wet_val) / (dry_val - wet_val) * 100)
                
                # Clamp to valid range
                moisture_percent = max(0, min(100, moisture_percent))
                
                return round(moisture_percent, 1)
            else:
                # Simulation mode - generate realistic moisture values
                return round(random.uniform(30, 85), 1)
                
        except Exception as e:
            self.logger.error(f"Moisture sensor read error: {e}")
            return None
    
    def read_temperature_sensor(self):
        """Read soil temperature sensor value"""
        try:
            if HAS_GPIO and hasattr(self, 'temp_channel'):
                # Read voltage from temperature sensor (LM35)
                voltage = self.temp_channel.voltage
                
                # Convert voltage to temperature (LM35: 10mV per degree Celsius)
                temperature = (voltage * 100) + self.config.TEMP_CALIBRATION['offset']
                
                return round(temperature, 1)
            else:
                # Simulation mode - generate realistic temperature values
                return round(random.uniform(15, 35), 1)
                
        except Exception as e:
            self.logger.error(f"Temperature sensor read error: {e}")
            return None
    
    def read_npk_sensor(self):
        """Read NPK sensor values (if available)"""
        try:
            if HAS_GPIO:
                # This would be implemented based on your specific NPK sensor
                # For now, return simulated values
                pass
            
            # Simulation mode - generate realistic NPK values
            return {
                'nitrogen': random.randint(20, 80),
                'phosphorus': random.randint(15, 60),
                'potassium': random.randint(25, 70)
            }
            
        except Exception as e:
            self.logger.error(f"NPK sensor read error: {e}")
            return None
    
    def collect_sensor_data(self):
        """Collect data from all sensors"""
        timestamp = datetime.now().isoformat()
        
        sensor_data = {
            'device_id': self.config.DEVICE_ID,
            'location': self.config.LOCATION,
            'timestamp': timestamp,
            'soil_ph': self.read_ph_sensor(),
            'soil_moisture': self.read_moisture_sensor(),
            'soil_temperature': self.read_temperature_sensor(),
        }
        
        # Add NPK data if available
        npk_data = self.read_npk_sensor()
        if npk_data:
            sensor_data.update(npk_data)
        
        # Filter out None values
        sensor_data = {k: v for k, v in sensor_data.items() if v is not None}
        
        self.logger.info(f"Collected sensor data: {sensor_data}")
        return sensor_data
    
    def send_data_to_server(self, data):
        """Send sensor data to KrishiMitra server"""
        try:
            url = f"{self.config.SERVER_URL}{self.config.API_ENDPOINT}"
            
            headers = {
                'Content-Type': 'application/json',
                'X-Device-ID': self.config.DEVICE_ID
            }
            
            response = requests.post(
                url, 
                json=data, 
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("Data sent successfully to server")
                return True
            else:
                self.logger.error(f"Server responded with status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send data to server: {e}")
            return False
    
    def save_data_locally(self, data):
        """Save data locally as backup"""
        try:
            filename = f"sensor_data_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Read existing data
            try:
                with open(filename, 'r') as f:
                    local_data = json.load(f)
            except FileNotFoundError:
                local_data = []
            
            # Append new data
            local_data.append(data)
            
            # Save back to file
            with open(filename, 'w') as f:
                json.dump(local_data, f, indent=2)
            
            self.logger.info("Data saved locally")
            
        except Exception as e:
            self.logger.error(f"Failed to save data locally: {e}")
    
    def run_data_collection_loop(self):
        """Main data collection loop"""
        self.logger.info(f"Starting data collection loop (interval: {self.config.COLLECTION_INTERVAL}s)")
        
        while True:
            try:
                # Collect sensor data
                sensor_data = self.collect_sensor_data()
                
                # Save locally first (as backup)
                self.save_data_locally(sensor_data)
                
                # Try to send to server
                success = self.send_data_to_server(sensor_data)
                
                if not success:
                    self.logger.warning("Failed to send to server - data saved locally")
                
                # Wait for next collection
                time.sleep(self.config.COLLECTION_INTERVAL)
                
            except KeyboardInterrupt:
                self.logger.info("Data collection stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in collection loop: {e}")
                time.sleep(5)  # Wait a bit before retrying

# Arduino Code Template (C++)
ARDUINO_CODE_TEMPLATE = '''
/*
KrishiMitra IoT Sensor - Arduino Code
This code reads sensors and sends data to Raspberry Pi or directly to server
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Server settings
const char* serverURL = "http://your-krishimitra-server.com/api/sensor-data-upload";

// Sensor pins
const int PH_SENSOR_PIN = A0;
const int MOISTURE_SENSOR_PIN = A1;
const int TEMP_SENSOR_PIN = A2;

// Calibration constants
const float PH_SLOPE = -0.1;
const float PH_INTERCEPT = 7.0;
const int MOISTURE_DRY = 1023;
const int MOISTURE_WET = 300;

void setup() {
  Serial.begin(115200);
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
}

void loop() {
  // Read sensors
  float ph = readPHSensor();
  float moisture = readMoistureSensor();
  float temperature = readTemperatureSensor();
  
  // Create JSON payload
  DynamicJsonDocument doc(1024);
  doc["device_id"] = "KRISHIMITRA_ARDUINO_001";
  doc["location"] = "West Bengal Farm A";
  doc["timestamp"] = millis();
  doc["soil_ph"] = ph;
  doc["soil_moisture"] = moisture;
  doc["soil_temperature"] = temperature;
  
  // Send data
  sendDataToServer(doc);
  
  // Wait 30 seconds
  delay(30000);
}

float readPHSensor() {
  int sensorValue = analogRead(PH_SENSOR_PIN);
  float voltage = sensorValue * (5.0 / 1023.0);
  float ph = (voltage * PH_SLOPE) + PH_INTERCEPT;
  return constrain(ph, 0, 14);
}

float readMoistureSensor() {
  int sensorValue = analogRead(MOISTURE_SENSOR_PIN);
  float moisture = 100 - ((float(sensorValue - MOISTURE_WET) / (MOISTURE_DRY - MOISTURE_WET)) * 100);
  return constrain(moisture, 0, 100);
}

float readTemperatureSensor() {
  int sensorValue = analogRead(TEMP_SENSOR_PIN);
  float voltage = sensorValue * (5.0 / 1023.0);
  float temperature = voltage * 100; // LM35 conversion
  return temperature;
}

void sendDataToServer(DynamicJsonDocument& data) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverURL);
    http.addHeader("Content-Type", "application/json");
    
    String jsonString;
    serializeJson(data, jsonString);
    
    int httpResponseCode = http.POST(jsonString);
    
    if (httpResponseCode > 0) {
      Serial.println("Data sent successfully");
    } else {
      Serial.println("Error sending data");
    }
    
    http.end();
  }
}
'''

def main():
    """Main function to run the IoT sensor"""
    print("KrishiMitra IoT Sensor Starting...")
    
    # Initialize configuration
    config = SensorConfig()
    
    # Create sensor instance
    sensor = KrishiMitraIoTSensor(config)
    
    # Start data collection
    sensor.run_data_collection_loop()

if __name__ == "__main__":
    main()

"""
Hardware Setup Instructions:

1. Raspberry Pi Setup:
   - Install required packages:
     pip install adafruit-circuitpython-ads1x15 requests
   
   - Connect sensors:
     - pH sensor to ADC channel 0
     - Soil moisture sensor to ADC channel 1  
     - Temperature sensor to ADC channel 2
     - Power and ground connections

2. Arduino Setup:
   - Install required libraries in Arduino IDE:
     - WiFi library
     - HTTPClient library
     - ArduinoJson library
   
   - Use the Arduino code template above
   - Connect sensors to analog pins A0, A1, A2

3. Sensor Wiring:
   - pH Sensor: VCC to 3.3V/5V, GND to Ground, Signal to ADC
   - Moisture Sensor: VCC to 3.3V/5V, GND to Ground, Signal to ADC
   - Temperature (LM35): VCC to 3.3V/5V, GND to Ground, OUT to ADC
   
4. Network Configuration:
   - Ensure device can connect to internet
   - Update SERVER_URL in configuration
   - Configure WiFi credentials for Arduino

5. Calibration:
   - Calibrate sensors using known reference values
   - Update calibration constants in configuration
"""