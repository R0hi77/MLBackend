#!/usr/bin/env python3
"""
Realistic Home Power Data Simulator
Generates and publishes realistic household power consumption data to MQTT broker
"""
import json
import time
import random
from datetime import datetime, timezone
import paho.mqtt.client as mqtt
import numpy as np

# ===== CONFIGURATION =====
MQTT_BROKER = "broker.peterzzburg.online"
MQTT_PORT = 8884
MQTT_TOPIC = "sensor-data/00124"  # Matches backend subscription: sensor-data/#
DEVICE_ID = "00124"

# Realistic power consumption patterns (in Watts)
BASE_LOAD = 150  # Always-on devices (router, fridge idle, etc.)
FRIDGE_CYCLE = {"on": 180, "off": 50, "cycle_duration": 45}  # 45 min cycles
PATTERNS = {
    "night": {"min": 150, "max": 400, "noise": 30},      # 00:00-06:00
    "morning": {"min": 800, "max": 2500, "noise": 200},  # 06:00-09:00 (kettle, toaster, microwave)
    "day": {"min": 200, "max": 800, "noise": 100},       # 09:00-17:00
    "evening": {"min": 600, "max": 2200, "noise": 150},  # 17:00-22:00 (cooking, TV, washing)
    "late": {"min": 300, "max": 600, "noise": 50}        # 22:00-00:00
}

# ===== MQTT CALLBACKS =====
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"✅ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"❌ Connection failed with code {rc}")

def on_publish(client, userdata, mid, reason_code=None, properties=None):
    """Callback when message is published"""
    pass  # Silent publishing

# ===== POWER GENERATION =====
class PowerSimulator:
    def __init__(self):
        self.fridge_state = "off"
        self.fridge_timer = 0
        self.last_spike_time = 0
        self.spike_duration = 0
        
    def get_time_pattern(self):
        """Determine current time pattern"""
        hour = datetime.now().hour
        if 0 <= hour < 6:
            return "night"
        elif 6 <= hour < 9:
            return "morning"
        elif 9 <= hour < 17:
            return "day"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "late"
    
    def simulate_fridge(self):
        """Simulate fridge compressor cycling"""
        self.fridge_timer += 1
        if self.fridge_timer >= FRIDGE_CYCLE["cycle_duration"]:
            self.fridge_state = "on" if self.fridge_state == "off" else "off"
            self.fridge_timer = 0
        
        return FRIDGE_CYCLE[self.fridge_state]
    
    def random_appliance_spike(self):
        """Simulate random appliance usage (kettle, microwave, etc.)"""
        current_time = time.time()
        
        # Check if we're in a spike
        if self.spike_duration > 0:
            self.spike_duration -= 1
            return random.uniform(1200, 2500)  # High power appliance
        
        # Random chance of new spike (5% per reading)
        if random.random() < 0.05 and (current_time - self.last_spike_time) > 60:
            self.last_spike_time = current_time
            self.spike_duration = random.randint(3, 10)  # 3-10 readings
            return random.uniform(1200, 2500)
        
        return 0
    
    def generate_power(self):
        """Generate realistic power reading"""
        pattern = PATTERNS[self.get_time_pattern()]
        
        # Base consumption for time of day
        base_power = random.uniform(pattern["min"], pattern["max"])
        
        # Add fridge cycling
        fridge_power = self.simulate_fridge()
        
        # Add random appliance spikes
        spike_power = self.random_appliance_spike()
        
        # Add realistic noise
        noise = random.gauss(0, pattern["noise"] / 3)
        
        # Total power (ensure non-negative)
        total_power = max(0, base_power + fridge_power + spike_power + noise)
        
        return round(total_power, 2)

# ===== MAIN SIMULATION =====
def main():
    print("="*60)
    print("🏠 Home Power Consumption Simulator")
    print("="*60)
    print(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Device ID: {DEVICE_ID}")
    print(f"Topic: {MQTT_TOPIC}")
    print("="*60)
    print("\nStarting simulation... (Press Ctrl+C to stop)\n")
    
    # Initialize MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_publish = on_publish
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        
        # Wait for connection
        time.sleep(2)
        
        # Initialize simulator
        simulator = PowerSimulator()
        reading_count = 0
        
        while True:
            # Generate power reading
            power = simulator.generate_power()
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Create message
            message = {
                "power": power,
                "device_id": DEVICE_ID,
                "timestamp": timestamp
            }
            
            # Publish to MQTT
            result = client.publish(MQTT_TOPIC, json.dumps(message))
            
            reading_count += 1
            
            # Display status
            pattern = simulator.get_time_pattern()
            print(f"[{reading_count:04d}] {timestamp[:19]} | "
                  f"Power: {power:7.2f}W | Pattern: {pattern:8s} | "
                  f"Fridge: {simulator.fridge_state:3s}")
            
            # Wait before next reading (1 second intervals)
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation stopped by user")
        print(f"Total readings sent: {reading_count}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        print("✅ Disconnected from MQTT broker")

if __name__ == "__main__":
    main()
