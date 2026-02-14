# Power Data Simulator

## Quick Start

```bash
# From MLBackend directory
source venv/bin/activate
python3 tests/simulate_power_data.py
```

## What It Does

Generates realistic household power consumption data and publishes to MQTT:
- **Base load**: 150W (always-on devices)
- **Fridge cycling**: 50W (off) / 180W (on) every 45 readings
- **Time-based patterns**: Different consumption for night/morning/day/evening
- **Random spikes**: Simulates kettle, microwave, toaster usage
- **Realistic noise**: Adds natural variation

## Configuration

Edit `tests/simulate_power_data.py`:
```python
MQTT_BROKER = "broker.peterzzburg.online"
MQTT_PORT = 8884
MQTT_TOPIC = "sensor-data/00124"
DEVICE_ID = "00124"
```

## Output Format

```json
{
    "power": 1500.48,
    "device_id": "00124",
    "timestamp": "2025-12-08T19:52:12.659480Z"
}
```

## Power Patterns by Time

| Time | Pattern | Range | Description |
|------|---------|-------|-------------|
| 00:00-06:00 | Night | 150-400W | Minimal usage |
| 06:00-09:00 | Morning | 800-2500W | Breakfast appliances |
| 09:00-17:00 | Day | 200-800W | Moderate usage |
| 17:00-22:00 | Evening | 600-2200W | Cooking, TV, washing |
| 22:00-00:00 | Late | 300-600W | Winding down |

## Stop Simulation

Press `Ctrl+C` to stop gracefully.
