# MLbackend

ML-backed microservice for non-intrusive load monitoring (NILM).

It accepts aggregate power readings (HTTP + MQTT), buffers them per-device, runs a TensorFlow Keras model to disaggregate appliance-level power, and publishes or returns predictions.

---

## Table of contents

- Project overview
- Repository layout
- Quick start
- Configuration
- HTTP API
- MQTT contract
- Core internals
- Logging & broker connection visibility
- Troubleshooting
- Development & testing
- Security & deployment notes

---

## Project overview

This service consumes aggregate power sensor readings and runs a trained NILM model to estimate per-appliance power and on/off state. It supports ingestion via MQTT and HTTP, keeps a per-device circular buffer, and runs inference when sufficient historical context is available.

---

## Repository layout

Important files and folders:

- `run.py` — Flask development entrypoint
- `gunicorn.py` — Gunicorn configuration used by the Docker image
- `Dockerfile` — Container build
- `requirements.txt` — Python dependencies
- `models/` — saved Keras models and scaler pickles
- `app/` — Flask application package
  - `__init__.py` — app factory (`create_app`), app config and MQTT init
  - `configurations/config.py` — configuration (reads env vars)
  - `handlers/mqtt_handler.py` — MQTT callbacks, subscriptions, parsing, publish flow
  - `routes/` — HTTP routes (prediction, status, health)
  - `utils/model_manager.py` — `NILMModelManager`: model/scaler loading, device buffers, input preparation, prediction and post-processing
  - `utils/httpStatusCodes.py` — HTTP status constants

---

## Quick start (local)

1. Create and activate a virtualenv (Python 3.12 recommended):

   python3.12 -m venv venv
   source venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Provide required environment variables (example `.env` or export):

- `MODEL_PATH` — path to Keras model (e.g. `./models/multi_appliance_model_continued_4.keras`)
- `SCALER_PATH` — path to scaler pickle (e.g. `./models/original_power_scaler_5.pkl`)
- `MQTT_BROKER_URL` — broker host
- `MQTT_BROKER_PORT` — broker port (e.g. 1883 or 8883/8884 for TLS)
- `MQTT_USERNAME` / `MQTT_PASSWORD` — optional
- (optional) `MQTT_CONNECT_TIMEOUT` — seconds to wait at startup if synchronous connect-check implemented

4. Run development server:

   python run.py

5. Production (example):

   gunicorn -c gunicorn.py run:app

or build the Docker image using the provided `Dockerfile`.

---

## Configuration

Primary configuration is in `app/configurations/config.py` and is populated from environment variables. Key variables are described above. Keep secrets (MQTT credentials) out of source control and use environment or secret stores in production.

---

## HTTP API

- GET `/api/v1/health`
  - Basic service health check.

- POST `/api/v1/predict`
  - Accepts JSON payloads in one of two forms:
    - Single reading: `{ "device_id": "dev1", "aggregate_power": 123.4 }` (also supports `power`/`aggregate` keys)
    - Batch: `{ "device_id": "dev1", "batch_data": [123.4, 122.3, ...] }`
  - Behavior: data is appended to the device buffer. If the buffer is not yet large enough for the model, the endpoint returns buffer status. If sufficient, the endpoint runs prediction and returns a JSON object containing per-appliance `power`, `state` and `confidence`.

- GET `/api/v1/status/<device_id>`
  - Returns buffer length, required length and list of appliances.

---

## MQTT contract

- Subscribed topic pattern: `sensor-data/#` — expected topic form `sensor-data/{device_id}`.
- Incoming payload: typically JSON such as `{ "aggregate_power": 123.4 }` but handlers accept simple numeric payloads in many cases — check `handlers/mqtt_handler.py` for exact parsing.
- Published predictions topic: `predictions/{device_id}` — JSON payload with per-appliance entries:

  {
    "appliance_name": { "power": <watts>, "state": 0|1, "confidence": 0.0-1.0 },
    ...
  }

---

## Core internals

- `NILMModelManager` (in `app/utils/model_manager.py`) is the central component.
  - Loads a TensorFlow Keras model and a scikit-learn scaler (joblib pickle).
  - Holds per-device `deque` buffers with size `window_size + t1_window`.
  - `add_sensor_data(device_id, value)` appends values thread-safely.
  - `can_predict(device_id)` returns true when the buffer length meets the model's required context.
  - `prepare_sequence(device_id)` builds a DataFrame placeholder matching scaler column layout, scales inputs, and slices two input windows used by the model:
    - `main_window_input` (length `window_size`)
    - `t1_window_input` (length `t1_window`)
  - `predict_appliances(device_id)` runs `model.predict(...)`, inverse-scales regression outputs to watts using the scaler, thresholds state outputs at 0.5, and returns a dict of results.

Important design notes:
- The scaler must match the column layout used during training (aggregate + N appliances). The code constructs a placeholder DataFrame to satisfy that layout before applying `scaler.transform` or `inverse_transform`.
- Model.predict may not be thread-safe when used concurrently; consider single-threaded inference, a dedicated inference process, or a synchronization strategy if serving multiple simultaneous requests.

---

## Logging & broker connection visibility

- `mqtt.init_app(app)` configures the MQTT client using app config but connects asynchronously.
- Connection success/failure is logged from `on_connect` and `on_disconnect` callbacks in `app/handlers/mqtt_handler.py`.
- If you require startup-time confirmation of broker connectivity, the codebase supports a `threading.Event` pattern where `on_connect` sets the event and `create_app()` waits with a short timeout (3–5s) to avoid blocking worker startup.

---

## Troubleshooting

- No "connected" startup log: expected because connection is asynchronous; check `on_connect` logs or implement the `connect_event` wait if you need confirmation at startup.
- Hangs during `model.predict`: verify model input shapes vs prepared inputs, add logging around shapes and inspect saved model using `tf.keras.models.load_model(...); model.summary()` in an isolated script. Consider serialization/locking if predicts are called concurrently.
- Scaler unpickle warnings (InconsistentVersionWarning): caused by mismatched scikit-learn versions between the environment used to create the pickle and the runtime. Keep versions aligned or re-create scaler.
- TF GPU messages: informational when GPU drivers are missing — TF will fall back to CPU.

---

## Development & testing

- Unit-test `NILMModelManager.prepare_sequence`, `add_sensor_data` and the HTTP handler logic. Use a mocked model to avoid heavy TF dependency when running fast tests.
- Integration tests for MQTT handlers can use a local test broker (e.g., mosquitto) or mock the `mqtt` client.

---

## Security & deployment notes

- Do not store credentials in source control. Use env vars, Docker secrets, or a secret manager.
- When running behind Gunicorn with multiple workers, be deliberate about model loading (`preload_app=True` is enabled in `gunicorn.py`) and inference concurrency semantics.

---


