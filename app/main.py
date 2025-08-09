# app/main.py

# =================================================================
# ML Model Serving API
# =================================================================
# This Flask application serves a trained scikit-learn model.
# It is designed as a containerized microservice that dynamically loads
# a model from a remote MLflow server on startup. It includes
# production-ready features like input validation, performance
# monitoring, and persistent logging.
# =================================================================

# --- Core Python Libraries ---
import os
import logging
from logging.handlers import RotatingFileHandler
import time
import subprocess

# --- Data Science and ML Libraries ---
import joblib
import pandas as pd
import mlflow

# --- Web Framework and Extensions ---
from flask import Flask, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter
from flask_pydantic import validate
from pydantic import BaseModel


# --- 1. APPLICATION INITIALIZATION ---
app = Flask(__name__)
metrics = PrometheusMetrics(app)  # Exposes /metrics endpoint automatically


# --- 2. LOGGING CONFIGURATION ---
# Configure a persistent, rotating file logger for the application.
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Rotate logs when they reach 1MB, keeping the last 5 files.
file_handler = RotatingFileHandler(os.path.join(log_dir, 'api.log'), maxBytes=1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))

# Use Flask's built-in logger to avoid conflicts and ensure proper integration.
app.logger.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
logger = app.logger
logger.info('API Logger successfully configured.')


# --- 3. MONITORING & VALIDATION SETUP ---

# Define a custom Prometheus metric for tracking prediction details.
# This allows us to monitor model behavior, not just application performance.
prediction_log_metric = Counter(
    'prediction_log',
    'Logs each prediction request and its response',
    ['request_data', 'response_data']
)

# Define a Pydantic model to enforce the schema of incoming prediction requests.
# If a request does not match this structure, it will be automatically rejected
# with a 400 Bad Request error, protecting the application logic.
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# --- 4. DYNAMIC MODEL LOADING ---
# Global variables to hold the model and its associated preprocessor.
model = None
scaler = None

# Read configuration from environment variables for container portability.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.getenv("MODEL_NAME_TO_SERVE", "Iris-LogisticRegression")
MODEL_ALIAS = "production"
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 10

logger.info("--- Starting model loading process ---")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Implement a retry loop to make the service resilient. This handles cases
# where the API container starts before the MLflow server is fully available.
for attempt in range(MAX_RETRIES):
    try:
        logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES}: Connecting to MLflow server at {MLFLOW_TRACKING_URI}...")
        client = mlflow.tracking.MlflowClient()

        # Load the model version that is currently tagged with the specified alias.
        # This allows for dynamic model updates without changing the API's code.
        logger.info(f"Attempting to load model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'.")
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

        # To ensure predictions are correct, we must use the exact same scaler
        # that the model was trained with. We find it by getting the model's
        # run ID and downloading the scaler artifact from that same run.
        model_version_details = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
        run_id = model_version_details.run_id
        logger.info(f"Found model version {model_version_details.version} from run_id {run_id}.")

        temp_scaler_path = client.download_artifacts(run_id, "scaler/scaler.joblib")
        scaler = joblib.load(temp_scaler_path)

        logger.info("--- Model and scaler loaded successfully. API is ready. ---")
        break  # Exit the loop on success.
    except Exception as e:
        logger.error(f"Attempt {attempt + 1} failed: {e}", exc_info=True)
        if attempt + 1 == MAX_RETRIES:
            logger.critical("--- All attempts to load model failed. API will not serve predictions. ---")
        else:
            logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)


# --- 5. API ENDPOINTS ---

@app.route("/")
def read_root():
    """Root endpoint to confirm the API is running."""
    return jsonify(status="ok", message=f"Serving model: '{MODEL_NAME}' with alias '{MODEL_ALIAS}'")


@app.route("/health")
def health_check():
    """Health check for orchestrators (e.g., Kubernetes, Docker Swarm)."""
    if model is not None and scaler is not None:
        return jsonify(status="ok", message="Model and scaler are loaded."), 200
    else:
        # Return a 503 Service Unavailable if the model isn't ready.
        return jsonify(status="error", message="Model or scaler not available."), 503


@app.route("/predict", methods=['POST'])
@validate()  # Pydantic decorator automatically validates the request body.
def predict_species(body: PredictionInput):
    """
    Main prediction endpoint. Accepts JSON data, validates its schema,
    preprocesses it, and returns a model prediction.
    """
    if model is None or scaler is None:
        return jsonify(error="Model not loaded. Check server logs."), 503
    try:
        # Pydantic provides a validated 'body' object.
        input_data = pd.DataFrame([body.dict()])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        result = str(prediction[0])

        # Log detailed information for debugging and auditing.
        logger.info(f"Prediction Request: {body.dict()} ==> Prediction: {result}")

        # Increment the custom Prometheus metric.
        prediction_log_metric.labels(request_data=str(body.dict()), response_data=result).inc()

        return jsonify(predicted_species=result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify(error=f"Prediction failed. Details: {e}"), 500


@app.route("/retrain", methods=['POST'])
def retrain_model():
    """Endpoint to trigger the model training script as a background process."""
    logger.info("--- Received request to retrain model ---")
    try:
        script_path = "scripts/train.py"
        # Use Popen to run the script without blocking the API.
        # The API responds immediately while training happens in the background.
        subprocess.Popen(["python", script_path])
        message = f"Started retraining process with '{script_path}'."
        logger.info(message)
        return jsonify(status="ok", message=message), 202
    except Exception as e:
        logger.error(f"Failed to start retraining: {e}", exc_info=True)
        return jsonify(status="error", message=str(e)), 500


# This block is for local development and testing only.
# It is NOT executed when the app is run by a production server like Gunicorn.
if __name__ == '__main__':
    # Connect to the MLflow server via its exposed localhost port for local testing.
    mlflow.set_tracking_uri("http://127.0.0.1:5002")
    app.run(host='0.0.0.0', port=8000, debug=True)
