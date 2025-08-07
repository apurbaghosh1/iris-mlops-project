# app/main.py

import os
import logging
from logging.handlers import RotatingFileHandler
import time
import joblib
import pandas as pd
import mlflow
import subprocess
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter  # <-- Import the Counter class
from flask_pydantic import validate
from pydantic import BaseModel


# --- 1. SETUP ---
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# === LOGGING SETUP (Part 5) ===
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
file_handler = RotatingFileHandler(os.path.join(log_dir, 'api.log'), maxBytes=1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
logger = app.logger
logger.info('API Logger configured to write to file.')
# === END OF LOGGING SETUP ===


# === CUSTOM METRIC FOR PREDICTION LOGS (Your Request) ===
# This creates a new metric that will appear on the /metrics page.
# It's a counter with two labels: 'request_data' and 'response_data'.
prediction_log_metric = Counter(
    'prediction_log',
    'Logs each prediction request and its response',
    ['request_data', 'response_data']
)


# === PYDANTIC INPUT VALIDATION SCHEMA (Bonus) ===
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# --- 2. MODEL LOADING ---
# ... (The model loading code remains exactly the same) ...
model = None
scaler = None
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.getenv("MODEL_NAME_TO_SERVE", "Iris-LogisticRegression")
MODEL_ALIAS = "production"
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 10
logger.info("--- Starting Model Loading Process using Aliases ---")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
for attempt in range(MAX_RETRIES):
    try:
        logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES}: Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
        client = mlflow.tracking.MlflowClient()
        logger.info("--> Listing all registered models found by the client:")
        all_models = client.search_registered_models()
        if not all_models:
            logger.warning("--> No registered models found on the MLflow server.")
        else:
            for model_info in all_models:
                logger.info(f"-->   - Found model: '{model_info.name}'")
        logger.info(f"Attempting to load model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'.")
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        model_version_details = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
        run_id = model_version_details.run_id
        logger.info(f"Found model version {model_version_details.version} with alias '{MODEL_ALIAS}' from run_id {run_id}.")
        temp_scaler_path = client.download_artifacts(run_id, "scaler/scaler.joblib")
        scaler = joblib.load(temp_scaler_path)
        logger.info("--- Model and Scaler loaded successfully. API is ready. ---")
        break
    except Exception as e:
        logger.error(f"Attempt {attempt + 1} failed: {e}", exc_info=True)
        if attempt + 1 == MAX_RETRIES:
            logger.critical("--- All attempts to load the model failed. The API will not be able to serve predictions. ---")
        else:
            logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)


# --- 3. API ENDPOINTS ---
@app.route("/")
def read_root():
    return jsonify(status="ok", message=f"Welcome! Serving model: '{MODEL_NAME}' with alias '{MODEL_ALIAS}'")


@app.route("/health")
def health_check():
    if model is not None and scaler is not None:
        return jsonify(status="ok", message="Model and scaler are loaded."), 200
    else:
        return jsonify(status="error", message="Model and/or scaler are not available."), 503


@app.route("/predict", methods=['POST'])
@validate()
def predict_species(body: PredictionInput):
    if model is None or scaler is None:
        return jsonify(error="Model or scaler is not available. Check server logs for startup errors."), 503
    try:
        input_data = pd.DataFrame([body.dict()])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        result = str(prediction[0])
        
        # Log to the file (for human-readable debugging)
        logger.info(f"Prediction Request: {body.dict()} ==> Prediction: {result}")
        
        # === INCREMENT THE CUSTOM METRIC (Your Request) ===
        # This adds the request/response data to the /metrics endpoint.
        # We convert the dictionary to a string to use it as a label.
        prediction_log_metric.labels(request_data=str(body.dict()), response_data=result).inc()
        
        return jsonify(predicted_species=result)
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        return jsonify(error=f"Failed to make prediction. Details: {e}"), 500


@app.route("/retrain", methods=['POST'])
def retrain_model():
    logger.info("--- Received request to retrain model. ---")
    try:
        script_path = "scripts/train.py"
        subprocess.Popen(["python", script_path])
        return jsonify(status="ok", message=f"Started retraining process using '{script_path}'."), 202
    except Exception as e:
        logger.error(f"Failed to start retraining process. Error: {e}", exc_info=True)
        return jsonify(status="error", message=str(e)), 500


# Local testing block
if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5002")
    app.run(host='0.0.0.0', port=8000, debug=True)
