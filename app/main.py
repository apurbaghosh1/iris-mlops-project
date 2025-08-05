# app/main.py

import os
import logging
import time
import joblib
import pandas as pd
import mlflow
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics


# --- 1. SETUP ---
app = Flask(__name__)
metrics = PrometheusMetrics(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 2. MODEL LOADING WITH ALIASES AND DEBUGGING ---
model = None
scaler = None

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.getenv("MODEL_NAME_TO_SERVE", "Iris-LogisticRegression")
MODEL_ALIAS = "production"  # We use an alias instead of a stage
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 10

logger.info("--- Starting Model Loading Process using Aliases ---")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

for attempt in range(MAX_RETRIES):
    try:
        logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES}: Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
        client = mlflow.tracking.MlflowClient()

        # === ADDED BACK AS REQUESTED: List all registered models ===
        logger.info("--> Listing all registered models found by the client:")
        all_models = client.search_registered_models()
        if not all_models:
            logger.warning("--> No registered models found on the MLflow server.")
        else:
            for model_info in all_models:
                logger.info(f"-->   - Found model: '{model_info.name}'")
        # === END OF DEBUGGING CODE ===

        # Load the model using the '@' syntax for aliases
        logger.info(f"Attempting to load model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'.")
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

        # Get model version details using the alias
        model_version_details = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
        run_id = model_version_details.run_id
        logger.info(f"Found model version {model_version_details.version} with alias '{MODEL_ALIAS}' from run_id {run_id}.")

        # Download the scaler artifact from that run
        temp_scaler_path = client.download_artifacts(run_id, "scaler/scaler.joblib")
        scaler = joblib.load(temp_scaler_path)

        logger.info("--- Model and Scaler loaded successfully. API is ready. ---")
        break  # Exit the loop on success

    except Exception as e:
        logger.error(f"Attempt {attempt + 1} failed: {e}")
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
def predict_species():
    if model is None or scaler is None:
        return jsonify(error="Model or scaler is not available. Check server logs for startup errors."), 503
    try:
        json_data = request.get_json()
        input_data = pd.DataFrame([json_data])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        result = str(prediction[0])
        logger.info(f"Prediction Request: {json_data} ==> Prediction: {result}")
        return jsonify(predicted_species=result)
    except ValueError as ve:
        return jsonify(error=f"Prediction failed. Check your input data. Details: {ve}"), 400
    except Exception as e:
        # Using the exception variable 'e' to fix the F841 error
        return jsonify(error=f"Failed to make prediction. Details: {e}"), 500


# Local testing block
if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5002")
    app.run(host='0.0.0.0', port=8000, debug=True)