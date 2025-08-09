# app/main.py

# =================================================================
# ML Model Serving API using Flask
# =================================================================
# This script defines a Flask web application that serves a trained
# scikit-learn model. Its key responsibilities include:
# 1. Loading a specific model version from a remote MLflow server on startup.
# 2. Providing multiple API endpoints for health checks, predictions, and retraining.
# 3. Implementing robust features like persistent logging, input validation,
#    and real-time performance monitoring.
#
# This application is designed to be run as a containerized service,
# typically managed by Gunicorn.
# =================================================================

# --- Core Libraries ---
import os
import logging
from logging.handlers import RotatingFileHandler
import time
import joblib
import pandas as pd
import mlflow
import subprocess
import subprocess

# --- Flask and Extension Libraries ---
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter
from flask_pydantic import validate
from pydantic import BaseModel


# --- 1. APPLICATION SETUP ---
# Initialize the core Flask application object.
app = Flask(__name__)
# Initialize the Prometheus metrics exporter, which automatically creates
# the /metrics endpoint and tracks standard HTTP request metrics.
metrics = PrometheusMetrics(app)


# === 2. PERSISTENT LOGGING CONFIGURATION (Part 5) ===
# This section configures a robust logging setup that writes to a file,
# ensuring that logs are not lost when the container restarts.
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Use a RotatingFileHandler to prevent log files from growing indefinitely.
# It creates a new file when the current one reaches 1MB and keeps the last 5 logs.
file_handler = RotatingFileHandler(os.path.join(log_dir, 'api.log'), maxBytes=1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))

# Configure Flask's built-in logger to use our file handler.
# This is the standard practice for logging in Flask applications.
app.logger.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
logger = app.logger  # Use the configured app logger throughout the file.
logger.info('API Logger configured to write to file.')


# === 3. CUSTOM METRICS AND VALIDATION SCHEMAS ===

# Define a custom Prometheus metric. This allows us to log specific business
# events, like the content of a prediction, to the /metrics endpoint.
prediction_log_metric = Counter(
    'prediction_log',
    'Logs each prediction request and its response',
    ['request_data', 'response_data']
)

# Define a Pydantic model for input validation (Bonus Feature).
# This class acts as a schema. If an incoming JSON request does not match
# this structure, Flask-Pydantic will automatically reject it with a 400 error.
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# --- 4. MODEL LOADING ON STARTUP ---
# Global variables to hold the loaded model and scaler.
model = None
scaler = None

# Configuration is read from environment variables for portability.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.getenv("MODEL_NAME_TO_SERVE", "Iris-LogisticRegression")
MODEL_ALIAS = "production"
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 10

logger.info("--- Starting Model Loading Process using Aliases ---")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Implement a retry loop to handle cases where the API container starts
# faster than the MLflow server, preventing a startup crash.
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

        # Load the model version that has the specified alias (e.g., 'production').
        # The '@' syntax is the modern way to reference models by alias.
        logger.info(f"Attempting to load model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'.")
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

        # To get the associated scaler, we first find the model version details.
        model_version_details = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
        run_id = model_version_details.run_id
        logger.info(f"Found model version {model_version_details.version} with alias '{MODEL_ALIAS}' from run_id {run_id}.")

        # Download the scaler artifact from the same MLflow run that produced the model.
        temp_scaler_path = client.download_artifacts(run_id, "scaler/scaler.joblib")
        scaler = joblib.load(temp_scaler_path)

        logger.info("--- Model and Scaler loaded successfully. API is ready. ---")
        break  # Exit the loop on successful loading.
    except Exception as e:
        logger.error(f"Attempt {attempt + 1} failed: {e}", exc_info=True)
        if attempt + 1 == MAX_RETRIES:
            logger.critical("--- All attempts to load the model failed. The API will not be able to serve predictions. ---")
        else:
            logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)


# --- 5. API ENDPOINTS ---

@app.route("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return jsonify(status="ok", message=f"Welcome! Serving model: '{MODEL_NAME}' with alias '{MODEL_ALIAS}'")


@app.route("/health")
def health_check():
    """A health check endpoint for load balancers or container orchestrators."""
    if model is not None and scaler is not None:
        return jsonify(status="ok", message="Model and scaler are loaded."), 200
    else:
        # Return a 503 Service Unavailable if the model failed to load.
        return jsonify(status="error", message="Model and/or scaler are not available."), 503


@app.route("/predict", methods=['POST'])
@validate()  # The Pydantic decorator automatically validates the request body.
def predict_species(body: PredictionInput):
    """
    The main prediction endpoint. It accepts JSON data, validates it,
    and returns a model prediction.
    """
    if model is None or scaler is None:
        return jsonify(error="Model or scaler is not available. Check server logs for startup errors."), 503
    try:
        # The 'body' argument is a validated Pydantic object, not a raw JSON dict.
        input_data = pd.DataFrame([body.dict()])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        result = str(prediction[0])

        # Log the detailed request and response to the persistent log file.
        logger.info(f"Prediction Request: {body.dict()} ==> Prediction: {result}")

        # Increment the custom Prometheus metric with the request/response as labels.
        prediction_log_metric.labels(request_data=str(body.dict()), response_data=result).inc()

        return jsonify(predicted_species=result)
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        return jsonify(error=f"Failed to make prediction. Details: {e}"), 500


@app.route("/retrain", methods=['POST'])
def retrain_model():
    """
    A simple endpoint to trigger the training script.
    NOTE: In a real production system, this would be handled by a more robust
    job queue system like Celery, Airflow, or a serverless function.
    """
    logger.info("--- Received request to retrain model ---")
    try:
        # The script must be available inside the container at this path.
        # The training script will use whatever version of iris.csv was
        # copied into the Docker image during the build process.
        script_path = "scripts/train.py"
        # Use Popen to run the script as a non-blocking background process
        subprocess.Popen(["python", script_path])
        message = f"Started retraining process using '{script_path}'."
        logger.info(message)
        return jsonify(status="ok", message=message), 202
    except Exception as e:
        error_message = f"Failed to start retraining process. Error: {e}"
        logger.error(error_message, exc_info=True)
        return jsonify(status="error", message=error_message), 500
    
# This block is only executed when you run `python app/main.py` directly.
# It is NOT used when the application is run by Gunicorn in the Docker container.
if __name__ == '__main__':
    # For local testing, we connect to the MLflow server via its exposed localhost port.
    mlflow.set_tracking_uri("http://127.0.0.1:5002")
    app.run(host='0.0.0.0', port=8000, debug=True)
