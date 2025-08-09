# scripts/train.py

# =================================================================
# Model Training and MLflow Tracking Script
# =================================================================
# This script is responsible for the core machine learning tasks:
# 1. Loading and preprocessing the Iris dataset.
# 2. Training two different classification models (Logistic Regression, Random Forest).
# 3. Using MLflow to track experiments, log parameters, metrics, and artifacts.
# 4. Registering the trained models in the MLflow Model Registry for versioning.
#
# This script is designed to be run as a standalone process and communicates
# with a running MLflow tracking server.
# =================================================================

# --- Core Python Libraries ---
import os
import warnings
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

# --- Data Science and ML Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Suppress common scikit-learn warnings for cleaner output.
warnings.filterwarnings("ignore")


def load_and_preprocess_data(data_path='data/iris.csv'):
    """
    Loads data, cleans it by removing rows with missing values, and splits
    it into scaled training and testing sets.

    Args:
        data_path (str): The file path for the dataset.

    Returns:
        tuple: A tuple containing (X_train_scaled, X_test_scaled, y_train,
               y_test, scaler).
    """
    print("1. Loading and preprocessing data...")
    df = pd.read_csv(data_path)

    # Data cleaning: remove any rows with missing values (NaNs).
    print(f"Original dataset shape: {df.shape}")
    df.dropna(inplace=True)
    print(f"Shape after dropping NaNs: {df.shape}")

    # Assume the last column is the target variable (y).
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split data. 'stratify=y' ensures the class distribution in the train
    # and test sets is the same as the original dataset, which is crucial
    # for classification tasks.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features. This is important for distance-based algorithms
    # like Logistic Regression.
    scaler = StandardScaler()
    # Fit the scaler ONLY on training data to prevent data leakage.
    X_train_scaled = scaler.fit_transform(X_train)
    # Apply the same scaling to the test data.
    X_test_scaled = scaler.transform(X_test)

    print("Data loading and preprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(model, model_name, params, X_train, y_train, X_test, y_test, scaler):
    """
    Trains a model, logs all relevant information to MLflow, and returns the
    MLflow run ID.

    Args:
        model: The scikit-learn model instance to be trained.
        model_name (str): A descriptive name for the model run.
        params (dict): A dictionary of model parameters to log.
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        scaler (StandardScaler): The fitted scaler object to be logged.

    Returns:
        str: The unique ID of the MLflow run.
    """
    print(f"\n--- Training {model_name} ---")
    # The 'with' statement ensures the MLflow run is properly managed.
    with mlflow.start_run(run_name=model_name):
        # Train the model.
        model.fit(X_train, y_train)

        # Evaluate the model on the test set.
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {model_name}: {accuracy:.4f}")

        # Log parameters and metrics for experiment tracking.
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Infer the model's signature to define its expected input/output schema.
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

        # Log the scikit-learn model to MLflow.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5]
        )

        # Log the scaler as an artifact. This is critical for ensuring
        # predictions use the exact same preprocessing as training.
        scaler_path = "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")

        print(f"Successfully logged '{model_name}'.")
        # Return the run ID for model registration.
        return mlflow.active_run().info.run_id


# This block executes only when the script is run directly.
if __name__ == "__main__":
    # Read the MLflow server URI from an environment variable for flexibility.
    # This allows the script to connect to 'mlflow-server' when run inside
    # a Docker container, and fall back to 'localhost' for local execution.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5002")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"--- Training script connecting to MLflow at {tracking_uri} ---")

    # Set the experiment name. MLflow will create it if it doesn't exist.
    EXPERIMENT_NAME = "Iris_Classification_Final"
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and preprocess the data.
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    # --- Train and Register Logistic Regression ---
    lr_params = {"model_type": "LogisticRegression"}
    lr_model = LogisticRegression(random_state=42)
    lr_run_id = train_model(lr_model, "LogisticRegression", lr_params, X_train, y_train, X_test, y_test, scaler)

    # Register the model in the MLflow Model Registry. The 'runs:/' URI is a
    # portable reference to the model logged in the specified run.
    lr_model_uri = f"runs:/{lr_run_id}/model"
    print(f"Registering Logistic Regression from URI: {lr_model_uri}")
    mlflow.register_model(model_uri=lr_model_uri, name="Iris-LogisticRegression")

    # --- Train and Register Random Forest ---
    rf_params = {"model_type": "RandomForest", "n_estimators": 100}
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_run_id = train_model(rf_model, "RandomForest", rf_params, X_train, y_train, X_test, y_test, scaler)

    rf_model_uri = f"runs:/{rf_run_id}/model"
    print(f"Registering Random Forest from URI: {rf_model_uri}")
    mlflow.register_model(model_uri=rf_model_uri, name="Iris-RandomForest")

    print("\n\nWorkflow Complete. Models have been sent to the MLflow server.")
    print("Check the UI at http://127.0.0.1:5002")
