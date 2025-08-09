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

# --- Core Libraries ---
import pandas as pd
import mlflow
import mlflow.sklearn
import warnings
import joblib

# --- Scikit-learn for ML tasks ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Suppress common warnings for cleaner output.
warnings.filterwarnings("ignore")


def load_and_preprocess_data(data_path='data/iris.csv'):
    """
    Loads the dataset from the specified path, preprocesses it, and splits it
    into training and testing sets.

    Args:
        data_path (str): The file path for the dataset.

    Returns:
        tuple: A tuple containing scaled training features, scaled testing features,
               training labels, testing labels, and the fitted scaler object.
    """
    print("1. Loading and preprocessing data...")
    df = pd.read_csv(data_path)

    # Assume the last column is the target and the rest are features.
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data into training and testing sets.
    # 'stratify=y' is crucial for classification to ensure that both the train
    # and test sets have a proportional representation of class labels.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize a standard scaler to normalize feature values.
    # Scaling is important for models like Logistic Regression.
    scaler = StandardScaler()
    # Fit the scaler ONLY on the training data to avoid data leakage from the test set.
    X_train_scaled = scaler.fit_transform(X_train)
    # Apply the same fitted transformation to the test data.
    X_test_scaled = scaler.transform(X_test)

    print("Data loading and preprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(model, model_name, params, X_train, y_train, X_test, y_test, scaler):
    """
    Trains a given model, logs all relevant information to MLflow, and returns
    the MLflow run ID for later use.

    Args:
        model: The scikit-learn model instance to be trained.
        model_name (str): A descriptive name for the model run.
        params (dict): A dictionary of parameters to log to MLflow.
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        scaler (StandardScaler): The fitted scaler object to be logged as an artifact.

    Returns:
        str: The unique ID of the MLflow run.
    """
    print(f"\n--- Training {model_name} ---")
    # The 'with' statement ensures that the MLflow run is properly started and ended.
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {model_name}: {accuracy:.4f}")

        # Log parameters and metrics to MLflow for tracking
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Infer the model's input/output signature. This is crucial for deployment
        # as it defines the expected data schema.
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

        # Log the trained scikit-learn model to MLflow.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # This creates a 'model' subdirectory for the artifacts
            signature=signature,
            input_example=X_train[:5]
        )

        # It's critical to also save the scaler used for preprocessing.
        # The API will need this exact scaler to transform live prediction data.
        scaler_path = "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")

        print(f"Successfully logged '{model_name}'.")
        # Return the active run's ID so we can reference it for model registration.
        return mlflow.active_run().info.run_id


# This block executes only when the script is run directly (e.g., `python scripts/train.py`)
if __name__ == "__main__":
    # Configure MLflow to communicate with the remote tracking server.
    # This URI points to the MLflow server running in Docker, exposed on port 5002.
    tracking_uri = "http://127.0.0.1:5002"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"--- Running in API mode. Logging to MLflow server at {tracking_uri} ---")

    # Set the experiment name. If it doesn't exist, MLflow will create it.
    EXPERIMENT_NAME = "Iris_Classification_Final"
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and preprocess the data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    # --- Train and Register Logistic Regression ---
    lr_params = {"model_type": "LogisticRegression"}
    lr_model = LogisticRegression(random_state=42)
    lr_run_id = train_model(lr_model, "LogisticRegression", lr_params, X_train, y_train, X_test, y_test, scaler)

    # Register the model in the MLflow Model Registry.
    # The 'runs:/<run_id>/model' URI is a portable way to reference the model
    # logged in a specific run, regardless of where the artifacts are stored.
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
    print("Check the UI at http://localhost:5002")
