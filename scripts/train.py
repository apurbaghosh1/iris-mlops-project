# scripts/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
import joblib


# Suppress warnings
warnings.filterwarnings("ignore")


def load_and_preprocess_data(data_path='data/iris.csv'):
    print("1. Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data loading and preprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(model, model_name, params, X_train, y_train, X_test, y_test, scaler):
    """Trains a model and logs it to MLflow, returning the run_id."""
    print(f"\n--- Training {model_name} ---")
    # Removed 'as run' since the 'run' variable was not used, fixing F841
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {model_name}: {accuracy:.4f}")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5]
        )

        scaler_path = "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")

        print(f"Successfully logged '{model_name}'.")
        return mlflow.active_run().info.run_id


if __name__ == "__main__":
    # --- THIS IS THE CRITICAL CHANGE ---
    # Instead of saving to a local file, we are now pointing to the
    # HTTP address of the MLflow server that is running in Docker.
    # The docker-compose.yml file exposes the server on port 5002.
    tracking_uri = "http://127.0.0.1:5002"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"--- Running in API mode. Logging to MLflow server at {tracking_uri} ---")

    EXPERIMENT_NAME = "Iris_Classification_Final"
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    # --- Train and Register Logistic Regression ---
    lr_params = {"model_type": "LogisticRegression"}
    lr_model = LogisticRegression(random_state=42)
    lr_run_id = train_model(lr_model, "LogisticRegression", lr_params, X_train, y_train, X_test, y_test, scaler)

    # This registration logic works perfectly with a remote server.
    # MLflow knows to make an API call to register the model.
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
    # Removed 'f' prefix from string without placeholders to fix F541
    print("Check the UI at http://127.0.0.1:5002")