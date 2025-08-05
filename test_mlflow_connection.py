# test_mlflow_connection.py

import mlflow


# --- Configuration ---
# This should be the address of your running MLflow server.
# For local testing, this is the correct address.
MLFLOW_TRACKING_URI = "http://127.0.0.1:5002"

# Removed 'f' prefix from string without placeholders to fix F541
print("--- MLflow Connection Test ---")
print(f"Attempting to connect to MLflow server at: {MLFLOW_TRACKING_URI}")

try:
    # Set the tracking URI for this script
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create a client to interact with the server
    client = mlflow.tracking.MlflowClient()
    print("\nStep 1: Successfully connected to MLflow client.")

    # Get a list of all registered models
    registered_models = mlflow.search_registered_models()

    if not registered_models:
        print("\nStep 2: Connection successful, but no registered models were found.")
        print("         Please make sure you have run the training script first.")
    else:
        print(f"\nStep 2: Connection successful! Found {len(registered_models)} registered models:")
        for model in registered_models:
            print(f"  - Model Name: {model.name}")
            # Print latest versions for more detail
            for version in model.latest_versions:
                # UPDATED: The 'aliases' attribute holds the new stage names like 'production'
                print(f"    - Version: {version.version}, Aliases: {version.aliases}, Run ID: {version.run_id}")

    # Removed 'f' prefix from string without placeholders to fix F541
    print("\n--- Test Complete ---")

except Exception as e:
    print("\n--- TEST FAILED ---")
    print("Could not connect to the MLflow server or an error occurred.")
    print(f"Error: {e}")
    print("\nPlease ensure your MLflow server is running with the command:")
    print("mlflow server --host 0.0.0.0 --port 5000 --artifacts-destination ./mlruns")
    