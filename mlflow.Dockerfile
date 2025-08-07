# mlflow.Dockerfile

# Use the same base Python image for consistency
FROM python:3.9-slim

# Install only the necessary packages for the server
RUN pip install mlflow gunicorn boto3

# The command to start the server.
# --host 0.0.0.0: Makes the server accessible from outside the container.
# --port 5000: The internal port the server listens on.
# --backend-store-uri /mlruns: Tells the server to use the /mlruns directory
#   (which is mapped to our local ./mlruns folder) to store experiment metadata.
#
# --artifacts-destination /mlruns: THIS IS THE CRITICAL FIX.
#   It tells the server to also use the /mlruns directory as the root for storing
#   all model artifacts (like the model files and scaler). This ensures that all
#   artifact paths are stored relative to this root, making them portable.
#   It also enables the server to act as a file server, serving artifacts over HTTP.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "/mlruns", \
     "--artifacts-destination", "/mlruns"]
