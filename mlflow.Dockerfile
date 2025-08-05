# mlflow.Dockerfile

# Use the same base Python image
FROM python:3.9-slim

# Install only the necessary packages for the server
RUN pip install mlflow gunicorn

# The command to start the server.
# --artifacts-destination: This forces the server to act as a proxy and
# serve the model files over HTTP, which solves the absolute path problem.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "/mlruns", "--artifacts-destination", "/mlruns"]
