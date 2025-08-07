#!/bin/bash

# =================================================================
# Deployment Script for the MLflow Server
# =================================================================
# This script starts only the MLflow server using Docker Compose.
# It should be run first, before training the model.
# =================================================================

echo "--- Starting MLflow Server Deployment ---"

# This command is smart: if the server is already running, it does nothing.
# If it's not running, it will build the image from its Dockerfile and start it.
docker-compose up -d mlflow-server

echo
echo "--- MLflow Server is up and running ---"
echo "You can access the UI at http://localhost:5002"
