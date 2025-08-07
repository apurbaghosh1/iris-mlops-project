#!/bin/bash

# =================================================================
# Deployment Script for the API Service
# =================================================================
# This script deploys only the API service. It should be run
# AFTER the MLflow server is running and the model has been trained
# and promoted (aliased) in the MLflow UI.
# =================================================================

echo "--- Starting API Service Deployment ---"

# Step 1: Pull the latest version of the API image from Docker Hub.
# This reads the 'image:' name from the 'api' service in docker-compose.yml.
echo
echo "==> STEP 1: Pulling latest API image from Docker Hub..."
docker-compose pull api

# Step 2: Start or update the 'api' service using the pulled image.
# The '--no-build' flag is crucial to ensure it uses the image
# we just pulled from Docker Hub.
echo
echo "==> STEP 2: Starting/Updating the 'api' service..."
docker-compose up -d --no-build api

echo
echo "--- API Deployment Complete ---"
echo "The API service is running. You can check its status with: docker-compose ps"
echo "The API should be accessible at http://localhost:8002"
