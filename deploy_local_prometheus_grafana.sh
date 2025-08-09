#!/bin/bash

# =================================================================
# Deployment Script for the Monitoring Stack (Prometheus & Grafana)
# =================================================================
# This script starts only the monitoring services defined in the
# docker-compose.yml file. It should be run after the API service
# is up and running.
# =================================================================

echo "--- Starting Monitoring Stack (Prometheus & Grafana) ---"

# This command will start the 'prometheus' and 'grafana' containers
# in detached mode. If they are already running, it will recreate them
# if there are any configuration changes.
docker-compose up -d prometheus grafana

echo
echo "--- Monitoring Stack is up and running ---"
echo "You can access Prometheus at http://localhost:9090"
echo "You can access Grafana at http://localhost:3000"
