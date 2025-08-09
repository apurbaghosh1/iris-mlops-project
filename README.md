# Iris MLOps Project

This project demonstrates a complete, end-to-end MLOps pipeline using the Iris classification dataset. It covers the entire machine learning lifecycle, from data versioning and model training to automated CI/CD, containerized deployment, and real-time monitoring.

## Core Features

* **Experiment Tracking:** Uses **MLflow** to log experiments, track model parameters, metrics, and artifacts.
* **Model Registry:** Leverages the MLflow Model Registry for model versioning and lifecycle management using aliases.
* **Containerization:** The prediction API is containerized using **Docker** for portability and consistent deployments.
* **CI/CD Automation:** A **GitHub Actions** workflow automatically lints, tests, builds, and pushes the API's Docker image to Docker Hub.
* **Data Versioning:** Uses **DVC** to version the dataset, ensuring full reproducibility.
* **Robust API:** A **Flask** application serves predictions and includes production-ready features like:
    * Input validation with **Pydantic**.
    * Persistent file-based logging.
    * A retraining trigger endpoint.
* **Monitoring:** Exposes a `/metrics` endpoint for **Prometheus** and includes a sample **Grafana** dashboard for live monitoring.

---

## Local Setup & Validation Workflow

This guide will walk you through running and validating the entire pipeline on your local machine.

### Prerequisites

* Docker & Docker Compose
* Python 3.9+ and `pip`
* Git
* A Docker Hub account

### Phase 1: Initial Setup and Baseline Deployment

First, let's get the initial version of our model trained and the application running.

#### 1. Clone and Set Up the Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd iris-mlops-project

# Install Python and DVC dependencies
pip install -r requirements.txt
```

#### 2. Initialize Data Versioning with DVC

```bash
# This is a one-time setup for DVC
dvc init
mkdir dvc-storage
dvc remote add -d local-remote dvc-storage
git add .dvc/config .gitignore
git commit -m "chore: Configure DVC remote"

# Track the initial version of the data
dvc add data/iris.csv
git add data/iris.csv.dvc
git commit -m "feat: Version initial iris dataset v1"
dvc push
```

#### 3. Start the MLflow Server

```bash
# This script starts the MLflow container
./deploy_local_mlflow.sh
```

#### 4. Train the Baseline Model

```bash
# This script trains the models and pushes them to the running MLflow server
python scripts/train.py
```

#### 5. Promote the Model to Production

* Open the MLflow UI in your browser at `http://localhost:5002`.
* Navigate to the **Models** tab, click on `Iris-LogisticRegression`.
* For **Version 1**, add a new **alias** and name it `production`.

#### 6. Deploy the API

* **Note:** The first time you run this, you may need to manually build and push your image to your own Docker Hub repository.

    ```bash
    # Login to Docker Hub
    docker login
    
    # Build and push the image (replace with your username)
    docker build -t <your-dockerhub-username>/iris-mlops-api:latest -f Dockerfile .
    docker push <your-dockerhub-username>/iris-mlops-api:latest
    ```
* Now, run the API deployment script.

    ```bash
    ./deploy_local_api.sh
    ```

#### 7. Start the Monitoring Stack

```bash
# This script starts the Prometheus and Grafana containers
./deploy_local_prometheus_grafana.sh
```

At this point, your entire application stack is running.

### Phase 2: Validation

Let's confirm that every part of the system is working correctly.

#### 1. Test the Prediction API

```bash
curl -X 'POST' \
  'http://localhost:8002/predict' \
  -H 'Content-Type: application/json' \
  -d '{
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
      }'
```

**Expected Output:** `{"predicted_species":"Iris-setosa"}`

#### 2. Validate Input Validation (Pydantic)

Send a request with a typo in a field name.

```bash
curl -X 'POST' \
  'http://localhost:8002/predict' \
  -H 'Content-Type: application/json' \
  -d '{
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal--width": 0.2
      }'
```

**Expected Output:** A `400 Bad Request` error explaining that `petal_width` is a required field.

#### 3. Validate Monitoring Endpoints

* **API Metrics:** Open `http://localhost:8002/metrics` in your browser. You should see a text page with metrics like `flask_http_requests_total`.
* **Prometheus:** Open `http://localhost:9090`. Go to **Status -> Targets** to confirm it's successfully scraping the API.
* **Grafana:** Open `http://localhost:3000` (login: admin/admin). Add Prometheus as a data source (`http://prometheus:9090`).

##### Sample Grafana Dashboard

Here are the queries to create the 4-panel dashboard shown in the screenshot:

**Panel 1: Request Latency / Duration (Time series)**

* **Title:** Request Latency / Duration
* **PromQL Query:** `histogram_quantile(0.95, sum(rate(flask_http_request_duration_seconds_bucket{method="POST", path="/predict"}[5m])) by (le))`
* **Unit:** `Duration -> seconds (s)`

**Panel 2: CPU/Memory Usage (Time series)**

* **Title:** CPU/Memory Usage
* **PromQL Query 1 (CPU):** `rate(process_cpu_seconds_total[1m])`
* **PromQL Query 2 (Memory):** `process_resident_memory_bytes`
* **Unit (CPU):** `Percent -> Percent (0.0-1.0)`
* **Unit (Memory):** `Data -> bytes(IEC)`

**Panel 3: Request Rate (Time series)**

* **Title:** Request Rate
* **PromQL Query:** `rate(flask_http_requests_total{method="POST", path="/predict"}[1m])`
* **Unit:** `Misc -> requests/sec (rps)`

**Panel 4: Prediction Distribution (Bar chart)**

* **Title:** Prediction Distribution
* **PromQL Query:** `sum(prediction_log_total) by (response_data)`

### Phase 3: Retraining on New Data

This workflow simulates updating your dataset and retraining the model.

#### 1. Update the Dataset

* Add the following new data to the end of your `data/iris.csv` file and save it.

    ```csv
    5.0,3.5,1.5,0.3,Iris-setosa
    6.0,2.8,4.4,1.4,Iris-versicolor
    7.0,3.1,6.0,2.2,Iris-virginica
    ```

#### 2. Create a New Data Version with DVC

```bash
# DVC will detect the change and update its pointer file
dvc add data/iris.csv

# Push the new data file to your DVC remote storage
dvc push

# Commit the new data version pointer to Git
git add data/iris.csv.dvc
git commit -m "feat: Update dataset to v2 with new samples"
```

#### 3. Trigger Retraining via API

* **Important:** For the API to use the new data, you first need to build and deploy a new image that contains it. Push your Git changes to trigger the CI/CD pipeline, then run `./deploy_local_api.sh`.
* Once the new API is running, trigger the training process:

    ```bash
    curl -X POST http://localhost:8002/retrain
    ```

#### 4. Validate the New Model

* Watch the API logs (`docker-compose logs -f api`) to see the training start.
* Go to the MLflow UI. You will see a **Version 2** of your `Iris-LogisticRegression` model. You can now evaluate it and promote it to production by moving the `production` alias.
