# Heart Disease MLOps Pipeline

End-to-end Machine Learning Operations (MLOps) project for predicting the presence of heart disease using the UCI Heart Disease dataset.

The project demonstrates a reproducible ML workflow covering data acquisition, exploratory data analysis, preprocessing, model training and tuning, experiment tracking, automated testing, CI/CD, FastAPI model serving, Docker containerization, Kubernetes deployment, and API monitoring.

## MLOps Pipeline

```text
UCI Heart Disease Dataset
          |
          v
Data Acquisition
          |
          v
EDA + Data Preparation
          |
          v
Preprocessing Pipeline
          |
          v
Model Training + Hyperparameter Tuning
          |
          v
Cross-Validation + Evaluation
          |
          +------> MLflow Experiment Tracking
          |
          v
Saved Production Model
          |
          v
FastAPI Prediction API
          |
          v
Docker Image
          |
          v
Kubernetes Deployment
          |
          v
Prediction API + Prometheus Metrics
```

GitHub Actions provides automated linting, testing, pipeline execution, and artifact generation for repository changes.

## Dataset

This project uses the UCI Heart Disease dataset (UCI dataset ID 45).

The dataset contains 303 observations and 13 predictor features. The original target `num` has values from 0 to 4 and is converted into a binary classification target:

- `0` = no heart disease
- `1` = heart disease (`num > 0`)

The project can download the dataset automatically using `ucimlrepo`.

## Models

Two classification algorithms are evaluated:

- Logistic Regression
- Random Forest

Hyperparameter tuning is performed using `GridSearchCV` with 5-fold cross-validation and ROC-AUC as the model-selection metric.

The final model is selected using cross-validation ROC-AUC rather than test-set performance.

The selected model is saved as:

```text
models/heart_disease_model.pkl
```

The saved object contains the complete Scikit-learn preprocessing and classification pipeline.

## Project Structure

```text
heart-disease-mlops/
|
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|
|-- data/
|   |-- raw/
|   |   `-- heart_disease.csv
|   `-- processed/
|       `-- heart_disease_clean.csv
|
|-- k8s/
|   |-- deployment.yaml
|   `-- service.yaml
|
|-- models/
|   `-- heart_disease_model.pkl
|
|-- notebooks/
|   `-- 01_eda.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- api.py
|   |-- download_data.py
|   |-- eda.py
|   |-- preprocess.py
|   `-- train.py
|
|-- tests/
|   |-- __init__.py
|   |-- test_api.py
|   |-- test_model.py
|   `-- test_preprocess.py
|
|-- .dockerignore
|-- .gitignore
|-- Dockerfile
|-- main.py
|-- requirements.txt
`-- README.md
```

## Prerequisites

For local reproduction:

- Python 3.13
- pip
- Internet connection for dependency installation and UCI dataset download
- Docker Desktop 26.1.1 for container execution
- Docker Desktop Kubernetes enabled for local Kubernetes deployment
- `kubectl`

The Python ML pipeline can be executed without Docker or Kubernetes. Docker Desktop is required only for the containerization and Kubernetes deployment stages.

## 1. Create the Python Environment

Open a terminal in the project root.

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. Run the Complete ML Pipeline

From the project root:

```cmd
python main.py
```

This performs:

1. UCI dataset download
2. Data preparation and EDA
3. Preprocessing
4. Logistic Regression training/tuning
5. Random Forest training/tuning
6. Cross-validation
7. Model evaluation
8. MLflow experiment tracking
9. Evaluation artifact generation
10. Final model selection and serialization

A successful execution ends with:

```text
============================================================
Pipeline completed successfully.
============================================================
```

## 3. Run Automated Tests

```cmd
python -m pytest -v
```

The test suite verifies preprocessing, the trained model, FastAPI prediction behavior, and monitoring endpoints.

Expected result:

```text
8 passed
```

Linting can also be checked with:

```cmd
ruff check src tests main.py
```

## 4. Run the FastAPI Application Locally

Start the API:

```cmd
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open Swagger UI in a browser:

```text
http://localhost:8000/docs
```

Use `POST /predict` to perform inference.

Example request:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}
```

Example response:

```json
{
  "prediction": 0,
  "prediction_label": "no heart disease",
  "confidence": 0.6864782350043445
}
```

The exact confidence can depend on the newly trained model.

Stop the development server with `Ctrl+C` before continuing if port 8000 is required by Docker or Kubernetes.

## 5. Build the Docker Image

Ensure Docker Desktop is running.

From the project root:

```cmd
docker build -t heart-disease-api .
```

The Docker image packages the FastAPI application together with the trained model.

## 6. Run the Docker Container Locally

```cmd
docker run --rm -p 8000:8000 heart-disease-api
```

Open:

```text
http://localhost:8000/docs
```

Test `POST /predict` using the sample JSON shown above.

Stop the container with `Ctrl+C` before deploying the Kubernetes service on the same port.

## 7. Deploy to Kubernetes

This project has been tested using the Kubernetes cluster supplied by Docker Desktop.

Ensure Docker Desktop Kubernetes is enabled and running.

Deploy the application:

```cmd
kubectl apply -f k8s\deployment.yaml
kubectl apply -f k8s\service.yaml
```

If the Docker image has just been rebuilt while an older deployment already exists, restart the deployment:

```cmd
kubectl rollout restart deployment heart-disease-api
kubectl rollout status deployment heart-disease-api
```

Check the application:

```cmd
kubectl get pods
kubectl get service heart-disease-api-service
```

With Docker Desktop Kubernetes, the `LoadBalancer` service is exposed locally on port 8000.

Open:

```text
http://localhost:8000/docs
```

Then execute `POST /predict`.

## 8. Monitoring and Logging

The FastAPI application records API request information including:

- HTTP method
- endpoint
- HTTP status code
- request duration

Prometheus-compatible application metrics are exposed at:

```text
http://localhost:8000/metrics
```

Metrics include:

```text
api_requests_total
api_request_duration_seconds
```

For a Kubernetes deployment, application logs can be inspected using:

```cmd
kubectl logs deployment/heart-disease-api
```

The logs provide request-level evidence such as successful `/predict` requests and their execution duration.

## 9. MLflow Experiment Tracking

Training runs are recorded using MLflow.

The project tracks information including:

- model parameters
- cross-validation results
- accuracy
- precision
- recall
- ROC-AUC
- evaluation plots
- trained model artifacts

The local MLflow backend is stored in `mlflow.db`.

After running the training pipeline, the MLflow UI can be used to inspect and compare experiments.

## 10. CI/CD

GitHub Actions configuration is located at:

```text
.github/workflows/ci.yml
```

The CI workflow runs automatically for configured pushes and pull requests to `main`.

The pipeline performs:

```text
Checkout repository
        |
        v
Set up Python 3.13
        |
        v
Install dependencies
        |
        v
Ruff linting
        |
        v
Unit tests
        |
        v
Run complete ML pipeline
        |
        v
Upload generated artifacts/model
```

A linting, testing, or pipeline failure causes the CI job to fail.

## Reproducing the Complete Project

For a fresh local reproduction, use the following sequence:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

python main.py
python -m pytest -v

docker build -t heart-disease-api .

kubectl apply -f k8s\deployment.yaml
kubectl apply -f k8s\service.yaml
kubectl rollout restart deployment heart-disease-api
kubectl rollout status deployment heart-disease-api

kubectl get pods
kubectl get service heart-disease-api-service
```

Then open:

```text
http://localhost:8000/docs
```

Use the sample JSON in this README to test `POST /predict`.

Monitoring is available at:

```text
http://localhost:8000/metrics
```

## Reproducibility Notes

`main.py` is responsible for the machine-learning workflow. Docker and Kubernetes deployment are intentionally separate infrastructure stages.

Therefore:

```text
python main.py
```

does **not** automatically build the Docker image or deploy Kubernetes.

When the model is retrained, rebuild the Docker image before deployment so the container contains the newly generated model.

The submitted dataset files are included for convenience, while `src/download_data.py` also allows the raw dataset to be acquired again from UCI.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API health/status |
| `/predict` | POST | Heart disease prediction and confidence |
| `/metrics` | GET | Prometheus-compatible monitoring metrics |

## Technology Stack

Python, Pandas, Scikit-learn, MLflow, Pytest, Ruff, FastAPI, Uvicorn, Prometheus Client, Docker, Kubernetes, GitHub Actions and the UCI Machine Learning Repository.