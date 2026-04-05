import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI()

# =========================
# LOAD MODEL (MLflow)
# =========================
model = mlflow.pyfunc.load_model("models:/model/latest")

# =========================
# PROMETHEUS METRICS
# =========================
REQUEST_COUNT = Counter(
    "app_requests_total", "Total number of requests"
)

PREDICTION_COUNT = Counter(
    "app_predictions_total", "Total number of predictions"
)

ERROR_COUNT = Counter(
    "app_errors_total", "Total number of errors"
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Request latency"
)

PREDICTION_LATENCY = Histogram(
    "app_prediction_latency_seconds", "Prediction latency"
)

# =========================
# INPUT SCHEMA
# =========================
class InputData(BaseModel):
    data: dict

# =========================
# HEALTH CHECK
# =========================
@app.get("/ping")
def ping():
    REQUEST_COUNT.inc()
    return {"status": "ok"}

# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(input_data: InputData):
    start_time = time.time()
    REQUEST_COUNT.inc()

    try:
        df = pd.DataFrame([input_data.data])

        pred_start = time.time()
        prediction = model.predict(df)
        pred_time = time.time() - pred_start

        PREDICTION_COUNT.inc()
        PREDICTION_LATENCY.observe(pred_time)

        total_time = time.time() - start_time
        REQUEST_LATENCY.observe(total_time)

        return {
            "prediction": prediction.tolist(),
            "latency": total_time
        }

    except Exception as e:
        ERROR_COUNT.inc()
        return {"error": str(e)}

# =========================
# PROMETHEUS ENDPOINT
# =========================
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")