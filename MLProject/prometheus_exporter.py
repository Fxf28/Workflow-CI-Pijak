from prometheus_client import start_http_server, Counter, Histogram
import requests
import time

# =========================
# METRICS
# =========================
REQUEST_COUNT = Counter("mlflow_requests_total", "Total requests")
ERROR_COUNT = Counter("mlflow_errors_total", "Total errors")
LATENCY = Histogram("mlflow_request_latency_seconds", "Request latency")

MODEL_URL = "http://localhost:5000/ping"

# =========================
# LOOP MONITORING
# =========================
def monitor():
    while True:
        start = time.time()
        try:
            res = requests.get(MODEL_URL)

            REQUEST_COUNT.inc()

            if res.status_code != 200:
                ERROR_COUNT.inc()

        except:
            ERROR_COUNT.inc()

        latency = time.time() - start
        LATENCY.observe(latency)

        time.sleep(2)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    start_http_server(8000)  # 🔥 metrics di sini
    monitor()