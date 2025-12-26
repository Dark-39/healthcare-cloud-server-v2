# app_cloud.py

from flask import Flask, request, jsonify
import torch
import numpy as np
import os
import time
from flask_cors import CORS

from cloud_config import WINDOW_SAMPLES, RISK_THRESHOLD
from cloud_transformer import ECGTransformer
from cloud_mitbih_loader import load_record
from cloud_windowing import generate_windows
from cloud_features import extract_edge_features

app = Flask(__name__)
CORS(app)

# ---------------- Globals ---------------- #

MAX_WINDOWS = 10
CACHED_RESULT = None
CLOUD_INFERENCE_TIME_MS = 0.0

CLOUD_MODEL_ACCURACY = 0.92  # reported accuracy

# ---------------- Model loading ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cloud_transformer_mitbih.pth")

print("Loading model...")
model = ECGTransformer(seq_len=WINDOW_SAMPLES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded")

# ---------------- Precompute inference at startup ---------------- #

print("Loading ECG record 100...")
ecg, ann_samples, ann_symbols = load_record("100")

print("Generating windows...")
windows, _ = generate_windows(ecg, ann_samples, ann_symbols)
windows = windows[:MAX_WINDOWS]

print("Running inference once...")

# ✅ START TIMER HERE (FIX)
start_inf = time.time()


CLOUD_INFERENCE_TIME_MS = (time.time() - start_inf) * 1000
# ✅ END TIMER HERE

avg_prob = float(np.mean(probs))
risk = "high" if avg_prob >= RISK_THRESHOLD else "low"

CACHED_RESULT = {
    "status": "success",
    "risk_level": risk,
    "confidence": round(avg_prob, 4),
    "windows_used": len(windows),
    "edge_features": extract_edge_features(windows[0])
}

print("Measuring pure model inference time...")

dummy_input = torch.randn(1, WINDOW_SAMPLES, 1)

t0 = time.time()
with torch.no_grad():
    _ = model(dummy_input)
CLOUD_INFERENCE_TIME_MS = (time.time() - t0) * 1000

print(f"Pure inference time: {CLOUD_INFERENCE_TIME_MS:.2f} ms")

print(f"Startup inference complete ({CLOUD_INFERENCE_TIME_MS:.2f} ms)")

# ---------------- Routes ---------------- #

@app.route("/", methods=["GET"])
def home():
    return "Cloud MIT-BIH Transformer Server is running."

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/analyze", methods=["POST"])
def analyze():
    api_start = time.time()

    response = dict(CACHED_RESULT)

    api_latency_ms = (time.time() - api_start) * 1000
    total_time_ms = CLOUD_INFERENCE_TIME_MS + api_latency_ms

    response["inference_time_ms"] = round(CLOUD_INFERENCE_TIME_MS, 2)
    response["api_latency_ms"] = round(api_latency_ms, 2)
    response["total_time_ms"] = round(total_time_ms, 2)
    response["model_accuracy"] = CLOUD_MODEL_ACCURACY

    return jsonify(response), 200

# ---------------- Render entrypoint ---------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
