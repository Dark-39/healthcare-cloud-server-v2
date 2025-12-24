# app_cloud.py

from flask import Flask, request, jsonify
import torch
import numpy as np
import os

from cloud_config import WINDOW_SAMPLES, RISK_THRESHOLD
from cloud_transformer import ECGTransformer
from cloud_mitbih_loader import load_record
from cloud_windowing import generate_windows
from cloud_features import extract_edge_features

# ---------------- Globals ---------------- #

# Cache to avoid repeated MIT-BIH downloads
RECORD_CACHE = {}

# Limit inference workload per request (CRITICAL for Render)
MAX_WINDOWS = 10

app = Flask(__name__)

# ---------------- Model loading ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cloud_transformer_mitbih.pth")

model = ECGTransformer(seq_len=WINDOW_SAMPLES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---------------- Routes ---------------- #

@app.route("/", methods=["GET"])
def home():
    return "Cloud MIT-BIH Transformer Server is running."

@app.route("/health", methods=["GET"])
def health():
    # Lightweight endpoint for warming Render
    return {"status": "ok"}, 200

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    record_id = data.get("record_id")

    if not record_id:
        return jsonify({
            "status": "error",
            "message": "record_id is required"
        }), 400

    # -------- Load & cache record -------- #
    if record_id in RECORD_CACHE:
        ecg, ann_samples, ann_symbols = RECORD_CACHE[record_id]
    else:
        ecg, ann_samples, ann_symbols = load_record(record_id)
        RECORD_CACHE[record_id] = (ecg, ann_samples, ann_symbols)

    # -------- Windowing -------- #
    windows, _ = generate_windows(ecg, ann_samples, ann_symbols)

    if len(windows) == 0:
        return jsonify({
            "status": "error",
            "message": "No valid ECG windows generated"
        }), 400

    # -------- Limit workload -------- #
    windows = windows[:MAX_WINDOWS]

    # -------- Inference -------- #
    probs = []
    for w in windows:
        t = torch.tensor(w, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            probs.append(model(t).item())

    avg_prob = float(np.mean(probs))
    risk = "high" if avg_prob >= RISK_THRESHOLD else "low"

    return jsonify({
        "status": "success",
        "risk_level": risk,
        "confidence": round(avg_prob, 4),
        "windows_used": len(windows),
        "edge_features": extract_edge_features(windows[0])
    }), 200

# ---------------- Render entrypoint ---------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
