import time
import requests
import numpy as np
from edge_mitbih_loader import load_record
from edge_windowing import generate_windows
from sklearn.metrics import classification_report, confusion_matrix

CLOUD_URL = "https://healthcare-cloud-server-v2.onrender.com/analyze_window"
RECORDS = ["100", "101", "102", "103", "104"]

y_true = []
y_pred = []
latencies = []

print("\nEvaluating CLOUD Transformer on REAL MIT-BIH data\n")

for rid in RECORDS:
    print(f"Processing record {rid}")
    ecg, ann_s, ann_sym = load_record(rid)
    windows, labels = generate_windows(ecg, ann_s, ann_sym)

    for w, label in zip(windows, labels):

        payload = {"ecg_window": w.tolist()}

        start = time.perf_counter()
        res = requests.post(CLOUD_URL, json=payload)
        end = time.perf_counter()

        data = res.json()

        prob = data["probability"]
        pred = 1 if prob >= 0.5 else 0  # same threshold logic

        y_true.append(label)
        y_pred.append(pred)
        latencies.append((end - start) * 1000)

print("\n===== CLOUD TRANSFORMER PERFORMANCE (REAL DATA) =====\n")
print(classification_report(y_true, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

latencies = np.array(latencies)

print("\n===== CLOUD LATENCY (END-TO-END) =====")
print(f"Total windows evaluated : {len(latencies)}")
print(f"Average latency         : {latencies.mean():.2f} ms")
print(f"Median latency          : {np.median(latencies):.2f} ms")
print(f"P95 latency             : {np.percentile(latencies, 95):.2f} ms")
print(f"Max latency             : {latencies.max():.2f} ms")
