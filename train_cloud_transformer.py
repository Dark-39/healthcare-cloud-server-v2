# train_cloud_transformer.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from cloud_mitbih_loader import load_record
from cloud_windowing import generate_windows
from cloud_transformer import ECGTransformer
from cloud_config import WINDOW_SAMPLES

# ---------------- Dataset ---------------- #

class ECGWindowDataset(Dataset):
    def __init__(self, record_ids):
        self.samples = []

        for rid in record_ids:
            ecg, ann_samples, ann_symbols = load_record(rid)
            windows, labels = generate_windows(ecg, ann_samples, ann_symbols)

            for w, l in zip(windows, labels):
                self.samples.append((w, l))

        print(f"Total windows: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, label = self.samples[idx]

        window = torch.from_numpy(window).float().unsqueeze(-1)
        label = torch.tensor(label, dtype=torch.float32)

        return window, label


# ---------------- Training ---------------- #

def train():
    # Start small; expand later
    record_ids = ["100", "101", "102", "103", "104"]

    dataset = ECGWindowDataset(record_ids)

    loader = DataLoader(
        dataset,
        batch_size=8,        # GPU friendly
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ECGTransformer(seq_len=WINDOW_SAMPLES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    model.train()

    EPOCHS = 5

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(X).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "cloud_transformer_mitbih.pth")
    print("✅ Model saved as cloud_transformer_mitbih.pth")


if __name__ == "__main__":
    train()
