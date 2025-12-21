# cloud_transformer.py

import torch
import torch.nn as nn

class ECGTransformer(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=4, layers=2):
        super().__init__()

        self.embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return torch.sigmoid(x)
