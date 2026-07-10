"""
models.py
=========
Neural architectures, faithful to Section III of the paper.

DL-LSTM detector  (Section III-A)
---------------------------------
  * Sequence input -> 3 stacked LSTM layers with 256 / 128 / 64 hidden units
    (Eqs. 3-8), each followed by dropout p = 0.01.
  * Final temporal feature -> dense + ReLU (Eq. 9) -> Softmax (Eq. 10).
  * Hierarchical (dual-layer) use, Eqs. (11)-(12):
        classifier1 : X -> {Normal, Single, Multi}
        classifier2 : X -> single-point location  (1-of-6 channel)
        classifier3 : X -> multi-point locations  (multi-label over 6 channels)

BiGRU defender  (Section III-B)
-------------------------------
  * One bidirectional GRU layer, 100 hidden units (Eqs. 13a-13d).
  * Learns the normal next-step mapping  x_hat_t = f(x_{t-1},...,x_{t-tau})
    (Eq. 14) and replaces compromised measurements (Eq. 16).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import config as C

torch.manual_seed(C.SEED)


# --------------------------------------------------------------------------- #
#  DL-LSTM backbone + heads
# --------------------------------------------------------------------------- #
class LSTMBackbone(nn.Module):
    """Three stacked LSTM layers (256/128/64) with dropout, then dense+ReLU."""

    def __init__(self, n_feat: int = C.N_CHANNELS,
                 hidden=C.LSTM_HIDDEN, p_drop: float = C.LSTM_DROPOUT,
                 dense_out: int = 64):
        super().__init__()
        self.l1 = nn.LSTM(n_feat, hidden[0], batch_first=True)
        self.d1 = nn.Dropout(p_drop)
        self.l2 = nn.LSTM(hidden[0], hidden[1], batch_first=True)
        self.d2 = nn.Dropout(p_drop)
        self.l3 = nn.LSTM(hidden[1], hidden[2], batch_first=True)
        self.d3 = nn.Dropout(p_drop)
        self.dense = nn.Linear(hidden[2], dense_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.l1(x); x = self.d1(x)
        x, _ = self.l2(x); x = self.d2(x)
        x, _ = self.l3(x); x = self.d3(x)
        h_T = x[:, -1, :]                    # last-step temporal feature (Eq. 9)
        return self.relu(self.dense(h_T))


class LSTMClassifier(nn.Module):
    """Backbone + Softmax/sigmoid head (Eq. 10)."""

    def __init__(self, n_classes: int, multilabel: bool = False,
                 n_feat: int = C.N_CHANNELS):
        super().__init__()
        self.backbone = LSTMBackbone(n_feat=n_feat)
        self.head = nn.Linear(64, n_classes)
        self.multilabel = multilabel

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)                 # logits (softmax/sigmoid applied in loss)


class DLLSTMDetector(nn.Module):
    """
    Dual-layer LSTM detector: one layer-1 classifier (3-way) plus two layer-2
    classifiers (single-location softmax, multi-location multi-label).
    """

    def __init__(self, n_channels: int = C.N_CHANNELS):
        super().__init__()
        self.clf1 = LSTMClassifier(n_classes=3)                 # Normal/Single/Multi
        self.clf2 = LSTMClassifier(n_classes=n_channels)        # single location
        self.clf3 = LSTMClassifier(n_classes=n_channels,        # multi locations
                                   multilabel=True)

    @torch.no_grad()
    def infer(self, x):
        """
        Hierarchical inference (Eqs. 11-12).
        Returns (layer1_label, per_channel_mask) for a batch.
        """
        self.eval()
        logits1 = self.clf1(x)
        lab1 = logits1.argmax(1)                                # 0/1/2
        B = x.shape[0]
        mask = torch.zeros(B, self.clf2.head.out_features, device=x.device)

        single = lab1 == 1
        multi = lab1 == 2
        if single.any():
            loc = self.clf2(x[single]).argmax(1)
            idx = torch.where(single)[0]
            mask[idx, loc] = 1.0
        if multi.any():
            probs = torch.sigmoid(self.clf3(x[multi]))
            idx = torch.where(multi)[0]
            mask[idx] = (probs > 0.5).float()
            # guarantee at least 2 channels flagged for a multi decision
            for r, row in zip(idx, probs):
                if mask[r].sum() < 2:
                    top2 = torch.topk(row, 2).indices
                    mask[r] = 0.0
                    mask[r, top2] = 1.0
        return lab1, mask


# --------------------------------------------------------------------------- #
#  BiGRU defender
# --------------------------------------------------------------------------- #
class BiGRUDefender(nn.Module):
    """Bidirectional GRU (100 units) predicting the clean next-step vector."""

    def __init__(self, n_feat: int = C.N_CHANNELS, hidden: int = C.BIGRU_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden, n_feat)   # forward+backward -> output

    def forward(self, x):
        out, _ = self.gru(x)                       # (B, L, 2H)
        return self.fc(out[:, -1, :])              # predict next-step (B, n_feat)
