"""
train.py
========
Trains and evaluates:
  * the DL-LSTM detector (layer-1 3-class + layer-2 localisation),
  * the BiGRU defender (next-step regression),
  * comparison baselines for Table II (detection) and Fig. 7 (prediction).

All metrics are computed on the held-out 30% test split (Section IV-B).
"""

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

import config as C
from models import DLLSTMDetector, BiGRUDefender, LSTMClassifier

torch.manual_seed(C.SEED)
np.random.seed(C.SEED)


def _pick_device() -> str:
    """CUDA > Apple-silicon Metal (MPS) > CPU.
    Force a choice with e.g.  LFC_DEVICE=cpu  in the environment."""
    override = os.environ.get("LFC_DEVICE", "").lower()
    if override in ("cpu", "cuda", "mps"):
        return override
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()
if DEVICE == "cpu":
    torch.set_num_threads(max(1, os.cpu_count() or 1))
print(f"[train] using device: {DEVICE}")


# --------------------------------------------------------------------------- #
#  Generic recurrent nets for baselines
# --------------------------------------------------------------------------- #
class RecurrentClassifier(nn.Module):
    """Single-layer RNN/LSTM/GRU (optionally bidirectional) -> softmax head."""
    def __init__(self, kind: str, n_classes: int, hidden: int = 64,
                 bidir: bool = False, n_feat: int = C.N_CHANNELS):
        super().__init__()
        rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.rnn = rnn(n_feat, hidden, batch_first=True, bidirectional=bidir)
        self.fc = nn.Linear(hidden * (2 if bidir else 1), n_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class RecurrentRegressor(nn.Module):
    """Single-layer RNN/LSTM/GRU (optionally bidirectional) -> linear head."""
    def __init__(self, kind: str, hidden: int = 64, bidir: bool = False,
                 n_feat: int = C.N_CHANNELS):
        super().__init__()
        rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.rnn = rnn(n_feat, hidden, batch_first=True, bidirectional=bidir)
        self.fc = nn.Linear(hidden * (2 if bidir else 1), n_feat)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


# --------------------------------------------------------------------------- #
#  Low-level training loops
# --------------------------------------------------------------------------- #
def _batches(n, bs, shuffle=True):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def train_classifier(model, X, y, epochs, class_weight=None,
                     multilabel=False, log=None):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(X, dtype=torch.float32)
    if multilabel:
        yt = torch.tensor(y, dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss()
    else:
        yt = torch.tensor(y, dtype=torch.long)
        w = (None if class_weight is None
             else torch.tensor(class_weight, dtype=torch.float32).to(DEVICE))
        lossf = nn.CrossEntropyLoss(weight=w)
    for ep in range(epochs):
        tot = 0.0
        for b in _batches(len(X), C.BATCH):
            opt.zero_grad()
            out = model(Xt[b].to(DEVICE))
            loss = lossf(out, yt[b].to(DEVICE))
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if log is not None:
            log.append(tot / len(X))
    return model


def train_regressor(model, X, Y, epochs, log=None):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    lossf = nn.MSELoss()
    for ep in range(epochs):
        tot = 0.0
        for b in _batches(len(X), C.BATCH):
            opt.zero_grad()
            out = model(Xt[b].to(DEVICE)); loss = lossf(out, Yt[b].to(DEVICE))
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if log is not None:
            # store RMSE for the training-curve figure
            log.append(np.sqrt(tot / len(X)))
    return model


# --------------------------------------------------------------------------- #
#  Detector: DL-LSTM (3 hierarchical classifiers)
# --------------------------------------------------------------------------- #
def _cap(X, *ys, cap):
    if cap is None or len(X) <= cap:
        return (X, *ys)
    idx = np.random.default_rng(C.SEED).choice(len(X), cap, replace=False)
    return (X[idx], *[y[idx] for y in ys])


def train_dllstm_detector(det_data):
    d = det_data
    Xtr, y1tr, yctr = _cap(d["Xtr"], d["y1tr"], d["yctr"], cap=C.DET_CAP)

    det = DLLSTMDetector()

    # class weights for the imbalanced 3-way layer-1 task
    counts = np.bincount(y1tr, minlength=3).astype(float)
    cw = counts.sum() / (3 * np.maximum(counts, 1))

    print("  training layer-1 classifier (Normal/Single/Multi) ...")
    train_classifier(det.clf1, Xtr, y1tr, C.DET_EPOCHS, class_weight=cw)

    # layer-2 single-location classifier: only windows with exactly one channel
    single_mask = y1tr == 1
    if single_mask.sum() > 0:
        Xs = Xtr[single_mask]
        loc = yctr[single_mask].argmax(1)
        print("  training layer-2 single-location classifier ...")
        train_classifier(det.clf2, Xs, loc, C.DET_EPOCHS)

    # layer-2 multi-location classifier: windows with >=2 channels
    multi_mask = y1tr == 2
    if multi_mask.sum() > 0:
        Xm = Xtr[multi_mask]
        ym = yctr[multi_mask].astype(np.float32)
        print("  training layer-2 multi-location classifier ...")
        train_classifier(det.clf3, Xm, ym, C.DET_EPOCHS, multilabel=True)
    return det


# --------------------------------------------------------------------------- #
#  Evaluation helpers
# --------------------------------------------------------------------------- #
def eval_detection(pred, true):
    """Weighted detection metrics on the 3-class layer-1 task (Table II)."""
    return {
        "Accuracy":  accuracy_score(true, pred),
        "Precision": precision_score(true, pred, average="weighted", zero_division=0),
        "Recall":    recall_score(true, pred, average="weighted", zero_division=0),
        "F1-score":  f1_score(true, pred, average="weighted", zero_division=0),
    }


def predict_classifier(model, X):
    model.to(DEVICE).eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        return out.argmax(1).cpu().numpy()


def eval_regression(pred, true):
    err = pred - true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# --------------------------------------------------------------------------- #
#  Baselines for Table II  (detection)
# --------------------------------------------------------------------------- #
def train_detection_baselines(det_data):
    d = det_data
    Xtr, y1tr = _cap(d["Xtr"], d["y1tr"], cap=C.DET_CAP)
    Xte, y1te = d["Xte"], d["y1te"]
    results = {}

    counts = np.bincount(y1tr, minlength=3).astype(float)
    cw = counts.sum() / (3 * np.maximum(counts, 1))

    # ---- SVM on flattened windows (subsample for speed) ----
    print("  baseline: SVM ...")
    sub = np.random.default_rng(C.SEED).choice(len(Xtr), min(2500, len(Xtr)), replace=False)
    svm = SVC(C=2.0, kernel="rbf", class_weight="balanced")
    svm.fit(Xtr[sub].reshape(len(sub), -1), y1tr[sub])
    p = svm.predict(Xte.reshape(len(Xte), -1))
    results["SVM"] = eval_detection(p, y1te)

    # ---- recurrent baselines ----
    specs = [("RNN", "RNN", False), ("LSTM", "LSTM", False),
             ("GRU", "GRU", False), ("BiLSTM", "LSTM", True),
             ("BiGRU", "GRU", True)]
    for name, kind, bidir in specs:
        print(f"  baseline: {name} ...")
        m = RecurrentClassifier(kind, n_classes=3, bidir=bidir)
        train_classifier(m, Xtr, y1tr, C.BASE_EPOCHS, class_weight=cw)
        results[name] = eval_detection(predict_classifier(m, Xte), y1te)
    return results


# --------------------------------------------------------------------------- #
#  Defender: BiGRU + baselines for Fig. 7 (prediction)
# --------------------------------------------------------------------------- #
def train_bigru_defender(def_data, loss_log):
    Xtr, Ytr = _cap(def_data["Xtr"], def_data["Ytr"], cap=C.DEF_CAP)
    model = BiGRUDefender()
    print("  training BiGRU defender ...")
    train_regressor(model, Xtr, Ytr, C.DEF_EPOCHS, log=loss_log)
    return model


def predict_regressor(model, X):
    model.to(DEVICE).eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()


def train_prediction_baselines(def_data):
    Xtr, Ytr = _cap(def_data["Xtr"], def_data["Ytr"], cap=C.DEF_CAP)
    Xte, Yte = def_data["Xte"], def_data["Yte"]
    results = {}

    print("  pred-baseline: SVM ...")
    # SVR per-output on flattened windows (subsample for speed)
    sub = np.random.default_rng(C.SEED).choice(len(Xtr), min(3000, len(Xtr)), replace=False)
    preds = np.zeros_like(Yte)
    for j in range(C.N_CHANNELS):
        svr = SVR(C=2.0, kernel="rbf")
        svr.fit(Xtr[sub].reshape(len(sub), -1), Ytr[sub, j])
        preds[:, j] = svr.predict(Xte.reshape(len(Xte), -1))
    results["SVM"] = eval_regression(preds, Yte)

    specs = [("LSTM", "LSTM", False), ("GRU", "GRU", False),
             ("BiLSTM", "LSTM", True)]
    for name, kind, bidir in specs:
        print(f"  pred-baseline: {name} ...")
        m = RecurrentRegressor(kind, bidir=bidir)
        train_regressor(m, Xtr, Ytr, C.BASE_EPOCHS)
        results[name] = eval_regression(predict_regressor(m, Xte), Yte)
    return results
