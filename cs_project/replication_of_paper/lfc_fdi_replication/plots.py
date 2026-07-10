"""
plots.py
========
Publication-style figures reproducing the paper's results.
All figures are written to outputs/.
"""

from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from lfc_model import ThreeAreaLFC, make_disturbances
from attacks import Attack
from dataset import simulate_episode

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.autolayout": False,
})
BLUE, RED, GREEN, ORANGE, GREY = "#1f4e79", "#c0392b", "#27ae60", "#e67e22", "#7f8c8d"


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def plot_disturbances(seed=1, t_end=60.0):
    """RES + load disturbances (Fig. 5 / Fig. 9)."""
    plant = ThreeAreaLFC()
    Pd, Pres = make_disturbances(seed=seed, t_end=t_end)
    t = np.arange(0, t_end, 0.05)
    fig, axes = plt.subplots(3, 1, figsize=(8, 5.5), sharex=True)
    for i, ax in enumerate(axes):
        pl = np.array([Pd(tt)[i] for tt in t])
        pr = np.array([Pres(tt)[i] for tt in t])
        ax.plot(t, pl, color=BLUE, lw=1.3, label="Load $\\Delta P_d$")
        ax.plot(t, pr, color=ORANGE, lw=1.3, label="RES $\\Delta P_{RES}$")
        ax.set_ylabel(f"Area {i+1}\n(p.u.)")
        if i == 0:
            ax.legend(ncol=2, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Random RES and load disturbances", fontweight="bold")
    return _save(fig, "fig_disturbances.png")


def plot_attack_types(t_end=60.0):
    """Pulse / Step / Sine attack waveforms (Fig. 10)."""
    t = np.arange(0, t_end, 0.05)
    specs = [("Pulse Attack", Attack(0, "pulse", 15, 45, mag=0.5), BLUE),
             ("Step Attack",  Attack(0, "step",  20, t_end, mag=0.5), RED),
             ("Sine Attack",  Attack(0, "sine",  10, 55, mag=0.4, freq=2.5), GREEN)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, (title, atk, col) in zip(axes, specs):
        ax.plot(t, atk.signal(t), color=col, lw=1.5)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Attack value")
    fig.suptitle("Three types of FDI attacks", fontweight="bold")
    return _save(fig, "fig_attack_types.png")


def plot_bigru_training(loss_log):
    """BiGRU training RMSE curve (Fig. 6)."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(np.arange(1, len(loss_log)+1), loss_log, color=BLUE, lw=1.6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training RMSE")
    ax.set_title("BiGRU defender training RMSE over epochs", fontweight="bold")
    return _save(fig, "fig_bigru_training.png")


def plot_detection_table(det_metrics: dict):
    """Detection performance comparison table (Table II)."""
    order = ["SVM", "RNN", "LSTM", "GRU", "BiLSTM", "BiGRU", "DL-LSTM"]
    cols = ["Accuracy", "Precision", "Recall", "F1-score"]
    rows = [m for m in order if m in det_metrics]
    data = [[f"{det_metrics[m][c]:.4f}" for c in cols] for m in rows]

    fig, ax = plt.subplots(figsize=(7.2, 0.5*len(rows)+1.2))
    ax.axis("off")
    tbl = ax.table(cellText=data, rowLabels=rows, colLabels=cols,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor(BLUE); tbl[0, j].set_text_props(color="w", weight="bold")
    best = rows.index("DL-LSTM") + 1 if "DL-LSTM" in rows else -1
    if best > 0:
        for j in range(len(cols)):
            tbl[best, j].set_facecolor("#eaf2fb")
    ax.set_title("TABLE II  —  Detection performance of DL-LSTM vs baselines",
                 fontweight="bold", pad=14)
    return _save(fig, "fig_table_detection.png")


def plot_prediction_comparison(pred_metrics: dict):
    """BiGRU vs other predictors (Fig. 7): grouped bars for MSE/RMSE/MAE/R2."""
    order = ["SVM", "LSTM", "GRU", "BiLSTM", "BiGRU"]
    rows = [m for m in order if m in pred_metrics]
    metrics = ["RMSE", "MAE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, met in zip(axes, metrics):
        vals = [pred_metrics[m][met] for m in rows]
        colors = [RED if m == "BiGRU" else GREY for m in rows]
        ax.bar(rows, vals, color=colors)
        ax.set_title(met, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Prediction performance: BiGRU vs baselines", fontweight="bold")
    return _save(fig, "fig_prediction_comparison.png")


def plot_bigru_prediction(scaler, defender, seed=321, t_end=60.0):
    """One-step BiGRU prediction vs actual for df1 and Ptie2 (Fig. 8)."""
    _, clean, _, _, _ = simulate_episode(seed, t_end, attacks=[])
    cs = scaler.transform(clean)
    L = C.WINDOW_LEN
    preds = []
    defender.to("cpu").eval()
    with torch.no_grad():
        for s in range(cs.shape[0] - L):
            w = torch.tensor(cs[s:s+L][None], dtype=torch.float32)
            preds.append(defender(w).numpy()[0])
    preds = scaler.inverse(np.array(preds))
    actual = clean[L:]
    t = np.arange(L, cs.shape[0]) * C.DT_ML

    fig, axes = plt.subplots(2, 1, figsize=(8, 5))
    axes[0].plot(t, actual[:, 0], color=BLUE, lw=1.4, label="Actual")
    axes[0].plot(t, preds[:, 0], "--", color=RED, lw=1.2, label="BiGRU predicted")
    axes[0].set_ylabel("$\\Delta f_1$ (Hz)"); axes[0].legend(fontsize=8)
    axes[0].set_title("(a) Frequency deviation — Area 1")
    axes[1].plot(t, actual[:, 4], color=BLUE, lw=1.4, label="Actual")
    axes[1].plot(t, preds[:, 4], "--", color=RED, lw=1.2, label="BiGRU predicted")
    axes[1].set_ylabel("$\\Delta P_{tie2}$ (p.u.)"); axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8); axes[1].set_title("(b) Tie-line power — Area 2")
    fig.suptitle("BiGRU prediction accuracy on normal operation", fontweight="bold")
    return _save(fig, "fig_bigru_prediction.png")


# --------------------------------------------------------------------------- #
def plot_case(res: dict):
    """4-panel detection+defense figure per case (Figs. 11,13,15,17)."""
    name = res["name"]; t = res["t"]; focus = res["focus"]
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 10), sharex=True)

    # (a) layer-1 classifier output
    axes[0].plot(t, res["lab1"], color=BLUE, lw=1.4, drawstyle="steps-post")
    axes[0].set_yticks([0, 1, 2]); axes[0].set_yticklabels(["Normal", "Single", "Multi"])
    axes[0].set_ylabel("Layer-1"); axes[0].set_title(
        f"(a) First-layer LSTM classifier output", loc="left")

    # (b) per-channel localisation
    for ch in range(C.N_CHANNELS):
        if res["mask"][:, ch].sum() > 0:
            axes[1].plot(t, res["mask"][:, ch]*(ch+1), lw=1.6,
                         drawstyle="steps-post", label=C.CHANNELS[ch])
    axes[1].set_yticks(range(1, C.N_CHANNELS+1)); axes[1].set_yticklabels(C.CHANNELS)
    axes[1].set_ylabel("Layer-2"); axes[1].legend(fontsize=7, ncol=3, loc="upper right")
    axes[1].set_title("(b) Second-layer localisation (flagged channels)", loc="left")

    # (c) defender activation
    axes[2].fill_between(t, 0, res["act"], color=GREEN, alpha=0.4, step="post")
    axes[2].plot(t, res["act"], color=GREEN, lw=1.2, drawstyle="steps-post")
    axes[2].set_ylabel("Enable"); axes[2].set_ylim(-0.1, 1.2)
    axes[2].set_title("(c) BiGRU defender activation signal", loc="left")

    # (d) attacked vs BiGRU-predicted on focus channel
    axes[3].plot(t, res["att_focus"], color=RED, lw=1.1, label="Attacked value")
    axes[3].plot(t, res["pred_focus"], color=BLUE, lw=1.4, label="BiGRU predicted")
    axes[3].plot(t, res["df_clean"][:, focus], color=GREY, lw=1.0, ls=":", label="True (clean)")
    axes[3].set_ylabel(f"$\\Delta f_{focus+1}$ (Hz)"); axes[3].set_xlabel("Time (s)")
    axes[3].legend(fontsize=8, loc="upper right")
    axes[3].set_title("(d) Attacked vs predicted (focus channel)", loc="left")

    fig.suptitle(f"Case {name}: DL-LSTM detection + BiGRU defense",
                 fontweight="bold", y=0.995)
    return _save(fig, f"fig_case_{name}.png")


def plot_case_mitigation(res: dict):
    """Before/after mitigation on the focus area frequency (Figs. 12,14,16,18)."""
    name = res["name"]; t = res["t"]; focus = res["focus"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(t, res["df_att"][:, focus], color=BLUE, lw=1.3, label="Attacked LFC (no defense)")
    ax.plot(t, res["df_def"][:, focus], "--", color=RED, lw=1.4, label="Attack-removal LFC (defended)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(f"$\\Delta f_{focus+1}$ (Hz)")
    ax.legend(fontsize=9)
    ax.set_title(f"Case {name}: frequency stability before vs after mitigation "
                 f"(Area {focus+1}) — {res['reduction_pct']:.0f}% reduction",
                 fontweight="bold")
    return _save(fig, f"fig_case_{name}_mitigation.png")
