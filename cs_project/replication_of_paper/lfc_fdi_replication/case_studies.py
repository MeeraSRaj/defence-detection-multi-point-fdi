"""
case_studies.py
===============
Reproduces the four attack scenarios of Section IV-C:

  Case A: t=10s  step 0.8 Hz on df1 (Area 1)                       [single-point]
  Case B: t=20s  sine on df1 (Area 1) + pulse 0.2 p.u. on Ptie1    [multi-point]
  Case C: t=30s  step on df1 & df3 + pulse 0.7 p.u. on Ptie2       [multi-point]
  Case D: t=10s  step 0.5 Hz on df2 (removed) ; t=20s pulse 0.6 on Ptie3

For each case it runs three closed-loop simulations of the true plant:
  * clean     : no attack (reference),
  * attacked  : AGC reacts to corrupted telemetry, NO defense,
  * defended  : online DL-LSTM detection + BiGRU correction (Eqs. 14, 16).

It returns per-case time series + detector traces used for the figures and the
frequency-fluctuation reduction (%) reported in the paper.
"""

from __future__ import annotations
import numpy as np
import torch
from collections import deque
from typing import List

import config as C
from lfc_model import ThreeAreaLFC, make_disturbances, N_STATES
from attacks import Attack

CH = {n: i for i, n in enumerate(C.CHANNELS)}   # channel-name -> index


# --------------------------------------------------------------------------- #
#  Paper's exact case definitions
# --------------------------------------------------------------------------- #
def case_attacks(name: str, t_end: float = 60.0) -> List[Attack]:
    if name == "A":
        return [Attack(CH["df1"], "step", 10, t_end, mag=0.8)]
    if name == "B":
        return [Attack(CH["df1"], "sine", 20, t_end, mag=0.30, freq=2.5),
                Attack(CH["Ptie1"], "pulse", 20, t_end, mag=0.20)]
    if name == "C":
        return [Attack(CH["df1"], "step", 30, t_end, mag=0.6),
                Attack(CH["df3"], "step", 30, t_end, mag=0.6),
                Attack(CH["Ptie2"], "pulse", 30, t_end, mag=0.7)]
    if name == "D":
        return [Attack(CH["df2"], "step", 10, 18, mag=0.5),
                Attack(CH["Ptie3"], "pulse", 20, t_end, mag=0.6)]
    raise ValueError(name)

# area whose frequency the paper plots for each case (0-indexed)
CASE_FOCUS_AREA = {"A": 0, "B": 0, "C": 1, "D": 1}


# --------------------------------------------------------------------------- #
def _attack_bias_fun(attacks: List[Attack]):
    """Physical-unit measurement bias (df[3], ptie[3]) from an attack list."""
    def bias(t):
        dfb = np.zeros(3); ptb = np.zeros(3)
        for a in attacks:
            val = a.signal(np.array([t]))[0]
            if a.channel < 3:
                dfb[a.channel] += val
            else:
                ptb[a.channel - 3] += val
        return dfb, ptb
    return bias


# --------------------------------------------------------------------------- #
def run_case(name: str, scaler, detector, defender,
             seed: int = 777, t_end: float = 60.0):
    plant = ThreeAreaLFC()
    Pd, Pres = make_disturbances(seed=seed, t_end=t_end)
    attacks = case_attacks(name, t_end)
    attack_bias = _attack_bias_fun(attacks)

    L = C.WINDOW_LEN
    sub = int(round(C.DT_ML / C.DT_SIM))
    n_ml = int(round(t_end / C.DT_ML)) + 1

    # ---- (1) clean reference ---------------------------------------------- #
    clean = plant.simulate(t_end, C.DT_SIM, Pd, Pres)
    stride = sub
    t_ml   = clean["t"][::stride][:n_ml]
    df_clean = clean["df"][::stride][:n_ml]

    # ---- (2) attacked closed-loop (no defense) ---------------------------- #
    att = plant.simulate(t_end, C.DT_SIM, Pd, Pres, bias_fun=attack_bias)
    df_att = att["df"][::stride][:n_ml]

    # ---- (3) defended closed-loop (online detection + correction) --------- #
    x = np.zeros(N_STATES)
    hist_corr: deque = deque(maxlen=L)     # corrected standardised measurements
    df_def = np.zeros((n_ml, 3))
    trace_lab1 = np.zeros(n_ml, dtype=int)
    trace_mask = np.zeros((n_ml, C.N_CHANNELS))
    trace_act  = np.zeros(n_ml)            # defender activation (any channel)
    pred_focus = np.full(n_ml, np.nan)     # BiGRU prediction on focus df channel
    att_focus  = np.zeros(n_ml)            # attacked value on focus df channel

    # Online, per-step inference runs on CPU regardless of the training device:
    # single-window calls in a Python loop are faster without host<->GPU copies.
    detector.to("cpu").eval(); defender.to("cpu").eval()
    focus = CASE_FOCUS_AREA[name]

    with torch.no_grad():
        for k in range(n_ml):
            t = k * C.DT_ML
            df_t, pt_t = plant.measurements(x)
            df_def[k] = df_t

            dfb, ptb = attack_bias(t)
            meas = np.concatenate([df_t + dfb, pt_t + ptb])     # attacked telemetry
            meas_std = scaler.transform(meas)
            att_focus[k] = meas[focus]

            corrected_std = meas_std.copy()
            lab1 = 0; mask = np.zeros(C.N_CHANNELS)

            if len(hist_corr) == L:
                past = np.array(hist_corr)                      # (L,6)
                det_win = np.vstack([past[1:], meas_std])[None]  # ends at current
                lab_t, mask_t = detector.infer(torch.tensor(det_win, dtype=torch.float32))
                lab1 = int(lab_t.item()); mask = mask_t.numpy()[0]
                pred_std = defender(torch.tensor(past[None], dtype=torch.float32)).numpy()[0]
                pred_phys = scaler.inverse(pred_std)
                pred_focus[k] = pred_phys[focus]
                if lab1 != 0:
                    for ch in range(C.N_CHANNELS):
                        if mask[ch] > 0.5:
                            corrected_std[ch] = pred_std[ch]

            trace_lab1[k] = lab1
            trace_mask[k] = mask
            trace_act[k]  = 1.0 if mask.sum() > 0.5 else 0.0

            corrected = scaler.inverse(corrected_std)
            dfb_ctrl = corrected[:3] - df_t
            ptb_ctrl = corrected[3:] - pt_t
            hist_corr.append(corrected_std)

            # integrate one control interval with held correction bias
            if k < n_ml - 1:
                for s in range(sub):
                    tt = t + s * C.DT_SIM
                    x = plant.rk4_step(tt, x, C.DT_SIM,
                                       Pd(tt), Pres(tt), dfb_ctrl, ptb_ctrl)

    # ---- fluctuation-reduction metric (paper reports % on focus area) ----- #
    aw = t_ml >= (attacks[0].t0)          # attack-active window
    def _fluct(sig):
        s = sig[aw]
        return float(s.max() - s.min())   # peak-to-peak fluctuation
    fl_att = _fluct(df_att[:, focus])
    fl_def = _fluct(df_def[:, focus])
    reduction = 100.0 * (1 - fl_def / max(fl_att, 1e-9))

    return {
        "name": name, "attacks": attacks, "focus": focus, "t": t_ml,
        "df_clean": df_clean, "df_att": df_att, "df_def": df_def,
        "lab1": trace_lab1, "mask": trace_mask, "act": trace_act,
        "pred_focus": pred_focus, "att_focus": att_focus,
        "fluct_attacked": fl_att, "fluct_defended": fl_def,
        "reduction_pct": reduction,
    }
