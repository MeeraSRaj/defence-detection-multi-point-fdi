"""
live_sim.py
===========
Real-time, animated view of the three-area Load-Frequency-Control (LFC) system
while a multi-point FDI attack is injected and the DL-LSTM detector + BiGRU
defender act online.  This is the *live* counterpart to the static case-study
figures produced by main.py.

For each of the three control areas you watch, drawn as the simulation runs:
    * CLEAN     frequency deviation (no attack)              -- what should happen
    * ATTACKED  deviation (attack, no defense)               -- the damage
    * DEFENDED  deviation (detector + BiGRU online)          -- the mitigation
plus a live status panel: the clock, an attack indicator, the detector's verdict
(Normal / Single / Multi + which channels), the defender state, and the running
peak-to-peak fluctuation reduction on the focus area.

Run this AFTER  python3 main.py  (which trains and saves the models):

    python3 live_sim.py                      # Case A, real time
    python3 live_sim.py --case C             # heavier multi-point attack
    python3 live_sim.py --case D --speed 3   # 3x faster playback
    python3 live_sim.py --case B --seconds 40

Options:
    --case  A|B|C|D        which paper case study to play      (default A)
    --speed FLOAT          playback speed multiplier           (default 1.0)
    --seconds FLOAT        simulated seconds to run            (default 60)
    --device cpu|mps|cuda  inference device                    (default cpu)
"""

from __future__ import annotations
import os
import argparse
from collections import deque

import numpy as np
import torch

import config as C
from lfc_model import ThreeAreaLFC, make_disturbances, N_STATES
from dataset import Scaler
from models import DLLSTMDetector, BiGRUDefender
from case_studies import case_attacks, _attack_bias_fun, CASE_FOCUS_AREA

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "outputs", "trained_models.pt")
CH_NAMES = ["df1", "df2", "df3", "Ptie1", "Ptie2", "Ptie3"]

# colour palette (matches the static figures)
BLUE, RED, GREEN, GREY = "#1f5fbf", "#c0392b", "#1e8449", "#7f8c8d"


def load_models(device: str = "cpu"):
    """Load the trained detector, defender and scaler from the checkpoint."""
    if not os.path.exists(CKPT):
        print("[live_sim] No outputs/trained_models.pt found -- run  python3 main.py  first.")
        print("[live_sim] Showing CLEAN vs ATTACKED only (no online defense).")
        return None, None, None
    ck = torch.load(CKPT, map_location="cpu")
    det = DLLSTMDetector(); det.load_state_dict(ck["detector"]); det.to(device).eval()
    dfn = BiGRUDefender();  dfn.load_state_dict(ck["defender"]); dfn.to(device).eval()
    mean = np.asarray(ck["scaler_mean"], dtype=np.float64)
    std = np.asarray(ck["scaler_std"], dtype=np.float64)
    print("[live_sim] loaded trained detector + BiGRU defender")
    return det, dfn, Scaler(mean=mean, std=std)


# --------------------------------------------------------------------------- #
class LiveLFC:
    """Steps three parallel plants (clean / attacked / defended) one control
    interval at a time, running detection + defense online on the defended one.
    Front-ends (the animation, or a headless test) just call `step()`."""

    def __init__(self, case="A", t_end=None, device="cpu",
                 detector=None, defender=None, scaler=None, dist_seed=777):
        self.case = case
        self.t_end = float(t_end if t_end is not None else C.T_END)
        self.device = device
        self.detector, self.defender, self.scaler = detector, defender, scaler
        self.have_models = detector is not None and scaler is not None

        self.plant = ThreeAreaLFC()
        self.Pd, self.Pres = make_disturbances(seed=dist_seed, t_end=self.t_end)
        self.attacks = case_attacks(case, self.t_end)
        self.attack_bias = _attack_bias_fun(self.attacks)
        self.focus = CASE_FOCUS_AREA[case]
        self.t0_attack = min(a.t0 for a in self.attacks)

        self.L = C.WINDOW_LEN
        self.sub = int(round(C.DT_ML / C.DT_SIM))
        # record n_ml points at t = 0, DT_ML, ..., t_end (matches run_case),
        # integrating one control interval *between* consecutive samples.
        self.n_ml = int(round(self.t_end / C.DT_ML)) + 1
        self.t_grid = np.arange(self.n_ml) * C.DT_ML

        self.x_clean = np.zeros(N_STATES)
        self.x_att = np.zeros(N_STATES)
        self.x_def = np.zeros(N_STATES)
        self.hist = deque(maxlen=self.L)

        self.df_clean = np.full((self.n_ml, 3), np.nan)
        self.df_att = np.full((self.n_ml, 3), np.nan)
        self.df_def = np.full((self.n_ml, 3), np.nan)

        self.k = 0
        self.verdict = "Normal"
        self.flagged: list[str] = []
        self.defending = False

    @property
    def done(self) -> bool:
        return self.k >= self.n_ml

    @torch.no_grad()
    def step(self):
        """Advance the simulation by one DT_ML control interval."""
        if self.done:
            return
        k = self.k
        t = k * C.DT_ML

        dfc, _ = self.plant.measurements(self.x_clean)
        dfa, _ = self.plant.measurements(self.x_att)
        dfd, ptd = self.plant.measurements(self.x_def)
        self.df_clean[k] = dfc; self.df_att[k] = dfa; self.df_def[k] = dfd

        dfb0, ptb0 = self.attack_bias(t)
        meas = np.concatenate([dfd + dfb0, ptd + ptb0])          # attacked telemetry
        meas_std = self.scaler.transform(meas) if self.have_models else meas
        corrected_std = meas_std.copy()
        self.verdict, self.flagged, self.defending = "Normal", [], False

        if self.have_models and len(self.hist) == self.L:
            past = np.array(self.hist)
            det_win = np.vstack([past[1:], meas_std])[None]
            lab_t, mask_t = self.detector.infer(
                torch.tensor(det_win, dtype=torch.float32).to(self.device))
            lab1 = int(lab_t.item()); mask = mask_t.cpu().numpy()[0]
            pred_std = self.defender(
                torch.tensor(past[None], dtype=torch.float32).to(self.device)
            ).cpu().numpy()[0]
            self.verdict = ["Normal", "Single", "Multi"][lab1]
            if lab1 != 0:
                for ch in range(C.N_CHANNELS):
                    if mask[ch] > 0.5:
                        corrected_std[ch] = pred_std[ch]
                        self.flagged.append(CH_NAMES[ch])
                self.defending = len(self.flagged) > 0

        if self.have_models:
            corrected = self.scaler.inverse(corrected_std)
            dfb_ctrl = corrected[:3] - dfd
            ptb_ctrl = corrected[3:] - ptd
            self.hist.append(corrected_std)
        else:
            dfb_ctrl = ptb_ctrl = np.zeros(3)

        if k < self.n_ml - 1:
            for s in range(self.sub):
                tt = t + s * C.DT_SIM
                dfb, ptb = self.attack_bias(tt)
                self.x_clean[:] = self.plant.rk4_step(tt, self.x_clean, C.DT_SIM,
                                                      self.Pd(tt), self.Pres(tt))
                self.x_att[:] = self.plant.rk4_step(tt, self.x_att, C.DT_SIM,
                                                    self.Pd(tt), self.Pres(tt), dfb, ptb)
                self.x_def[:] = self.plant.rk4_step(tt, self.x_def, C.DT_SIM,
                                                    self.Pd(tt), self.Pres(tt),
                                                    dfb_ctrl, ptb_ctrl)
        self.k = k + 1

    def reduction_pct(self) -> float:
        """Running peak-to-peak reduction on the focus area (%)."""
        k = self.k
        if k < 4:
            return 0.0
        aw = self.t_grid[:k] >= self.t0_attack
        if not aw.any():
            return 0.0
        a = self.df_att[:k, self.focus][aw]; d = self.df_def[:k, self.focus][aw]
        fl_a = np.nanmax(a) - np.nanmin(a); fl_d = np.nanmax(d) - np.nanmin(d)
        return 100.0 * (1 - fl_d / max(fl_a, 1e-9))

    def attacked_channels_now(self) -> str:
        t = self.k * C.DT_ML
        act = [f"{CH_NAMES[a.channel]}({a.kind})"
               for a in self.attacks if a.t0 <= t <= a.t1]
        return ", ".join(act) if act else "-"


# --------------------------------------------------------------------------- #
def run_animation(sim: LiveLFC, speed: float):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1.15], hspace=0.45, wspace=0.25)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_panel = fig.add_subplot(gs[:, 1]); ax_panel.axis("off")
    fig.suptitle(f"Live LFC simulation  -  Case {sim.case}  "
                 f"(three-area smart grid under multi-point FDI attack)",
                 fontweight="bold")

    lc_, la_, ld_ = [], [], []
    for i, ax in enumerate(axes):
        a, = ax.plot([], [], color=GREY, lw=1.3, label="clean (no attack)")
        b, = ax.plot([], [], color=RED, lw=1.6, label="attacked")
        c, = ax.plot([], [], color=GREEN, lw=1.6, label="defended")
        lc_.append(a); la_.append(b); ld_.append(c)
        ax.set_ylabel(f"$\\Delta f_{i+1}$ (Hz)")
        ax.set_xlim(0, sim.t_end); ax.set_ylim(-0.6, 0.6)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4); ax.grid(alpha=0.25)
        if i == sim.focus:
            ax.set_facecolor("#f4f8ff")
            ax.set_title(f"Area {i+1}  (focus area for Case {sim.case})",
                         loc="left", fontsize=9, color=BLUE)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.9)

    clock = ax_panel.text(0.05, 0.96, "", fontsize=13, fontweight="bold",
                          va="top", family="monospace", transform=ax_panel.transAxes)
    leds, led_txt = [], []
    for j, lab in enumerate(["ATTACK", "DETECTOR", "DEFENSE"]):
        y = 0.82 - j * 0.10
        circ = plt.Circle((0.10, y), 0.026, transform=ax_panel.transAxes,
                          color=GREY, ec="k", lw=0.6)
        ax_panel.add_patch(circ); leds.append(circ)
        led_txt.append(ax_panel.text(0.20, y, lab, fontsize=10, va="center",
                       transform=ax_panel.transAxes, family="monospace"))
    info = ax_panel.text(0.05, 0.46, "", fontsize=9, va="top",
                         family="monospace", transform=ax_panel.transAxes)
    metric = ax_panel.text(0.05, 0.18, "", fontsize=11, va="top",
                           fontweight="bold", transform=ax_panel.transAxes)

    steps_per_frame = max(1, int(round(speed)))

    def update(_):
        for _ in range(steps_per_frame):
            if sim.done:
                break
            sim.step()
        k = sim.k; xs = sim.t_grid[:k]
        for i in range(3):
            lc_[i].set_data(xs, sim.df_clean[:k, i])
            la_[i].set_data(xs, sim.df_att[:k, i])
            ld_[i].set_data(xs, sim.df_def[:k, i])
        t_now = k * C.DT_ML
        clock.set_text(f" t = {t_now:5.1f} s")
        dfb, ptb = sim.attack_bias(t_now)
        attack_on = (abs(dfb).sum() + abs(ptb).sum()) > 1e-9
        leds[0].set_color(RED if attack_on else GREY)
        leds[1].set_color("#e67e22" if sim.verdict != "Normal" else GREY)
        leds[2].set_color(GREEN if sim.defending else GREY)
        led_txt[1].set_text(f"DETECTOR: {sim.verdict}")
        led_txt[2].set_text("DEFENSE: active" if sim.defending else "DEFENSE: idle")
        flg = ", ".join(sim.flagged) if sim.flagged else "-"
        info.set_text(f"attacked channels now:\n  {sim.attacked_channels_now()}\n\n"
                      f"channels being corrected:\n  {flg}")
        red = sim.reduction_pct()
        if k > 3:
            aw = sim.t_grid[:k] >= sim.t0_attack
            fa = fd = 0.0
            if aw.any():
                fa = np.nanmax(sim.df_att[:k, sim.focus][aw]) - np.nanmin(sim.df_att[:k, sim.focus][aw])
                fd = np.nanmax(sim.df_def[:k, sim.focus][aw]) - np.nanmin(sim.df_def[:k, sim.focus][aw])
            metric.set_text(f"focus area {sim.focus+1}:\n"
                            f" attacked p-p = {fa:5.3f} Hz\n"
                            f" defended p-p = {fd:5.3f} Hz\n"
                            f" reduction   = {red:4.0f}%")
        return (*lc_, *la_, *ld_, clock, info, metric, *leds, *led_txt)

    total_frames = int(np.ceil(sim.n_ml / steps_per_frame)) + 1
    interval_ms = max(1, int(1000 * C.DT_ML / max(speed, 0.01)))
    _anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                    interval=interval_ms, blit=False, repeat=False)
    print(f"[live_sim] playing Case {sim.case} "
          f"({sim.t_end:.0f}s, speed x{speed}). Close the window to exit.")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Live LFC + FDI + defense animation")
    ap.add_argument("--case", default="A", choices=["A", "B", "C", "D"])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=C.T_END)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()
    torch.manual_seed(C.SEED)
    det, dfn, scaler = load_models(args.device)
    sim = LiveLFC(case=args.case, t_end=args.seconds, device=args.device,
                  detector=det, defender=dfn, scaler=scaler)
    run_animation(sim, args.speed)


if __name__ == "__main__":
    main()
