"""
control_room.py
===============
Interactive "control-room" dashboard for the three-area Load-Frequency-Control
(LFC) smart grid, the FDI attack, and the DL-LSTM + BiGRU cyber-defense.

Unlike the static figures, this is a live, *adjustable* simulator:

  * a labelled BLOCK DIAGRAM of the control system we are using
    (droop, governor, turbine, power system, PI/AGC, tie-lines, and the
    cyber layer that cleans the telemetry before it reaches the AGC);
  * live scrolling plots of the frequency deviation of all three areas
    (clean / attacked / defended);
  * sliders and buttons you can move WHILE it runs:
        - attack magnitude and type (step / pulse / sine)
        - which measurement channels are attacked (df1..Ptie3)
        - controller gains  Kp, Ki  (how aggressive the AGC is)
        - disturbance level (load / renewables)
        - playback speed
        - ATTACK on/off, DEFENSE on/off, Pause, Reset.

Run it (after  python3 main.py  has trained + saved the models):

    python3 control_room.py

If no window appears, install a GUI backend:  pip install PyQt5
"""

from __future__ import annotations
import os
from collections import deque

import numpy as np
import torch

import config as C
from lfc_model import ThreeAreaLFC, make_disturbances, N_STATES
from dataset import Scaler
from models import DLLSTMDetector, BiGRUDefender

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "outputs", "trained_models.pt")
CH_NAMES = ["df1", "df2", "df3", "Ptie1", "Ptie2", "Ptie3"]

# palette
BLUE, RED, GREEN, GREY, DARK = "#1f5fbf", "#c0392b", "#1e8449", "#95a5a6", "#2c3e50"
PANEL, ACCENT = "#f4f8ff", "#e67e22"


# --------------------------------------------------------------------------- #
def load_models(device="cpu"):
    if not os.path.exists(CKPT):
        return None, None, None
    ck = torch.load(CKPT, map_location="cpu")
    det = DLLSTMDetector(); det.load_state_dict(ck["detector"]); det.to(device).eval()
    dfn = BiGRUDefender();  dfn.load_state_dict(ck["defender"]); dfn.to(device).eval()
    mean = np.asarray(ck["scaler_mean"], dtype=np.float64)
    std = np.asarray(ck["scaler_std"], dtype=np.float64)
    return det, dfn, Scaler(mean=mean, std=std)


# --------------------------------------------------------------------------- #
class InteractiveLFC:
    """Live, adjustable three-area LFC engine (clean / attacked / defended)."""

    def __init__(self, detector=None, defender=None, scaler=None,
                 device="cpu", window_s=20.0):
        self.detector, self.defender, self.scaler = detector, defender, scaler
        self.device = device
        self.have_models = detector is not None and scaler is not None
        self.L = C.WINDOW_LEN
        self.sub = int(round(C.DT_ML / C.DT_SIM))
        self.window_s = window_s
        self.maxlen = int(window_s / C.DT_ML)

        # live-adjustable parameters (mutated by the UI widgets)
        self.P = dict(atk_mag=0.5, atk_type="step", atk_freq=2.5,
                      channels={0}, kp_scale=1.0, ki_scale=1.0,
                      dist_scale=1.0, attack_on=False, defense_on=True)

        self.plant = ThreeAreaLFC()
        self.reset()

    # -- (re)initialise state and buffers ----------------------------------- #
    def reset(self):
        self.Pd, self.Pres = make_disturbances(seed=777, t_end=120.0)
        self.x_clean = np.zeros(N_STATES)
        self.x_att = np.zeros(N_STATES)
        self.x_def = np.zeros(N_STATES)
        self.hist = deque(maxlen=self.L)
        self.t = 0.0
        self.attack_started_at = None
        self.verdict, self.flagged, self.defending = "Normal", [], False
        self.tb = deque(maxlen=self.maxlen)
        self.cb = [deque(maxlen=self.maxlen) for _ in range(3)]
        self.ab = [deque(maxlen=self.maxlen) for _ in range(3)]
        self.db = [deque(maxlen=self.maxlen) for _ in range(3)]

    # -- current attack bias (physical units) ------------------------------- #
    def _attack_bias(self, t):
        dfb = np.zeros(3); ptb = np.zeros(3)
        if not self.P["attack_on"] or not self.P["channels"]:
            return dfb, ptb
        t0 = self.attack_started_at if self.attack_started_at is not None else t
        dt = t - t0
        mag = self.P["atk_mag"]
        if self.P["atk_type"] == "step":
            val = mag
        elif self.P["atk_type"] == "pulse":
            val = mag if (dt % 1.0) < 0.5 else 0.0
        else:  # sine
            val = mag * np.sin(self.P["atk_freq"] * dt)
        for ch in self.P["channels"]:
            if ch < 3:
                dfb[ch] += val
            else:
                ptb[ch - 3] += val
        return dfb, ptb

    # -- advance one control interval (DT_ML) ------------------------------- #
    @torch.no_grad()
    def step(self):
        # push live gain scaling into the plant
        self.plant.kp_scale = self.P["kp_scale"]
        self.plant.ki_scale = self.P["ki_scale"]
        ds = self.P["dist_scale"]

        t = self.t
        dfd, ptd = self.plant.measurements(self.x_def)
        dfc, _ = self.plant.measurements(self.x_clean)
        dfa, _ = self.plant.measurements(self.x_att)

        dfb0, ptb0 = self._attack_bias(t)
        meas = np.concatenate([dfd + dfb0, ptd + ptb0])
        self.verdict, self.flagged, self.defending = "Normal", [], False

        dfb_ctrl = dfb0.copy(); ptb_ctrl = ptb0.copy()   # default: no cleaning
        if self.have_models and self.P["defense_on"]:
            meas_std = self.scaler.transform(meas)
            corrected_std = meas_std.copy()
            if len(self.hist) == self.L:
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
            corrected = self.scaler.inverse(corrected_std)
            dfb_ctrl = corrected[:3] - dfd
            ptb_ctrl = corrected[3:] - ptd
            self.hist.append(corrected_std)
        elif self.have_models:
            # defense off but keep history warm so toggling on works instantly
            self.hist.append(self.scaler.transform(meas))

        # record for plotting (focus on true frequencies)
        self.tb.append(t)
        for i in range(3):
            self.cb[i].append(dfc[i]); self.ab[i].append(dfa[i]); self.db[i].append(dfd[i])

        # integrate one control interval
        for s in range(self.sub):
            tt = t + s * C.DT_SIM
            dfb, ptb = self._attack_bias(tt)
            pd = self.Pd(tt) * ds; pr = self.Pres(tt) * ds
            self.x_clean[:] = self.plant.rk4_step(tt, self.x_clean, C.DT_SIM, pd, pr)
            self.x_att[:] = self.plant.rk4_step(tt, self.x_att, C.DT_SIM, pd, pr, dfb, ptb)
            self.x_def[:] = self.plant.rk4_step(tt, self.x_def, C.DT_SIM, pd, pr,
                                                dfb_ctrl, ptb_ctrl)
        self.t = t + C.DT_ML

    # -- helpers for the UI ------------------------------------------------- #
    def set_attack(self, on: bool):
        self.P["attack_on"] = on
        self.attack_started_at = self.t if on else None

    def reduction_now(self):
        if len(self.tb) < 5:
            return 0.0
        f = min(self.P["channels"]) if self.P["channels"] else 0
        f = f if f < 3 else 0
        a = np.array(self.ab[f]); d = np.array(self.db[f])
        fa = a.max() - a.min(); fd = d.max() - d.min()
        return 100.0 * (1 - fd / max(fa, 1e-9))


# --------------------------------------------------------------------------- #
#  Block diagram of the control system (drawn once)
# --------------------------------------------------------------------------- #
def draw_block_diagram(ax, engine=None):
    import matplotlib.patches as mp
    ax.clear()
    ax.set_xlim(0, 10); ax.set_ylim(-0.1, 4.4); ax.axis("off")
    ax.set_title("Control system: one LFC area (of three) + cyber layer",
                 fontsize=10, fontweight="bold", color=DARK, loc="left")

    def box(x, y, w, h, label, fc="#ffffff", ec=DARK, fs=8, bold=False):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color=DARK)

    def arrow(p1, p2, color=DARK, lw=1.4, ls="-"):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls))

    atk_on = bool(engine.P["attack_on"]) if engine is not None else True
    def_on = bool(engine.P["defense_on"]) if engine is not None else True

    # ---- forward path:  PI -> governor -> turbine -> power system -> df ---- #
    y = 2.9
    box(0.2, y, 1.5, 0.7, "PI / AGC\n$K_p+K_i/s$", fc="#eaf2ff", bold=True)
    box(2.2, y, 1.5, 0.7, "Governor\n$1/(1{+}sT_g)$", fc="#f0f0f0")
    box(4.2, y, 1.5, 0.7, "Turbine\n$1/(1{+}sT_t)$", fc="#f0f0f0")
    box(6.5, y, 1.9, 0.7, "Power system\n$1/(Ms{+}D)$", fc="#fdeeee", bold=True)
    arrow((1.7, y + 0.35), (2.2, y + 0.35))
    arrow((3.7, y + 0.35), (4.2, y + 0.35))
    arrow((5.7, y + 0.35), (6.5, y + 0.35))
    arrow((8.4, y + 0.35), (9.5, y + 0.35), color=BLUE, lw=2.2)
    ax.text(9.55, y + 0.35, r"$\Delta f$", color=BLUE, fontsize=12,
            fontweight="bold", va="center")

    # disturbance into the power system
    ax.text(7.45, 4.2, r"$\Delta P_d,\ \Delta P_{res}$  (load / RES)",
            ha="center", fontsize=8, color=DARK)
    arrow((7.45, 4.05), (7.45, y + 0.7), color=DARK, ls="--")

    # df feeds the tie-lines
    arrow((9.1, y), (9.1, 1.55), color=GREEN)

    # ---- telemetry / feedback row (RIGHT -> LEFT):  df,Ptie measured,
    #      FDI attack corrupts it, detector+BiGRU cleans it, then ACE -> PI --- #
    yb = 0.95
    box(6.6, yb, 1.9, 0.7, "Tie-lines (AC+HVDC)\n$2\\pi T_{ij}\\!\\int(\\Delta f_i{-}\\Delta f_j)$",
        fc="#eefaf1", fs=7)
    box(4.7, yb, 1.5, 0.7, "FDI attack\ninject on $z$",
        fc="#fdecea" if atk_on else "#f4f4f4",
        ec=RED if atk_on else GREY, fs=8, bold=atk_on)
    box(2.8, yb, 1.5, 0.7, "Detector\n+ BiGRU",
        fc="#eafaf0" if def_on else "#f4f4f4",
        ec=GREEN if def_on else GREY, fs=8, bold=def_on)
    box(0.2, yb, 1.5, 0.7, "ACE\n$\\beta\\Delta f{+}\\Delta P_{tie}$", fc="#fff7e6", fs=8)

    arrow((6.6, yb + 0.35), (6.2, yb + 0.35), color=DARK)                 # meas -> attack
    arrow((4.7, yb + 0.35), (4.3, yb + 0.35),
          color=RED if atk_on else GREY)                                  # attack -> defense
    arrow((2.8, yb + 0.35), (1.7, yb + 0.35),
          color=GREEN if def_on else GREY)                                # defense -> ACE
    arrow((0.95, yb + 0.7), (0.95, y), color=DARK)                        # ACE -> PI
    ax.text(3.55, yb + 0.95, "telemetry cleaned\nbefore AGC", ha="center",
            fontsize=7, color=GREEN if def_on else GREY)
    if atk_on:
        ax.text(5.45, yb + 0.92, "\u26a1", fontsize=13, ha="center", color=RED)

    # ---- droop 1/R feedback along the bottom (df -> governor) -------------- #
    arrow((9.1, 1.55), (9.1, 0.25), color=GREY)
    arrow((9.1, 0.25), (2.95, 0.25), color=GREY)
    arrow((2.95, 0.25), (2.95, y), color=GREY)
    ax.text(6.0, 0.12, r"droop  $1/R$   ($\Delta f \rightarrow$ governor)",
            ha="center", fontsize=7, color=GREY)


# --------------------------------------------------------------------------- #
def run_dashboard(engine: InteractiveLFC):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title("LFC control room  -  FDI attack & defense")
    gs = fig.add_gridspec(3, 2, width_ratios=[2.3, 1.0],
                          height_ratios=[1.05, 1.5, 1.0], hspace=0.42, wspace=0.18,
                          left=0.06, right=0.985, top=0.95, bottom=0.06)

    ax_diag = fig.add_subplot(gs[0, :])
    draw_block_diagram(ax_diag, engine)

    ax_f = [fig.add_subplot(gs[1, 0]) if i == 0 else None for i in range(1)]
    # three stacked frequency plots share the middle-left cell -> use a sub-grid
    sub = gs[1, 0].subgridspec(3, 1, hspace=0.12)
    axf = [fig.add_subplot(sub[i, 0]) for i in range(3)]
    lc, la, ld = [], [], []
    for i, ax in enumerate(axf):
        a, = ax.plot([], [], color=GREY, lw=1.2, label="clean")
        b, = ax.plot([], [], color=RED, lw=1.5, label="attacked")
        c, = ax.plot([], [], color=GREEN, lw=1.6, label="defended")
        lc.append(a); la.append(b); ld.append(c)
        ax.set_ylabel(f"$\\Delta f_{i+1}$", fontsize=9)
        ax.set_ylim(-0.6, 0.6); ax.grid(alpha=0.25)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        if i < 2:
            ax.set_xticklabels([])
    axf[0].legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
    axf[-1].set_xlabel("time (s)")

    # status panel (middle-right)
    ax_stat = fig.add_subplot(gs[1, 1]); ax_stat.axis("off")
    leds, led_txt = [], []
    for j, lab in enumerate(["ATTACK", "DETECTOR", "DEFENSE"]):
        yy = 0.9 - j * 0.12
        circ = plt.Circle((0.10, yy), 0.03, transform=ax_stat.transAxes,
                          color=GREY, ec="k", lw=0.6); ax_stat.add_patch(circ)
        leds.append(circ)
        led_txt.append(ax_stat.text(0.2, yy, lab, transform=ax_stat.transAxes,
                       va="center", family="monospace", fontsize=10))
    stat_txt = ax_stat.text(0.03, 0.48, "", transform=ax_stat.transAxes,
                            va="top", family="monospace", fontsize=9)
    metric_txt = ax_stat.text(0.03, 0.2, "", transform=ax_stat.transAxes,
                              va="top", family="monospace", fontsize=11,
                              fontweight="bold")

    # ---- controls (bottom band) ------------------------------------------ #
    def sax(x, y, w, h):
        a = fig.add_axes([x, y, w, h]); return a

    s_mag = Slider(sax(0.09, 0.30, 0.30, 0.02), "attack mag", 0.0, 1.0,
                   valinit=engine.P["atk_mag"])
    s_kp = Slider(sax(0.09, 0.26, 0.30, 0.02), "Kp x", 0.0, 3.0,
                  valinit=1.0)
    s_ki = Slider(sax(0.09, 0.22, 0.30, 0.02), "Ki x", 0.0, 3.0,
                  valinit=1.0)
    s_dist = Slider(sax(0.09, 0.18, 0.30, 0.02), "disturbance", 0.0, 3.0,
                    valinit=1.0)
    s_speed = Slider(sax(0.09, 0.14, 0.30, 0.02), "speed", 1, 8, valinit=2,
                     valstep=1)

    r_type = RadioButtons(sax(0.44, 0.14, 0.11, 0.16),
                          ["step", "pulse", "sine"], active=0)
    r_type.ax.set_title("attack type", fontsize=8)

    chk = CheckButtons(sax(0.57, 0.14, 0.12, 0.16), CH_NAMES,
                       [i in engine.P["channels"] for i in range(6)])
    chk.ax.set_title("channels", fontsize=8)

    b_atk = Button(sax(0.72, 0.28, 0.11, 0.035), "ATTACK: off", color="#f4d7d3")
    b_def = Button(sax(0.72, 0.235, 0.11, 0.035), "DEFENSE: on", color="#d4f0dd")
    b_pause = Button(sax(0.72, 0.19, 0.11, 0.035), "Pause")
    b_reset = Button(sax(0.72, 0.145, 0.11, 0.035), "Reset")

    running = {"on": True}

    # ---- widget callbacks ------------------------------------------------- #
    def on_mag(v): engine.P["atk_mag"] = float(v)
    def on_kp(v): engine.P["kp_scale"] = float(v)
    def on_ki(v): engine.P["ki_scale"] = float(v)
    def on_dist(v): engine.P["dist_scale"] = float(v)
    s_mag.on_changed(on_mag); s_kp.on_changed(on_kp)
    s_ki.on_changed(on_ki); s_dist.on_changed(on_dist)

    def on_type(lbl): engine.P["atk_type"] = lbl
    r_type.on_clicked(on_type)

    def on_chk(lbl):
        i = CH_NAMES.index(lbl)
        if i in engine.P["channels"]:
            engine.P["channels"].discard(i)
        else:
            engine.P["channels"].add(i)
    chk.on_clicked(on_chk)

    def on_atk(_):
        engine.set_attack(not engine.P["attack_on"])
        b_atk.label.set_text(f"ATTACK: {'on' if engine.P['attack_on'] else 'off'}")
        b_atk.color = "#e74c3c" if engine.P["attack_on"] else "#f4d7d3"
        draw_block_diagram(ax_diag, engine)
    b_atk.on_clicked(on_atk)

    def on_def(_):
        engine.P["defense_on"] = not engine.P["defense_on"]
        b_def.label.set_text(f"DEFENSE: {'on' if engine.P['defense_on'] else 'off'}")
        b_def.color = "#2ecc71" if engine.P["defense_on"] else "#e6e6e6"
        draw_block_diagram(ax_diag, engine)
    b_def.on_clicked(on_def)

    def on_pause(_):
        running["on"] = not running["on"]
        b_pause.label.set_text("Resume" if not running["on"] else "Pause")
    b_pause.on_clicked(on_pause)

    def on_reset(_):
        engine.reset()
        b_atk.label.set_text("ATTACK: off"); b_atk.color = "#f4d7d3"
    b_reset.on_clicked(on_reset)

    if not engine.have_models:
        stat_txt.set_text("No trained models found.\nRun  python3 main.py  first\n"
                          "to enable the defense.")

    # ---- animation -------------------------------------------------------- #
    def update(_):
        if running["on"]:
            for _ in range(int(s_speed.val)):
                engine.step()
        if len(engine.tb) < 2:
            return []
        ts = np.array(engine.tb)
        for i in range(3):
            lc[i].set_data(ts, np.array(engine.cb[i]))
            la[i].set_data(ts, np.array(engine.ab[i]))
            ld[i].set_data(ts, np.array(engine.db[i]))
            axf[i].set_xlim(max(0, ts[-1] - engine.window_s), max(engine.window_s, ts[-1]))
        leds[0].set_color(RED if engine.P["attack_on"] else GREY)
        leds[1].set_color(ACCENT if engine.verdict != "Normal" else GREY)
        leds[2].set_color(GREEN if engine.defending else GREY)
        led_txt[1].set_text(f"DETECTOR: {engine.verdict}")
        led_txt[2].set_text("DEFENSE: active" if engine.defending else "DEFENSE: idle")
        if engine.have_models:
            flg = ", ".join(engine.flagged) if engine.flagged else "-"
            atkc = ", ".join(CH_NAMES[i] for i in sorted(engine.P["channels"])) or "-"
            stat_txt.set_text(f"t = {engine.t:6.1f} s\n\n"
                              f"attacking:\n  {atkc if engine.P['attack_on'] else '-'}\n\n"
                              f"correcting:\n  {flg}")
            metric_txt.set_text(f"fluctuation\nreduction:\n   {engine.reduction_now():4.0f}%")
        return []

    anim = animation.FuncAnimation(fig, update, interval=40, blit=False,
                                   cache_frame_data=False)
    fig._anim = anim  # keep a reference alive
    print("[control_room] launched. Move the sliders, toggle ATTACK / DEFENSE. "
          "Close the window to exit.")
    plt.show()


def main():
    torch.manual_seed(C.SEED)
    det, dfn, scaler = load_models("cpu")
    if det is None:
        print("[control_room] Note: no outputs/trained_models.pt found -- the "
              "defense will be disabled until you run  python3 main.py  once.")
    engine = InteractiveLFC(detector=det, defender=dfn, scaler=scaler)
    run_dashboard(engine)


if __name__ == "__main__":
    main()
