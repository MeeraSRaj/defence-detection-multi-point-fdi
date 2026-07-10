"""
config.py
=========
Central configuration for the replication of:

    X.-C. Shangguan, M.-H. Yu, C.-K. Zhang, Y. He,
    "Detection and Defense Against Multi-Point False Data Injection Attacks
     of Load Frequency Control in Smart Grid,"
    IEEE Transactions on Smart Grid, vol. 16, no. 5, pp. 4143-4154, Sep. 2025.

All physical parameters are taken from Table I of the paper. Each control area
in the paper lists two turbine-governor sets (the "a / b" values, e.g. 0.32/0.32).
Because the state-space model in Eqs. (1a)-(1h) is written per-area with a single
mechanical/valve state, we collapse each area to a single-machine equivalent by
averaging the two unit values (documented below). This preserves the closed-loop
dynamics described by the paper's equations exactly.

Units: p.u. on a 3000 MVA system base, frequency deviations in Hz.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np

# --------------------------------------------------------------------------- #
#  System-wide ratings (Section IV-A)
# --------------------------------------------------------------------------- #
S_BASE_MVA      = 3000.0     # system power base
RATED_CAP_MVA   = 3000.0     # rated power capacity per area
RATED_LOAD_MW   = 2000.0     # rated load demand per area

# --------------------------------------------------------------------------- #
#  Per-area machine parameters  (Table I)
#  Dual "a / b" entries are averaged into a single-machine equivalent.
# --------------------------------------------------------------------------- #
def _avg(a: float, b: float) -> float:
    return 0.5 * (a + b)

@dataclass
class AreaParams:
    Tg:  float   # governor time constant  [s]
    Tt:  float   # turbine  time constant  [s]
    R:   float   # speed droop             [Hz/p.u.]
    M:   float   # 2H, generator inertia   [p.u.s]
    D:   float   # damping coefficient     [p.u./Hz]
    beta: float  # frequency bias factor   [p.u./Hz]  (= 1/R + D)
    Kp:  float   # PI proportional gain
    Ki:  float   # PI integral gain
    Kdc: float   # HVDC link gain
    Tdc: float   # HVDC link time constant [s]


# Table I  (Area 1, 2, 3)
AREAS: Dict[int, AreaParams] = {
    1: AreaParams(Tg=_avg(0.32, 0.32), Tt=_avg(0.06, 0.07), R=_avg(2.67, 3.00),
                  M=0.1500, D=0.0640, beta=0.4250, Kp=0.0554, Ki=0.2285,
                  Kdc=0.90, Tdc=0.50),
    2: AreaParams(Tg=_avg(0.30, 0.31), Tt=_avg(0.08, 0.08), R=_avg(2.78, 3.11),
                  M=0.1875, D=0.0964, beta=0.3966, Kp=0.0475, Ki=0.1764,
                  Kdc=1.10, Tdc=0.55),
    3: AreaParams(Tg=_avg(0.30, 0.34), Tt=_avg(0.06, 0.06), R=_avg(2.78, 2.67),
                  M=0.1440, D=0.0850, beta=0.3552, Kp=0.0444, Ki=0.1866,
                  Kdc=0.86, Tdc=0.48),
}

# Tie-line synchronizing coefficients  T_ij  (Table I)
T_SYNC: Dict[Tuple[int, int], float] = {
    (1, 2): 0.243,
    (1, 3): 0.321,
    (2, 3): 0.195,
}

# The three interconnection pairs (undirected) present in the 3-area ring
TIE_PAIRS = [(1, 2), (1, 3), (2, 3)]
AREA_IDS  = [1, 2, 3]

# --------------------------------------------------------------------------- #
#  Nonlinear constraints  (Section IV-A)
# --------------------------------------------------------------------------- #
GDB_DEADZONE = 0.0006     # governor dead-band width [p.u.]
GRC_LIMIT    = 0.05       # generation rate constraint [p.u./s]  (5% p.u.)

# --------------------------------------------------------------------------- #
#  Simulation settings
# --------------------------------------------------------------------------- #
DT_SIM   = 0.01           # integration step [s]
DT_ML    = 0.05           # ML sampling step [s]  (downsample factor = 5)
T_END    = 60.0           # default simulation horizon [s]

# Random RES / load disturbance envelope (Section IV-A / Fig. 5, Fig. 9)
DISTURB_MIN = 0.01        # p.u.
DISTURB_MAX = 0.05        # p.u.

# --------------------------------------------------------------------------- #
#  FDI attack model parameters  (Section II-B)
#     (1) Step : A(t) = lam_s                 t >= tau
#     (2) Pulse: A(t) = lam_p                 t in tau-window
#     (3) Sine : A(t) = lam_sin1 sin(lam_sin2 t)
# --------------------------------------------------------------------------- #
@dataclass
class AttackDefaults:
    step_mag:  float = 0.5          # Hz or p.u.
    pulse_mag: float = 0.4
    sine_amp:  float = 0.3
    sine_freq: float = 2.0          # rad/s
    pulse_width: float = 3.0        # s
DEFAULT_ATTACK = AttackDefaults()

# Measurement channels that can be attacked (Section II-B: df_i and P_tie-i)
#   The 6-dim measurement vector fed to the detector/defender is:
#     [ df1, df2, df3, Ptie1, Ptie2, Ptie3 ]
CHANNELS = ["df1", "df2", "df3", "Ptie1", "Ptie2", "Ptie3"]
N_CHANNELS = len(CHANNELS)

# --------------------------------------------------------------------------- #
#  Machine-learning settings  (Section IV-B)
#  Architecture is kept faithful to the paper; dataset / epoch counts are
#  configurable so the pipeline runs to completion inside a CPU sandbox.
#  Set FULL_SCALE = True to reproduce the paper's training budget.
# --------------------------------------------------------------------------- #
WINDOW_LEN = 20               # length of each time-series window fed to the nets

# DL-LSTM detector (three stacked LSTM layers 256/128/64, dropout 0.01)
LSTM_HIDDEN   = (256, 128, 64)
LSTM_DROPOUT  = 0.01

# BiGRU defender (single BiGRU layer, 100 hidden units)
BIGRU_HIDDEN  = 100

FULL_SCALE = False
if FULL_SCALE:
    # Paper-faithful training budget. Heavier: expect a long run on CPU,
    # much faster on an Apple-silicon GPU (MPS) or CUDA machine.
    N_TRAIN_EPISODES = 400     # simulated episodes used to build the dataset
    DET_EPOCHS = 230           # detector epochs (main DL-LSTM)
    DEF_EPOCHS = 200           # defender epochs
    BASE_EPOCHS = 120          # epochs for comparison-baseline models
    BATCH = 256
    DET_CAP = None             # None = use every training window (no subsampling)
    DEF_CAP = None
else:
    N_TRAIN_EPISODES = 60      # simulated episodes used to build the dataset
    DET_EPOCHS = 16            # detector epochs (main DL-LSTM)
    DEF_EPOCHS = 18            # defender epochs
    BASE_EPOCHS = 8            # epochs for comparison-baseline models
    BATCH = 256
    DET_CAP = 5000             # cap on detector train windows (sandbox)
    DEF_CAP = 6000             # cap on defender train windows (sandbox)

SEED = 2025

def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)
