"""
attacks.py
==========
Multi-point False Data Injection (FDI) attack models from Section II-B.

Three attack signal types are supported (Eqs. in Section II-B):

    (1) Step  :  A(t) = lam_s                        for t in [tau0, tau1]
    (2) Pulse :  A(t) = lam_p                        for t in a short window
    (3) Sine  :  A(t) = lam_sin1 * sin(lam_sin2 * t) for t in [tau0, tau1]

An attack corrupts a *measurement channel* by adding A(t) to the true signal:
    z_A = z + A            (Eq. 2, additive injection on RTU telemetry)

The 6 attackable channels are  [df1, df2, df3, Ptie1, Ptie2, Ptie3].
"multi-point" attacks corrupt >= 2 channels simultaneously.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List
import config as C


@dataclass
class Attack:
    """A single-channel FDI attack."""
    channel: int          # index into C.CHANNELS (0..5)
    kind: str             # 'step' | 'pulse' | 'sine'
    t0: float             # attack onset [s]
    t1: float             # attack end   [s]
    mag: float = 0.4      # step / pulse magnitude, or sine amplitude
    freq: float = 2.0     # sine angular frequency [rad/s]

    def signal(self, t: np.ndarray) -> np.ndarray:
        """Vectorised attack waveform A(t) over a time array."""
        a = np.zeros_like(t, dtype=float)
        active = (t >= self.t0) & (t <= self.t1)
        if self.kind == "step":
            a[active] = self.mag
        elif self.kind == "pulse":
            # short rectangular pulse train inside the window
            width = C.DEFAULT_ATTACK.pulse_width
            phase = ((t - self.t0) % (2 * width)) < width
            a[active & phase] = self.mag
        elif self.kind == "sine":
            a[active] = self.mag * np.sin(self.freq * (t[active] - self.t0))
        else:
            raise ValueError(f"unknown attack kind: {self.kind}")
        return a


def apply_attacks(meas: np.ndarray, t: np.ndarray,
                  attacks: List[Attack]) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject a list of attacks into a measurement matrix.

    meas : (N, 6) clean measurements [df1,df2,df3,Ptie1,Ptie2,Ptie3]
    t    : (N,)  time vector
    Returns
        meas_att : (N, 6) attacked measurements
        labels   : (N, 6) per-channel binary attack mask (1 = under attack)
    """
    meas_att = meas.copy()
    labels = np.zeros_like(meas, dtype=int)
    for atk in attacks:
        a = atk.signal(t)
        meas_att[:, atk.channel] += a
        labels[:, atk.channel] = np.where(a != 0.0, 1, labels[:, atk.channel])
        # mark full active window (even zero-crossings of a sine are "under attack")
        active = (t >= atk.t0) & (t <= atk.t1)
        labels[active, atk.channel] = 1
    return meas_att, labels


# --------------------------------------------------------------------------- #
#  Random attack sampler (for building the training dataset)
# --------------------------------------------------------------------------- #
def sample_random_attacks(g: np.random.Generator, t_end: float,
                          multi_prob: float = 0.45) -> List[Attack]:
    """
    Draw a random attack scenario:
      * with prob (1-multi_prob) -> normal or single-point,
      * otherwise                -> multi-point (2-3 channels).
    """
    kinds = ["step", "pulse", "sine"]
    roll = g.random()
    if roll < 0.30:
        return []                                   # normal (no attack)

    n_pts = 1 if g.random() > multi_prob else int(g.integers(2, 4))
    chans = g.choice(C.N_CHANNELS, size=n_pts, replace=False)

    attacks = []
    for ch in chans:
        kind = g.choice(kinds)
        t0 = g.uniform(0.1 * t_end, 0.6 * t_end)
        dur = g.uniform(0.2 * t_end, 0.4 * t_end)
        t1 = min(t_end, t0 + dur)
        is_freq_ch = ch < 3
        if kind == "sine":
            mag = g.uniform(0.15, 0.45)
            freq = g.uniform(1.0, 4.0)
        else:
            mag = g.uniform(0.3, 0.9) if is_freq_ch else g.uniform(0.3, 0.8)
            freq = 2.0
        attacks.append(Attack(channel=int(ch), kind=str(kind),
                              t0=float(t0), t1=float(t1),
                              mag=float(mag), freq=float(freq)))
    return attacks
