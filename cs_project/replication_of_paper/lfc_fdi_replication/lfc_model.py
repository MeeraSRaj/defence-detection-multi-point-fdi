"""
lfc_model.py
============
Three-area interconnected Load Frequency Control (LFC) plant, implemented
exactly from Eqs. (1a)-(1h) of the paper, including:

  * AC + DC (HVDC) tie-lines            -- Eqs. (1e), (1f), (1g)
  * PI-based secondary control (AGC)    -- Eq.  (1d) in stabilising form
  * Area Control Error                  -- Eq.  (1h)
  * Governor dead-band (GDB)            -- nonlinearity, Section IV-A
  * Generation rate constraint (GRC)    -- nonlinearity, Section IV-A
  * RES + load disturbances             -- Section IV-A / Fig. 5, Fig. 9

State vector layout (18 states)
-------------------------------
  per area i in {1,2,3}:   [ df_i , dPm_i , dPv_i , xACE_i ]      (4 x 3 = 12)
  AC tie states:           [ Ptie_ac(1,2) , Ptie_ac(1,3) , Ptie_ac(2,3) ]  (3)
  DC tie states:           [ Ptie_dc(1,2) , Ptie_dc(1,3) , Ptie_dc(2,3) ]  (3)

The AGC control command (Eq. 1d) is realised as
      dPc_i = -( Kp_i * ACE_i + Ki_i * xACE_i ),   d/dt xACE_i = ACE_i
which is the physically-stabilising (negative-feedback) integral of Eq. (1d).
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Optional
import config as C


# --------------------------------------------------------------------------- #
#  Nonlinear helper blocks
# --------------------------------------------------------------------------- #
def dead_band(x: np.ndarray | float, width: float) -> np.ndarray | float:
    """Symmetric governor dead-band (GDB)."""
    if width <= 0.0:
        return x
    return np.sign(x) * np.maximum(np.abs(x) - width, 0.0)


# --------------------------------------------------------------------------- #
#  State index bookkeeping
# --------------------------------------------------------------------------- #
def _area_slice(i: int) -> slice:
    base = (i - 1) * 4
    return slice(base, base + 4)

_TIE_AC_BASE = 12
_TIE_DC_BASE = 15
_TIE_INDEX = {(1, 2): 0, (1, 3): 1, (2, 3): 2}
N_STATES = 18


class ThreeAreaLFC:
    """Continuous-time nonlinear plant integrated with fixed-step RK4."""

    def __init__(self,
                 gdb: float = C.GDB_DEADZONE,
                 grc: float = C.GRC_LIMIT,
                 enable_hvdc: bool = True):
        self.gdb = gdb
        self.grc = grc
        self.enable_hvdc = enable_hvdc
        self.A = C.AREAS
        self.Tsync = C.T_SYNC

    # ---- signed AC/DC tie power seen by an area ---------------------------- #
    def _ptie_area(self, x: np.ndarray, i: int) -> float:
        """Net tie-line power ΔP_tie-i (Eq. 1e), sum of AC + DC over neighbours."""
        total = 0.0
        for (a, b), idx in _TIE_INDEX.items():
            if i not in (a, b):
                continue
            sign = 1.0 if i == a else -1.0            # antisymmetry P_ji = -P_ij
            ac = x[_TIE_AC_BASE + idx]
            dc = x[_TIE_DC_BASE + idx] if self.enable_hvdc else 0.0
            total += sign * (ac + dc)
        return total

    # ---- full state derivative -------------------------------------------- #
    def deriv(self, t: float, x: np.ndarray,
              Pd: np.ndarray, Pres: np.ndarray,
              df_bias: np.ndarray | None = None,
              ptie_bias: np.ndarray | None = None) -> np.ndarray:
        """
        dx/dt for the whole 3-area plant.
        Pd, Pres  : length-3 load-change and RES-output disturbances.
        df_bias   : length-3 additive bias on the frequency measurement seen by
                    the AGC (models an FDI attack / defender correction on df_i).
        ptie_bias : length-3 additive bias on the tie-line power measurement seen
                    by the AGC (models an attack / correction on P_tie-i).
        Only the *controller's* view of the measurements is biased; the physical
        states evolve from the true dynamics -- so a corrupted measurement drives
        the AGC to the wrong action and the true frequency drifts.
        """
        dx = np.zeros_like(x)
        f = {i: x[_area_slice(i)][0] for i in C.AREA_IDS}   # df_i
        if df_bias is None:
            df_bias = np.zeros(3)
        if ptie_bias is None:
            ptie_bias = np.zeros(3)

        # --- tie-line dynamics (Eq. 1f AC, Eq. 1g DC) ----------------------- #
        for (a, b), idx in _TIE_INDEX.items():
            Tij = self.Tsync[(a, b)]
            # Eq. (1f): d/dt Ptie_ac = 2*pi*Tij*(df_a - df_b)
            dx[_TIE_AC_BASE + idx] = 2.0 * np.pi * Tij * (f[a] - f[b])
            # Eq. (1g): first-order HVDC lag
            if self.enable_hvdc:
                Kdc = 0.5 * (self.A[a].Kdc + self.A[b].Kdc)
                Tdc = 0.5 * (self.A[a].Tdc + self.A[b].Tdc)
                dc = x[_TIE_DC_BASE + idx]
                dx[_TIE_DC_BASE + idx] = (Kdc * (f[a] - f[b]) - dc) / Tdc

        # --- per-area machine dynamics ------------------------------------- #
        for i in C.AREA_IDS:
            p = self.A[i]
            s = _area_slice(i)
            df, dPm, dPv, xACE = x[s]

            Ptie_i = self._ptie_area(x, i)
            # measurements as seen by the AGC (true value + attack/correction bias)
            df_meas = df + df_bias[i - 1]
            ptie_meas = Ptie_i + ptie_bias[i - 1]
            ACE_i = p.beta * df_meas + ptie_meas         # Eq. (1h) on telemetry

            # AGC / PI command (stabilising form of Eq. 1d)
            dPc = -(p.Kp * ACE_i + p.Ki * xACE)

            # Governor input frequency passes through the dead-band (GDB)
            df_gov = dead_band(df, self.gdb)

            # Eq. (1a): swing equation
            ddf = (dPm - Pd[i - 1] + Pres[i - 1] - Ptie_i - p.D * df) / p.M
            # Eq. (1c): governor
            ddPv = (-df_gov / p.R - dPv + dPc) / p.Tg
            # Eq. (1b): turbine  (with GRC rate limit applied to output)
            ddPm = (dPv - dPm) / p.Tt
            if self.grc > 0.0:
                ddPm = float(np.clip(ddPm, -self.grc, self.grc))
            # integrator of ACE
            dxACE = ACE_i

            dx[s] = np.array([ddf, ddPm, ddPv, dxACE])
        return dx

    # ---- fixed-step RK4 rollout ------------------------------------------- #
    def rk4_step(self, t, x, dt, Pd, Pres, df_bias=None, ptie_bias=None):
        """One fixed-step RK4 update with (optional) held measurement biases."""
        k1 = self.deriv(t,            x,               Pd, Pres, df_bias, ptie_bias)
        k2 = self.deriv(t + 0.5*dt,   x + 0.5*dt*k1,   Pd, Pres, df_bias, ptie_bias)
        k3 = self.deriv(t + 0.5*dt,   x + 0.5*dt*k2,   Pd, Pres, df_bias, ptie_bias)
        k4 = self.deriv(t + dt,       x + dt*k3,       Pd, Pres, df_bias, ptie_bias)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def measurements(self, x):
        """Return (df[3], ptie[3]) for the current state."""
        df = np.array([x[_area_slice(i)][0] for i in C.AREA_IDS])
        pt = np.array([self._ptie_area(x, i) for i in C.AREA_IDS])
        return df, pt

    def simulate(self,
                 t_end: float,
                 dt: float,
                 Pd_fun: Callable[[float], np.ndarray],
                 Pres_fun: Callable[[float], np.ndarray],
                 x0: Optional[np.ndarray] = None,
                 bias_fun: Optional[Callable[[float], tuple]] = None
                 ) -> Dict[str, np.ndarray]:
        """
        Integrate the plant and return the measurement time series.

        Returns dict with keys:
           t      (N,)
           df     (N,3)   frequency deviation per area   [Hz]
           Ptie   (N,3)   net tie-line power per area     [p.u.]
           Pm     (N,3)   mechanical power per area
        """
        n = int(round(t_end / dt)) + 1
        x = np.zeros(N_STATES) if x0 is None else x0.copy()

        t_arr   = np.zeros(n)
        df_arr  = np.zeros((n, 3))
        pt_arr  = np.zeros((n, 3))
        pm_arr  = np.zeros((n, 3))

        for k in range(n):
            t = k * dt
            Pd   = np.asarray(Pd_fun(t),   dtype=float)
            Pres = np.asarray(Pres_fun(t), dtype=float)

            # record measurements
            t_arr[k] = t
            for i in C.AREA_IDS:
                df_arr[k, i - 1] = x[_area_slice(i)][0]
                pt_arr[k, i - 1] = self._ptie_area(x, i)
                pm_arr[k, i - 1] = x[_area_slice(i)][1]

            if k == n - 1:
                break

            # measurement biases seen by the AGC (attack / correction), if any
            if bias_fun is not None:
                dfb, ptb = bias_fun(t)
            else:
                dfb, ptb = None, None

            # RK4 step (disturbances + biases held over the step)
            x = self.rk4_step(t, x, dt, Pd, Pres, dfb, ptb)

        return {"t": t_arr, "df": df_arr, "Ptie": pt_arr, "Pm": pm_arr}


# --------------------------------------------------------------------------- #
#  Disturbance generators (RES + load), Section IV-A
# --------------------------------------------------------------------------- #
def make_disturbances(seed: int,
                      t_end: float,
                      lo: float = C.DISTURB_MIN,
                      hi: float = C.DISTURB_MAX):
    """
    Build smooth random RES/load disturbance profiles per area, with magnitudes
    in [lo, hi] p.u. (Fig. 5 / Fig. 9).  Returns (Pd_fun, Pres_fun).

    The returned functions are PURE functions of time: the first-order smoothing
    is pre-computed once on a fine grid at build time, so calling them in any
    order (and re-running the plant several times, as the case studies do for the
    clean / attacked / defended passes) always yields the *same* disturbance.
    """
    g = np.random.default_rng(seed)

    def _raw_profile():
        # piecewise-constant random steps (the un-smoothed target)
        n_seg = g.integers(4, 8)
        seg_t = np.sort(g.uniform(0, t_end, n_seg))
        seg_v = g.uniform(lo, hi, n_seg) * g.choice([-1, 1], n_seg)

        def f(t):
            v = 0.0
            for st, sv in zip(seg_t, seg_v):
                if t >= st:
                    v = sv
            return v
        return f

    load_raw = [_raw_profile() for _ in range(3)]
    res_raw = [_raw_profile() for _ in range(3)]

    # pre-integrate a simple first-order lag on a fine grid (order-independent)
    tau = 1.5
    dt = C.DT_SIM
    grid = np.arange(0.0, t_end + dt, dt)
    pd_grid = np.zeros((len(grid), 3))
    pr_grid = np.zeros((len(grid), 3))
    sd = np.zeros(3)
    sr = np.zeros(3)
    for k, t in enumerate(grid):
        if k > 0:
            for i in range(3):
                sd[i] += (load_raw[i](t) - sd[i]) * min(dt / tau, 1.0)
                sr[i] += (res_raw[i](t) - sr[i]) * min(dt / tau, 1.0)
        pd_grid[k] = sd
        pr_grid[k] = sr

    last = len(grid) - 1

    def _idx(t):
        # hold the disturbance constant over each DT_SIM interval
        return int(min(max(t / dt, 0.0), last))

    def Pd_fun(t):
        return pd_grid[_idx(t)].copy()

    def Pres_fun(t):
        return pr_grid[_idx(t)].copy()

    return Pd_fun, Pres_fun
