# Detection and Defense Against Multi-Point FDI Attacks in Load Frequency Control

A from-scratch Python replication of:

> Shangguan *et al.*, "Detection and Defense Against Multi-Point False Data
> Injection Attacks of Load Frequency Control in Smart Grid," *IEEE Transactions
> on Smart Grid*, vol. 16, no. 5, pp. 4143–4154, Sep. 2025.

It builds the **three-area interconnected LFC power system** (with HVDC links,
renewable sources, governor dead-band and generation-rate limits), injects
**multi-point false-data-injection (FDI) attacks** on the frequency and
tie-line measurement channels, and runs the paper's two machine-learning
components:

* a **DL-LSTM hierarchical detector** — decides Normal / Single / Multi-point
  attack and localises the compromised channels;
* a **BiGRU defender** — predicts clean measurements and replaces the corrupted
  ones online, inside the closed control loop.

Running `main.py` reproduces the paper's tables, figures and the four case
studies A–D end to end.

---

## 1. Requirements

* **macOS** (Apple-silicon or Intel), or any Linux/Windows machine
* **Python 3.10 – 3.12**
* About 1 GB free disk for the virtual environment

On Apple-silicon Macs the code automatically uses the **Metal (MPS) GPU** for
training; on other machines it uses CUDA if present, otherwise the CPU. Nothing
to configure.

---

## 2. Setup (macOS)

Open **Terminal**, `cd` into this folder, then:

```bash
# 1. create and activate an isolated environment
python3 -m venv .venv
source .venv/bin/activate

# 2. install dependencies (this also pulls the correct PyTorch wheel for your Mac)
pip install --upgrade pip
pip install -r requirements.txt
```

> If `python3` is not found, install Python from https://www.python.org/downloads/macos/
> (or `brew install python`), then re-open Terminal.

---

## 3. Run everything

```bash
python3 main.py
```

That single command executes all six stages and prints a summary:

1. Build the dataset by simulating the LFC system under normal operation and
   under random single / multi-point FDI attacks.
2. Train the **DL-LSTM detector** and the comparison baselines (SVM, RNN, LSTM,
   GRU, BiLSTM, BiGRU) → **Table II**.
3. Train the **BiGRU defender** and its prediction baselines → **Fig. 7**.
4. Generate all figures.
5. Run **case studies A–D** with online detection + defense and measure the
   frequency-fluctuation reduction.
6. Save `metrics.json` and the trained weights.

Everything is written to the **`outputs/`** folder.

### Force a specific device (optional)

```bash
LFC_DEVICE=cpu  python3 main.py     # ignore the GPU, run on CPU
LFC_DEVICE=mps  python3 main.py     # force Apple-silicon GPU
```

---

## 3b. Watch the control system LIVE (animation)

The case-study PNGs are static snapshots. To *watch* the three-area grid run in
real time — the attack landing, the detector firing, and the BiGRU defender
correcting the frequency live — run (after `main.py` has trained the models):

```bash
python3 live_sim.py                      # Case A, real time
python3 live_sim.py --case C             # heavier multi-point attack
python3 live_sim.py --case D --speed 3   # 3x faster playback
python3 live_sim.py --case B --seconds 40
```

A window opens showing, for each of the three areas, three live traces —
**clean** (no attack), **attacked** (no defense) and **defended** (detector +
BiGRU online) — plus a status panel with a clock, an attack indicator, the
detector's verdict (Normal / Single / Multi and which channels), the defender
state, and the running peak-to-peak reduction on the focus area. It reproduces
the same closed-loop run as the corresponding case-study figure, but animated.

> If no window appears, your Python may be missing a GUI backend. Install one
> with `pip install PyQt5` (or run inside an IDE that shows Matplotlib windows).

---

## 4. What you get in `outputs/`

| File | Corresponds to |
|------|----------------|
| `fig_disturbances.png` | RES / load disturbance profiles (Figs. 5 & 9) |
| `fig_attack_types.png` | Step / pulse / sine FDI attack signals (Fig. 10) |
| `fig_bigru_training.png` | BiGRU training-loss curve (Fig. 6) |
| `fig_table_detection.png` | Detection metrics for all models (Table II) |
| `fig_prediction_comparison.png` | Predictor comparison MSE/RMSE/MAE/R² (Fig. 7) |
| `fig_bigru_prediction.png` | BiGRU prediction vs. actual (Fig. 8) |
| `fig_case_A … D.png` | Case-study frequency / tie-line responses (Figs. 11,13,15,17) |
| `fig_case_A … D_mitigation.png` | Before/after defense (Figs. 12,14,16,18) |
| `metrics.json` | All numeric results |
| `trained_models.pt` | Trained detector + defender weights and the scaler |

---

## 5. Two training budgets

The physics and every network architecture match the paper exactly. Only the
**amount of training** is configurable, in `config.py`:

* **`FULL_SCALE = False`** (default) — a lighter budget that finishes quickly
  on a laptop and already reproduces the paper's qualitative findings and
  near-paper metrics (detector accuracy ≈ 0.95, BiGRU R² ≈ 0.99, large
  fluctuation reductions across the case studies).

* **`FULL_SCALE = True`** — the paper-faithful budget (400 simulated episodes,
  hundreds of epochs, no subsampling). This reproduces the exact headline
  numbers but takes considerably longer; use a GPU (MPS/CUDA) if you can.

To switch, open `config.py` and set `FULL_SCALE = True`.

---

## 6. File overview

| File | Role |
|------|------|
| `config.py` | All Table-I physical parameters, attack/ML settings, budgets |
| `lfc_model.py` | Three-area LFC state-space model (Eqs. 1a–1h), RK4 integration |
| `attacks.py` | FDI attack signal models and injection (Section II-B) |
| `dataset.py` | Episode simulation, windowing, scaling, train/test split |
| `models.py` | DL-LSTM detector and BiGRU defender networks |
| `train.py` | Training loops, metrics, and all comparison baselines |
| `case_studies.py` | Closed-loop cases A–D with online detection + defense |
| `plots.py` | All publication-style figures |
| `main.py` | Orchestrates the whole pipeline |
| `live_sim.py` | Real-time animated viewer of the closed loop under attack/defense |

---

## 7. Notes

* Results vary slightly run to run because the attack scenarios and
  disturbances are randomly generated (a fixed seed keeps this reproducible).
* Each area's dual turbine–governor entries from Table I are combined into a
  single-machine equivalent so the per-area single-state equations apply
  exactly; this is documented in `config.py`.
* The AGC responds to the *attacked/corrected* telemetry while the physical
  states keep evolving from the *true* dynamics — this is what lets the attacks
  actually destabilise the loop and lets the defender correct it online.
