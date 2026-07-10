"""
main.py
=======
End-to-end replication pipeline.

Steps
-----
  1. Build datasets from LFC + FDI simulations.
  2. Train the DL-LSTM detector and comparison baselines  -> Table II.
  3. Train the BiGRU defender and comparison baselines     -> Fig. 7.
  4. Generate all paper-style figures.
  5. Run case studies A-D (online detection + defense)     -> Figs. 11-18.
  6. Save all metrics to outputs/metrics.json and print a summary.

Run:  python main.py
"""

from __future__ import annotations
import os, json, time
import numpy as np
import torch

import config as C
import dataset as D
import train as T
import plots as P
import case_studies as CS

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def banner(msg):
    print("\n" + "=" * 68 + f"\n  {msg}\n" + "=" * 68)


def main():
    t_start = time.time()
    results = {}

    banner("STEP 1/6  Building datasets (LFC + FDI simulations)")
    data = D.build_datasets()
    scaler = data["scaler"]
    dist = np.bincount(data["det"]["y1tr"], minlength=3)
    print(f"  detector windows : train {data['det']['Xtr'].shape[0]}, "
          f"test {data['det']['Xte'].shape[0]}  (N/S/M = {dist.tolist()})")
    print(f"  defender windows : train {data['def']['Xtr'].shape[0]}, "
          f"test {data['def']['Xte'].shape[0]}")

    banner("STEP 2/6  Training DL-LSTM detector + baselines (Table II)")
    detector = T.train_dllstm_detector(data["det"])
    # DL-LSTM detection metrics (layer-1 3-class task on test set)
    pred1 = T.predict_classifier(detector.clf1, data["det"]["Xte"])
    det_metrics = {"DL-LSTM": T.eval_detection(pred1, data["det"]["y1te"])}
    det_metrics.update(T.train_detection_baselines(data["det"]))
    results["detection"] = det_metrics
    print("\n  Detection performance (test set):")
    for m in ["SVM", "RNN", "LSTM", "GRU", "BiLSTM", "BiGRU", "DL-LSTM"]:
        if m in det_metrics:
            r = det_metrics[m]
            print(f"    {m:8s}  Acc={r['Accuracy']:.4f}  Prec={r['Precision']:.4f}"
                  f"  Rec={r['Recall']:.4f}  F1={r['F1-score']:.4f}")

    banner("STEP 3/6  Training BiGRU defender + baselines (Fig. 7)")
    loss_log = []
    defender = T.train_bigru_defender(data["def"], loss_log)
    dpred = T.predict_regressor(defender, data["def"]["Xte"])
    pred_metrics = {"BiGRU": T.eval_regression(dpred, data["def"]["Yte"])}
    pred_metrics.update(T.train_prediction_baselines(data["def"]))
    results["prediction"] = pred_metrics
    print("\n  Prediction performance (test set):")
    for m in ["SVM", "LSTM", "GRU", "BiLSTM", "BiGRU"]:
        if m in pred_metrics:
            r = pred_metrics[m]
            print(f"    {m:8s}  RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}  R2={r['R2']:.4f}")

    banner("STEP 4/6  Generating figures")
    figs = []
    figs.append(P.plot_disturbances())
    figs.append(P.plot_attack_types())
    figs.append(P.plot_bigru_training(loss_log))
    figs.append(P.plot_detection_table(det_metrics))
    figs.append(P.plot_prediction_comparison(pred_metrics))
    figs.append(P.plot_bigru_prediction(scaler, defender))
    for f in figs:
        print("  saved", os.path.basename(f))

    banner("STEP 5/6  Running case studies A-D (online detection + defense)")
    case_summary = {}
    for name in ["A", "B", "C", "D"]:
        print(f"  simulating Case {name} ...")
        res = CS.run_case(name, scaler, detector, defender)
        P.plot_case(res)
        P.plot_case_mitigation(res)
        case_summary[name] = {
            "focus_area": res["focus"] + 1,
            "fluct_attacked_Hz": round(res["fluct_attacked"], 4),
            "fluct_defended_Hz": round(res["fluct_defended"], 4),
            "reduction_pct": round(res["reduction_pct"], 1),
            "attacks": [f"{a.kind} on {C.CHANNELS[a.channel]} "
                        f"[{a.t0:.0f}-{a.t1:.0f}s]" for a in res["attacks"]],
        }
        print(f"     -> fluctuation {res['fluct_attacked']:.3f} Hz "
              f"-> {res['fluct_defended']:.3f} Hz "
              f"({res['reduction_pct']:.0f}% reduction)")
    results["cases"] = case_summary

    banner("STEP 6/6  Saving metrics")
    avg_red = np.mean([case_summary[c]["reduction_pct"] for c in case_summary])
    results["summary"] = {
        "dllstm_accuracy": det_metrics["DL-LSTM"]["Accuracy"],
        "dllstm_f1": det_metrics["DL-LSTM"]["F1-score"],
        "bigru_rmse": pred_metrics["BiGRU"]["RMSE"],
        "bigru_r2": pred_metrics["BiGRU"]["R2"],
        "avg_freq_reduction_pct": round(float(avg_red), 1),
        "runtime_sec": round(time.time() - t_start, 1),
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # persist trained weights (pure tensors -> loads cleanly on modern PyTorch)
    torch.save({"detector": detector.state_dict(),
                "defender": defender.state_dict(),
                "scaler_mean": torch.as_tensor(scaler.mean, dtype=torch.float32),
                "scaler_std":  torch.as_tensor(scaler.std,  dtype=torch.float32)},
               os.path.join(OUT, "trained_models.pt"))

    banner("DONE")
    s = results["summary"]
    print(f"  DL-LSTM  accuracy = {s['dllstm_accuracy']:.4f} | F1 = {s['dllstm_f1']:.4f}")
    print(f"  BiGRU    RMSE     = {s['bigru_rmse']:.4f} | R2 = {s['bigru_r2']:.4f}")
    print(f"  Avg. frequency-fluctuation reduction across A-D = "
          f"{s['avg_freq_reduction_pct']:.0f}%")
    print(f"  Runtime = {s['runtime_sec']:.0f} s")
    print(f"  Outputs written to: {OUT}")


if __name__ == "__main__":
    main()
