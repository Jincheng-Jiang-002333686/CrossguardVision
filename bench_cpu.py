"""CPU latency benchmark (batch=1, warmup, median over frames) for the 6 best.pt.
Threads capped to 4 to approximate the Raspberry-Pi core count.
Adds cpu_inf_ms / cpu_postproc_ms / cpu_total_ms to model_comparison_2026-06-23.csv.
inference + postprocess are model-intrinsic (resolution-independent after letterbox);
total = preprocess+inference+postprocess = realistic per-frame deployment latency."""
import os, glob, csv, statistics
import torch
torch.set_num_threads(4)            # mimic Pi 4-core CPU
from ultralytics import YOLO

REPO = r"E:/NEU files/TA/CrossguardVision"
SEG = os.path.join(REPO, "CrossguardVision", "runs", "seg")
VAL = os.path.join(REPO, "CrossguardVision", "dataset", "yolo_seg", "images", "val")
CSV = os.path.join(REPO, "CrossguardVision", "logs", "model_comparison_2026-06-23.csv")
IMGSZ, CONF, WARMUP, N_TIMED = 896, 0.25, 3, 20

MODELS = [
    ("yolo11n-seg", "11n_seg_e120"), ("yolo26n-seg", "26n_seg_e120"),
    ("yolo11s-seg", "11s_seg_e120"), ("yolo26s-seg", "26s_seg_e120"),
    ("yolo11m-seg", "11m_seg_e120"), ("yolo26m-seg", "26m_seg_e120"),
]

imgs = sorted(glob.glob(os.path.join(VAL, "*.jpg")))
need = WARMUP + N_TIMED
idx = sorted(set(round(i * (len(imgs) - 1) / (need - 1)) for i in range(need)))
sample = [imgs[i] for i in idx]
warm, timed = sample[:WARMUP], sample[WARMUP:]
print(f"threads={torch.get_num_threads()} imgsz={IMGSZ} warmup={len(warm)} timed={len(timed)}", flush=True)

res = {}
for label, name in MODELS:
    model = YOLO(os.path.join(SEG, name, "weights", "best.pt"))
    for im in warm:
        model.predict(im, device="cpu", imgsz=IMGSZ, conf=CONF, verbose=False, save=False)
    inf, post, tot = [], [], []
    for im in timed:
        s = model.predict(im, device="cpu", imgsz=IMGSZ, conf=CONF, verbose=False, save=False)[0].speed
        inf.append(s["inference"]); post.append(s["postprocess"])
        tot.append(s["preprocess"] + s["inference"] + s["postprocess"])
    def p95(x): return sorted(x)[max(0, round(0.95 * len(x)) - 1)]
    res[label] = (statistics.median(inf), statistics.median(post), statistics.median(tot), p95(tot))
    print(f"{label}: inf {res[label][0]:.1f}  post {res[label][1]:.2f}  "
          f"total {res[label][2]:.1f}  total_p95 {res[label][3]:.1f} ms", flush=True)

rows = list(csv.DictReader(open(CSV, newline="")))
fields = list(rows[0].keys()) + ["cpu_inf_ms", "cpu_postproc_ms", "cpu_total_ms"]
for row in rows:
    i, p, t, _ = res[row["model"]]
    row["cpu_inf_ms"] = round(i, 1); row["cpu_postproc_ms"] = round(p, 2); row["cpu_total_ms"] = round(t, 1)
with open(CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print("\nWROTE:", CSV, flush=True)
