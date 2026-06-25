"""Build the 6-model comparison CSV from each best.pt.
Columns: model, parameters, box_mAP50(all), box_mAP50-95(all),
crosswalk_mask_mAP50, crosswalk_mask_mAP50-95,
crosswalk_mask_precision, crosswalk_mask_recall, gpu_inf_ms.
One back-to-back val() pass per model (workers=0) -> consistent speed batch."""
import os, csv
from ultralytics import YOLO

REPO = r"E:/NEU files/TA/CrossguardVision"
DATA = r"E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/yolo_seg/crossguard_seg.yaml"
SEG = os.path.join(REPO, "CrossguardVision", "runs", "seg")
OUT = os.path.join(REPO, "CrossguardVision", "logs", "model_comparison_2026-06-23.csv")

# (csv label, run name) — ordered by scale then version
MODELS = [
    ("yolo11n-seg", "11n_seg_e120"),
    ("yolo26n-seg", "26n_seg_e120"),
    ("yolo11s-seg", "11s_seg_e120"),
    ("yolo26s-seg", "26s_seg_e120"),
    ("yolo11m-seg", "11m_seg_e120"),
    ("yolo26m-seg", "26m_seg_e120"),
]

HEADER = ["model", "parameters", "box_mAP50_all", "box_mAP50-95_all",
          "crosswalk_mask_mAP50", "crosswalk_mask_mAP50-95",
          "crosswalk_mask_precision", "crosswalk_mask_recall", "gpu_inf_ms"]

rows = []
for label, name in MODELS:
    w = os.path.join(SEG, name, "weights", "best.pt")
    model = YOLO(w)
    params = sum(p.numel() for p in model.model.parameters())  # unfused, before val fuses
    m = model.val(data=DATA, imgsz=896, split="val", batch=8, device="0",
                  workers=0, plots=False, verbose=False)
    cw = {m.names[c]: i for i, c in enumerate(m.seg.ap_class_index)}["crosswalk"]
    row = [
        label,
        params,
        round(float(m.box.map50), 3),     # all-class box mAP50
        round(float(m.box.map), 3),       # all-class box mAP50-95
        round(float(m.seg.ap50[cw]), 3),  # crosswalk mask mAP50
        round(float(m.seg.ap[cw]), 3),    # crosswalk mask mAP50-95
        round(float(m.seg.p[cw]), 3),     # crosswalk mask precision
        round(float(m.seg.r[cw]), 3),     # crosswalk mask recall
        round(float(m.speed["inference"]), 2),
    ]
    rows.append(row)
    print(label, row[1:])

with open(OUT, "w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(HEADER)
    wtr.writerows(rows)
print("\nWROTE:", OUT)
