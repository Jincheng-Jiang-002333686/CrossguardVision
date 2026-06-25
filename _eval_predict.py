"""Eval best.pt on the held-out test + predict on test images.
Usage: python _eval_predict.py <run_name>   e.g. 26s_seg_e120
Reusable across the 5 comparison models. Bakes in workers=0 (Windows val hang fix).
"""
import os
import sys
from ultralytics import YOLO

REPO = os.path.dirname(os.path.abspath(__file__))
NAME = sys.argv[1] if len(sys.argv) > 1 else "26s_seg_e120"
DATA = r"E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/yolo_seg/crossguard_seg.yaml"
PROJECT = os.path.join(REPO, "CrossguardVision", "runs", "seg")
WEIGHTS = os.path.join(PROJECT, NAME, "weights", "best.pt")
TEST_IMAGES = os.path.join(REPO, "CrossguardVision", "dataset", "yolo_seg", "images", "val")

print("name:", NAME, "| weights:", WEIGHTS)
model = YOLO(WEIGHTS)

print("\n===================== EVAL (held-out test, split=val) =====================")
m = model.val(data=DATA, imgsz=896, split="val", batch=8, device="0", workers=0,
              project=PROJECT, name=NAME + "_val", exist_ok=True, plots=True, verbose=True)


def row(d):
    return f"P {d[0]:.3f}  R {d[1]:.3f}  mAP50 {d[2]:.3f}  mAP50-95 {d[3]:.3f}"


print("\n--- overall (mean) ---")
print("  Box :", row(m.box.mean_results()))
print("  Mask:", row(m.seg.mean_results()))
print("--- per class ---")
names = m.names
for i, c in enumerate(m.box.ap_class_index):
    print(f"  [{names[c]}] Box  mAP50 {m.box.ap50[i]:.3f}  mAP50-95 {m.box.ap[i]:.3f}")
for i, c in enumerate(m.seg.ap_class_index):
    print(f"  [{names[c]}] Mask mAP50 {m.seg.ap50[i]:.3f}  mAP50-95 {m.seg.ap[i]:.3f}")
print("  speed (ms/img):", {k: round(v, 2) for k, v in m.speed.items()})
print("  eval artifacts:", os.path.join(PROJECT, NAME + "_val"))

print("\n===================== PREDICT (test images) =====================")
res = model.predict(source=TEST_IMAGES, imgsz=896, conf=0.25, device="0", save=True,
                    save_txt=True, project=PROJECT, name=NAME + "_predict",
                    exist_ok=True, verbose=False)
n_person = sum(int((r.boxes.cls == 0).sum()) for r in res if r.boxes is not None)
n_cross = sum(int((r.boxes.cls == 1).sum()) for r in res if r.boxes is not None)
print(f"  images predicted : {len(res)}")
print(f"  detections       : {n_person} person, {n_cross} crosswalk")
print(f"  predict dir      : {os.path.join(PROJECT, NAME + '_predict')}")
print("DONE")
