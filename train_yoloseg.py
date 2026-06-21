#!/usr/bin/env python
"""
train_yoloseg.py
----------------
Fine-tune a YOLO segmentation model on the CrossguardVision dataset produced by
convert_xanylabeling_to_yoloseg.py (classes: person, crosswalk).

The crosswalk must be a segmentation mask (not just a box) because the
downstream spatial-reasoning stage needs the actual crosswalk area.

This is written so a tiny "smoke" run (few epochs, nano model) and a real
training run are the same command with different numbers.

Usage (from the `myenv` conda env), smoke test:
    python train_yoloseg.py \
        --data "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/yolo_seg/crossguard_seg.yaml" \
        --model yolo11n-seg.pt --epochs 3 --imgsz 640 --batch 8 --name smoke

Notes:
  * --model yolo11n-seg.pt  downloads pretrained weights (needs internet once).
    Offline fallback: --model yolo11n-seg.yaml trains from scratch (no download),
    which still proves the pipeline runs.
  * On the 8 GB laptop GPU keep imgsz<=640 + a modest batch to avoid CUDA OOM.
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description="Fine-tune YOLO-Seg for CrossguardVision")
    ap.add_argument("--data", required=True, help="Path to dataset YAML")
    ap.add_argument("--model", default="yolo11n-seg.pt",
                    help="Base weights (.pt pretrained) or .yaml (from scratch). Default yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=3, help="Training epochs (default 3 = smoke test)")
    ap.add_argument("--imgsz", type=int, default=640, help="Training image size (default 640)")
    ap.add_argument("--batch", type=int, default=8, help="Batch size (default 8; -1 = auto)")
    ap.add_argument("--patience", type=int, default=100, help="Early-stop patience in epochs (default 100)")
    ap.add_argument("--device", default="0", help="CUDA device id or 'cpu' (default 0)")
    ap.add_argument("--workers", type=int, default=4, help="Dataloader workers (default 4)")
    ap.add_argument("--project", default=None,
                    help="Output project dir (default: <repo>/CrossguardVision/runs/seg)")
    ap.add_argument("--name", default="train", help="Run name (default 'train')")
    ap.add_argument("--cache", action="store_true", help="Cache images in RAM (off by default)")
    ap.add_argument("--resume", action="store_true", help="Resume the last run")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.abspath(__file__))
    project = args.project or os.path.join(repo, "CrossguardVision", "runs", "seg")

    import torch
    from ultralytics import YOLO

    cuda = torch.cuda.is_available() and str(args.device) != "cpu"
    print(f"device={'cuda:'+str(args.device) if cuda else 'cpu'}  model={args.model}  "
          f"epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}")
    print(f"data={args.data}")
    print(f"project={project}  name={args.name}")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=project,
        name=args.name,
        cache=args.cache,
        resume=args.resume,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    save_dir = getattr(results, "save_dir", os.path.join(project, args.name))
    best = os.path.join(str(save_dir), "weights", "best.pt")
    print("\n========== TRAINING DONE ==========")
    print(f"  run dir   : {save_dir}")
    print(f"  best.pt   : {best}  (exists={os.path.exists(best)})")
    print("===================================")


if __name__ == "__main__":
    main()
