#!/usr/bin/env python
"""
auto_label_person.py
--------------------
Pre-label PERSON bounding boxes for CrossguardVision frames using a pre-trained
YOLO model (COCO class 0 = person), writing X-AnyLabeling / LabelMe-style JSON
next to each image.  You then open the folder in X-AnyLabeling and only have to
draw / fix the `crosswalk` polygons (and tidy any wrong person boxes).

Key behaviour
  * person-only (COCO class 0); other COCO classes are ignored.
  * writes ONE .json per image, with the exact schema X-AnyLabeling expects,
    including 4-corner-point rectangles (TL, TR, BR, BL).
  * by default it SKIPS any image that already has a .json, so existing
    hand-labeled files (e.g. the 17 in IMG_2456) are never overwritten.
    Use --overwrite to regenerate them.
  * image width/height are read from each image individually (handles any
    portrait/landscape mix), so coordinates are always correct.

Usage (from the activated `myenv` conda env):
    python auto_label_person.py \
        "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/IMG_0227" \
        "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/IMG_0232" \
        "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/IMG_2456" \
        "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/IMG_4453"

Common options:
    --model yolo11m.pt   # medium model (default). yolo11x.pt = max recall, yolo11n.pt = fastest
    --conf 0.25          # confidence threshold
    --imgsz 1280         # inference resolution (higher = better small/distant people)
    --overwrite          # regenerate JSONs that already exist
    --label person       # output class name (must match your classes.txt)
"""

import argparse
import json
import os
import sys

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Matches the X-AnyLabeling version found in the existing hand-labeled JSONs.
XAL_VERSION = "4.0.0-beta.8"


def list_images(folder):
    out = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            out.append(os.path.join(folder, name))
    return out


def rect_points(x1, y1, x2, y2, w, h):
    """Clamp to image bounds and return 4 corner points TL, TR, BR, BL."""
    x1 = max(0.0, min(float(x1), w))
    y1 = max(0.0, min(float(y1), h))
    x2 = max(0.0, min(float(x2), w))
    y2 = max(0.0, min(float(y2), h))
    # ensure x1<=x2, y1<=y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    r = lambda v: round(v, 4)
    return [[r(x1), r(y1)], [r(x2), r(y1)], [r(x2), r(y2)], [r(x1), r(y2)]]


def make_shape(label, pts, score):
    return {
        "label": label,
        "score": score,                 # YOLO confidence (float) or None
        "points": pts,
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
        "kie_linking": [],
    }


def make_doc(shapes, image_name, h, w):
    return {
        "version": XAL_VERSION,
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
        "description": "",
    }


def main():
    ap = argparse.ArgumentParser(description="Auto-label person bounding boxes -> X-AnyLabeling JSON")
    ap.add_argument("folders", nargs="+", help="Image folders to process")
    ap.add_argument("--model", default="yolo11m.pt", help="YOLO weights (default yolo11m.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (default 0.7)")
    ap.add_argument("--imgsz", type=int, default=1024, help="Inference image size (default 1024)")
    ap.add_argument("--label", default="person", help="Output class name (default 'person')")
    ap.add_argument("--person-class", type=int, default=0, help="COCO class id for person (default 0)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .json files")
    ap.add_argument("--no-score", action="store_true", help="Write score as null instead of confidence")
    ap.add_argument("--device", default=None, help="cuda device e.g. 0, or 'cpu' (default: auto)")
    ap.add_argument("--no-half", action="store_true", help="Disable FP16 inference (FP16 is default on CUDA)")
    args = ap.parse_args()

    import gc
    import torch
    from ultralytics import YOLO  # imported here so --help works without the dep

    on_cuda = torch.cuda.is_available() and (args.device is None or str(args.device) != "cpu")
    use_half = on_cuda and not args.no_half
    print(f"device={'cuda' if on_cuda else 'cpu'}  half={use_half}  model={args.model}  imgsz={args.imgsz}  conf={args.conf}")

    model = YOLO(args.model)

    grand = {"images": 0, "processed": 0, "skipped_existing": 0, "persons": 0, "empty": 0}

    for folder in args.folders:
        if not os.path.isdir(folder):
            print(f"[WARN] not a folder, skipping: {folder}")
            continue

        images = list_images(folder)
        # Decide which images to run on (respect existing JSONs unless --overwrite)
        todo, skipped = [], 0
        for img in images:
            json_path = os.path.splitext(img)[0] + ".json"
            if os.path.exists(json_path) and not args.overwrite:
                skipped += 1
            else:
                todo.append(img)

        print(f"\n=== {folder} ===")
        print(f"  images={len(images)}  to_label={len(todo)}  skipped_existing={skipped}")
        grand["images"] += len(images)
        grand["skipped_existing"] += skipped
        if not todo:
            continue

        # Stream predictions to keep memory flat over large folders.
        results = model.predict(
            source=todo,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=[args.person_class],
            device=args.device,
            half=use_half,
            stream=True,
            verbose=False,
        )

        folder_persons = 0
        # results are yielded in input order; pair each with its known path
        # (do NOT trust res.path -- with a list source it collapses to "image0").
        for img_path, res in zip(todo, results):
            h, w = res.orig_shape  # (height, width) of the original image
            shapes = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    pts = rect_points(x1, y1, x2, y2, w, h)
                    score = None if args.no_score else round(float(c), 6)
                    shapes.append(make_shape(args.label, pts, score))

            doc = make_doc(shapes, os.path.basename(img_path), h, w)
            json_path = os.path.splitext(img_path)[0] + ".json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)

            grand["processed"] += 1
            grand["persons"] += len(shapes)
            folder_persons += len(shapes)
            if not shapes:
                grand["empty"] += 1

        print(f"  -> wrote {len(todo)} JSON, {folder_persons} person boxes")

        # release VRAM/cache between folders to avoid OOM on small GPUs
        if on_cuda:
            torch.cuda.empty_cache()
        gc.collect()

    print("\n========== SUMMARY ==========")
    print(f"  images seen         : {grand['images']}")
    print(f"  JSON written        : {grand['processed']}")
    print(f"  skipped (existing)  : {grand['skipped_existing']}")
    print(f"  person boxes total  : {grand['persons']}")
    print(f"  frames w/ 0 persons : {grand['empty']}")
    print("=============================")


if __name__ == "__main__":
    main()
