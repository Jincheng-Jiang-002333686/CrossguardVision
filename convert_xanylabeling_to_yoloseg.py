#!/usr/bin/env python
"""
convert_xanylabeling_to_yoloseg.py
----------------------------------
Convert X-AnyLabeling / LabelMe-style JSON annotations into the YOLO
*segmentation* training format and lay out a ready-to-train dataset folder.

Why segmentation: the crosswalk area must be a mask/polygon (a rectangular
crosswalk box is not accurate enough for the downstream spatial-reasoning
safety logic). `person` rectangles are emitted as 4-corner polygons so they
live happily in the same segmentation dataset.

Per-shape handling (keyed on the LABEL, not the shape_type):
  * rectangle    -> its 4 corner points (or 2 -> expanded to 4)  -> polygon
  * quadrilateral-> its 4 points                                  -> polygon
  * polygon      -> its N points (>=3)                            -> polygon
All coordinates are normalised to [0,1] and CLAMPED to the image, which also
fixes the slightly-out-of-bounds points flagged by validate_labels.py.

Output layout (standard Ultralytics layout; labels dir mirrors images dir):
    <out>/
      images/train/crossguard_0001.jpg ...
      labels/train/crossguard_0001.txt ...
      images/val/...        (only if --val-split > 0)
      labels/val/...
      crossguard_seg.yaml

Images are HARD-LINKED by default (instant, no extra disk when out dir is on
the same drive as the source); falls back to copy across drives.

Usage (from the `myenv` conda env):
    python convert_xanylabeling_to_yoloseg.py \
        "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/Crossguard_data" \
        --out "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/yolo_seg"

Options:
    --val-split 0.0     fraction held out for val (0 -> all train, val=train)
    --link hardlink|copy|symlink   how images are placed (default hardlink)
    --classes person,crosswalk     class order -> ids 0,1,...
    --seed 0            shuffle seed for the val split
"""

import argparse
import json
import os
import random
import shutil
import sys

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def find_image_for(stem, folder):
    for ext in IMG_EXTS:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return None


def rect_to_poly(pts):
    """Return 4 corner points for a rectangle shape (handles 2- or 4-point form)."""
    if len(pts) >= 4:
        return pts[:4]
    if len(pts) == 2:
        (x1, y1), (x2, y2) = pts[0], pts[1]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return None


def shape_to_polygon(shape):
    """Extract a list of (x,y) points for any supported shape, or None to skip."""
    st = shape.get("shape_type")
    pts = shape.get("points", [])
    if st == "rectangle":
        return rect_to_poly(pts)
    # polygon / quadrilateral / anything point-based
    if len(pts) >= 3:
        return pts
    return None


def normalize(points, W, H):
    """Normalise to [0,1] and clamp to the image. Returns flat [x1,y1,...]."""
    flat = []
    for x, y in points:
        nx = min(max(x / W, 0.0), 1.0)
        ny = min(max(y / H, 0.0), 1.0)
        flat.extend([round(nx, 6), round(ny, 6)])
    return flat


def place_image(src, dst, mode):
    if os.path.exists(dst):
        os.remove(dst)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # cross-drive or unsupported -> fall back to copy
    elif mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="X-AnyLabeling JSON -> YOLO-Seg dataset")
    ap.add_argument("folders", nargs="+", help="Folders containing image + .json pairs")
    ap.add_argument("--out", required=True, help="Output dataset directory")
    ap.add_argument("--classes", default="person,crosswalk",
                    help="Comma-separated class names; index = class id (default person,crosswalk)")
    ap.add_argument("--val-split", type=float, default=0.0,
                    help="Fraction of images held out for val (default 0 -> all train, val=train)")
    ap.add_argument("--link", choices=["hardlink", "copy", "symlink"], default="hardlink",
                    help="How to place images into the dataset (default hardlink)")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed for val split")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    cls_id = {name: i for i, name in enumerate(classes)}

    # 1. Collect every (image, json) pair across all input folders.
    pairs = []  # (stem, image_path, json_path)
    for folder in args.folders:
        if not os.path.isdir(folder):
            print(f"[WARN] not a folder, skipping: {folder}")
            continue
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(".json"):
                continue
            stem = os.path.splitext(name)[0]
            img = find_image_for(stem, folder)
            if img is None:
                print(f"[WARN] json without image, skipping: {name}")
                continue
            pairs.append((stem, img, os.path.join(folder, name)))

    if not pairs:
        print("[ERROR] no image/json pairs found.")
        sys.exit(1)

    # 2. Decide train/val membership.
    rng = random.Random(args.seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_val = int(round(len(pairs) * args.val_split))
    val_set = set(idx[:n_val])
    has_val = n_val > 0

    # 3. Prepare output dirs.
    splits = ["train", "val"] if has_val else ["train"]
    for sp in splits:
        os.makedirs(os.path.join(args.out, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", sp), exist_ok=True)

    stats = {"train": 0, "val": 0}
    inst = {c: 0 for c in classes}
    skipped_shapes = 0
    unknown_labels = {}

    # 4. Convert each pair.
    for i, (stem, img_path, json_path) in enumerate(pairs):
        split = "val" if i in val_set else "train"
        try:
            d = json.load(open(json_path, encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] cannot parse {json_path}: {e}")
            continue
        W = d.get("imageWidth")
        H = d.get("imageHeight")
        if not W or not H:
            print(f"[WARN] {stem}: missing image dims, skipping")
            continue

        lines = []
        for sh in d.get("shapes", []):
            label = sh.get("label")
            if label not in cls_id:
                unknown_labels[label] = unknown_labels.get(label, 0) + 1
                continue
            poly = shape_to_polygon(sh)
            if poly is None:
                skipped_shapes += 1
                continue
            flat = normalize(poly, W, H)
            if len(flat) < 6:  # need >=3 points
                skipped_shapes += 1
                continue
            lines.append(str(cls_id[label]) + " " + " ".join(f"{v:g}" for v in flat))
            inst[label] += 1

        # place image (hardlink/copy) + write label txt (may be empty = background)
        ext = os.path.splitext(img_path)[1]
        place_image(img_path, os.path.join(args.out, "images", split, stem + ext), args.link)
        with open(os.path.join(args.out, "labels", split, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        stats[split] += 1

    # 5. Write dataset YAML.
    out_abs = os.path.abspath(args.out).replace("\\", "/")
    yaml_path = os.path.join(args.out, "crossguard_seg.yaml")
    val_dir = "images/val" if has_val else "images/train"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by convert_xanylabeling_to_yoloseg.py\n")
        f.write(f"path: {out_abs}\n")
        f.write("train: images/train\n")
        f.write(f"val: {val_dir}\n")
        f.write("names:\n")
        for i, name in enumerate(classes):
            f.write(f"  {i}: {name}\n")

    # 6. Report.
    print("\n========== CONVERSION SUMMARY ==========")
    print(f"  pairs found        : {len(pairs)}")
    print(f"  train images       : {stats['train']}")
    print(f"  val images         : {stats['val']}" + ("" if has_val else "  (val -> train)"))
    for c in classes:
        print(f"  instances [{c}]    : {inst[c]}")
    print(f"  skipped shapes     : {skipped_shapes}")
    if unknown_labels:
        print(f"  UNKNOWN labels     : {unknown_labels}")
    print(f"  image link mode    : {args.link}")
    print(f"  dataset yaml       : {yaml_path}")
    print("========================================")


if __name__ == "__main__":
    main()
