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
      images/val/...        (if a val split exists)
      labels/val/...
      crossguard_seg.yaml

Images are HARD-LINKED by default (instant, no extra disk when out dir is on
the same drive as the source); falls back to copy across drives.

TWO WAYS TO USE IT
------------------
1) EXPLICIT-SPLIT mode (use this when you already split the data into
   train/test folders — it maps each folder to a fixed split and does NOT
   reshuffle, so a pre-made split is preserved exactly):

    python convert_xanylabeling_to_yoloseg.py \
        --train "E:/.../CrossguardVision/dataset/Crossguard_data_train" \
        --val   "E:/.../CrossguardVision/dataset/Crossguard_data_test" \
        --out   "E:/.../CrossguardVision/dataset/yolo_seg"

   (Ultralytics evaluates on the `val` split, so point --val at the held-out
   test set when there is no separate validation set.)

2) LEGACY pooled mode (one or more folders pooled, then a random val split):

    python convert_xanylabeling_to_yoloseg.py \
        "E:/.../dataset/Crossguard_data" \
        --out "E:/.../dataset/yolo_seg" --val-split 0.2

Options:
    --train / --val / --test  folders -> that split (explicit-split mode)
    --val-split 0.0     (legacy) fraction held out for val (0 -> val=train)
    --link hardlink|copy|symlink   how images are placed (default hardlink)
    --classes person,crosswalk     class order -> ids 0,1,...
    --seed 0            shuffle seed for the legacy val split
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


def collect_pairs(folders):
    """Return list of (stem, image_path, json_path) for every JSON with an image."""
    out = []
    for folder in folders:
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
            out.append((stem, img, os.path.join(folder, name)))
    return out


def main():
    ap = argparse.ArgumentParser(description="X-AnyLabeling JSON -> YOLO-Seg dataset")
    ap.add_argument("folders", nargs="*",
                    help="(legacy) folders of image+json pairs; pooled then split by --val-split")
    ap.add_argument("--out", required=True, help="Output dataset directory")
    ap.add_argument("--classes", default="person,crosswalk",
                    help="Comma-separated class names; index = class id (default person,crosswalk)")
    ap.add_argument("--train", nargs="*", default=[],
                    help="Folder(s) -> train split (explicit-split mode, no reshuffle)")
    ap.add_argument("--val", nargs="*", default=[],
                    help="Folder(s) -> val split (explicit-split mode). Ultralytics evaluates on this.")
    ap.add_argument("--test", nargs="*", default=[],
                    help="Folder(s) -> test split (optional, explicit-split mode)")
    ap.add_argument("--val-split", type=float, default=0.0,
                    help="(legacy pooled mode) fraction held out for val (0 -> all train, val=train)")
    ap.add_argument("--link", choices=["hardlink", "copy", "symlink"], default="hardlink",
                    help="How to place images into the dataset (default hardlink)")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed for the legacy val split")
    ap.add_argument("--clean", action="store_true",
                    help="Delete existing images/ and labels/ under --out first "
                         "(prevents mixing with a previous, differently-named dataset)")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    cls_id = {name: i for i, name in enumerate(classes)}

    # 1. Build a tagged list of (stem, image, json, split).
    explicit = bool(args.train or args.val or args.test)
    tagged = []
    if explicit:
        for sp, folders in (("train", args.train), ("val", args.val), ("test", args.test)):
            for stem, img, jp in collect_pairs(folders):
                tagged.append((stem, img, jp, sp))
        if not tagged:
            print("[ERROR] no image/json pairs found in --train/--val/--test folders.")
            sys.exit(1)
    else:
        pairs = collect_pairs(args.folders)
        if not pairs:
            print("[ERROR] no image/json pairs found (pass folders, or use --train/--val).")
            sys.exit(1)
        rng = random.Random(args.seed)
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        n_val = int(round(len(pairs) * args.val_split))
        val_set = set(idx[:n_val])
        for i, (stem, img, jp) in enumerate(pairs):
            tagged.append((stem, img, jp, "val" if i in val_set else "train"))

    # 2. Prepare output dirs for the splits that actually have data.
    if args.clean:
        for sub in ("images", "labels"):
            d = os.path.join(args.out, sub)
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"[clean] removed {d}")
    splits_present = [s for s in ("train", "val", "test") if any(t[3] == s for t in tagged)]
    for sp in splits_present:
        os.makedirs(os.path.join(args.out, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", sp), exist_ok=True)

    stats = {s: 0 for s in ("train", "val", "test")}
    inst = {c: 0 for c in classes}
    skipped_shapes = 0
    unknown_labels = {}

    # 3. Convert each pair into its assigned split.
    for stem, img_path, json_path, split in tagged:
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

    # 4. Write dataset YAML.
    out_abs = os.path.abspath(args.out).replace("\\", "/")
    yaml_path = os.path.join(args.out, "crossguard_seg.yaml")
    has_val = stats["val"] > 0
    has_test = stats["test"] > 0
    val_dir = "images/val" if has_val else "images/train"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by convert_xanylabeling_to_yoloseg.py\n")
        f.write(f"path: {out_abs}\n")
        f.write("train: images/train\n")
        f.write(f"val: {val_dir}\n")
        if has_test:
            f.write("test: images/test\n")
        f.write("names:\n")
        for i, name in enumerate(classes):
            f.write(f"  {i}: {name}\n")

    # 5. Report.
    print("\n========== CONVERSION SUMMARY ==========")
    print(f"  mode               : {'explicit-split' if explicit else 'legacy pooled'}")
    print(f"  pairs found        : {len(tagged)}")
    print(f"  train images       : {stats['train']}")
    print(f"  val images         : {stats['val']}" + ("" if has_val else "  (val -> train)"))
    if has_test:
        print(f"  test images        : {stats['test']}")
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
