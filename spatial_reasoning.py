#!/usr/bin/env python
"""
spatial_reasoning.py
--------------------
Stage 8 of CrossguardVision: after YOLO-Seg inference, classify each detected
pedestrian's risk as `high_risk`, `medium_risk`, or `low_risk` using geometry only.

Design rule: high_risk / medium_risk / low_risk are NOT model classes. They are
produced here from the person boxes and the crosswalk segmentation mask.
The geometric state maps to risk: in crossing-zone -> high_risk, waiting near the
crosswalk -> medium_risk, far away / too small -> low_risk.

Perspective note
----------------
The camera flattens a 3-D scene: a pedestrian standing AT the crosswalk looks
large, one across the intersection looks tiny. So a fixed pixel distance to the
crosswalk is meaningless. We instead measure each foot's distance to the
crosswalk and normalise it by the person's apparent BODY HEIGHT (box height) ->
a perspective-invariant "how many body-lengths away" measure, and ignore people
too small (too far) to be at the monitored crosswalk.

Crossing ZONE (not just the painted stripes)
---------------------------------------------
People also cross *between* the painted crosswalks, through the intersection
interior. So the "crossing" region is the CONVEX HULL of all crosswalk masks --
this encloses the arms plus the road area between them. A foot inside that hull
counts as crossing even if it is on bare asphalt.

Logic (per person; feet = bottom-centre of box; bh = box height; hull = crossing zone):
  * too small (bh < min_h_frac * imageHeight)              -> low_risk (too far to assess)
  * foot inside hull  OR  dist_to_crosswalk <= crossing_k*bh -> high_risk (in crossing zone)
  * dist_to_crosswalk <= waiting_k * bh                     -> medium_risk (waiting near)
  * otherwise                                              -> low_risk

The core classifier `classify_people()` is pure geometry (boxes + binary mask)
so it can be unit-tested without a model. `run_image()` wraps a YOLO-Seg model.

CLI:
    python spatial_reasoning.py --selftest
    # single image:
    python spatial_reasoning.py \
        --weights ".../runs/seg/m_seg_e120/weights/best.pt" \
        --image   ".../dataset/IMG_0232/corss2_0150.jpg"
    # whole folder (batch) -> <stem>_state.jpg per image + 2 CSVs:
    python spatial_reasoning.py \
        --weights ".../runs/seg/26s_seg/weights/best.pt" \
        --images-dir ".../dataset/Crossguard_data_test" \
        --out ".../runs/spatial/26s_test"
"""

import argparse
import os

import cv2
import numpy as np

# ---- states & colors (BGR) -------------------------------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
# Risk labels (the OUTPUT classes). Geometric state -> risk:
#   in crossing-zone / on crosswalk -> HIGH_RISK
#   waiting near the crosswalk       -> MEDIUM_RISK
#   far away / too small to assess   -> LOW_RISK
HIGH_RISK, MEDIUM_RISK, LOW_RISK = "high_risk", "medium_risk", "low_risk"
STATE_COLOR = {HIGH_RISK: (0, 0, 255), MEDIUM_RISK: (0, 165, 255), LOW_RISK: (0, 200, 0)}
ZONE_COLOR = (0, 255, 255)  # yellow hull outline


# ---- core geometry (no ultralytics dependency) -----------------------------
def crosswalk_distance_map(crosswalk_mask):
    """Per-pixel L2 distance (px) to the nearest crosswalk pixel, or None if
    there is no crosswalk in the frame."""
    if not crosswalk_mask.any():
        return None
    # distanceTransform measures distance to the nearest ZERO pixel, so invert.
    return cv2.distanceTransform((1 - crosswalk_mask).astype(np.uint8), cv2.DIST_L2, 5)


def _inner_edge_corners(arm_mask, cx, cy):
    """For one crosswalk arm, return the 2 endpoints of its INNER long edge --
    the edge facing the intersection centre (cx,cy). These are the corners we
    connect between arms; the outer corners are left out so the zone does not
    sweep across the corner waiting areas."""
    cnts, _ = cv2.findContours(arm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    box = cv2.boxPoints(cv2.minAreaRect(max(cnts, key=cv2.contourArea)))  # 4 corners
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    elen = [float(np.hypot(*(box[i] - box[j]))) for i, j in edges]
    # opposite edges (0,2) and (1,3) are parallel; one pair is the long sides.
    long_pair = (0, 2) if (elen[0] + elen[2]) >= (elen[1] + elen[3]) else (1, 3)
    # of the two long sides, keep the one whose midpoint is nearer the centre.
    best, best_d = None, None
    for ei in long_pair:
        i, j = edges[ei]
        mx, my = (box[i] + box[j]) / 2.0
        d = (mx - cx) ** 2 + (my - cy) ** 2
        if best_d is None or d < best_d:
            best_d, best = d, (box[i], box[j])
    return [best[0], best[1]]


def crossing_zone_from_mask(crosswalk_mask, min_arms=2, erode_frac=0.0, min_arm_px=50):
    """Crossing zone = the painted crosswalk arms PLUS the intersection interior
    bounded by the arms' INNER edges -> (filled zone mask, outline polygon).

    Why not a plain convex hull of all crosswalk pixels: that hull connects the
    arms' FAR (outer) corners with straight chords that cut across the concave
    corner curb-ramps / waiting areas, wrongly pulling waiters into the zone.
    Instead, for each arm we take its inner-edge corners (nearest the
    intersection centre) and hull only those -> the interior polygon hugs the
    road and leaves the corners out.

    min_arms : need >= this many crosswalk blobs (a lone crosswalk has no
               interior, so no zone is built and we fall back to the distance
               rule). erode_frac : optional extra inward shrink (off by default).
    Returns (zone_mask, outline) or (None, None) when no usable zone."""
    if not crosswalk_mask.any():
        return None, None
    num, labels = cv2.connectedComponents(crosswalk_mask)
    ys, xs = np.nonzero(crosswalk_mask)
    cx, cy = float(xs.mean()), float(ys.mean())

    inner_pts = []
    n_arms = 0
    for lbl in range(1, num):
        arm = (labels == lbl).astype(np.uint8)
        if int(arm.sum()) < min_arm_px:
            continue
        n_arms += 1
        inner_pts.extend(_inner_edge_corners(arm, cx, cy))
    if n_arms < min_arms or len(inner_pts) < 3:
        return None, None

    zone = crosswalk_mask.copy()  # the painted arms are themselves crossing
    interior = cv2.convexHull(np.array(inner_pts, dtype=np.float32))
    cv2.fillConvexPoly(zone, interior.astype(np.int32), 1)

    k = int(round(erode_frac * crosswalk_mask.shape[0]))
    if k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        zone = cv2.erode(zone, kernel, iterations=1)
    if not zone.any():
        return None, None
    cnts, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = max(cnts, key=cv2.contourArea) if cnts else None
    return zone, outline


def classify_people(boxes, crosswalk_mask, crossing_k=0.10, waiting_k=0.50,
                    min_h_frac=0.05, use_hull=True, crossing_zone=None,
                    min_arms=2, hull_erode_frac=0.0):
    """
    boxes          : list/array of [x1,y1,x2,y2] in pixel coords.
    crosswalk_mask : HxW uint8 {0,1} union of crosswalk masks.
    crossing_k     : crossing if foot-to-crosswalk dist <= crossing_k * box_height.
    waiting_k      : waiting  if dist <= waiting_k * box_height (and not crossing).
    min_h_frac     : people shorter than this * imageHeight -> low_risk (too far).
    use_hull       : treat the intersection interior (inner-edge zone) as crossing.
    crossing_zone  : optional precomputed zone mask (else computed from the mask).
    Returns a list of dicts: {box, state, dist_px, ratio, box_h, in_zone, foot_point}.
    """
    H, W = crosswalk_mask.shape[:2]
    dist_map = crosswalk_distance_map(crosswalk_mask)
    if use_hull and crossing_zone is None:
        crossing_zone, _ = crossing_zone_from_mask(crosswalk_mask, min_arms=min_arms,
                                                   erode_frac=hull_erode_frac)
    min_h = min_h_frac * H

    results = []
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        bh = max(1.0, y2 - y1)
        # standing point = bottom-centre of the box (where the feet are)
        px = int(round(min(max((x1 + x2) / 2.0, 0), W - 1)))
        py = int(round(min(max(y2 - 1, 0), H - 1)))
        in_zone = bool(crossing_zone[py, px]) if crossing_zone is not None else False

        if dist_map is None:
            state, d, ratio = LOW_RISK, None, None
        elif bh < min_h:
            state, d, ratio = LOW_RISK, float(dist_map[py, px]), None  # too far to assess
        else:
            d = float(dist_map[py, px])
            ratio = d / bh
            if in_zone or ratio <= crossing_k:
                state = HIGH_RISK
            elif ratio <= waiting_k:
                state = MEDIUM_RISK
            else:
                state = LOW_RISK

        results.append({
            "box": [x1, y1, x2, y2], "state": state,
            "dist_px": round(d, 1) if d is not None else None,
            "ratio": round(ratio, 3) if ratio is not None else None,
            "box_h": round(bh, 1), "in_zone": in_zone, "foot_point": [px, py],
        })
    return results


# ---- model wrapper ---------------------------------------------------------
def crosswalk_mask_from_result(result, shape, crosswalk_id=1):
    """Rasterise all crosswalk-class instance polygons into one binary mask.

    Uses masks.xy (polygons already in ORIGINAL-image pixel coords), which
    sidesteps letterbox/resolution mismatch.
    """
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    if result.masks is None or result.boxes is None:
        return mask
    cls = result.boxes.cls.cpu().numpy().astype(int)
    polys = result.masks.xy  # list of (k,2) float arrays in image coords
    for c, poly in zip(cls, polys):
        if c == crosswalk_id and poly is not None and len(poly) >= 3:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask


def person_boxes_from_result(result, person_id=0):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    cls = result.boxes.cls.cpu().numpy().astype(int)
    xyxy = result.boxes.xyxy.cpu().numpy()
    return [xyxy[i].tolist() for i in range(len(cls)) if cls[i] == person_id]


def annotate(image, crosswalk_mask, people, hull=None):
    """Draw crosswalk overlay + crossing-zone outline + per-person boxes. Returns a copy."""
    out = image.copy()
    if crosswalk_mask.any():
        overlay = out.copy()
        overlay[crosswalk_mask.astype(bool)] = (255, 0, 0)  # blue crosswalk
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    if hull is not None:
        cv2.polylines(out, [hull], isClosed=True, color=ZONE_COLOR, thickness=2)
    for p in people:
        x1, y1, x2, y2 = [int(round(v)) for v in p["box"]]
        col = STATE_COLOR[p["state"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.circle(out, tuple(p["foot_point"]), 4, col, -1)
        r = p["ratio"]
        label = p["state"] if r is None else f'{p["state"]} {r:.2f}'
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    return out


def predict_states(model, image_path, conf=0.25, imgsz=640,
                   crossing_k=0.10, waiting_k=0.50, min_h_frac=0.05, use_hull=True,
                   min_arms=2, hull_erode_frac=0.0, device="0"):
    """Run an already-loaded model on one image + classify.
    Returns (img, crosswalk_mask, people, zone_outline)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]
    result = model.predict(image_path, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    cw_mask = crosswalk_mask_from_result(result, (H, W))
    boxes = person_boxes_from_result(result)
    zone, outline = (crossing_zone_from_mask(cw_mask, min_arms=min_arms, erode_frac=hull_erode_frac)
                     if use_hull else (None, None))
    people = classify_people(boxes, cw_mask, crossing_k, waiting_k, min_h_frac,
                             use_hull=use_hull, crossing_zone=zone)
    return img, cw_mask, people, outline


def _save_annotated(img, cw_mask, people, outline, image_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + "_state.jpg")
    cv2.imwrite(out_path, annotate(img, cw_mask, people, hull=outline))
    return out_path


def _counts(people):
    return {s: sum(p["state"] == s for p in people) for s in (HIGH_RISK, MEDIUM_RISK, LOW_RISK)}


def run_image(weights, image_path, out_dir, conf=0.25, imgsz=640,
              crossing_k=0.10, waiting_k=0.50, min_h_frac=0.05, use_hull=True,
              min_arms=2, hull_erode_frac=0.0, device="0", model=None):
    """Single image: load model (unless one is passed), classify, save, print."""
    if model is None:
        from ultralytics import YOLO
        model = YOLO(weights)
    img, cw_mask, people, outline = predict_states(
        model, image_path, conf, imgsz, crossing_k, waiting_k, min_h_frac,
        use_hull, min_arms, hull_erode_frac, device)
    out_path = _save_annotated(img, cw_mask, people, outline, image_path, out_dir)
    H, W = img.shape[:2]
    counts = _counts(people)
    print(f"\nimage      : {image_path}  ({W}x{H})")
    print(f"crosswalk  : {'detected' if cw_mask.any() else 'NONE detected'}  hull={'on' if use_hull else 'off'}")
    print(f"persons    : {len(people)}  -> {counts}")
    for i, p in enumerate(people):
        print(f"   person{i:>2}: {p['state']:8s} in_zone={p['in_zone']} ratio={p['ratio']} "
              f"dist={p['dist_px']} boxH={p['box_h']} foot={p['foot_point']}")
    print(f"annotated  : {out_path}")
    return people


def run_folder(weights, images_dir, out_dir, conf=0.25, imgsz=640,
               crossing_k=0.10, waiting_k=0.50, min_h_frac=0.05, use_hull=True,
               min_arms=2, hull_erode_frac=0.0, device="0"):
    """Batch: load the model ONCE and classify every image in images_dir.
    Writes <stem>_state.jpg per image + two CSVs (per-image counts and
    per-person predictions)."""
    import csv
    from ultralytics import YOLO

    imgs = sorted(f for f in os.listdir(images_dir)
                  if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images found in {images_dir}")
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(weights)
    print(f"loaded weights : {weights}")
    print(f"processing     : {len(imgs)} images  {images_dir} -> {out_dir}")

    per_image, per_person = [], []
    agg = {HIGH_RISK: 0, MEDIUM_RISK: 0, LOW_RISK: 0}
    no_cw = 0
    for k, name in enumerate(imgs, 1):
        path = os.path.join(images_dir, name)
        try:
            img, cw_mask, people, outline = predict_states(
                model, path, conf, imgsz, crossing_k, waiting_k, min_h_frac,
                use_hull, min_arms, hull_erode_frac, device)
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
            continue
        _save_annotated(img, cw_mask, people, outline, path, out_dir)
        c = _counts(people)
        for s in agg:
            agg[s] += c[s]
        has_cw = bool(cw_mask.any())
        no_cw += 0 if has_cw else 1
        per_image.append([name, len(people), c[HIGH_RISK], c[MEDIUM_RISK], c[LOW_RISK], int(has_cw)])
        for i, p in enumerate(people):
            per_person.append([name, i, p["state"], p["ratio"], p["dist_px"], p["box_h"],
                               int(p["in_zone"]), p["foot_point"][0], p["foot_point"][1]])
        if k % 10 == 0 or k == len(imgs):
            print(f"  [{k}/{len(imgs)}] {name}: {len(people)} persons {c}")

    sum_path = os.path.join(out_dir, "spatial_summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "persons", "high_risk", "medium_risk", "low_risk", "crosswalk_detected"])
        w.writerows(per_image)
    det_path = os.path.join(out_dir, "spatial_predictions.csv")
    with open(det_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "person", "state", "ratio", "dist_px", "box_h", "in_zone", "foot_x", "foot_y"])
        w.writerows(per_person)

    total_people = sum(r[1] for r in per_image)
    print("\n========== BATCH SPATIAL SUMMARY ==========")
    print(f"  images processed : {len(per_image)}")
    print(f"  no crosswalk det : {no_cw}")
    print(f"  persons total    : {total_people}  -> {agg}")
    print(f"  annotated images : {out_dir} (<stem>_state.jpg)")
    print(f"  per-image  csv   : {sum_path}")
    print(f"  per-person csv   : {det_path}")
    print("===========================================")
    return per_image


# ---- synthetic self-test (proves the branches without a model) -------------
def selftest():
    ok = True

    # --- Scenario A: single crosswalk arm (hull == the arm) ---
    H, W = 400, 600
    m = np.zeros((H, W), np.uint8)
    m[250:350, 200:400] = 1
    boxesA = [
        [280, 200, 320, 350],   # feet on the crosswalk, big box      -> high_risk
        [150, 200, 190, 360],   # feet ~30px left, big box            -> medium_risk
        [500, 50, 540, 150],    # far from crosswalk, big box         -> low_risk
        [290, 255, 302, 272],   # feet ON crosswalk but tiny (far)    -> low_risk (size gate)
    ]
    sA = [p["state"] for p in classify_people(boxesA, m)]
    expA = [HIGH_RISK, MEDIUM_RISK, LOW_RISK, LOW_RISK]
    print("scenario A:", sA, "OK" if sA == expA else f"FAIL exp {expA}")
    ok &= sA == expA

    # --- Scenario B: two arms -> hull encloses the gap between them ---
    m2 = np.zeros((H, W), np.uint8)
    m2[250:350, 100:180] = 1   # left arm
    m2[250:350, 420:500] = 1   # right arm
    boxesB = [
        [285, 250, 315, 345],   # standing in the GAP (bare asphalt), big -> high_risk (in hull)
        [40, 250, 80, 345],     # left of the left arm, outside hull, near -> medium_risk
    ]
    pB = classify_people(boxesB, m2)
    sB = [p["state"] for p in pB]
    expB = [HIGH_RISK, MEDIUM_RISK]
    print("scenario B:", sB, "(in_zone:", [p["in_zone"] for p in pB], ")",
          "OK" if sB == expB else f"FAIL exp {expB}")
    ok &= sB == expB

    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description="CrossguardVision spatial reasoning (state classification)")
    ap.add_argument("--weights", help="YOLO-Seg weights (best.pt)")
    ap.add_argument("--image", help="Single image to classify")
    ap.add_argument("--images-dir", help="Folder of images -> classify every one (batch) + write CSVs")
    ap.add_argument("--out", default=None, help="Output dir (default <repo>/CrossguardVision/runs/spatial)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--crossing-k", type=float, default=0.10,
                    help="crossing if foot-to-crosswalk dist <= crossing_k * box_height")
    ap.add_argument("--waiting-k", type=float, default=0.50,
                    help="waiting if dist <= waiting_k * box_height (and not crossing)")
    ap.add_argument("--min-h-frac", type=float, default=0.05,
                    help="people shorter than this * imageHeight -> low_risk (too far)")
    ap.add_argument("--no-hull", action="store_true",
                    help="disable the intersection-interior (inner-edge) crossing zone")
    ap.add_argument("--min-arms", type=int, default=2,
                    help="min separate crosswalk blobs before an interior zone is built")
    ap.add_argument("--hull-erode-frac", type=float, default=0.0,
                    help="optional extra inward shrink of the zone (fraction of imageHeight)")
    ap.add_argument("--selftest", action="store_true", help="Run the synthetic geometry self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    if not (args.weights and (args.image or args.images_dir)):
        ap.error("provide --weights and (--image OR --images-dir), or use --selftest")

    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(repo, "CrossguardVision", "runs", "spatial")
    common = dict(conf=args.conf, imgsz=args.imgsz, crossing_k=args.crossing_k,
                  waiting_k=args.waiting_k, min_h_frac=args.min_h_frac,
                  use_hull=not args.no_hull, min_arms=args.min_arms,
                  hull_erode_frac=args.hull_erode_frac, device=args.device)
    if args.images_dir:
        run_folder(args.weights, args.images_dir, out_dir, **common)
    else:
        run_image(args.weights, args.image, out_dir, **common)


if __name__ == "__main__":
    main()
