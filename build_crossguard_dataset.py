#!/usr/bin/env python
"""
build_crossguard_dataset.py
---------------------------
Consolidate the labeled source folders into a single full dataset
(Crossguard_data) and split it 80:20 into train/test.

Steps:
  1. Read the 5 labeled source folders (read-only), collect image/JSON pairs.
  2. Renumber to crossguard_####.jpg / .json (contiguous), rewriting each
     JSON's "imagePath" to the new filename. Keep a rename_mapping.csv.
  3. Split 80:20 into Crossguard_data_train / Crossguard_data_test.

Split strategy (default): per-clip STRATIFIED RANDOM, seed=42 -> each source
clip contributes ~80/20, reproducible. NOTE: the frames come from only ~3
video scenes, so a random split shares scenes between train and test (mild
leakage); this is fine for RANKING the candidate models (all share the split)
but optimistic for novel-scene generalization. Use --group-holdout to hold out
whole clips instead (honest generalization, but coarse with 3 scenes).

Source folders are NEVER modified. The existing Crossguard_data, if present, is
moved to Crossguard_data_OLD<count> as a backup (not deleted).

Default is a DRY RUN (prints the plan, writes nothing). Pass --execute to write.

Usage:
  python build_crossguard_dataset.py                  # dry run / preview
  python build_crossguard_dataset.py --execute        # actually build
  python build_crossguard_dataset.py --execute --group-holdout 4453   # hold out a clip
"""

import argparse
import csv
import json
import os
import random
import shutil

# source folder -> clip id (which underlying video the frames came from)
SOURCES = [
    ("IMG_4453_done", "4453"),
    ("4453_2_done",   "4453"),
    ("181-210_done",  "4453"),
    ("IMG_2456_done", "2456"),
    ("IMG_0227_done", "0227"),
]
IMG_EXTS = (".jpg", ".jpeg", ".png")


def natural_key(name):
    # split digits so corss_0009 < corss_0010 < corss_0100
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def collect_pairs(dataset_root):
    """Return ordered list of dicts: {clip, src_folder, src_img, src_json, stem}."""
    pairs = []
    for folder, clip in SOURCES:
        fdir = os.path.join(dataset_root, folder)
        if not os.path.isdir(fdir):
            raise SystemExit(f"ERROR: source folder missing: {fdir}")
        imgs = [f for f in os.listdir(fdir)
                if os.path.splitext(f)[1].lower() in IMG_EXTS]
        imgs.sort(key=natural_key)
        for img in imgs:
            stem = os.path.splitext(img)[0]
            jpath = os.path.join(fdir, stem + ".json")
            if not os.path.exists(jpath):
                print(f"  WARN: no JSON for {folder}/{img} -- skipped")
                continue
            pairs.append({
                "clip": clip,
                "src_folder": folder,
                "src_img": os.path.join(fdir, img),
                "src_json": jpath,
                "src_name": img,
            })
    return pairs


def count_instances(json_paths):
    c = {}
    for jp in json_paths:
        d = json.load(open(jp, encoding="utf-8"))
        for s in d.get("shapes", []):
            c[s.get("label")] = c.get(s.get("label"), 0) + 1
    return c


def link_or_copy(src, dst, use_hardlink):
    if os.path.exists(dst):
        os.remove(dst)
    if use_hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="Consolidate + 80:20 split CrossguardVision dataset")
    ap.add_argument("--dataset-root",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "CrossguardVision", "dataset"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--group-holdout", default=None,
                    help="Clip id to hold out ENTIRELY as test (e.g. 4453). "
                         "Overrides stratified split.")
    ap.add_argument("--execute", action="store_true",
                    help="Actually write files (default is a dry run).")
    args = ap.parse_args()

    root = args.dataset_root
    full_dir = os.path.join(root, "Crossguard_data")
    train_dir = os.path.join(root, "Crossguard_data_train")
    test_dir = os.path.join(root, "Crossguard_data_test")

    print(f"dataset root : {root}")
    print(f"mode         : {'EXECUTE' if args.execute else 'DRY RUN (no writes)'}")
    print(f"seed={args.seed}  test_frac={args.test_frac}  "
          f"split={'group-holdout '+args.group_holdout if args.group_holdout else 'stratified-random'}")
    print()

    pairs = collect_pairs(root)
    # deterministic global numbering order: clip-grouped, then natural filename
    pairs.sort(key=lambda p: (p["clip"], natural_key(p["src_name"])))
    width = max(4, len(str(len(pairs))))
    for i, p in enumerate(pairs, start=1):
        p["new_stem"] = f"crossguard_{i:0{width}d}"

    # per-clip tallies
    by_clip = {}
    for p in pairs:
        by_clip.setdefault(p["clip"], []).append(p)
    print(f"collected {len(pairs)} image/JSON pairs:")
    for clip, ps in sorted(by_clip.items()):
        print(f"  clip {clip:6s}: {len(ps):4d} frames  "
              f"(folders: {', '.join(sorted({x['src_folder'] for x in ps}))})")
    print()

    # ---- split ----
    rng = random.Random(args.seed)
    test_set = set()
    if args.group_holdout:
        if args.group_holdout not in by_clip:
            raise SystemExit(f"ERROR: --group-holdout '{args.group_holdout}' not a known clip {list(by_clip)}")
        test_set = {p["new_stem"] for p in by_clip[args.group_holdout]}
    else:
        for clip, ps in by_clip.items():
            idx = list(range(len(ps)))
            rng.shuffle(idx)
            n_test = max(1, round(len(ps) * args.test_frac))
            for j in idx[:n_test]:
                test_set.add(ps[j]["new_stem"])

    train = [p for p in pairs if p["new_stem"] not in test_set]
    test = [p for p in pairs if p["new_stem"] in test_set]

    tr_inst = count_instances([p["src_json"] for p in train])
    te_inst = count_instances([p["src_json"] for p in test])
    print(f"SPLIT  train={len(train)}  test={len(test)}  "
          f"({len(train)/len(pairs)*100:.1f}% / {len(test)/len(pairs)*100:.1f}%)")
    print(f"  train instances: {tr_inst}")
    print(f"  test  instances: {te_inst}")
    print("  per-clip test counts:", {c: sum(1 for p in test if p['clip'] == c) for c in sorted(by_clip)})
    print()

    if not args.execute:
        print("DRY RUN complete -- no files written. Re-run with --execute to build.")
        return

    # ---- backup existing Crossguard_data ----
    if os.path.isdir(full_dir):
        n_old = len([f for f in os.listdir(full_dir) if f.lower().endswith(".jpg")])
        bak = os.path.join(root, f"Crossguard_data_OLD{n_old}")
        if os.path.exists(bak):
            shutil.rmtree(bak)
        shutil.move(full_dir, bak)
        print(f"backed up existing Crossguard_data ({n_old} imgs) -> {os.path.basename(bak)}")

    # ---- build full Crossguard_data ----
    os.makedirs(full_dir, exist_ok=True)
    mapping_rows = []
    for p in pairs:
        new_img = os.path.join(full_dir, p["new_stem"] + ".jpg")
        new_json = os.path.join(full_dir, p["new_stem"] + ".json")
        shutil.copy2(p["src_img"], new_img)
        d = json.load(open(p["src_json"], encoding="utf-8"))
        d["imagePath"] = p["new_stem"] + ".jpg"
        json.dump(d, open(new_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        mapping_rows.append([p["new_stem"], p["clip"], p["src_folder"], p["src_name"]])
    with open(os.path.join(full_dir, "rename_mapping.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["new_name", "clip", "source_folder", "original_name"])
        w.writerows(mapping_rows)
    print(f"built Crossguard_data: {len(pairs)} pairs + rename_mapping.csv")

    # ---- build train/test (hardlink from full set, fallback copy) ----
    for name, subset, ddir in [("train", train, train_dir), ("test", test, test_dir)]:
        if os.path.isdir(ddir):
            shutil.rmtree(ddir)
        os.makedirs(ddir)
        for p in subset:
            link_or_copy(os.path.join(full_dir, p["new_stem"] + ".jpg"),
                         os.path.join(ddir, p["new_stem"] + ".jpg"), use_hardlink=True)
            link_or_copy(os.path.join(full_dir, p["new_stem"] + ".json"),
                         os.path.join(ddir, p["new_stem"] + ".json"), use_hardlink=True)
        with open(os.path.join(ddir, f"{name}_list.txt"), "w", encoding="utf-8") as fh:
            for p in subset:
                fh.write(p["new_stem"] + ".jpg\n")
        print(f"built Crossguard_data_{name}: {len(subset)} pairs (+ {name}_list.txt)")

    # ---- split summary ----
    summary = {
        "seed": args.seed, "test_frac": args.test_frac,
        "strategy": f"group-holdout {args.group_holdout}" if args.group_holdout else "per-clip stratified random",
        "total": len(pairs), "train": len(train), "test": len(test),
        "train_instances": tr_inst, "test_instances": te_inst,
        "clips": {c: len(ps) for c, ps in sorted(by_clip.items())},
    }
    json.dump(summary, open(os.path.join(root, "split_summary.json"), "w", encoding="utf-8"), indent=2)
    print(f"wrote split_summary.json")
    print("\nDONE.")


if __name__ == "__main__":
    main()
