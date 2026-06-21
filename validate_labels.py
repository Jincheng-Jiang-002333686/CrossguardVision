#!/usr/bin/env python
"""Validate X-AnyLabeling JSONs against the required schema + image geometry."""
import json, os, sys
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TOP_KEYS = {"version","flags","checked","shapes","imagePath","imageData","imageHeight","imageWidth","description"}
SHAPE_KEYS = {"label","score","points","group_id","description","difficult","shape_type","flags","attributes","kie_linking"}
ALLOWED_LABELS = {"person","crosswalk"}

def validate_folder(folder):
    errors, warns = [], []
    imgs = {os.path.splitext(f)[0] for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXTS}
    jsons = {os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith(".json")}

    img_no_json = sorted(imgs - jsons)
    json_no_img = sorted(jsons - imgs)
    for s in img_no_json: warns.append(f"image without json: {s}")
    for s in json_no_img: errors.append(f"json without image: {s}")

    n_person = n_crosswalk = n_files = n_manual = 0
    for stem in sorted(jsons & imgs):
        jp = os.path.join(folder, stem + ".json")
        n_files += 1
        try:
            d = json.load(open(jp, encoding="utf-8"))
        except Exception as e:
            errors.append(f"{stem}.json: cannot parse: {e}"); continue

        missing = TOP_KEYS - set(d.keys())
        extra = set(d.keys()) - TOP_KEYS
        if missing: errors.append(f"{stem}.json: missing top keys {missing}")
        if extra: warns.append(f"{stem}.json: extra top keys {extra}")

        # real image dims
        ip = os.path.join(folder, [f for f in os.listdir(folder)
              if os.path.splitext(f)[0]==stem and os.path.splitext(f)[1].lower() in IMG_EXTS][0])
        with Image.open(ip) as im: W, H = im.size
        if d.get("imageWidth") != W or d.get("imageHeight") != H:
            errors.append(f"{stem}.json: dims {d.get('imageWidth')}x{d.get('imageHeight')} != real {W}x{H}")
        if d.get("imagePath") != os.path.basename(ip):
            errors.append(f"{stem}.json: imagePath '{d.get('imagePath')}' != '{os.path.basename(ip)}'")

        has_polygon = False
        for i, sh in enumerate(d.get("shapes", [])):
            sk_missing = SHAPE_KEYS - set(sh.keys())
            if sk_missing: errors.append(f"{stem}.json shape{i}: missing keys {sk_missing}")
            lbl = sh.get("label")
            if lbl not in ALLOWED_LABELS: warns.append(f"{stem}.json shape{i}: label '{lbl}' not in {ALLOWED_LABELS}")
            if lbl == "person": n_person += 1
            if lbl == "crosswalk": n_crosswalk += 1
            st = sh.get("shape_type")
            pts = sh.get("points", [])
            if st == "rectangle":
                if len(pts) != 4:
                    errors.append(f"{stem}.json shape{i}: rectangle has {len(pts)} pts (want 4)")
            elif st == "polygon":
                has_polygon = True
                if len(pts) < 3:
                    errors.append(f"{stem}.json shape{i}: polygon has {len(pts)} pts (<3)")
            # bounds check
            for (x, y) in pts:
                if not (-0.5 <= x <= W + 0.5 and -0.5 <= y <= H + 0.5):
                    errors.append(f"{stem}.json shape{i}: point ({x:.1f},{y:.1f}) out of bounds {W}x{H}")
        if has_polygon:
            n_manual += 1
    return dict(folder=folder, files=n_files, person=n_person, crosswalk=n_crosswalk,
                manual_like=n_manual, img_no_json=len(img_no_json), json_no_img=len(json_no_img),
                errors=errors, warns=warns)

def main():
    total_err = 0
    for folder in sys.argv[1:]:
        r = validate_folder(folder)
        print(f"\n=== {r['folder']} ===")
        print(f"  json files       : {r['files']}")
        print(f"  person boxes     : {r['person']}")
        print(f"  crosswalk polys  : {r['crosswalk']}  (in {r['manual_like']} files w/ polygons)")
        print(f"  images w/o json  : {r['img_no_json']}")
        print(f"  json w/o image   : {r['json_no_img']}")
        print(f"  ERRORS           : {len(r['errors'])}")
        for e in r['errors'][:15]: print(f"     ! {e}")
        if len(r['errors']) > 15: print(f"     ... +{len(r['errors'])-15} more")
        if r['warns']:
            print(f"  warnings         : {len(r['warns'])}")
            for w in r['warns'][:5]: print(f"     ~ {w}")
            if len(r['warns'])>5: print(f"     ... +{len(r['warns'])-5} more")
        total_err += len(r['errors'])
    print(f"\n>>> TOTAL ERRORS ACROSS ALL FOLDERS: {total_err}")
    sys.exit(1 if total_err else 0)

if __name__ == "__main__":
    main()
