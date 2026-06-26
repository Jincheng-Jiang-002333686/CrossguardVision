# CrossguardVision Pipeline

CrossguardVision is a real-time crosswalk safety pipeline. The system detects pedestrians and the crosswalk area from camera frames, reasons about each pedestrian's position relative to the crosswalk, and sends a warning signal to a Raspberry Pi output such as LEDs or a warning light.

The predesign pipeline is documented in `ppt/CrossguardVision_predesign.pptx`. This README records the working pipeline and the current project state.

## Project Goal

Build a real-time system that can:

- Detect pedestrians.
- Detect the crosswalk area.
- Classify each pedestrian's risk as `high_risk`, `medium_risk`, or `low_risk`.
- Trigger a Raspberry Pi warning output when the scene requires attention.

Important design rule: `high_risk`, `medium_risk`, and `low_risk` are not YOLO training classes. They are produced after model inference by spatial reasoning (in the crossing zone -> `high_risk`, waiting near the crosswalk -> `medium_risk`, far away / too small to assess -> `low_risk`).

## Current Data

The current labeled dataset is here:

```text
CrossguardVision/dataset/Crossguard_data
```

Current contents:

- 269 `.jpg` images.
- 269 matching `.json` annotation files.
- 1 `rename_mapping.csv` file mapping normalized names back to source folders and frame names.
- File naming pattern: `crossguard_####.jpg` and `crossguard_####.json`.
- Annotation format: X-AnyLabeling / LabelMe-style JSON.

Current label counts:

- `person`: 3,896 labels.
- `crosswalk`: 945 labels.

Current shape types:

- `rectangle`: 3,897 shapes.
- `polygon`: 592 shapes.
- `quadrilateral`: 352 shapes.

The expected annotation meaning is:

- `person`: one rectangle around every visible pedestrian.
- `crosswalk`: polygon or quadrilateral marking the crosswalk area.

## Repository Layout

This repo tracks the code and logs only. Datasets, images/videos, model weights (`*.pt`),
and office documents are kept out (too large / private) — see `.gitignore`.

```text
.
|-- auto_label_person.py                 # pre-label pedestrians with a pretrained YOLO
|-- validate_labels.py                   # check X-AnyLabeling JSON before training
|-- build_crossguard_dataset.py          # consolidate + rename frames into Crossguard_data
|-- convert_xanylabeling_to_yoloseg.py   # JSON -> YOLO-Seg labels + train/val/test split
|-- crossguard_seg.yaml                  # dataset YAML (0: person, 1: crosswalk)
|-- train_yoloseg.py                     # fine-tune a YOLO-Seg model
|-- _eval_predict.py                     # eval + predict on the held-out test set
|-- bench_cpu.py                         # CPU latency benchmark
|-- make_comparison_csv.py               # build the model-comparison table
|-- spatial_reasoning.py                 # stage 8: geometry-only risk classification (+ --selftest)
|-- live_camera.py                       # stage 9: live Pi camera -> MJPEG risk stream
|-- command.txt                          # end-to-end pipeline commands
`-- logs/                                # work logs + model-comparison results
```

## Pipeline

### 1. Collect Road Videos

Record crosswalk scenes that include:

- Pedestrians crossing.
- Pedestrians waiting near the crosswalk.
- Pedestrians outside the risk area.
- Different distances, lighting conditions, vehicles, and partial occlusions.

Raw videos are stored in folders such as `CrossguardVision/photo/` and `CrossguardVision/dataset/`.

### 2. Extract Training Images

Convert video files into representative image frames. The predesign recommends sampling rather than labeling every frame because nearby frames are usually too similar.

Recommended extraction policy:

- Save about 1 frame every 30 frames.
- Remove blurry frames.
- Remove near-duplicates.
- Keep diverse examples across viewpoint, lighting, distance, crowd density, and occlusion.

### 3. Pre-label Pedestrians

Use `auto_label_person.py` to run a pretrained YOLO model over extracted frames and create initial X-AnyLabeling JSON files with `person` rectangles.

Example:

```bash
python auto_label_person.py \
  "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/IMG_0227" \
  "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/IMG_0232" \
  "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/IMG_2456" \
  "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/IMG_4453" \
  --model "E:/NEU files/TA/CrossguardVision/models/yolo11m.pt" \
  --conf 0.25 \
  --imgsz 1024
```

By default, the script skips images that already have a `.json` file, which protects manual annotation work. Use `--overwrite` only when you intentionally want to regenerate existing JSON files.

### 4. Manually Annotate and Correct Labels

Open the image folders in X-AnyLabeling and review the generated labels.

Manual work:

- Fix missing, incorrect, or oversized `person` boxes.
- Add `crosswalk` shapes as polygons or quadrilaterals.
- Make sure labels use only the agreed class names: `person` and `crosswalk`.
- Save one JSON file next to each image.

The final training dataset has been consolidated into `CrossguardVision/dataset/Crossguard_data`.

### 5. Validate Labels

Run the validation script before training:

```bash
python validate_labels.py "E:/NEU files/TA/CrossguardVision/CrossguardVision/dataset/Crossguard_data"
```

Latest validation result for `Crossguard_data`:

- 269 JSON files.
- 0 images without JSON.
- 0 JSON files without images.
- 15 validation errors.

Known cleanup items:

- `crossguard_0012.json`, `crossguard_0013.json`, `crossguard_0014.json`, and `crossguard_0015.json` are missing the top-level `description` key.
- Several annotations have points slightly outside the image boundary and should be clamped or manually corrected.
- Some `crosswalk` labels use `shape_type: quadrilateral`; this may need conversion to `polygon` depending on the training converter.

### 6. Convert Dataset for YOLO-Seg

The training target is YOLO segmentation because the crosswalk area needs a mask or polygon, not only a bounding box. Use `convert_xanylabeling_to_yoloseg.py`:

- Convert LabelMe/X-AnyLabeling JSON into YOLO segmentation format.
- Preserve `person` annotations as boxes or segmentation-compatible objects according to the training setup.
- Preserve `crosswalk` polygon/quadrilateral points as segmentation masks.
- Create a train/validation/test split.
- Create a dataset YAML with the classes:

```yaml
names:
  0: person
  1: crosswalk
```

### 7. Fine-tune and Compare YOLO-Seg Models

Train with `train_yoloseg.py`, evaluate/predict with `_eval_predict.py`. Each model outputs a class name, bounding box, confidence, and a crosswalk segmentation mask (a rectangular crosswalk box is not accurate enough for risk classification).

Six models were trained and compared end-to-end on the honest 80:20 split (YOLO11 vs YOLO26 at n/s/m scales) — see `logs/worklog_2026-06-23.md` and `logs/model_comparison_2026-06-23.csv`. **`yolo11n-seg`** was chosen for deployment: best crosswalk-mask mAP50 point estimate, smallest, and fastest on CPU.

### 8. Spatial Reasoning

`spatial_reasoning.py` classifies each pedestrian from geometry only (person boxes + the crosswalk mask), so it can be unit-tested without a model (`python spatial_reasoning.py --selftest`):

- The standing point is the bottom-centre of the person box (where the feet are).
- Distance from the feet to the crosswalk is normalised by the person's box height, giving a perspective-invariant "how many body-lengths away" measure (someone far across the intersection looks tiny, so a fixed pixel distance is meaningless).
- A "crossing zone" is built from the crosswalk arms plus the intersection interior bounded by their inner edges, because people also cross *between* the painted stripes.
- Feet inside the crossing zone (or within `crossing_k` body-lengths of the crosswalk) -> `high_risk`; within `waiting_k` body-lengths -> `medium_risk`; otherwise, or too small/far to assess -> `low_risk`.

This stage produces the safety risk labels. They are not drawn as YOLO training classes.

### 9. Raspberry Pi Deployment (live)

The pipeline runs live on a Raspberry Pi 4 (Cortex-A72, OV5647 camera, Debian Bookworm). `live_camera.py` captures frames with Picamera2, runs the model plus the spatial-reasoning geometry on every frame (reusing `spatial_reasoning.py` unchanged), overlays each person's risk colour, and serves the annotated video as an MJPEG stream at `http://<pi-ip>:8000/` (head-less / SSH-friendly). Use `--display` for a wired monitor, or `--source` for a video file / USB camera.

The chosen `yolo11n-seg` model is exported to **NCNN** on the Pi for a ~2.8x CPU speedup:

```bash
# on the Pi, inside the venv:
python live_camera.py --weights yolo11n-seg-crossguard_ncnn_model --imgsz 480
```

Measured Pi-4 latency (imgsz 640, CPU, full pipeline per frame):

| Runtime | ms/frame | FPS |
|---|---:|---:|
| torch 2.3.1 | ~1450 | 0.7 |
| NCNN | ~515 | 1.9 |

Pi setup notes are in `logs/worklog_2026-06-26.md`. The key gotcha: the default aarch64 PyTorch wheel crashes with `Illegal instruction` on the Pi's Cortex-A72 — pin `torch==2.3.1` (with `numpy<2` / `opencv-python<4.10`). The remaining step is wiring the `high_risk` decision into a Raspberry Pi warning output (LED / light / picar-x actuation).

## Status and Next Steps

Done: dataset labeled and validated, converted to YOLO-Seg with an 80:20 split, six models trained and compared end-to-end, the spatial-reasoning stage implemented, and `yolo11n-seg` deployed live on the Raspberry Pi via NCNN (see `logs/`).

Next:

1. Validate the `medium_risk` / `high_risk` levels against a real crosswalk scene on the Pi.
2. Wire the risk decision into a Raspberry Pi warning output (LED / light / picar-x).
3. Optional: auto-start `live_camera.py` as a systemd service; draw the accuracy-vs-FPS plot.
