# CrossguardVision Pipeline

CrossguardVision is a real-time crosswalk safety pipeline. The system detects pedestrians and the crosswalk area from camera frames, reasons about each pedestrian's position relative to the crosswalk, and sends a warning signal to a Raspberry Pi output such as LEDs or a warning light.

The predesign pipeline is documented in `../ppt/CrossguardVision_predesign.pptx`. This README records the working pipeline and the current project state.

## Project Goal

Build a real-time system that can:

- Detect pedestrians.
- Detect the crosswalk area.
- Classify each pedestrian as `crossing`, `waiting`, or `no_risk`.
- Trigger a Raspberry Pi warning output when the scene requires attention.

Important design rule: `crossing`, `waiting`, and `no_risk` are not YOLO training classes. They are produced after model inference by spatial reasoning.

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

```text
.
|-- auto_label_person.py
|-- validate_labels.py
|-- models/
|   `-- yolo11m.pt
|-- ppt/
|   `-- CrossguardVision_predesign.pptx
|-- CrossguardVision/
|   |-- dataset/
|   |   |-- Crossguard_data/
|   |   |-- IMG_0227/
|   |   |-- IMG_0232/
|   |   |-- IMG_2456/
|   |   |-- IMG_4453/
|   |   `-- *_done/
|   |-- logs/
|   |-- photo/
|   |-- annotations/
|   `-- generated/
`-- X-AnyLabeling/
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

The training target is YOLO segmentation because the crosswalk area needs a mask or polygon, not only a bounding box.

Expected conversion:

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

### 7. Fine-tune YOLO-Seg

Train and evaluate a YOLO segmentation model on the converted dataset.

The model should output:

- Class name.
- Bounding box.
- Confidence.
- Segmentation mask for the crosswalk area.

The crosswalk mask is required because a rectangular crosswalk box is not accurate enough for state classification.

### 8. Spatial Reasoning

After inference, classify each pedestrian using geometry:

- Use the bottom region or bottom point of the person box because it represents where the pedestrian is standing.
- If the bottom region overlaps the crosswalk mask, classify as `crossing`.
- If the bottom point is in a waiting zone near the crosswalk but not inside it, classify as `waiting`.
- Otherwise classify as `no_risk`.

This stage produces the safety state labels. These labels should not be drawn as YOLO training classes.

### 9. Raspberry Pi Deployment

The final runtime loop is:

1. Camera captures video frames.
2. YOLO-Seg predicts pedestrians and crosswalk masks.
3. Python spatial reasoning classifies pedestrian state.
4. Raspberry Pi controls LEDs or a warning light.

## Immediate Next Steps

1. Clean the 15 validation errors in `Crossguard_data`.
2. Decide whether to normalize all `crosswalk` shapes to `polygon`.
3. Add or run the LabelMe-to-YOLO-Seg conversion script.
4. Split the dataset into train/validation/test sets.
5. Fine-tune YOLO-Seg.
6. Implement and test the spatial reasoning module.
7. Integrate Raspberry Pi warning output.
