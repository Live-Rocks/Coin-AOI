# Coin-AOI

Coin-AOI is a learning prototype for coin surface inspection. This first stage
only proves that a local Python environment can run an Ultralytics pretrained
YOLO model on one image.

It is **not** a coin-defect detector yet. The pretrained model knows the
general COCO object classes, not defects such as scratches, stains, or dents.
It may therefore produce no boxes for a coin image. That is a valid result for
this environment check; custom coin-defect training comes later.

## 1. Create and activate the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

A virtual environment keeps this project's Python packages separate from other
projects on your computer.

## 2. Install YOLO

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`ultralytics` provides the YOLO model, image preprocessing, inference, and
bounding-box drawing. On its first use, it downloads the small `yolo11n.pt`
pretrained weights.

## 3. Run inference on one coin image

```bash
mkdir -p data/local
# Copy your image to data/local/coin.jpg, then run:
python src/inference.py --source data/local/coin.jpg
```

`data/local/` is ignored by Git, so your original inspection images stay on
your computer rather than being committed to the repository.

Optional arguments:

```bash
# Keep only detections with confidence at least 0.50
python src/inference.py --source data/local/coin.jpg --confidence 0.50

# Force Apple Silicon acceleration when available
python src/inference.py --source data/local/coin.jpg --device mps
```

The script writes these files to `outputs/inference/`:

- `annotated_<image-name>.jpg`: your source image with any detected bounding
  boxes drawn on it.
- `detections_<image-name>.json`: each prediction's class, confidence, and
  bounding-box coordinates.

## Key terms

- **Pretrained model**: a model already trained on a large general-purpose
  dataset. Here it confirms that the software pipeline works.
- **Bounding box**: a rectangle locating an object in an image.
- **Confidence**: the model's estimated certainty, from 0 to 1, for a detected
  class.
- **Inference**: using a trained model to make predictions on a new image.

## Important limitation

Do not interpret an empty result as `PASS`. At this stage it only means the
general-purpose model found no supported COCO object at the selected confidence
threshold. A future fine-tuned model and a deterministic QC rule will be needed
before Coin-AOI can make a PASS/FAIL decision.

## Dataset and annotation

The first custom dataset uses three known defect classes:

```text
0 dent
1 scratch
2 stain_corrosion
```

Read the [dataset workflow](data/README.md) before collecting or exporting
images, and follow the [annotation guidelines](docs/annotation-guidelines.md)
when using Roboflow. These documents define the class boundaries, image-source
records, and group split rule that prevents photos of the same physical coin
from leaking across train, validation, and test data.

The numeric IDs mirror the current Roboflow YOLO export. The order is not a
quality ranking; it only has to remain consistent in `data.yaml`, label files,
training, and inference.

Download a Roboflow Object Detection export to `data/roboflow/`, then copy the
selected image/label pairs into `datasets/coin-defect-hybrid/`. Create
`data/manifest.csv` from the supplied template and validate the local dataset:

```bash
cp data/manifest_template.csv data/manifest.csv
python src/validate_dataset.py
```

The validator checks that images and labels are paired, normal images have empty
label files, class IDs are valid, and each `coin_id` appears in only one split.
Raw images and exported datasets are deliberately ignored by Git; source URLs,
licenses, and capture details belong in the manifest.

## Fine-tuning smoke test

The earlier one-epoch smoke test verified the initial local data, training,
validation, and checkpoint pipeline:

```bash
python src/train_smoke.py
```

The script validates the dataset first, then fine-tunes `yolo11n.pt` for one
epoch at 320px with batch size 2. It attempted MPS but used CPU because MPS was
unavailable at runtime. Its local outputs are ignored by Git.

## CPU training baseline

The initial five-epoch, 320px runs used an earlier Roboflow `Stretch to`
export. They remain local debugging artifacts only: stretching changed the
coin geometry, so do not use those runs for model conclusions.

The corrected export uses Roboflow `Fit within`, which preserves image aspect
ratio. Run the current exploratory baseline at the YOLO11 reference input size:

```bash
python src/train_baseline.py \
  --epochs 100 \
  --imgsz 640 \
  --run-name yolo11n-cpu-e100-s0-fit640
```

The script validates the 26-image grouped dataset first, then fine-tunes
`yolo11n.pt` using CPU, 640px images, batch size 2, seed 0,
deterministic mode, zero dataloader workers, and no cache. It then loads
`best.pt` and runs an Ultralytics validation pass on the test split. Outputs are
local-only under `runs/detect/baseline/`, so this does not overwrite the smoke
run.

This baseline is a reproducible pipeline artifact, not model evaluation. The
test split contains three normal images plus one `dent` and one
`stain_corrosion` image, but no `scratch` ground truth. The validation and test
sets are far too small for mAP, precision, recall, prediction thresholds, empty
predictions, or PASS/FAIL results to have a quality interpretation.

## Documentation

- [Product scope and limitations](docs/PRODUCT.md)
- [Implemented and planned architecture](docs/ARCHITECTURE.md)
- [Current verified project status](docs/PROJECT_STATUS.md)
- [Technical and data decisions](docs/DECISIONS.md)
- [Annotation guidelines](docs/annotation-guidelines.md)
