# Coin-AOI

Coin-AOI is a computer-vision learning prototype: capture coin photos, label
defects, fine-tune YOLO11n, and evaluate whether the model generalises. It is
**not** a usable coin-defect detector and not an industrial AOI system.

The pipeline that exists today:

```text
self-captured coins → Roboflow labels → YOLO11n fine-tune
→ train can memorise reviewed rim labels
→ held-out coins fail
→ hosted Roboflow API is wired, but the cloud endpoint returns HTTP 500
```

That negative result is the main finding. After 100 epochs, `last.pt` localised
all seven training `rim_dent` boxes at confidence 0.25 and rejected the six
training target-negatives. The same checkpoint missed both held-out validation
rims and the sealed test coin. This is overfitting / failed cross-coin
generalisation, not a missing confidence threshold.

## What this repo demonstrates

- Python: isolated venv, Ultralytics YOLO, dataset validation, and evaluation
  scripts that turn one idea into a runnable experiment.
- Object detection: YOLO11n fine-tuning, grouped train/val/test splits, and
  single-image inference with JSON plus annotated-image output.
- ML basics: training versus validation, leakage-aware `coin_id` splits, loss
  and mAP as diagnostics, and an explicit train-memorisation versus
  generalisation check.
- AI-assisted development: the repo was built and diagnosed in Cursor, with
  API keys kept out of Git.

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

The default `src/inference.py` path is an environment check with pretrained
COCO YOLO11n. Empty boxes are expected; they are not a coin-defect result.

```bash
mkdir -p data/local
# Copy your image to data/local/coin.jpg, then run:
python src/inference.py --source data/local/coin.jpg
```

`data/local/` is ignored by Git. Optional `--confidence` and `--device`
flags are documented in `python src/inference.py --help`. Outputs go to
`outputs/inference/` as an annotated image and a JSON file.

## 4. Hosted Roboflow client (known external blocker)

`src/inference_roboflow.py` is a dependency-free REST client for the published
v9 Workflow. It reads `ROBOFLOW_API_KEY` from the environment, retries
transient failures, and writes compact JSON. As last verified on 2026-08-17,
Roboflow serverless returns HTTP 500 for both the model and the Workflow.
That is a cloud-side blocker; the local client and its unit tests already
exist. Do not treat a 500 as a passing detector.

```bash
set -a
source .env
set +a
python src/inference_roboflow.py --source data/local/coin.jpg
```

The fixed `C005_03` acceptance command, SHA-256, and opt-in live test are in
the Hosted Workflow notes below if you need to re-run the smoke gate.

## Key terms

- **Pretrained model**: a model already trained on a large general-purpose
  dataset. Here it confirms that the software pipeline works.
- **Bounding box**: a rectangle locating an object in an image.
- **Confidence**: the model's estimated certainty, from 0 to 1, for a detected
  class.
- **Inference**: using a trained model to make predictions on a new image.

## Important limitation

Do not interpret an empty result as `PASS`. A missing box can mean the
pretrained COCO model saw no supported class, the fine-tuned model failed to
generalise, or a hosted endpoint returned an error. Coin-AOI has no
deterministic quality-control rule.

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

The script validates the grouped dataset first, then fine-tunes
`yolo11n.pt` using CPU, 640px images, batch size 2, seed 0,
deterministic mode, zero dataloader workers, and no cache. It then loads
`best.pt` and runs an Ultralytics validation pass on the test split. Outputs are
local-only under `runs/detect/baseline/`, so this does not overwrite the smoke
run.

This baseline is a reproducible pipeline artifact, not model evaluation. The
current test split contains five normal images plus one `dent` and one
`stain_corrosion` image, but no `scratch` ground truth. The validation and test
sets are far too small for mAP, precision, recall, prediction thresholds, empty
predictions, or PASS/FAIL results to have a quality interpretation.

## Dent-only recognition experiment

`src/build_dent_dataset.py` creates a separate one-class dataset from manifest
rows that contain only `dent` or are confirmed normal. It excludes images with
`scratch` or `stain_corrosion` so they are not incorrectly treated as negative
examples:

```bash
python src/build_dent_dataset.py --replace
python src/validate_dataset.py \
  --dataset-root datasets/coin-dent-v1 \
  --manifest data/dent_dataset_manifest.csv \
  --class-names dent
```

The first controlled experiment used 640px inputs, CPU, 100 epochs, and Mosaic
disabled:

```bash
python src/train_baseline.py \
  --dataset-config datasets/coin-dent-v1/data.yaml \
  --manifest data/dent_dataset_manifest.csv \
  --class-names dent \
  --epochs 100 --imgsz 640 --mosaic 0 \
  --run-name dent-v1-yolo11n-cpu-e100-i640-m0-s0
```

It did not detect the held-out C009 dent at confidence 0.25 and produced false
positive dent boxes on normal test images. This is an honest negative result,
not a usable detector. The per-image evidence can be reproduced with:

```bash
python src/evaluate_dent.py \
  --model runs/detect/baseline/dent-v1-yolo11n-cpu-e100-i640-m0-s0/weights/best.pt \
  --output-dir outputs/dent-evaluation/dent-v1-yolo11n-cpu-e100-i640-m0-s0
```

## Reviewed rim-dent experiment

A manual review narrowed the target to visible outer-rim deformation. Ten
existing labels were retained as `rim_dent`; two face dents and two ambiguous
design-region marks became target-negative hard examples. The complete C011
coin group moved from train to validation, while C009 remained sealed in test:

```bash
python src/build_rim_dent_dataset.py --replace
python src/validate_dataset.py \
  --dataset-root datasets/coin-rim-dent-v2 \
  --manifest data/rim_dent_v2_manifest.csv \
  --class-names rim_dent
```

The resulting 25-image dataset contains 7/2/1 positive images and 6/4/5
target-negative images across train/validation/test. The controlled run kept
the previous model and training settings:

```bash
python src/train_baseline.py \
  --dataset-config datasets/coin-rim-dent-v2/data.yaml \
  --manifest data/rim_dent_v2_manifest.csv \
  --class-names rim_dent \
  --epochs 100 --imgsz 640 --mosaic 0 \
  --run-name rim-dent-v2-yolo11n-cpu-e100-i640-m0-s0
```

`last.pt` memorised all seven train rim labels at confidence 0.25 without a
false positive on the six train target-negatives. Neither checkpoint detected
the two held-out C011 validation rims at that threshold. The frozen `best.pt`
then missed C009 and produced no boxes on any of the five test negatives. This
isolates a cross-coin generalisation failure; it is not a usable rim detector.

## Hosted Workflow notes

Store `ROBOFLOW_API_KEY` in a local `.env` file that Git ignores. The
acceptance gate for the published v9 Workflow is:

```bash
python src/inference_roboflow.py \
  --source data/local/roboflow-smoke/C005_03.png \
  --source-url https://source.roboflow.com/RBizhxjW0kge5Flii3Wlp7jUK8g2/vGVZc5ZRWDsD2KfIlyLh/original.jpg \
  --expect-output predictions \
  --expect-class scratch \
  --output-dir outputs/roboflow-smoke
```

The fixed image SHA-256 is
`d1ac6fdf0170266bf02bf89b91176e9c64cee415eb303e397e4857167a34aec0`.
Live failures write `outputs/roboflow-smoke/failure_C005_03.json`. Offline
client tests:

```bash
python -m unittest discover -s tests -p 'test_inference_roboflow.py' -v
```

The credit-consuming live test stays opt-in with `RUN_ROBOFLOW_LIVE=1`.
Roboflow reference IDs for the 2026-08-17 HTTP 500s are
`46ecb284e6bd8db56380dbf5732bf501` (model) and
`6dedb323169641584dd70b12ebc8eafe` (Workflow).

## Documentation

- [Product scope and limitations](docs/PRODUCT.md)
- [Implemented and planned architecture](docs/ARCHITECTURE.md)
- [Current verified project status](docs/PROJECT_STATUS.md)
- [Technical and data decisions](docs/DECISIONS.md)
- [Annotation guidelines](docs/annotation-guidelines.md)
