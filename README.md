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
