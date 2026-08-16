# Project Status

Last verified: 2026-08-17

## Current capability

- [x] Create an isolated Python environment and install Ultralytics YOLO.
- [x] Run pretrained YOLO single-image inference and save an annotated image
  plus JSON detections.
- [x] Define a three-class annotation taxonomy and validate a local YOLO export.
- [x] Complete a one-epoch YOLO fine-tuning smoke test and load its checkpoint
  for inference.
- [x] Run a corrected 640px CPU exploratory baseline and validate its
  checkpoint on the test split.
- [ ] Train a training-ready coin-defect detector.
- [ ] Evaluate a trained detector on an independent test set.
- [ ] Implement a deterministic PASS/FAIL rule.

## Dataset reality

The current dataset is a complete pipeline smoke test, not a training-ready
release.

| Item | Current state |
| --- | --- |
| Usable annotated images | 26 |
| Physical coins represented | 9 (`C001`, `C002`, `C004`–`C010`) |
| Available split | train: 15, val: 6, test: 5 |
| Class mapping | `0=dent`, `1=scratch`, `2=stain_corrosion` |
| Image source | self-captured only |
| Normal images | 7 empty-label images |
| Image geometry | Roboflow `Fit within`; original aspect ratios preserved |
| Validation status | `src/validate_dataset.py` passed |

The image and label files are local-only. Their metadata is recorded in
[`data/manifest.csv`](../data/manifest.csv).

## Smoke training result

`python src/train_smoke.py` completed one epoch after validating the dataset and
produced `best.pt`, `last.pt`, training logs, and validation artifacts under
`runs/detect/runs/smoke/yolo11n-mps-s0/`. The checkpoint loaded successfully
for a single-image inference smoke test.

PyTorch was built with MPS support, but MPS was unavailable at runtime, so
Ultralytics trained on CPU (Apple M2). This verifies CPU fallback, not MPS
training performance. The one-epoch run produced zero detections at the default
confidence threshold; that result has no quality interpretation.

## Corrected 640px CPU baseline result

Earlier 320px runs used a Roboflow `Stretch to` export that altered the coin
geometry. They are retained as pipeline-debugging artifacts, not model
evidence.

`python src/train_baseline.py --epochs 100 --imgsz 640 --run-name
yolo11n-cpu-e100-s0-fit640` completed on CPU (Apple M2) after the corrected
`Fit within` dataset passed validation. The run used `yolo11n.pt`, 100 epochs,
640px images, batch size 2, seed 0, deterministic mode, zero workers, and no
cache. It produced `args.yaml`, `results.csv`, plots, `best.pt`, and `last.pt`
under `runs/detect/baseline/yolo11n-cpu-e100-s0-fit640/`.

The saved `best.pt` completed test-split validation and generated plots under
`runs/detect/baseline/yolo11n-cpu-e100-s0-fit640-test/`. At confidence 0.25,
it produced no boxes for any test image, including the one `dent` and one
`stain_corrosion` image. The test split has no `scratch` ground truth, so no
metric, prediction, or threshold from this run is evidence of detector quality.

## Current blocker

The grouped train/val/test flow now works and validation contains all three
classes, but the dataset remains too small. The test split lacks `scratch` and
has only one labelled example each for `dent` and `stain_corrosion`, so it
cannot support a credible performance claim or defect-recall measurement.

## Next milestone

Collect more independent coins so validation and test contain all defect
classes and normal images. Every physical coin must remain in one split before
augmentation.

## Claims we can make

- The local pretrained inference path works.
- The current YOLO export uses the documented three-class mapping.
- The pilot manifest and local export pass structural validation.
- The corrected 640px CPU baseline and its checkpoint test-validation workflow
  completed reproducibly on the aspect-ratio-preserving local dataset.

## Claims we cannot make

- Fine-tuned defect-detection accuracy, mAP, precision, or recall.
- Generalisation to unseen coins, lighting, or defects.
- PASS/FAIL accuracy or production readiness.

## Maintenance rule

Update this file only when a capability, dataset fact, blocker, or evidence-based
milestone changes. Each completed item should have a reproducible command,
artifact, or commit behind it.
