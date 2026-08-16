# Project Status

Last verified: 2026-08-17

## Current capability

- [x] Create an isolated Python environment and install Ultralytics YOLO.
- [x] Run pretrained YOLO single-image inference and save an annotated image
  plus JSON detections.
- [x] Define a three-class annotation taxonomy and validate a local YOLO export.
- [ ] Fine-tune a coin-defect detector.
- [ ] Evaluate a trained detector on an independent test set.
- [ ] Implement a deterministic PASS/FAIL rule.

## Dataset reality

The current dataset is a pipeline smoke test, not a training-ready release.

| Item | Current state |
| --- | --- |
| Usable annotated images | 12 |
| Physical coins represented | 3 (`C002`, `C004`, `C005`) |
| Available split | train only |
| Class mapping | `0=dent`, `1=scratch`, `2=stain_corrosion` |
| Image source | self-captured only |
| Normal images | none in the validated pilot |
| Validation status | `src/validate_dataset.py` passed |

The image and label files are local-only. Their metadata is recorded in
[`data/manifest.csv`](../data/manifest.csv).

## Current blocker

There is no representative dataset with normal samples and independent
validation/test partitions. The current 12-image train-only pilot can test data
format and pipeline behaviour, but cannot support a credible performance claim.

## Next milestone

Create and validate a grouped train/val/test dataset release. Every physical
coin must be assigned to a single split before augmentation.

## Claims we can make

- The local pretrained inference path works.
- The current YOLO export uses the documented three-class mapping.
- The pilot manifest and local export pass structural validation.

## Claims we cannot make

- Fine-tuned defect-detection accuracy, mAP, precision, or recall.
- Generalisation to unseen coins, lighting, or defects.
- PASS/FAIL accuracy or production readiness.

## Maintenance rule

Update this file only when a capability, dataset fact, blocker, or evidence-based
milestone changes. Each completed item should have a reproducible command,
artifact, or commit behind it.
