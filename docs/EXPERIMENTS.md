# Experiment Log

This file preserves Coin-AOI's experiment history, including negative results.
The [README](../README.md) presents only the current v13 portfolio path.

## Experiment matrix

| Experiment | Dataset | Model and budget | Purpose | Outcome |
| --- | --- | --- | --- | --- |
| Initial smoke | train-only pilot | YOLO11n, 1 epoch, 320px | Verify local fine-tuning and checkpoints | Pipeline completed; no quality claim |
| Corrected baseline | 26 images, `15/6/5` | YOLO11n, 100 epochs, 640px | Replace geometry-distorting export | Test predictions empty; split too small |
| Dent-only v1 | 25 images, `15/4/6` | YOLO11n, 100 epochs, Mosaic off | Isolate one defect class | Missed C009 and produced normal-image false positives |
| Rim-dent v2 | 25 images, `13/6/6` | YOLO11n, 100 epochs, Mosaic off | Test reviewed rim labels | Memorised train labels; missed held-out coins |
| Roboflow v13 cloud | v13 cloud version | YOLO11n, 300 epochs | Roboflow-managed training | Configuration recorded; no verified local metrics comparison |
| Local v13 smoke | 102 exported images, `93/6/3` | YOLO11n, 100 epochs, augmentation off | Reproducible fixed gate | Dent and normal passed; scratch missed |

Cloud and local epoch counts are not an architecture comparison. The Roboflow
training engine, augmentation handling, and checkpoint policy were not frozen
to the local Ultralytics configuration.

## Initial one-epoch smoke

The first train-only pilot verified that the environment, Roboflow export,
Ultralytics training loop, and checkpoint loading worked. It did not contain a
valid held-out split and therefore produced no detector-quality evidence.

```bash
python -m experiments.legacy.train_smoke
```

The machine had an Apple M2 CPU. MPS was unavailable at runtime, so this result
was retained only as pipeline-debugging history.

## Corrected 640px baseline

An early Roboflow export used `Stretch to 512×512`, which changed coin geometry.
The corrected export used `Fit within 640×640` and a deterministic CPU recipe:

```bash
python -m experiments.legacy.train_baseline \
  --epochs 100 --imgsz 640 \
  --run-name yolo11n-cpu-e100-s0-fit640
```

The held-out split contained five normal images, one dent, and one
stain-corrosion instance, but no scratch ground truth. `best.pt` produced no
test boxes at confidence 0.25. This remained a reproducible baseline rather
than a performance claim.

## Dent-only v1

Mixed-defect images were excluded so unlabelled scratch or corrosion would not
become false negatives in a one-class detector. The derived dataset retained
only exact dent labels and confirmed normal images.

```bash
python -m experiments.legacy.build_dent_dataset --replace
python src/validate_dataset.py \
  --dataset-root datasets/coin-dent-v1 \
  --manifest data/dent_dataset_manifest.csv \
  --class-names dent
python -m experiments.legacy.train_baseline \
  --dataset-config datasets/coin-dent-v1/data.yaml \
  --manifest data/dent_dataset_manifest.csv \
  --class-names dent --epochs 100 --imgsz 640 --mosaic 0 \
  --run-name dent-v1-yolo11n-cpu-e100-i640-m0-s0
```

At confidence 0.25 the checkpoint missed the held-out C009 dent and produced
false-positive dent boxes on normal C006 and C018 images. A train-side
diagnostic at confidence 0.05 matched 6 of 10 ground-truth dents but produced
75 boxes across five normal training images. Lowering the threshold did not
create a usable model.

## Reviewed rim-dent v2

Manual review narrowed the target to visible outer-rim deformation. Face dents
and ambiguous design-region marks became target-negative hard examples. C011
moved as a complete coin group to validation; C009 remained sealed in test.

```bash
python -m experiments.legacy.build_rim_dent_dataset --replace
python src/validate_dataset.py \
  --dataset-root datasets/coin-rim-dent-v2 \
  --manifest data/rim_dent_v2_manifest.csv \
  --class-names rim_dent
python -m experiments.legacy.train_baseline \
  --dataset-config datasets/coin-rim-dent-v2/data.yaml \
  --manifest data/rim_dent_v2_manifest.csv \
  --class-names rim_dent --epochs 100 --imgsz 640 --mosaic 0 \
  --run-name rim-dent-v2-yolo11n-cpu-e100-i640-m0-s0
```

`last.pt` localised 7/7 train rim labels at confidence 0.25 with no detections
on six train target-negatives. Neither checkpoint localised the two C011
validation rims, and frozen `best.pt` missed C009. This separated train
memorisation from cross-coin generalisation failure.

## Roboflow v13 cloud training

Roboflow's v13 model was configured for 300 epochs. It remains a separate cloud
run: no cloud metric or test result is copied into this repository without a
verifiable export, and its epoch count is not used to reinterpret the local
100-epoch result.

The optional v9 hosted Workflow integration is implemented in
`src/inference_roboflow.py`. It keeps `ROBOFLOW_API_KEY` out of URLs and Git,
validates responses, retries transient failures, and externalises image-shaped
outputs. The fixed C005 source SHA-256 is
`d1ac6fdf0170266bf02bf89b91176e9c64cee415eb303e397e4857167a34aec0`.

On 2026-08-17 the published serverless model and Workflow returned HTTP 500.
Roboflow reference IDs were `46ecb284e6bd8db56380dbf5732bf501` for the model
and `6dedb323169641584dd70b12ebc8eafe` for the Workflow. This is preserved as
an integration failure, not accepted as a prediction result.

## Local v13 YOLO11n smoke

The tracked descriptor points to a 102-image Roboflow export: 93 train, 6
validation, and 3 fixed test images. Training images include offline
augmentations, so additional Ultralytics augmentation was disabled.

```bash
python src/train_v13.py \
  --model yolo11n.pt --epochs 1 \
  --run-name v13-yolo11n-preflight-e1-i640-s0
python src/train_v13.py \
  --model yolo11n.pt --epochs 100 \
  --run-name v13-yolo11n-cpu-e100-i640-s0 \
  --test-after-train
```

Preflight verified `93/6/3`, image/label pairing, class IDs, and six polygon
rows converted to detection boxes. The formal run restarted from the original
`yolo11n.pt`, completed in 1.157 hours, and used `best.pt` as the primary
checkpoint.

At confidence 0.25 and matching IoU 0.50:

- C005 scratch: no detection, fail.
- C006 normal: no detections, pass.
- C013 dent: confidence 0.321, IoU 0.744, pass.

Primary status remained failed. The prescribed `last.pt @ 0.25` and
`best.pt @ 0.05` diagnostics also missed scratch. Public evidence is under
[`artifacts/v13-yolo11n`](../artifacts/v13-yolo11n/).
