# Project Status

Last verified: 2026-08-17

## Portfolio versus product

This repository is ready to show as a freshman/junior CV side project. A
reviewer can clone it, run the Python/YOLO pipeline, and read an honest
train-memorisation versus held-out failure. It is not a finished inspection
product: no generalising detector, no PASS/FAIL rule, and no successful
hosted inference.

## Current capability

- [x] Create an isolated Python environment and install Ultralytics YOLO.
- [x] Run pretrained YOLO single-image inference and save an annotated image
  plus JSON detections.
- [x] Define a three-class annotation taxonomy and validate a local YOLO export.
- [x] Complete a one-epoch YOLO fine-tuning smoke test and load its checkpoint
  for inference.
- [x] Run a corrected 640px CPU exploratory baseline and validate its
  checkpoint on the test split.
- [x] Build a leakage-checked dent-only dataset and save held-out qualitative
  inference evidence.
- [x] Build a reviewed rim-only dataset and verify train memorisation versus
  cross-coin validation failure.
- [x] Implement an API-key-safe Roboflow Workflow REST client with bounded
  retries, typed errors, and compact output persistence.

Product gaps (not required to treat this repo as a reviewable side project):

- [ ] Complete a successful published Roboflow Workflow inference; the live
  endpoint currently returns HTTP 500.
- [ ] Achieve qualitative dent detection on a held-out physical coin.
- [ ] Train a training-ready coin-defect detector.
- [ ] Evaluate a trained detector on an independent test set.
- [ ] Implement a deterministic PASS/FAIL rule.

## Dataset reality

The current dataset is a complete pipeline smoke test, not a training-ready
release.

| Item | Current state |
| --- | --- |
| Usable annotated images | 38 |
| Physical coins represented | 17 (`C001`, `C002`, `C004`–`C018`) |
| Available split | train: 24, val: 7, test: 7 |
| Class mapping | `0=dent`, `1=scratch`, `2=stain_corrosion` |
| Image source | self-captured only |
| Normal images | 11 empty-label images |
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

## Dent-only experiment result

`src/build_dent_dataset.py` produced a single-class dent dataset with 25
images: train 15, val 4, test 6. Its groups are isolated by `coin_id`; train
contains six dent coins, validation two, and test only one dent coin (`C009`).
Images containing other defect classes are excluded rather than converted to
negative samples.

The 100-epoch, 640px CPU run with Mosaic disabled completed under
`runs/detect/baseline/dent-v1-yolo11n-cpu-e100-i640-m0-s0/`. Held-out inference
at confidence 0.25 saved annotated images and JSON under
`outputs/dent-evaluation/dent-v1-yolo11n-cpu-e100-i640-m0-s0/`. It missed the
C009 dent and produced false-positive dent boxes on normal C006 and C018
images. This is a useful failure analysis artifact, not a successful detector.

A follow-up diagnosis used the same `best.pt` without retraining. On the 15
train images at confidence 0.05, 6 of 10 ground-truth dents matched at
IoU ≥ 0.5, but matched confidences were 0.06–0.48 (none ≥ 0.5), C015 was a
complete miss, and all five empty-label train images produced false positives
(75 boxes total). That is result B: the model does not fit its own training
samples at a usable score. At confidence 0.01 the C009 ground-truth box is
matched (IoU 0.73) at confidence 0.018, while the same image is flooded with
weak `dent` boxes; `last.pt` at 0.05 produced zero test boxes. Evidence is
under `outputs/dent-evaluation/diag-train-conf005/`,
`diag-test-conf001/`, and `diag-test-lastpt/`. The next dent experiment is an
augmentation-off overfit run, not more data and not a C009 reshoot.

## Reviewed rim-dent experiment result

Manual review retained ten outer-rim deformation labels and converted two face
dents plus two ambiguous design-region labels into target-negative hard
examples. `src/build_rim_dent_dataset.py` produced 25 leakage-checked images:
train 13 (7 positive), validation 6 (2 positive), and test 6 (1 positive).
The full C011 group moved to validation; C009 remained sealed in test.

The controlled `yolo11n.pt` run kept 640px inputs, CPU, 100 epochs, batch size
2, seed 0, deterministic mode, and Mosaic disabled. `last.pt` correctly
localised 7/7 train rim labels at confidence 0.25 with 0/6 target-negative
false positives. `best.pt` localised 0/7 train and 0/2 C011 validation rims at
that threshold; `last.pt` also localised 0/2 validation rims. The frozen
`best.pt` then produced no boxes on C009 or any of the five test negatives.

This proves the current model and pipeline can memorise the reviewed training
labels, but it does not learn a cross-coin rim representation. The validation
fitness peak (`mAP50=0.054`, `mAP50-95=0.011`) is diagnostic only because it
comes from two positive images of one physical coin at low validation
confidence.

## Hosted Workflow integration status

`src/inference_roboflow.py` now sends one local image to the published
`coin-defect-hybrid` version 9 Workflow. Its contract was verified through
Roboflow MCP: one `image` input, no declared parameters, and one JSON
`predictions` output. The client keeps the API key in `ROBOFLOW_API_KEY`,
validates TLS, retries transient failures, and externalises base64 image
outputs.

Local request encoding, retries, response handling, source hashing, and
acceptance gates pass eight standard-library unit tests. The opt-in live test
is fixed to v9 `C005_03` with SHA-256
`d1ac6fdf0170266bf02bf89b91176e9c64cee415eb303e397e4857167a34aec0`
and requires both the `predictions` output and a `scratch` detection.

Live execution reached Roboflow with a valid key but returned HTTP 500 after
all retries on 2026-08-17. MCP reproduced the failure for both the direct t2
model (`46ecb284e6bd8db56380dbf5732bf501`) and the published Workflow
(`6dedb323169641584dd70b12ebc8eafe`); the bug report was recorded by Roboflow.
Failure runs now save timestamped, secret-free evidence under ignored outputs.
No real prediction response has been accepted as evidence.

## Current blocker

For a job-reviewable prototype, nothing is blocked: the pipeline, negative
rim-dent result, and hosted client are documented.

For a working inspector, two product gaps remain. The modelling gap is
cross-coin generalisation: both C011 validation rims and the C009 test rim
were missed at confidence 0.25. There are only five physical rim-positive
coins in v2 training, one in validation, and one in test. The integration gap
is Roboflow hosted HTTP 500; that is external and is not the next modelling
experiment.

## Next milestone

If continuing as a detector: collect independently confirmed rim-deformed
coins under the same capture protocol, keep every physical coin in one split,
expand validation and test beyond one positive coin each, then repeat the
frozen evaluation protocol. If only using this repo as a portfolio piece, stop
here and do not wait on hosted inference.

## Claims we can make

- This is a reviewable CV side project: capture, label, train, evaluate.
- The local pretrained inference path works.
- The current YOLO export uses the documented three-class mapping.
- The pilot manifest and local export pass structural validation.
- The corrected 640px CPU baseline and its checkpoint test-validation workflow
  completed reproducibly on the aspect-ratio-preserving local dataset.
- The dent-only curation and held-out evidence workflow works, including an
  explicit negative result.
- The reviewed rim-only model can memorise its seven training positives while
  rejecting its six training target-negatives at confidence 0.25.
- The hosted REST client and offline tests exist; live execution is blocked by
  Roboflow HTTP 500.

## Claims we cannot make

- Fine-tuned defect-detection accuracy, mAP, precision, or recall.
- Generalisation to unseen coins, lighting, or defects.
- A completed hosted inference demo.
- PASS/FAIL accuracy or production readiness.

## Maintenance rule

Update this file only when a capability, dataset fact, blocker, or evidence-based
milestone changes. Each completed item should have a reproducible command,
artifact, or commit behind it.
