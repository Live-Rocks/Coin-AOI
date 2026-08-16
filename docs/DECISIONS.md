# Decisions

This file records decisions that affect architecture, data governance, or
claims made about the project. Add a new decision when a choice changes; do not
rewrite old decisions. Mark replaced decisions as `Superseded`.

## D001 — Treat the current dataset as a pipeline smoke test

Date: 2026-08-17  
Status: Accepted

### Context

The validated pilot contains 12 annotated images from three physical coins. All
images are in the train split, and there are no normal, validation, or test
samples.

### Decision

Use the pilot only to verify annotation, YOLO export, manifest, and validation
behaviour. Do not use it to make model-quality claims.

### Consequences

Any experimental training run is a technical smoke test, not a portfolio metric
or generalisation result.

## D002 — Preserve the Roboflow class mapping

Date: 2026-08-17  
Status: Accepted

### Context

The Roboflow YOLO export maps class IDs as:

```text
0 dent
1 scratch
2 stain_corrosion
```

### Decision

Use this mapping in `data.yaml`, labels, validation, training, and inference.

### Consequences

Class IDs are not a quality ranking. Any mapping change requires a new dataset
version and a new decision record.

## D003 — Treat normal images as negative detection samples

Date: 2026-08-17  
Status: Accepted

### Context

YOLO object detection represents missing objects by an empty label file.

### Decision

Do not create a `normal` bounding-box class. Include normal coin images with
empty label files.

### Consequences

The detector can learn a no-defect scene, but an empty prediction alone is not
a quality guarantee or a PASS result.

## D004 — Require independent evaluation before performance claims

Date: 2026-08-17  
Status: Accepted

### Context

Pretrained COCO inference and train-only pilot data do not demonstrate
coin-defect performance.

### Decision

Report mAP, precision, recall, error analysis, or threshold conclusions only
after evaluation on a held-out test set with no shared physical `coin_id` values
with training data.

### Consequences

The current project may claim a working environment and validated dataset
pipeline, but not a working defect detector.

## D005 — Fix the first exploratory baseline to CPU and five epochs

Date: 2026-08-17
Status: Superseded by D006

### Context

The current v3 local dataset has 26 images from nine physical coins, split
15/6/5 across train/validation/test. PyTorch has an MPS build, but MPS is
unavailable at runtime on the current Apple M2 machine. The test split has
three normal images, one `dent`, and one `stain_corrosion` image; it has no
`scratch` ground truth.

### Decision

Use `yolo11n.pt` with Ultralytics `8.4.120` for a five-epoch exploratory
baseline on CPU, using 320px images, batch size 2, seed 0, deterministic mode,
zero dataloader workers, and no cache. Validate the resulting `best.pt` on the
test split, while preserving the earlier one-epoch smoke run.

### Consequences

The generated checkpoints, logs, and validation plots are reproducible
technical artifacts for this exact local dataset and setup. Do not report any
validation or test mAP, precision, recall, prediction threshold, or empty
prediction as defect-detection quality, generalisation, or a PASS/FAIL result.

## D006 — Preserve image geometry and use 640px for the corrected baseline

Date: 2026-08-17
Status: Accepted

### Context

The first Roboflow export used `Stretch to 512×512`, which changed the geometry
of portrait coin images. A source image of `768×1024` was converted to
`512×512`, horizontally widening the coin relative to its original shape.

### Decision

Regenerate the local YOLO export using Roboflow `Fit within 640×640`, preserving
each image's aspect ratio. Train a separate `yolo11n.pt` CPU experiment at
`imgsz=640` for 100 epochs with batch size 2, seed 0, deterministic mode, zero
workers, and no cache. Keep all prior stretched-image outputs only as
pipeline-debugging artifacts.

### Consequences

Training and inference now use correctly proportioned coin images. The
corrected dataset remains only 26 images from nine coins, with no `scratch`
ground truth in test; the 640px run and its test predictions must not be used
as evidence of detector accuracy, generalisation, or PASS/FAIL performance.
