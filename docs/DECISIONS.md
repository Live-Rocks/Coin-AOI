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
