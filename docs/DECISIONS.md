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

## D007 — Isolate a dent-only experiment and retain negative evidence

Date: 2026-08-17
Status: Accepted

### Context

The hybrid taxonomy includes `dent`, `scratch`, and `stain_corrosion`, but the
first milestone is to localise one defect type. Mixed-defect images would be
unsafe as negative samples in a one-class detector because their non-dent
defects would be unlabelled. The expanded hybrid dataset has 38 annotated
images from 17 coins, while the curated dent-only dataset has 25 images and
only one dent coin in test.

### Decision

Build `coin-dent-v1` only from rows labelled exactly `dent` and confirmed normal
rows; retain class `0=dent` labels and exclude mixed-defect rows. Train
`yolo11n.pt` at 640px on CPU for 100 epochs with Mosaic disabled, then save one
fixed-threshold qualitative inference pass on the held-out test split.

### Consequences

The initial dent run missed the C009 dent and produced false positives on
normal C006 and C018 images at confidence 0.25. Preserve these annotated images
and JSON as portfolio evidence of an honest failed experiment. Do not adjust
the test threshold or claim dent recognition from this run; collect another
independent test dent coin and more train-side variation before rerunning.

## D008 — Narrow the target to reviewed outer-rim deformation

Date: 2026-08-17
Status: Accepted

### Context

The dent-v1 labels mix outer-rim deformation, coin-face dents, and two
ambiguous marks near intentional relief. A checkpoint comparison showed that
`last.pt` could memorise reviewed rim labels, while the validation split
contained no rim-positive image and therefore could not select a checkpoint
for the intended task.

### Decision

Define the next target as `rim_dent`: a visible impact deformation of the outer
coin rim. Retain the four non-rim images as target-negative hard examples,
move the complete C011 coin group to validation, and keep C009 sealed in test.
Train `yolo11n.pt` with the same 100-epoch, 640px, CPU, batch-2, seed-0,
Mosaic-disabled recipe so annotation scope and split are the controlled
variables.

### Consequences

The generated v2 dataset has 7/2/1 positive images across
train/validation/test. `last.pt` localised all seven train rims at confidence
0.25 with no train target-negative false positives, but neither checkpoint
localised either C011 validation rim. Frozen `best.pt` also missed C009 while
remaining empty on all five test negatives. Preserve this as evidence that the
pipeline can memorise the reviewed task but does not yet generalise across
physical coins. Do not tune from the C009 result or claim rim-detection
performance.
