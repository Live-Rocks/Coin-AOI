# Coin-AOI Case Study

## Problem

Manual inspection is inconsistent when a surface contains small marks,
reflections, intentional relief, and normal wear. Coin-AOI uses this constrained
problem to demonstrate how an object-detection idea becomes a testable Python
system rather than a notebook-only model demo.

The goal is defect localisation and failure analysis. It is not to claim that a
small pilot dataset can support an industrial inspection decision.

## Approach

The project follows an end-to-end computer-vision workflow:

```text
capture → annotate → freeze a dataset version → validate → train
→ select checkpoint → evaluate held-out cases → preserve evidence
```

The current taxonomy is `dent`, `scratch`, and `stain_corrosion`. Confirmed
normal images use empty YOLO label files rather than a synthetic `normal`
object class. Captures from one physical coin stay in one split to reduce
identity leakage.

## Engineering decisions

- Use explicit Python CLIs so training and evaluation settings are reviewable.
- Keep the Roboflow dataset version frozen for each experiment.
- Validate image/label pairing, split counts, normalized coordinates, class
  IDs, and polygon conversion before training.
- Fix CPU, 640px, batch 2, seed 0, deterministic mode, and zero workers for the
  local v13 comparison point.
- Disable online augmentation because the v13 export already contains offline
  augmented training images.
- Select `best.pt` before inspecting the three fixed test cases.
- Freeze confidence `0.25` and same-class matching IoU `0.50`; do not tune from
  a failed test result.
- Preserve failed predictions and diagnostic runs instead of reporting only a
  favourable example.

## Result

The one-epoch preflight parsed the `93/6/3` dataset, converted six polygon rows
to detection boxes, and produced both checkpoints. A fresh YOLO11n run then
completed 100 epochs on an Apple M2 CPU in 1.157 hours.

The frozen smoke gate passed two of its three cases:

- C013 dent matched at IoU 0.744 with confidence 0.321.
- C006 normal produced no detections at confidence 0.25.
- C005 scratch produced no detection and caused the overall gate to fail.

The prescribed `last.pt @ 0.25` and `best.pt @ 0.05` diagnostics also missed
the scratch. This supports a narrow conclusion: the software pipeline is
reproducible, but the current detector has not learned a reliable scratch
representation.

## Lessons learned

- A falling training loss does not demonstrate cross-coin generalisation.
- Small validation and test splits make aggregate metrics unstable and easy to
  overstate; per-image evidence is more honest for this smoke dataset.
- Group assignment must happen before augmentation, or the same physical
  object can leak into evaluation.
- A negative sample is useful only when its annotation meaning is explicit.
- Fixed gates prevent post-hoc threshold tuning from turning a miss into an
  apparent success.
- AI coding tools accelerate scaffolding and review, but tests, data rules, and
  final claims still need human verification.

## Production gap

Coin-AOI does not provide production accuracy, calibrated quality decisions,
real-time capture, drift monitoring, or a deployable inspection service. The
next evidence-based milestone is to collect more independently confirmed dent
and scratch examples across physical coins, expand validation and test, and
repeat the same frozen protocol before comparing another architecture.
