# Project Status

Last verified: 2026-08-18

## Current capability

- [x] Capture and annotate a three-class coin-defect dataset in Roboflow.
- [x] Validate image/label pairing, class IDs, normalized boxes, split counts,
  physical-coin grouping, and polygon conversion.
- [x] Run deterministic YOLO11n training from a reusable Python CLI.
- [x] Load frozen checkpoints for Ultralytics test validation.
- [x] Evaluate three fixed cases with confidence and same-class IoU gates.
- [x] Save per-image JSON, annotated predictions, matches, and a summary.
- [x] Test local pipeline helpers and the optional secret-safe hosted client.

The repository is reviewable as a Junior AI/CV portfolio project. It is not a
validated inspection product.

## Current dataset

| Layer | Count | Meaning |
| --- | ---: | --- |
| Curated historical dataset | 38 | Pre-v13 captured and reviewed images |
| v13 source images | 40 | Images before Roboflow offline augmentation |
| v13 export | 102 | `93 train / 6 validation / 3 test` files |

The class mapping is `0=dent`, `1=scratch`, `2=stain_corrosion`. Raw images,
exports, weights, and full run directories remain local; small portable
evidence is tracked under `artifacts/`.

## Latest frozen gate

Local YOLO11n trained for 100 epochs on CPU at 640px, batch 2, seed 0,
deterministic mode, and no additional online augmentation.

| Case | Expected | Result |
| --- | --- | --- |
| C005_03 | scratch | Failed: no detection |
| C006_01 | no defect | Passed: no detections |
| C013_01 | dent | Passed: confidence 0.321, IoU 0.744 |

Overall status is **failed** because every condition was required. Diagnostics
with `last.pt @ 0.25` and `best.pt @ 0.05` still missed scratch. The three-image
test split is pipeline evidence, not a generalisation or accuracy claim.

## Known limitations

- No evaluation-grade independent test set or validated production metrics.
- No reliable scratch representation across physical coins.
- No calibrated PASS/FAIL quality-control rule or real-time camera loop.
- The optional Roboflow hosted v9 endpoint last returned HTTP 500.
- The Roboflow 300-epoch cloud run is not directly comparable to the local
  Ultralytics 100-epoch configuration.

## Next milestone

Collect more independently confirmed dent and scratch examples, preserve
coin-level split isolation, expand validation and test, and repeat the frozen
gate. Compare another architecture only after the data supports a meaningful
held-out evaluation.
