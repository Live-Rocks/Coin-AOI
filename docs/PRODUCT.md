# Product

## Summary

Coin-AOI is a computer-vision learning prototype for inspecting coin surfaces.
It explores how a YOLO-based detector could locate known defect types in coin
images and support a future AOI-style workflow.

This repository is a portfolio prototype, not an industrial inspection system.

## Problem

Visual inspection can miss small or inconsistent surface defects. Coin-AOI
provides a constrained experiment for building an end-to-end workflow:

```text
Image → annotated dataset → detector training → evaluation → inspection result
```

The project focuses on defect localisation and classification before it attempts
any quality-control decision.

## Target defect classes

The current annotation taxonomy has three classes:

| Class | Meaning |
| --- | --- |
| `dent` | Visible indentation, impact mark, or deformation. |
| `scratch` | Visible, non-design linear surface damage. |
| `stain_corrosion` | Local unexpected stain, oxidation, or corrosion. |

The exact numeric IDs are defined by
[`datasets/coin-defect-hybrid/data.yaml`](../datasets/coin-defect-hybrid/data.yaml).

## Current scope

Implemented and reviewable as a portfolio prototype:

- A reproducible local Python environment using Ultralytics YOLO.
- Single-image inference with annotated-image and JSON output.
- A documented Roboflow annotation workflow, grouped `coin_id` splits, and
  dataset validator.
- Controlled YOLO11n fine-tunes, including a reviewed rim-dent experiment that
  memorises training labels and fails on held-out coins.
- A secret-safe Roboflow Workflow REST client with retries and an acceptance
  gate. Live hosted inference is blocked by HTTP 500.

This is enough to show an end-to-end CV experiment. It is not enough to ship
an inspector.

## Not currently provided

Coin-AOI does not currently provide:

- a fine-tuned detector that generalises to unseen coins;
- validated precision, recall, or mAP as quality claims;
- a deterministic PASS/FAIL rule;
- real-time camera inspection;
- a working hosted Roboflow inference result;
- production-quality guarantees.

An empty prediction must not be interpreted as PASS.

## Next evidence-based milestone

The product gap is cross-coin generalisation, not missing training code.
Collect independently confirmed rim-deformed coins under the same capture
protocol, keep each physical coin in one split, and expand validation and test
beyond one positive coin each. Hosted HTTP 500 is an external blocker and is
not the next modelling experiment.
