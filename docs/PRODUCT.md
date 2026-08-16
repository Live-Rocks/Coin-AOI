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

Implemented:

- A reproducible local Python environment using Ultralytics YOLO.
- Single-image pretrained-model inference with annotated-image and JSON output.
- A documented Roboflow annotation workflow and dataset validator.

In progress:

- Building a representative, leakage-resistant dataset with self-captured and
  clearly licensed public coin images.

## Not currently provided

Coin-AOI does not currently provide:

- a fine-tuned coin-defect detector;
- validated precision, recall, or mAP results;
- an independent validation or test set;
- a deterministic PASS/FAIL rule;
- real-time camera inspection;
- root-cause diagnosis or production-quality guarantees.

An empty prediction from the pretrained COCO model must not be interpreted as
PASS.

## Next evidence-based milestone

Create a versioned dataset release with normal images and all three defect
classes, split by physical `coin_id` into train, validation, and test sets.
Only then can a baseline detector be trained and evaluated honestly.
