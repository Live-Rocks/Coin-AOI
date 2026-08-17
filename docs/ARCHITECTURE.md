# Architecture

## System boundary

Coin-AOI currently runs locally and processes still images. It has no deployed
service, camera loop, dashboard, or automatic quality-control decision.

## Implemented flow

```mermaid
flowchart TD
    localImage[Local coin image] --> inference[src/inference.py]
    inference --> pretrained[Pretrained YOLO11n COCO model]
    pretrained --> prediction[Class confidence and bounding boxes]
    prediction --> artifacts[Annotated image and JSON output]
```

`src/inference.py` verifies the environment and output format. Its default
model is trained on COCO classes, so its predictions are not coin-defect
predictions.

The optional hosted path uses the published Roboflow version 9 Workflow:

```mermaid
flowchart TD
    localCoin[Local coin image] --> hostedClient[src/inference_roboflow.py]
    hostedClient --> workflow[Roboflow serverless Workflow]
    workflow --> dynamicOutputs[Workflow-defined outputs]
    dynamicOutputs --> compactJson[Compact JSON]
    dynamicOutputs --> imageArtifacts[Decoded image artifacts]
```

The API key is read only from `ROBOFLOW_API_KEY`. The client validates the
documented list response, discovers output names from the response, and writes
base64 image-shaped outputs to separate files. The integration exists locally,
but a successful hosted execution is not yet verified because the published
endpoint returned HTTP 500 on 2026-08-17.

## Dataset preparation flow

```mermaid
flowchart TD
    rawImages[Self-captured or licensed public images] --> manifest[data/manifest.csv]
    manifest --> roboflow[Roboflow Object Detection annotation]
    roboflow --> export[Local YOLO export]
    export --> dataset[datasets/coin-defect-hybrid]
    manifest --> validator[src/validate_dataset.py]
    dataset --> validator
    validator --> checkedData[Validated local dataset]
```

Raw images and Roboflow exports are local-only. Git tracks the manifest,
annotation rules, class configuration, and validation code rather than the
image files or model weights.

## Data contract

The current Roboflow export uses this mapping:

```text
0 dent
1 scratch
2 stain_corrosion
```

[`data.yaml`](../datasets/coin-defect-hybrid/data.yaml) is the local source of
truth for this mapping.

Each manifest row records an `image_id`, a physical `coin_id`, source metadata,
the assigned split, and annotation status. All views of the same physical coin
must remain in one split to prevent data leakage.

A normal image is a negative detection sample: it has no defect bounding boxes
and must export with an empty label file. It is not a `normal` detection class.

## Planned experiment flow

```mermaid
flowchart TD
    checkedData[Validated dataset] --> train[YOLO fine-tuning]
    train --> model[Custom coin-defect weights]
    model --> evaluation[Independent test-set evaluation]
    evaluation --> analysis[Metrics and failure analysis]
    analysis --> qcRule[Deterministic QC rule]
    qcRule --> result[PASS FAIL or review]
```

The planned flow is not implemented. Fine-tuning, evaluation, and QC rules must
not be described as current functionality until each has reproducible evidence.
