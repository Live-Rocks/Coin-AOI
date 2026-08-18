# Dataset workflow

Coin-AOI uses a hybrid dataset: self-captured coin photos are the primary
source, while clearly licensed public coin photos may supplement material,
lighting, and corrosion variation.

## Local-only directories

These directories are intentionally ignored by Git:

- `data/raw/self_captured/`: original photos captured for this project.
- `data/raw/public/`: downloaded public images before annotation.
- `data/roboflow/`: raw YOLO exports downloaded from Roboflow.

Do not upload an image to Roboflow until its row exists in the manifest and its
source and license have been checked.

## Dataset versions

| Dataset layer | Count | Split or purpose |
| --- | ---: | --- |
| Curated historical dataset | 38 | Pre-v13 manifest-backed captures |
| v13 source images | 40 | Before Roboflow offline augmentation |
| v13 YOLO export | 102 | `93 train / 6 validation / 3 test` |

The counts describe different layers, not conflicting current totals. The v13
train split contains three exported variants for each of 31 source images;
validation and test contain six and three unaugmented images respectively.
Group assignment happens before augmentation.

## MVP phone capture checklist

Use this simple setup to verify the full dataset flow:

1. Use the phone's main 1x camera with one fixed desk lamp and background.
   Disable filters, portrait mode, and night mode.
2. For each coin face, take three photos: front 45-degree light, left raking
   light, and right raking light.
3. Start new physical coin IDs at `C006`; use names such as
   `self_C006_01.jpg` and store originals in
   `data/raw/self_captured/`.
4. In the manifest, use `capture_setup=phone_desk_lamp`. Assign every photo
   from one `coin_id` to the same manual split: `train`, `val`, or `test`.
5. Include at least one confirmed normal coin. It receives no bounding box and
   later exports with an empty YOLO label file.

Do not damage coins or create digital defects to make this MVP dataset. If a
feature is uncertain, leave that image out of the dataset.

## Manifest

Copy `manifest_template.csv` to `manifest.csv` and add one row per image before
annotation. `image_id` must be the image filename without its extension, and
`coin_id` identifies the physical coin.

All images of one physical coin—different angles, lighting, crops, or sequential
captures—must use the same `coin_id` and belong to exactly one split. Use
`train`, `val`, or `test` in the `split` column.

Example:

```csv
image_id,coin_id,source_type,source_url,license,capture_setup,defect_classes,split,annotation_status
self_c01_a01,C01,self_captured,,,front_light_45deg,scratch,train,annotated
public_smithsonian_001,P001,public,https://example.org/item,CC0,unknown,stain_corrosion,val,annotated
```

## YOLO export

Download the Roboflow project in YOLO format to `data/roboflow/`. Copy the
selected image/label pairs into `datasets/coin-defect-hybrid/`, which must use
this local structure:

```text
datasets/coin-defect-hybrid/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

When generating a Roboflow version, use `Fit within 640×640` to preserve the
coin's aspect ratio. Do not use `Stretch to`, which distorts portrait coin
photos and their visible defects.

The export's image and label files remain local. The tracked `data.yaml` defines
the class order used by the current Roboflow export:

```text
0 dent
1 scratch
2 stain_corrosion
```

## Current v13 export

Download the selected v13 YOLO export to:

```text
data/roboflow/v13-cloud-augmented/
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

The tracked descriptor is `datasets/coin-defect-v13/data.yaml`. The current
preflight requires exactly 93 train, 6 validation, and 3 test image/label
pairs. It validates normalized class IDs and boxes and converts the six polygon
scratch rows to enclosing detection boxes.

The export already contains offline rotation, brightness, and grayscale
augmentation. The local reference run therefore disables additional
Ultralytics augmentation. Roboflow's separate cloud model was configured for
300 epochs; the reproducible local reference uses 100 epochs and is documented
as a different run.

## Dent-only experiment dataset

Use `python -m experiments.legacy.build_dent_dataset --replace` to derive
`datasets/coin-dent-v1/` and `data/dent_dataset_manifest.csv`. The builder
keeps only rows labelled exactly `dent` plus confirmed normal rows, then
preserves only YOLO class `0` labels. It excludes mixed-defect images so
unlabelled scratch or corrosion does not become a false negative for the
single-class experiment.
