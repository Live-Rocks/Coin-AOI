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

The export's image and label files remain local. The tracked `data.yaml` defines
the class order used by the current Roboflow export:

```text
0 dent
1 scratch
2 stain_corrosion
```
