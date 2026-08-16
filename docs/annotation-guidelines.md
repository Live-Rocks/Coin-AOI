# Coin-AOI annotation guidelines

Use these rules for every image annotated in the Roboflow **Object Detection**
project. The goal is a consistent first YOLO dataset, not an industrial defect
taxonomy.

## Class list

| ID | Class | Include | Exclude |
| --- | --- | --- | --- |
| 0 | `scratch` | A visible, continuous, non-design linear surface mark. | Coin lettering, engraved artwork, mint design lines, rim edges, or a narrow highlight from lighting. |
| 1 | `stain_corrosion` | A local, unexpected discoloration, oxidation, stain, or corrosion patch. | Expected patina, a shadow, removable dust, background, or an uncertain reflection. |
| 2 | `dent` | A visible indentation, impact mark, or deformation in the metal surface. | A feature identified only from glare, intentional relief, lettering, or an uncertain depth cue. |

`stain_corrosion` deliberately combines stains and corrosion. Their visual
boundary is too subjective for the first dataset; do not create extra classes
in Roboflow.

## Bounding-box rules

1. Draw a box around the **defect region**, never around the full coin.
2. Keep the box tight but include the complete visible defect. A small margin
   is acceptable; do not include unrelated coin artwork.
3. Mark separate defects with separate boxes, even if they share a class.
4. Assign exactly one class to each box. If a feature could be two classes but
   cannot be resolved confidently, leave it unannotated and record the reason
   in the image review notes.
5. If a defect continues beyond the image boundary, box only its visible part.
6. A normal coin image has no boxes. Export it with an empty YOLO label file.

## Image inclusion rules

Include images only when the coin surface is sufficiently visible to support a
reliable annotation. Record the physical coin's `coin_id`, source, license,
capture setup, and split in `data/manifest.csv` before uploading.

Exclude or mark for review when any of these prevent reliable judgment:

- strong reflection or shadow obscures the suspected feature;
- motion blur or focus blur hides the surface texture;
- the coin is too small for the smallest defect to be visible;
- a suspected defect is indistinguishable from normal wear, artwork, or patina.

## Source and split rules

- Use `self_captured` or `public` in `source_type`.
- For a public image, keep its exact source URL and license in the manifest.
  Use only images with a license compatible with your intended portfolio use.
- Assign all views of one physical coin to one split before augmentation:
  target `70% train`, `15% val`, `15% test` by `coin_id`.
- Never place different photos, crops, or lighting variations of the same
  physical coin in different splits. That would leak visual identity from
  training into evaluation.

## Roboflow checklist

1. Create an **Object Detection** project named `coin-defect-hybrid`.
2. Create only these labels, with this exact spelling and order:
   `scratch`, `stain_corrosion`, `dent`.
3. Upload only images that have manifest rows.
4. Annotate a 10–20 image pilot, review ambiguous cases, and revise these
   guidelines before scaling up.
5. Export in YOLO format without relying on augmentation to create train/val/test
   splits. Group assignment by `coin_id` remains the source of truth.
