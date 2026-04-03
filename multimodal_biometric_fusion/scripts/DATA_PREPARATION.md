# Data Preparation Scripts

This directory contains utility scripts to prepare raw biometric datasets into the
**Layout A** structure expected by `prepare_dataset.py`.

---

## Overview

`prepare_dataset.py` builds a virtual homologous multimodal dataset by pairing
identities across three modalities. Each raw dataset must be organized as:

```
<dataset_root>/
    <identity_id>/
        image1.jpg
        image2.png
        ...
```

This is called **Layout A** — one subdirectory per identity, with all sample images
directly inside it. Each identity must have at least `--min_samples` images (default 5).

---

## Scripts

### `extract_faces.py`

Extracts face images from a CASIA-WebFace **MXNet RecordIO** archive (`.rec`/`.idx`/`.lst`)
into Layout A folders.

**Requirements:** `numpy`, `opencv-python` (no MXNet needed).

**Input directory must contain:**
- `train.rec` — MXNet RecordIO binary data
- `train.idx` — text index file (`record_index  byte_offset` per line)
- `train.lst` — text list file (tab-separated: `label  path  label  [landmarks...]`)

**Usage:**

```bash
python scripts/extract_faces.py \
    --input_dir  /path/to/face_casia_webface \
    --output_dir /path/to/face_casia_webface_extracted
```

**Output:**

```
face_casia_webface_extracted/
    00000/
        000.jpg  001.jpg  002.jpg  ...
    00001/
        000.jpg  001.jpg  ...
    ...
```

### `debug_rec.py`

Diagnostic tool that dumps the first 5 records from a MXNet RecordIO file,
showing raw hex bytes and parsed header fields. Useful for verifying the binary format.

**Usage:**

```bash
python scripts/debug_rec.py \
    --input_dir /path/to/face_casia_webface
```

---

## Raw Dataset Normalization

Below are the steps used to normalize each raw dataset into Layout A.

### 1. CASIA-IrisV4-Interval

**Original structure:** `{id}/L/*.jpg` and `{id}/R/*.jpg` (left/right eye subfolders).

**Normalization** — flatten `L/` and `R/` into the identity folder:

```bash
cd /path/to/iris_casia_interval

for id_dir in */; do
    find "$id_dir" -mindepth 2 -name '*.jpg' -exec mv -n {} "$id_dir" \;
    find "$id_dir" -name 'Thumbs.db' -delete
    find "$id_dir" -mindepth 1 -type d -empty -delete
done
```

**Result:** ~249 identities, ~20 images each.

### 2. SOCOFing (Fingerprint)

**Original structure:** flat directory with files like `{id}__{gender}_{finger}.BMP`.

**Normalization** — group by identity prefix into zero-padded folders:

```bash
cd /path/to/fingerprint_socofing_real

for f in *.BMP; do
    id=$(echo "$f" | sed 's/__.*//')
    padded=$(printf '%03d' "$id")
    mkdir -p "$padded"
    mv "$f" "$padded/"
done
```

**Result:** 600 identities, 10 images each.

### 3. CASIA-WebFace (Face)

**Original structure:** MXNet RecordIO format (`train.rec`, `train.idx`, `train.lst`).

**Normalization** — use the extraction script:

```bash
python scripts/extract_faces.py \
    --input_dir  /path/to/face_casia_webface \
    --output_dir /path/to/face_casia_webface_extracted
```

**Result:** ~10,431 identities, ~47 images each on average.

---

## Running the Dataset Builder

Once all three datasets are in Layout A:

```bash
python scripts/prepare_dataset.py \
    --iris_dir        /path/to/iris_casia_interval \
    --fingerprint_dir /path/to/fingerprint_socofing_real \
    --face_dir        /path/to/face_casia_webface_extracted \
    --output_dir      data/virtual_dataset \
    --num_identities  249 \
    --min_samples     5 \
    --seed            42
```

> **Note:** `--num_identities` is bounded by the smallest dataset (iris = 249 identities).
