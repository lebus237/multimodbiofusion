#!/usr/bin/env python3
"""
extract_faces.py
================
Extract face images from CASIA-WebFace MXNet RecordIO format into Layout A:
    {output_dir}/{label:05d}/{idx:03d}.jpg

No MXNet dependency required — reads .rec/.idx/.lst directly.

Usage
-----
python scripts/extract_faces.py \
    --input_dir  /path/to/face_casia_webface \
    --output_dir /path/to/face_casia_webface_extracted

The input directory must contain:
    train.rec   — MXNet RecordIO binary data
    train.idx   — text index file (record_index  byte_offset)
    train.lst   — text list  file (label  path  label  [landmarks...])
"""

import argparse
import struct
import sys
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

RECORD_MAGIC = 0xCED7230A


def read_idx(idx_path: Path):
    """Read .idx text file → list of (record_index, byte_offset) sorted by index."""
    entries = []
    with open(idx_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                entries.append((int(parts[0]), int(parts[1])))
    entries.sort(key=lambda x: x[0])
    return entries


def read_lst(lst_path: Path):
    """Read .lst file → dict mapping record_index to label (int).
    Format: label_col0 \\t path \\t label \\t [landmarks...]
    Line number (1-indexed) maps to the .idx record index.
    """
    index_to_label = {}
    with open(lst_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                label = int(float(parts[2]))
                index_to_label[line_num] = label
            except (ValueError, IndexError):
                pass
    return index_to_label


def extract_jpeg(data: bytes) -> bytes:
    """Find JPEG data within a record's payload by locating the SOI marker."""
    pos = data.find(b"\xff\xd8")
    if pos < 0:
        return b""
    return data[pos:]


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract face images from CASIA-WebFace MXNet RecordIO into Layout A."
    )
    p.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing train.rec, train.idx, train.lst",
    )
    p.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for extracted identity folders",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    rec_file = input_dir / "train.rec"
    idx_file = input_dir / "train.idx"
    lst_file = input_dir / "train.lst"

    for f in (rec_file, idx_file, lst_file):
        if not f.exists():
            print(f"ERROR: Missing {f}")
            sys.exit(1)

    print(f"Reading .lst for labels...")
    index_to_label = read_lst(lst_file)
    print(f"  → {len(index_to_label)} label mappings loaded")

    print(f"Reading .idx for offsets...")
    entries = read_idx(idx_file)
    print(f"  → {len(entries)} records indexed")

    out_dir.mkdir(parents=True, exist_ok=True)
    counters = defaultdict(int)
    extracted = 0
    errors = 0

    print(f"Extracting images to {out_dir}...")
    with open(rec_file, "rb") as fh:
        for rec_idx, offset in entries:
            # Get label from .lst
            if rec_idx not in index_to_label:
                continue

            label = index_to_label[rec_idx]

            # Read record header: 4-byte magic + 4-byte data length
            fh.seek(offset)
            hdr = fh.read(8)
            if len(hdr) < 8:
                errors += 1
                continue

            magic, data_len = struct.unpack("<II", hdr)
            if magic != RECORD_MAGIC:
                errors += 1
                continue

            data = fh.read(data_len)
            if len(data) < data_len:
                errors += 1
                continue

            # Find and extract JPEG
            jpg_bytes = extract_jpeg(data)
            if not jpg_bytes:
                errors += 1
                continue

            # Decode to validate
            img = cv2.imdecode(
                np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                errors += 1
                continue

            label_dir = out_dir / f"{label:05d}"
            label_dir.mkdir(parents=True, exist_ok=True)

            sample_idx = counters[label]
            counters[label] += 1

            out_path = label_dir / f"{sample_idx:03d}.jpg"
            cv2.imwrite(str(out_path), img)
            extracted += 1

            if extracted % 10000 == 0:
                print(
                    f"  ... {extracted} images extracted "
                    f"({errors} errors, {len(counters)} identities so far)"
                )

    print(f"\nDone!")
    print(f"  Total extracted: {extracted}")
    print(f"  Total identities: {len(counters)}")
    print(f"  Errors/skipped: {errors}")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
