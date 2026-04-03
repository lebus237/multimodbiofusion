#!/usr/bin/env python3
"""
debug_rec.py
============
Inspect the first few records in a MXNet RecordIO .rec file.

Usage
-----
python scripts/debug_rec.py --input_dir /path/to/face_casia_webface
"""
import argparse
import struct
from pathlib import Path

parser = argparse.ArgumentParser(description="Debug MXNet RecordIO .rec/.idx files.")
parser.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing train.rec and train.idx")
args = parser.parse_args()

input_dir = Path(args.input_dir)
REC_FILE = input_dir / "train.rec"
IDX_FILE = input_dir / "train.idx"

# Read first 5 offsets from idx
offsets = []
with open(IDX_FILE, "r") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        parts = line.strip().split()
        offsets.append((int(parts[0]), int(parts[1])))

print(f"First 5 idx entries: {offsets}")

with open(REC_FILE, "rb") as fh:
    for idx, offset in offsets:
        fh.seek(offset)
        raw = fh.read(64)
        print(f"\n--- Record idx={idx}, offset={offset} ---")
        print(f"  Raw hex (first 64 bytes): {raw[:64].hex()}")
        
        # Parse RecordIO header
        magic, lrec = struct.unpack("II", raw[:8])
        cflag = (magic >> 29) & 7
        length = magic & ((1 << 29) - 1)
        print(f"  magic=0x{magic:08x}, lrec={lrec}, cflag={cflag}, length={length}")
        
        # Parse IRHeader from data (after 8-byte record header)
        data = raw[8:]
        flag = struct.unpack("I", data[0:4])[0]
        print(f"  IRHeader flag={flag}")
        
        if flag == 0:
            label = struct.unpack("f", data[4:8])[0]
            rid = struct.unpack("Q", data[8:16])[0]
            print(f"  label={label}, id={rid}")
            print(f"  Image data starts at offset+24, first bytes: {data[16:24].hex()}")
        else:
            # Multiple labels
            n = flag
            labels = struct.unpack(f"{n}f", data[4:4+4*n])
            rid = struct.unpack("Q", data[4+4*n:4+4*n+8])[0]
            hdr_size = 4 + 4*n + 8
            print(f"  labels={labels}, id={rid}")
            print(f"  Image data starts at offset+8+{hdr_size}, first bytes: {data[hdr_size:hdr_size+8].hex()}")
