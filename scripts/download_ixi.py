#!/usr/bin/env python3
"""Download the IXI T1w archive with parallel HTTP range requests.

The public IXI mirror is slow per connection (~3 MB/s), so we fetch
byte ranges concurrently and stitch the parts back together.
"""
import os
import sys
import time
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "IXI-T1.tar")
PARTS = os.path.join(ROOT, "data", "parts")
TOTAL = 4_840_816_640
CHUNK = 64 * 1024 * 1024
NTHREADS = 12


def fetch_range(idx, start, end, attempt=0):
    path = os.path.join(PARTS, f"{idx:04d}.part")
    want = end - start + 1
    if os.path.exists(path) and os.path.getsize(path) == want:
        return idx, want
    for a in range(5):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if os.path.getsize(path) == want:
                return idx, want
        except Exception as e:
            print(f"chunk {idx} attempt {a}: {e}", flush=True)
            time.sleep(2)
    raise RuntimeError(f"failed chunk {idx}")


def main():
    os.makedirs(PARTS, exist_ok=True)
    ranges = []
    for i, s in enumerate(range(0, TOTAL, CHUNK)):
        ranges.append((i, s, min(s + CHUNK - 1, TOTAL - 1)))
    print(f"{len(ranges)} chunks, {NTHREADS} threads", flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=NTHREADS) as ex:
        futs = [ex.submit(fetch_range, *r) for r in ranges]
        for f in as_completed(futs):
            _, n = f.result()
            done += n
            if done % (CHUNK * 4) < CHUNK:
                print(f"{done/1e9:.2f} GB  {done/1e6/(time.time()-t0):.1f} MB/s", flush=True)
    with open(OUT, "wb") as out:
        for i, _, _ in ranges:
            with open(os.path.join(PARTS, f"{i:04d}.part"), "rb") as f:
                while True:
                    b = f.read(1 << 22)
                    if not b:
                        break
                    out.write(b)
    print(f"wrote {OUT} {os.path.getsize(OUT)} bytes in {time.time()-t0:.0f}s", flush=True)
    assert os.path.getsize(OUT) == TOTAL


if __name__ == "__main__":
    main()
