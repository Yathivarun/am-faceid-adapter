"""Resumable download of a Hugging Face dataset folder (for weak connections).

    python prep/download_hf.py --repo Ryan-sjtu/ffhq512-caption --folder data --out data/raw/ffhq512

Lists the files through the HF API, then downloads each one with HTTP Range
resume and retries. Finished files (size matches) are skipped, so it can be
stopped and restarted any time. Partial files are kept as <name>.part.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1 << 20


def list_files(repo, folder):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{folder}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return [f for f in json.load(r) if f["type"] == "file"]


def download(url, dest: Path, size: int, retries: int):
    part = dest.with_name(dest.name + ".part")
    if size == 0:                         # empty files (e.g. a blank README.md): nothing to fetch
        dest.write_bytes(b"")
        print(f"  {dest.name}: done (empty)")
        return True
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            break
        if have > size:                   # a damaged over-long partial file: start over
            part.unlink()
            have = 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, part.open("ab" if have else "wb") as out:
                if have and r.status != 206:           # server ignored Range: start over
                    out.truncate(0)
                t0, got = time.time(), 0
                for chunk in iter(lambda: r.read(CHUNK), b""):
                    out.write(chunk)
                    got += len(chunk)
                    if time.time() - t0 > 5:
                        done = have + got
                        print(f"\r  {dest.name}: {done / 1e6:7.0f}/{size / 1e6:.0f} MB "
                              f"({got / 1e6 / (time.time() - t0):.1f} MB/s)", end="", flush=True)
                        t0, have, got = time.time(), done, 0
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"\n  {dest.name}: attempt {attempt} failed ({e}); retrying in {min(60, 5 * attempt)} s")
            time.sleep(min(60, 5 * attempt))
    if part.exists() and part.stat().st_size == size:
        part.replace(dest)
        print(f"\r  {dest.name}: done ({size / 1e6:.0f} MB)" + " " * 20)
        return True
    print(f"\n  {dest.name}: incomplete after {retries} attempts")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="Ryan-sjtu/ffhq512-caption")
    ap.add_argument("--folder", default="data")
    ap.add_argument("--out", default="data/raw/ffhq512")
    ap.add_argument("--retries", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(list_files(args.repo, args.folder), key=lambda f: f["path"])
    total = sum(f["size"] for f in files)
    print(f"{args.repo}/{args.folder}: {len(files)} files, {total / 1e9:.1f} GB -> {out}")
    failed = 0
    for i, f in enumerate(files, 1):
        dest = out / Path(f["path"]).name
        if dest.exists() and dest.stat().st_size == f["size"]:
            continue
        print(f"[{i}/{len(files)}] {dest.name}")
        url = f"https://huggingface.co/datasets/{args.repo}/resolve/main/{f['path']}"
        failed += not download(url, dest, f["size"], args.retries)
    done = sum(1 for f in files if (out / Path(f["path"]).name).exists())
    print(f"finished: {done}/{len(files)} files complete" + (f", {failed} incomplete (re-run to resume)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
