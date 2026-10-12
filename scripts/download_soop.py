"""
Downloads the Stroke Outcome Optimization Project (SOOP, OpenNeuro ds004889 v1.1.2) from the public OpenNeuro
S3 bucket. Python standard library only (Python >= 3.8).

  - Resumable: run it again at any time; finished files are skipped and partial files restart.
  - Verified: every file is checked against the SHA-256 in data/soop_manifest.csv.
  - By default only what CALM-Stroke needs is downloaded: metadata, lesion masks and DWI (trace + ADC), about 2.8 GB.
    Add "--groups meta deriv dwi flair t1" for the full release (about 72 GB).

Usage (from the repository root):
    python scripts/download_soop.py
    python scripts/download_soop.py --workers 12          # more parallel downloads
    python scripts/download_soop.py --verify-existing     # re-hash files already on disk
"""
import argparse, csv, hashlib, os, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URLS = ["https://s3.amazonaws.com/openneuro.org/ds004889/",
             "https://openneuro.org/crn/datasets/ds004889/snapshots/1.1.2/files/"]
ORDER = ["meta", "deriv", "dwi", "flair", "t1"]
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = Path(os.environ.get("CALM_DATA", str(REPO / "data")))


def sha256_of(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def url_for(base, rel):
    if "snapshots" in base:                      # OpenNeuro API uses ':' as path separator
        return base + rel.replace("/", ":")
    return base + rel


def fetch(rel, size, sha, dest, retries=5):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_err = None
    for attempt in range(retries):
        for base in BASE_URLS:
            try:
                req = urllib.request.Request(url_for(base, rel), headers={"User-Agent": "soop-downloader/1.0"})
                with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
                    h = hashlib.sha256()
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk); h.update(chunk)
                if tmp.stat().st_size != size or h.hexdigest() != sha:
                    raise IOError(f"checksum/size mismatch from {base}")
                os.replace(tmp, dest)
                return True, None
            except Exception as e:                # try next mirror / retry
                last_err = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2 ** attempt))
    try:
        tmp.unlink()
    except OSError:
        pass
    return False, last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA / "ds004889"), help="target folder")
    ap.add_argument("--manifest", default=str(REPO / "data" / "soop_manifest.csv"))
    ap.add_argument("--groups", nargs="+", default=["meta", "deriv", "dwi"], choices=ORDER)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify-existing", action="store_true")
    a = ap.parse_args()

    root = Path(a.root)
    rows = [r for r in csv.DictReader(open(a.manifest, newline="")) if r["group"] in a.groups]
    rows.sort(key=lambda r: ORDER.index(r["group"]))
    todo, done_bytes = [], 0
    for r in rows:
        p = root / r["path"]
        ok = p.exists() and p.stat().st_size == int(r["size"])
        if ok and a.verify_existing:
            ok = sha256_of(p) == r["sha256"]
        if ok:
            done_bytes += int(r["size"])
        else:
            todo.append(r)
    total = sum(int(r["size"]) for r in rows)
    print(f"Target: {root}")
    print(f"Groups: {' '.join(a.groups)} | files: {len(rows)} | to download: {len(todo)} "
          f"({sum(int(r['size']) for r in todo)/1e9:.2f} GB of {total/1e9:.2f} GB)")
    if not todo:
        print("Everything is already downloaded and verified."); return

    lock, t0, got, n_ok, fails = threading.Lock(), time.time(), 0, 0, []
    for g in ORDER:                                   # finish one group before the next
        batch = [r for r in todo if r["group"] == g]
        if not batch:
            continue
        print(f"\n== {g}: {len(batch)} files, {sum(int(r['size']) for r in batch)/1e9:.2f} GB")
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(fetch, r["path"], int(r["size"]), r["sha256"], root / r["path"]): r for r in batch}
            for i, f in enumerate(as_completed(futs), 1):
                r = futs[f]; ok, err = f.result()
                with lock:
                    if ok:
                        n_ok += 1; got += int(r["size"])
                    else:
                        fails.append((r["path"], err))
                    if i % 25 == 0 or i == len(batch):
                        el = max(1e-6, time.time() - t0)
                        print(f"  {g}: {i}/{len(batch)} | {got/1e9:6.2f} GB this run | {got/el/1e6:5.1f} MB/s | "
                              f"failures {len(fails)}", flush=True)
    log = Path(a.root).parent / "download_failures.txt"
    if fails:
        with open(log, "w") as fh:
            for pth, err in fails:
                fh.write(f"{pth}\t{err}\n")
        print(f"\n{len(fails)} files failed (see {log}). Just run the script again to retry them.")
    else:
        if log.exists():
            log.unlink()
        print(f"\nDone: {n_ok} files, {got/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min. All checksums verified.")


if __name__ == "__main__":
    main()
