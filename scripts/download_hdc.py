"""Download the Helsinki Deblur Challenge 2021 data from Zenodo and unpack it.

Two Zenodo records are used:
  - training set: record 4916176 (version 1.1, 20 zip files, one per defocus step, about 50 GB);
  - test set:     record 5713637 (version 1.0, 4 zip files, five steps each, about 13 GB).
Record 4772280 is an older version of the training set with only steps 0-9 and must not be used.

For every zip file the script downloads it (resuming a partial download if one exists), checks its MD5
checksum against the one Zenodo publishes, extracts it, and deletes the zip once extraction succeeded.
A file that is already extracted is skipped, so the script can be re-run after an interruption.

Layout produced under $DEFOCUS_ROOT (default ~/csc2529):
  data/hdc2021/train/<contents of the training zips>
  data/hdc2021/test/<contents of the test zips>
  data/hdc2021/_downloads/   partial downloads, removed when done

Usage:
  python3 scripts/download_hdc.py [test|train|all]
Only the Python standard library is used, so it runs before any project environment exists.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile

RECORDS = {"test": 5713637, "train": 4916176}
API = "https://zenodo.org/api/records/{}"
CHUNK = 8 * 1024 * 1024
HEADERS = {"User-Agent": "defocus-vlm course project data download"}


def log(msg):
    """Print a timestamped message and flush it, so it appears in the Slurm output file at once.

    Args:
        msg: the text to print.
    """
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def list_zip_files(record_id):
    """Ask the Zenodo API which zip files a record contains.

    Args:
        record_id: the Zenodo record number.

    Returns:
        A list of dicts with keys "name", "url", "size" (bytes) and "md5", one per zip file,
        sorted by name.
    """
    # fetch the record's metadata as JSON
    req = urllib.request.Request(API.format(record_id), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        record = json.load(resp)
    # keep only the zip files; the preview images in the record are not needed
    files = []
    for f in record["files"]:
        if not f["key"].endswith(".zip"):
            continue
        files.append({"name": f["key"], "url": f["links"]["self"], "size": f["size"],
                      "md5": f["checksum"].split(":", 1)[1]})
    return sorted(files, key=lambda f: f["name"])


def download(url, dest, size, retries=20):
    """Download a file, resuming from where a previous attempt stopped.

    Args:
        url: where to download from.
        dest: local path of the (possibly partial) file.
        size: expected size in bytes, used to know when the download is complete.
        retries: how many times to retry after a network error before giving up.

    Returns:
        None. Raises RuntimeError if the file cannot be completed.
    """
    for attempt in range(1, retries + 1):
        # resume from the bytes already on disk
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have == size:
            return
        if have > size:
            # a corrupted partial file: start again
            os.remove(dest)
            have = 0
        headers = dict(HEADERS)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "ab") as out:
                # report progress roughly every 256 MB
                start, done, next_report = time.time(), 0, 256 * 1024 * 1024
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if done >= next_report:
                        rate = done / max(time.time() - start, 1e-6) / 1e6
                        log(f"  {os.path.basename(dest)}: {(have + done) / 1e9:.2f}/{size / 1e9:.2f} GB, {rate:.2f} MB/s")
                        next_report += 256 * 1024 * 1024
        except Exception as err:  # network errors, timeouts, HTTP errors
            log(f"  attempt {attempt} failed: {err!r}; retrying in 60 s")
            time.sleep(60)
    if not os.path.exists(dest) or os.path.getsize(dest) != size:
        raise RuntimeError(f"could not complete {dest}")


def md5_of(path):
    """Compute the MD5 checksum of a file.

    Args:
        path: file to read.

    Returns:
        The checksum as a lowercase hex string.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch_record(split, root):
    """Download, check and extract every zip file of one split.

    Args:
        split: "test" or "train".
        root: the data folder, e.g. ~/csc2529/data/hdc2021.

    Returns:
        None. Raises RuntimeError on a checksum mismatch.
    """
    out_dir = os.path.join(root, split)
    tmp_dir = os.path.join(root, "_downloads")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    files = list_zip_files(RECORDS[split])
    log(f"{split}: {len(files)} zip files, {sum(f['size'] for f in files) / 1e9:.1f} GB in total")
    for f in files:
        # a marker file records that this zip was fully extracted before
        marker = os.path.join(out_dir, f".done_{f['name']}")
        if os.path.exists(marker):
            log(f"{f['name']}: already extracted, skipped")
            continue
        zip_path = os.path.join(tmp_dir, f["name"])
        # download (or resume) the zip
        log(f"{f['name']}: downloading {f['size'] / 1e9:.2f} GB")
        download(f["url"], zip_path, f["size"])
        # make sure the file is exactly what Zenodo published
        got = md5_of(zip_path)
        if got != f["md5"]:
            os.remove(zip_path)
            raise RuntimeError(f"{f['name']}: MD5 mismatch ({got} != {f['md5']}); the file was removed")
        # extract, count what came out, then free the space taken by the zip
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            z.extractall(out_dir)
        open(marker, "w").write(f"{len(names)} entries\n")
        os.remove(zip_path)
        log(f"{f['name']}: checksum ok, {len(names)} entries extracted, zip removed")


def main():
    """Read which split to fetch from the command line and fetch it."""
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in ("test", "train", "all"):
        sys.exit("usage: python3 scripts/download_hdc.py [test|train|all]")
    # all data lives under $DEFOCUS_ROOT/data/hdc2021
    base = os.environ.get("DEFOCUS_ROOT", os.path.expanduser("~/csc2529"))
    root = os.path.join(base, "data", "hdc2021")
    log(f"data folder: {root}")
    # the test set first: it is smaller and is needed for every final number
    for split in (["test", "train"] if which == "all" else [which]):
        fetch_record(split, root)
    log("all done")


if __name__ == "__main__":
    main()
