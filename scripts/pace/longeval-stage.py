"""Verify the transferred checkout and record packages staged for offline compute."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "build/pace-longeval"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_source():
    base = json.loads((ROOT / "longeval-source-manifest.json").read_text())
    supplement = json.loads((ROOT / "longeval-bootstrap-manifest.json").read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != base["base_commit"] or head != supplement["base_commit"]:
        raise ValueError("Unexpected checkout base commit")
    files = dict(base["files"])
    if files.keys() & supplement["files"].keys():
        raise ValueError("Bootstrap supplement must not replace original source files")
    files.update(supplement["files"])
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        if ROOT not in path.parents or digest(path) != expected:
            raise ValueError(f"Source identity mismatch: {name}")
    print(f"Verified {len(files)} source files at {head}", flush=True)


def stage():
    verify_source()
    target = STAGE / "googletest-v1.15.2.tar.gz"
    # This is the same release URL as engine/CMakeLists.txt. Fetch only on login.
    if not target.exists():
        temporary = target.with_suffix(".part")
        with urllib.request.urlopen("https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz", timeout=120) as response:
            with temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
        temporary.replace(target)
    paths = sorted((STAGE / "wheelhouse").glob("*.whl")) + [target]
    if len(paths) < 2:
        raise ValueError("Empty wheelhouse")
    manifest = {"files": {p.relative_to(STAGE).as_posix(): digest(p) for p in paths}}
    (STAGE / "staged-files.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Recorded checksums for {len(paths)} staged packages", flush=True)


def verify_stage():
    verify_source()
    manifest = json.loads((STAGE / "staged-files.json").read_text())
    actual = {p.relative_to(STAGE).as_posix() for p in (STAGE / "wheelhouse").glob("*.whl")}
    expected = {name for name in manifest["files"] if name.startswith("wheelhouse/")}
    if actual != expected:
        raise ValueError("Wheelhouse membership changed")
    for name, expected_hash in manifest["files"].items():
        path = (STAGE / name).resolve()
        if STAGE not in path.parents or digest(path) != expected_hash:
            raise ValueError(f"Staged package identity mismatch: {name}")
    print("Verified staged package checksums", flush=True)


if __name__ == "__main__":
    {"source": verify_source, "stage": stage, "verify": verify_stage}[sys.argv[1]]()
