"""Verify the additive GPU diagnostic bundle and original validated checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, str(root / 'scripts/pace/longeval-stage.py'), 'source'], cwd=root, check=True)
manifest = json.loads((root / 'longeval-gpu-manifest.json').read_text())
for name, expected in manifest['files'].items():
    path = (root / name).resolve()
    if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'GPU bundle identity mismatch: {name}')
print(f"Verified {len(manifest['files'])} GPU diagnostic bundle files", flush=True)
