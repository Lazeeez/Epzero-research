"""Check every file listed in the FINAL_PROTOCOL_V1 freeze receipt against its recorded SHA-256.

    python scripts/verify_freeze.py
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
receipt = json.loads((ROOT / 'research/longeval/FINAL_PROTOCOL_V1_FREEZE.json').read_text(encoding='utf-8'))
changed = [name for name, digest in receipt['files'].items()
           if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
if changed:
    print('Frozen files that differ from the receipt:', *changed, sep='\n  ')
    sys.exit(1)
print(f"All {len(receipt['files'])} frozen files match the receipt (frozen {receipt['frozen_utc'][:10]}).")
