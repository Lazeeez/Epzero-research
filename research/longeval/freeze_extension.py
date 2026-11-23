"""Write the freeze receipt of the extension protocol; refuses while a decision is open.

The extension covers follow-up experiments designed after the held-out results were seen
(docs/protocol/EXTENSION_PROTOCOL_V1.md). FINAL_PROTOCOL_V1 and its receipt are not touched.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import subprocess
import sys

from .release import checksum

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'docs/protocol/EXTENSION_PROTOCOL_V1.md'
RECEIPT = HERE / 'EXTENSION_PROTOCOL_V1_FREEZE.json'
OPEN = 'TO DECIDE'
FROZEN = ('research/longeval/docs/protocol/EXTENSION_PROTOCOL_V1.md',
          'research/longeval/docs/protocol/JUDGE_VALIDATION_THRESHOLDS.md',
          'research/longeval/docs/protocol/E4_SECOND_ENCODER.md',
          'research/longeval/judge.py', 'research/longeval/judge-requirements.txt', 'research/longeval/judge-kernels.txt',
          'research/longeval/judge-model.json',
          'research/longeval/coverage.py', 'research/longeval/audit.py', 'research/longeval/insertion_pilot.py',
          'research/longeval/heldout_pool.py', 'research/longeval/evaluation.py',
          'scripts/pace/longeval-judge-pilot.sbatch', 'scripts/pace/longeval-judge-server.yaml',
          'scripts/pace/LONGEVAL_JUDGE_PILOT.md', 'scripts/pace/longeval-insertion-order.sbatch',
          'scripts/pace/LONGEVAL_INSERTION_ORDER.md',
          'research/longeval/order_pairs.py', 'research/longeval/insertion_strata.py',
          'scripts/pace/longeval-insertion-order-ids.sbatch', 'scripts/pace/LONGEVAL_INSERTION_ORDER_IDS.md',
          'scripts/pace/package-longeval-insertion-order-ids.py')
# Follow-up outcomes that may already exist when the protocol is frozen; their presence is recorded, not refused.
OUTCOMES = {'judge pilot evidence': 'build/pace-longeval-judge-pilot',
            'full-scale insertion-order evidence': 'build/pace-longeval-insertion-order',
            'order rerun with IDs and confirmation queries': 'build/pace-longeval-insertion-order-ids',
            'second-encoder vectors': 'build/longeval-bge-vectors',
            'human audit grades': 'build/longeval-human-audit-01/grades'}


def freeze(*, root=ROOT, protocol=PROTOCOL, receipt=RECEIPT, files=FROZEN, outcomes=None, build=None):
    """`build` is the folder that holds the study's `build/` outputs, if it is not the repository root."""
    root, protocol, receipt = Path(root), Path(protocol), Path(receipt)
    if receipt.exists():
        raise FileExistsError('the extension protocol is already frozen')
    if not protocol.is_file():
        raise FileNotFoundError('the protocol has not been finalised (docs/protocol/EXTENSION_PROTOCOL_V1.md is missing)')
    text = protocol.read_text(encoding='utf-8')
    if OPEN in text:
        raise ValueError(f'the protocol still holds {text.count(OPEN)} open decisions')
    base = Path(build) if build is not None else root
    outcomes = OUTCOMES if outcomes is None else outcomes
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--', *files], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    if dirty:
        raise ValueError('a file to be frozen has uncommitted changes')
    record = {'protocol': 'EXTENSION_PROTOCOL_V1',
              'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'git_commit': commit,
              'statement': 'Follow-up to FINAL_PROTOCOL_V1 on the same test queries. The held-out results and the '
                           'exploratory analyses listed in the protocol were known when it was frozen.',
              'outcomes_present_at_freeze': {name: (base / path).exists() for name, path in outcomes.items()},
              'files': {name: checksum(root / name) for name in files}}
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return record


if __name__ == '__main__':
    result = freeze(build=sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps({key: result[key] for key in ('protocol', 'frozen_utc', 'git_commit', 'outcomes_present_at_freeze')}
                     | {'files': len(result['files'])}, indent=2))
