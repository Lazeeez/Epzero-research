"""Write the FINAL_PROTOCOL_V1 freeze receipt; refuses if any held-out outcome already exists."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import subprocess
import sys

from .release import checksum

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RECEIPT = HERE / 'FINAL_PROTOCOL_V1_FREEZE.json'
FROZEN = ('research/longeval/FINAL_PROTOCOL_V1.md', 'research/longeval/trajectory-development.json',
          'research/longeval/ivf-development.json', 'research/longeval/tests/test_heldout.py',
          'scripts/pace/longeval-heldout.sbatch', 'scripts/pace/LONGEVAL_HELDOUT.md',
          'scripts/pace/package-longeval-heldout.py', 'scripts/pace/longeval-node-env.sh')
# Any of these existing means a held-out retrieval outcome may have been produced.
OUTCOMES = ('build/longeval-heldout-inputs-01/test/exact', 'build/longeval-heldout-exact-01',
            'build/longeval-heldout-01.tar.gz', 'build/pace-longeval-heldout', 'build/longeval-heldout-analysis-01')


def freeze(dry_run_report):
    if RECEIPT.exists():
        raise FileExistsError('protocol v1 is already frozen')
    present = [name for name in OUTCOMES if (ROOT / name).exists()]
    if present:
        raise ValueError(f'held-out outcomes already exist: {present}')
    sys.path.insert(0, str(ROOT))
    from research.longeval.heldout import load_heldout
    _, _, _, inputs = load_heldout(ROOT / 'build/longeval-heldout-inputs-01', references=False)
    dry = json.loads(Path(dry_run_report).read_text())
    if dry.get('status') != 'completed' or dry.get('pipeline_test_only') is not True:
        raise ValueError('requires a completed development pipeline test')
    files = {name: checksum(ROOT / name) for name in FROZEN}
    files.update({p.relative_to(ROOT).as_posix(): checksum(p) for p in sorted(HERE.glob('*.py'))})
    git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
    receipt = {'protocol': 'FINAL_PROTOCOL_V1', 'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'approved': '2026-11-19 (user approval of FINAL_PROTOCOL_V1_CANDIDATE.md)',
               'git_commit': git('rev-parse', 'HEAD'),
               'statement': 'No held-out retrieval outcome existed at freeze: no test-query exact reference, '
                            'ANN run or judged score had been computed. Test-query vectors (job 7164678) existed.',
               'checked_absent': list(OUTCOMES), 'heldout_inputs': inputs,
               'development_pipeline_test': {'report': Path(dry_run_report).resolve().relative_to(ROOT).as_posix(),
                                             'sha256': checksum(dry_run_report), 'conditions': dry['conditions'],
                                             'runs': dry['runs']},
               'files': files}
    with RECEIPT.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return receipt


if __name__ == '__main__':
    result = freeze(sys.argv[1])
    print(json.dumps({k: result[k] for k in ('protocol', 'frozen_utc', 'git_commit')} | {'files': len(result['files'])}, indent=2))
