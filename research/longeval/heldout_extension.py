"""Post-freeze EXPLORATORY held-out extension: shuffled-order HNSW arm and an ef sweep.

Not part of FINAL_PROTOCOL_V1. Designed after the held-out results, to test two exploratory
findings on the 381 test queries:

1. Shuffled insertion order. Does rebuilding each snapshot in a random insertion order remove
   HNSW's excess misses of later-inserted documents (HELDOUT_EXPLORATORY.md §1), and does it
   change the effectiveness gap?
2. ef sweep. The effectiveness gap as a function of neighbour recall at each snapshot, for
   incremental and shuffled indexes: ef in {100, 200, 400, 800, 1600, 3200}.

The incremental arm must reproduce the analysed held-out states and rankings exactly. Searches
are batched (not timed), as in the primary runs. No judgments are present in the job.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .contracts import digest
from .evaluation import ann_recall, read_run, write_run
from .heldout import ROLE, heldout_provenance, load_heldout, snapshot_specs
from .heldout_cost import check_baseline
from .release import checksum
from .storage import atomic_json

EFS = (100, 200, 400, 800, 1600, 3200)


class SweepObserver:
    """Searches the evaluation queries at each ef after every snapshot; verifies an optional baseline."""

    def __init__(self, output, baseline=None, efs=EFS):
        if baseline is not None and (baseline.get('status') != 'completed' or baseline.get('mode') != 'incremental'
                                     or baseline.get('evaluation_role') != ROLE):
            raise ValueError('requires a completed held-out incremental baseline')
        if not efs or list(efs) != sorted(set(efs)):
            raise ValueError('ef grid must be increasing and unique')
        self.output, self.baseline, self.efs = Path(output), baseline, tuple(efs)
        self.output.mkdir(parents=True, exist_ok=False)

    def __call__(self, backend, stage, queries, reference, config):
        step = stage['snapshot']
        if self.baseline is not None:
            expected = self.baseline['snapshots'][step - 1]
            if digest(stage['state']) != digest(expected['state']) or config != self.baseline['config']:
                raise ValueError('held-out baseline state or configuration did not reproduce')
            for regime in ('frozen', 'matched_recall'):
                if stage['runs'][regime]['sha256'] != expected['runs'][regime]['sha256']:
                    raise ValueError('held-out baseline ranking did not reproduce')
        destination = self.output / f'snapshot-{step}'
        destination.mkdir()
        original_ef = backend.index.hnsw.efSearch if backend.kind == 'faiss_hnsw' else None
        before = backend.state()
        cohort = queries[ROLE]
        refs = {q: reference[q] for q in cohort.ids}
        report = {'status': 'running', 'snapshot': step, 'backend': backend.kind, 'seed': backend.seed,
                  'state_sha256': stage['state']['state_sha256'], 'runs': {}}
        try:
            for ef in self.efs:
                if ef < config['k']:
                    raise ValueError('ef below result depth')
                run = backend.search(cohort, config['k'], ef)
                path = destination / f'ef-{ef}.run'
                write_run(path, run)
                if read_run(path) != run:
                    raise ValueError('sweep run export changed results')
                recalls = ann_recall(run, refs)
                report['runs'][str(ef)] = {'ef': ef, 'file': path.name, 'sha256': checksum(path),
                                           'ann_recall_per_query': recalls,
                                           'ann_recall_mean': float(np.mean(list(recalls.values())))}
                atomic_json(destination / 'report.json', report)
        finally:
            if original_ef is not None:
                # FAISS stores efSearch in the index; restore whatever the trajectory left there.
                backend.index.hnsw.efSearch = original_ef
        if digest(backend.state()) != digest(before):
            raise ValueError('sweep changed index state')
        report['status'] = 'completed'
        atomic_json(destination / 'report.json', report)
        return {'status': 'completed', 'report_sha256': checksum(destination / 'report.json'),
                'baseline_reproduced': self.baseline is not None}


def run(folder, audit_path, inputs, output, config_path, *, backend, mode, seed, baseline_path, package_root):
    from .trajectory import trajectory_core
    if mode not in ('incremental', 'shuffled') or seed not in (11, 29, 47) or backend not in ('epzero', 'faiss_hnsw'):
        raise ValueError('extension runs incremental (baseline-checked) or shuffled HNSW at seeds 11, 29, 47')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('single-thread environment required')
    baseline = json.loads(Path(baseline_path).read_text())
    check_baseline(baseline, inputs, config_path)
    if (baseline['backend'], baseline['seed']) != (backend, seed):
        raise ValueError('baseline does not match backend and seed')
    config = json.loads(Path(config_path).read_text())
    queries, refs, receipts, evidence = load_heldout(inputs)
    specs, audits = snapshot_specs(folder, audit_path, inputs, receipts)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'inputs.json', {'baseline_report_sha256': checksum(baseline_path),
                                         'source_sha256': checksum(__file__), 'mode': mode})
    record = heldout_provenance(package_root, 'checkout-build')
    record['heldout_inputs'] = {**evidence, **audits, 'config_sha256': checksum(config_path)}
    record['exploratory_extension'] = True
    observer = SweepObserver(output / 'sweep', baseline if mode == 'incremental' else None)
    return trajectory_core(specs, queries, refs, output / 'trajectory', config, kind=backend, mode=mode,
                           seed=seed, provenance_record=record, snapshot_observer=observer, evaluation_role=ROLE)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('folder', 'audit', 'inputs', 'output', 'config', 'baseline'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--backend', choices=('epzero', 'faiss_hnsw'), required=True)
    parser.add_argument('--mode', choices=('incremental', 'shuffled'), required=True)
    parser.add_argument('--seed', type=int, choices=(11, 29, 47), required=True)
    parser.add_argument('--package-root', type=Path, required=True)
    a = parser.parse_args()
    r = run(a.folder, a.audit, a.inputs, a.output, a.config, backend=a.backend, mode=a.mode, seed=a.seed,
            baseline_path=a.baseline, package_root=a.package_root)
    print(json.dumps({k: r[k] for k in ('status', 'backend', 'mode', 'seed')}, indent=2))


if __name__ == '__main__':
    main()
