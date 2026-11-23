"""More construction seeds for the held-out primary arms (EXPLORATORY follow-up).

The frozen analysis used seeds 11, 29 and 47. The sensitivity analysis (HELDOUT_EXPLORATORY.md,
section 17) showed that most of Epzero's change comes from one of them, and that with seeds resampled
its interval includes 0: three seeds cannot separate construction randomness from the effect. This
module runs the same three primary arms (Epzero HNSW, FAISS HNSW and IVF with frozen training, all
maintained incrementally) with seven further seeds, through the frozen code paths, unchanged:

- HNSW: `trajectory.trajectory_core` with the held-out inputs and configuration, as
  `heldout.run_hnsw` does, plus an observer that records each calibration query's neighbour recall at
  every width of the calibration grid (the frozen report keeps only the mean);
- IVF: `heldout.run_ivf`, unchanged.

The frozen package refuses these seeds on purpose; this module is the only entry point for them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contracts import digest
from .evaluation import ann_recall
from .heldout import DEPTH, ROLE, _require_single_thread, heldout_provenance, load_heldout, run_ivf, snapshot_specs
from .release import checksum

# Fixed before any of these runs existed (see scripts/pace/LONGEVAL_HELDOUT_SEEDS.md).
EXTRA_SEEDS = (101, 211, 307, 401, 503, 601, 701)


class CalibrationObserver:
    """Per-query neighbour recall of the calibration queries at every width of the calibration grid.
    Searches only; the index state must be unchanged afterwards."""

    def __call__(self, backend, stage, queries, reference, config):
        cohort = queries['calibration']
        truth = {q: reference[q] for q in cohort.ids}

        def fingerprint():
            # FAISS serializes efSearch, so it is reset to the configured value before every fingerprint,
            # as trajectory_core does.
            if backend.kind == 'faiss_hnsw':
                backend.index.hnsw.efSearch = config['ef']
            return digest(backend.state())

        before = fingerprint()
        per_width = {}
        for ef in config['ef_grid']:
            recalls = ann_recall(backend.search(cohort, config['k'], ef), truth)
            per_width[str(ef)] = {q: float(recalls[q]) for q in cohort.ids}
        if fingerprint() != before:
            raise ValueError('calibration observer changed index state')
        selected = stage['calibration']
        means = {ef: float(np.mean(list(v.values()))) for ef, v in per_width.items()}
        # The calibration step is deterministic: the observer must find exactly the recall it recorded.
        if means[str(selected['effort'])] != selected['calibration_recall']:
            raise ValueError('observer recall does not reproduce the calibration step')
        return {'calibration_recall_per_query': per_width, 'calibration_recall_mean': means}


def run_hnsw(folder, audit_path, inputs, output, config_path, *, kind, seed, package_root, native_origin):
    """heldout.run_hnsw for one of EXTRA_SEEDS, incremental only, with the calibration observer."""
    from .trajectory import trajectory_core, validate_settings
    if seed not in EXTRA_SEEDS:
        raise ValueError(f'seed must be one of {EXTRA_SEEDS}')
    config = json.loads(Path(config_path).read_text())
    validate_settings(config, kind, 'incremental', seed)
    if config['k'] != DEPTH:
        raise ValueError('held-out HNSW arms are at depth 100')
    _require_single_thread()
    queries, refs, receipts, evidence = load_heldout(inputs)
    specs, audits = snapshot_specs(folder, audit_path, inputs, receipts)
    record = heldout_provenance(package_root, native_origin)
    record['heldout_inputs'] = {**evidence, **audits, 'config_sha256': checksum(config_path)}
    record['follow_up'] = 'heldout_seeds: further construction seeds, exploratory'
    return trajectory_core(specs, queries, refs, output, config, kind=kind, mode='incremental', seed=seed,
                           provenance_record=record, snapshot_observer=CalibrationObserver(), evaluation_role=ROLE)


def run_ivf_seed(folder, audit_path, inputs, output, config_path, *, seed, package_root, native_origin):
    if seed not in EXTRA_SEEDS:
        raise ValueError(f'seed must be one of {EXTRA_SEEDS}')
    return run_ivf(folder, audit_path, inputs, output, config_path, mode='frozen_incremental', seed=seed,
                   package_root=package_root, native_origin=native_origin)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('hnsw', 'ivf'):
        p = sub.add_parser(command)
        for name in ('folder', 'audit', 'inputs', 'output', 'config'):
            p.add_argument(name, type=Path)
        if command == 'hnsw':
            p.add_argument('--backend', choices=('epzero', 'faiss_hnsw'), required=True)
        p.add_argument('--seed', type=int, choices=EXTRA_SEEDS, required=True)
        p.add_argument('--package-root', type=Path, required=True)
        p.add_argument('--native-origin', default='checkout-build')
    a = parser.parse_args()
    if a.command == 'hnsw':
        r = run_hnsw(a.folder, a.audit, a.inputs, a.output, a.config, kind=a.backend, seed=a.seed,
                      package_root=a.package_root, native_origin=a.native_origin)
        r = {k: r[k] for k in ('status', 'backend', 'mode', 'seed')}
    else:
        r = run_ivf_seed(a.folder, a.audit, a.inputs, a.output, a.config, seed=a.seed,
                         package_root=a.package_root, native_origin=a.native_origin)
        r = {k: r[k] for k in ('status', 'mode', 'seed')}
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
