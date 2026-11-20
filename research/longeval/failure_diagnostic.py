"""Post-hoc search diagnostics gated on reproduction of verified Epzero trajectories."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from .contracts import digest
from .evaluation import ann_recall, rank, read_run, write_run
from .query_diagnostic import validate_inputs
from .release import checksum
from .storage import atomic_json
from .trajectory import run_trajectory

EFFORTS = (100, 200, 400, 800, 1600, 3200)
TARGETS = ('161820e44e7022cf44fd99a59bce5792', '21e8c323e168b0813eff14fdac972f1f')
MAX_SAVED_EVENTS = 20000


def trace_selection(development_ids, targets):
    if len(set(targets)) != len(targets) or not set(targets) <= set(development_ids):
        raise ValueError('diagnostic targets must be distinct development queries')
    others = sorted(set(development_ids) - set(targets), key=lambda q: (
        hashlib.sha256(('longeval-trace-control-v1:' + q).encode()).hexdigest(), q))
    if len(others) < 2:
        raise ValueError('requires two additional development controls')
    return list(targets) + others[:2]


def trace_run(backend, trace):
    """Match the adapter's float32 conversion of native 1-dot distances."""
    pairs = []
    for label, distance in trace['results']:
        label = int(label)
        if not 0 <= label < len(backend.ids) or backend.ids[label] not in backend.live:
            raise ValueError('invalid traced result label')
        pairs.append((backend.ids[label], float(np.float32(1) - np.float32(distance))))
    return rank(pairs)


def trace_evidence(backend, trace, reference, path, *, event_cap=MAX_SAVED_EVENTS):
    """Bound the saved event prefix; hash and summarize the entire native trace."""
    if type(event_cap) is not int or event_cap < 0:
        raise ValueError('invalid trace prefix cap')
    entry, layer = int(trace['entry_label']), int(trace['entry_layer'])
    exact_labels = {backend.labels[key] for key, _ in reference}
    seen, layer0_seen = {entry}, set()
    start = entry
    first_layer0 = None
    upper_accepted = []
    h = hashlib.sha256()
    path = Path(path)
    with path.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
        header = {'entry_label': entry, 'entry_layer': layer, 'event_fields': ['layer','from','to','distance','accepted']}
        stream.write((json.dumps(header, separators=(',', ':'))+'\n').encode())
        for number, event in enumerate(trace['events']):
            level, source, target, distance, accepted = event
            level, source, target = int(level), int(source), int(target)
            if (not 0 <= source < len(backend.ids) or not 0 <= target < len(backend.ids)
                    or backend.ids[source] not in backend.live or backend.ids[target] not in backend.live
                    or not np.isfinite(distance)):
                raise ValueError('invalid trace event')
            seen.update((source,target))
            if level == 0:
                layer0_seen.update((source,target))
                if first_layer0 is None:
                    first_layer0 = source
            elif accepted:
                start = target
                upper_accepted.append([level,source,target,float(distance)])
            encoded = (json.dumps([level,source,target,float(distance),bool(accepted)], separators=(',', ':'))+'\n').encode()
            h.update(encoded)
            if number < event_cap:
                stream.write(encoded)
    if first_layer0 is not None and first_layer0 != start:
        raise ValueError('trace layer-0 start disagrees with upper-layer route')
    layer0_seen.add(start)
    count = len(trace['events'])
    return {'entry_label':entry, 'entry_id':backend.ids[entry], 'entry_layer':layer,
        'layer0_start_label':start, 'layer0_start_id':backend.ids[start], 'upper_accepted':upper_accepted,
        'distance_computations':int(trace['distance_computations']), 'nodes_expanded':int(trace['nodes_expanded']),
        'events_total':count, 'events_saved':min(count,event_cap), 'events_truncated':max(0,count-event_cap),
        'all_events_sha256':h.hexdigest(), 'trace_prefix_file':path.name, 'trace_prefix_sha256':checksum(path),
        'exact_neighbors_encountered_any_layer':len(seen & exact_labels),
        'exact_neighbors_encountered_layer0':len(layer0_seen & exact_labels)}


def directed_reachability(backend, root, references):
    """Exhaust layer-0 directed reachability using bounded numeric queues/flags."""
    capacity = len(backend.ids)
    live = np.zeros(capacity, dtype=np.bool_)
    for key in backend.live:
        live[backend.labels[key]] = True
    if not 0 <= root < capacity or not live[root]:
        raise ValueError('reachability root is not live')
    visited = np.zeros(capacity, dtype=np.bool_)
    queue = np.empty(len(backend.live), dtype=np.uint32)
    queue[0] = root; visited[root] = True
    head, tail, edges = 0, 1, 0
    while head < tail:
        label = int(queue[head]); head += 1
        for other in backend.index.neighbors(label, 0):
            other = int(other); edges += 1
            if not 0 <= other < capacity or not live[other]:
                raise ValueError('graph links to an invalid/nonlive label')
            if not visited[other]:
                visited[other] = True
                queue[tail] = other; tail += 1
    return {'root_label':root, 'root_id':backend.ids[root], 'live_count':len(backend.live),
        'reachable_count':tail, 'edges_examined':edges, 'visited_labels_sha256':hashlib.sha256(visited.tobytes()).hexdigest(),
        'queries':{q:{'reachable_exact_count':sum(bool(visited[backend.labels[d]]) for d,_ in pairs),
                     'unreachable_exact_ids':[d for d,_ in pairs if not visited[backend.labels[d]]]}
                   for q,pairs in references.items()}}


class DiagnosticObserver:
    def __init__(self, baseline, output, work, *, targets=TARGETS, efforts=EFFORTS, event_cap=MAX_SAVED_EVENTS):
        self.baseline, self.output, self.work = baseline, Path(output), Path(work)
        if (baseline.get('status') != 'completed' or baseline.get('backend') != 'epzero'
                or baseline.get('mode') != 'incremental' or [s['snapshot'] for s in baseline['snapshots']] != [1,2,3]):
            raise ValueError('requires a completed Epzero incremental baseline')
        self.targets, self.efforts, self.event_cap = tuple(targets), tuple(efforts), event_cap
        if not self.efforts or list(self.efforts) != sorted(set(self.efforts)) or min(self.efforts) < baseline['config']['k']:
            raise ValueError('invalid diagnostic effort grid')
        self.output.mkdir(parents=True, exist_ok=False)
        self.work.mkdir(parents=True, exist_ok=False)

    def __call__(self, backend, stage, queries, reference, config):
        step = stage['snapshot']
        expected = self.baseline['snapshots'][step-1]
        # Native metadata contains tuples; archived JSON represents those as lists.
        # Compare their canonical JSON identities, retaining every state field.
        if config != self.baseline['config'] or digest(stage['state']) != digest(expected['state']):
            raise ValueError('baseline graph/state reproduction failed')
        for regime in ('frozen','matched_recall'):
            if stage['runs'][regime]['sha256'] != expected['runs'][regime]['sha256']:
                raise ValueError('baseline ranking reproduction failed')
        selected = trace_selection(queries['development'].ids, self.targets)
        dest = self.output/f'snapshot-{step}'; dest.mkdir()
        work = self.work/f'snapshot-{step}'; work.mkdir()
        result = {'status':'running', 'purpose':'post-hoc mechanism diagnostic; no primary threshold selection',
            'snapshot':step, 'seed':backend.seed, 'baseline_reproduced':True,
            'baseline_state_sha256':expected['state']['state_sha256'], 'efforts':list(self.efforts),
            'trace_targets':list(self.targets), 'trace_controls':selected[len(self.targets):],
            'control_rule':'lowest SHA256(longeval-trace-control-v1:ID) among other development IDs',
            'event_prefix_cap':self.event_cap, 'sweeps':[], 'traces':[], 'reachability':[],
            'reachability_trigger':'snapshot 3; target recall below 0.95 at largest effort; each unique layer-0 start'}
        atomic_json(dest/'report.json',result)
        before = backend.state()
        positions = {q:i for i,q in enumerate(queries['development'].ids)}
        starts, last_recalls = {}, {}
        try:
            for effort in self.efforts:
                for role in ('calibration','development'):
                    run = backend.search(queries[role], config['k'], effort)
                    name = f'{role}.ef-{effort}.run'
                    write_run(dest/name,run)
                    if read_run(dest/name) != run:
                        raise ValueError('diagnostic run roundtrip failed')
                    recalls = ann_recall(run,{q:reference[q] for q in queries[role].ids})
                    result['sweeps'].append({'role':role,'effort':effort,'file':name,'sha256':checksum(dest/name),
                        'ann_recall_mean':float(np.mean(list(recalls.values()))),'per_query':recalls})
                    if role != 'development':
                        continue
                    last_recalls = recalls
                    for number,q in enumerate(selected):
                        trace = backend.index.trace(queries[role].vectors[positions[q]],k=config['k'],ef=effort)
                        if trace_run(backend,trace) != run[q]:
                            raise ValueError('trace and ordinary search results differ')
                        path = work/f'trace-{number}.ef-{effort}.jsonl.gz'
                        detail = trace_evidence(backend,trace,reference[q],path,event_cap=self.event_cap)
                        if path.stat().st_size > 4 * 1024**2:
                            raise ValueError('compressed trace prefix exceeds 4 MiB output budget')
                        shutil.copyfile(path,dest/path.name)
                        if q in starts and starts[q] != detail['layer0_start_label']:
                            raise ValueError('upper-layer route unexpectedly changes with search effort')
                        starts[q] = detail['layer0_start_label']
                        result['traces'].append({'query_id':q,'effort':effort,'ann_recall':recalls[q],**detail})
                        del trace
                atomic_json(dest/'report.json',result)
            if step == 3:
                roots = sorted({starts[q] for q in self.targets if last_recalls[q] < .95})
                for root in roots:
                    result['reachability'].append(directed_reachability(backend,root,{q:reference[q] for q in selected}))
                    atomic_json(dest/'report.json',result)
            if backend.state() != before:
                raise ValueError('diagnostics changed index state')
            result.update(status='completed',state_unchanged=True)
            atomic_json(dest/'report.json',result)
            return {'status':'completed','report_sha256':checksum(dest/'report.json'),'baseline_reproduced':True}
        except BaseException as error:
            result.update(status='failed',error_type=type(error).__name__)
            atomic_json(dest/'report.json',result)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('folder','audit','text_audit','inputs','encoded','references','baseline','output','work'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--seed',type=int,choices=(11,29),required=True)
    parser.add_argument('--check-only',action='store_true',help='validate baseline inputs/runtime before any corpus download')
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    if baseline['seed'] != args.seed:
        raise ValueError('baseline seed mismatch')
    config_path = Path(__file__).with_name('trajectory-development.json')
    if json.loads(config_path.read_text()) != baseline['config']:
        raise ValueError('baseline config mismatch')
    contract,_ = validate_inputs(args.inputs)
    trace_selection(contract['roles']['development'],TARGETS)
    for key,path in (('text_audit_sha256',args.text_audit), ('query_inputs_sha256',args.inputs/'query-inputs.json'),
                     ('query_encoding_sha256',args.encoded/'query-encoding.json'), ('config_sha256',config_path)):
        if baseline['provenance']['development_inputs'][key] != checksum(path):
            raise ValueError('baseline input identity mismatch')
    for step in (1,2,3):
        folder = args.references/f'snapshot-{step}'
        if baseline['provenance']['development_inputs']['exact_references'][f'snapshot-{step}'] != {
                'report_sha256':checksum(folder/'report.json'),'run_sha256':checksum(folder/'float32.run')}:
            raise ValueError('baseline exact-reference identity mismatch')
    # The optional observer hook is the only modified preexisting source module.
    root = Path(__file__).resolve().parents[2]
    for name,expected in baseline['provenance']['source_files'].items():
        if name != 'research/longeval/trajectory.py' and checksum(root/name) != expected:
            raise ValueError(f'baseline source changed: {name}')
    import epzero._core
    import faiss
    if checksum(epzero._core.__file__) != baseline['provenance']['epzero_binary_sha256']:
        raise ValueError('baseline native binary mismatch')
    import importlib.metadata
    for name,version in baseline['provenance']['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'baseline package changed: {name}')
    native = {name:{'file':Path(module.__file__).name,'sha256':checksum(module.__file__)}
        for name,module in list(sys.modules.items()) if name.startswith('faiss.')
        and getattr(module,'__file__',None) and Path(module.__file__).suffix in ('.so','.pyd')}
    if not native:
        raise ValueError('could not identify loaded FAISS native module')
    if args.check_only:
        print(json.dumps({'status':'validated','seed':args.seed,'purpose':'baseline input/runtime preflight; no graph built'}))
        return
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output/'diagnostic-inputs.json',{'baseline_report_sha256':checksum(args.baseline),
        'seed':args.seed,'faiss_native_modules':native,'source_sha256':checksum(__file__),
        'efforts':EFFORTS,'targets':TARGETS,'controls':trace_selection(contract['roles']['development'],TARGETS)[2:]})
    observer = DiagnosticObserver(baseline,args.output/'diagnostics',args.work)
    result = run_trajectory(args.folder,args.audit,args.text_audit,args.inputs,args.encoded,args.references,
        args.output/'trajectory',config_path,kind='epzero',mode='incremental',seed=args.seed,
        native_origin='checkout-build',snapshot_observer=observer)
    print(json.dumps({'status':result['status'],'seed':args.seed,'diagnostic_snapshots':len(result['snapshots'])}))


if __name__ == '__main__':
    main()
