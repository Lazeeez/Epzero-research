"""Isolated, fully observed search replay; never changes the native index.

Restricted to GNU/Linux Epzero uncompressed float32 inner-product indexes. Uses
the loaded library's own exported distance kernel through its Itanium ABI symbol.
Every experimental use must first pass native event/ranking/work-count parity.
"""
from bisect import bisect_right
import ctypes
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .release import checksum


class NativeScorer:
    def __init__(self, index, query):
        import epzero._core
        if sys.platform != 'linux':
            raise RuntimeError('retention diagnostic requires the validated Linux native ABI')
        params = index.params
        if (params['metric'] != 'ip' or params['quantization'] != 'none'
                or params['half_vectors'] or params['nav_dim'] or index.deleted_count):
            raise ValueError('requires uncompressed, undeleted, full-vector inner-product index')
        self.index = index
        self.query = np.ascontiguousarray(query, dtype=np.float32)
        if self.query.ndim != 1 or not np.isfinite(self.query).all():
            raise ValueError('requires one finite float32 query')
        self.library = ctypes.CDLL(epzero._core.__file__)
        factory = getattr(self.library, '_ZN6epzero15get_distance_fnENS_6MetricE')
        factory.argtypes, factory.restype = [ctypes.c_int], ctypes.c_void_p
        pointer = factory(1)  # epzero::Metric::InnerProduct, checked by native parity.
        if not pointer:
            raise RuntimeError('native inner-product kernel unavailable')
        self.pointer_type = ctypes.POINTER(ctypes.c_float)
        self.kernel = ctypes.CFUNCTYPE(ctypes.c_float, self.pointer_type,
                                       self.pointer_type, ctypes.c_size_t)(pointer)
        self.query_pointer = self.query.ctypes.data_as(self.pointer_type)
        self.binary_sha256 = checksum(epzero._core.__file__)

    def __call__(self, label):
        vector = self.index.get_vector(label)
        if vector.dtype != np.float32 or vector.shape != self.query.shape or not vector.flags.c_contiguous:
            raise ValueError('unexpected native vector layout')
        value = float(self.kernel(self.query_pointer, vector.ctypes.data_as(self.pointer_type), self.query.size))
        if not np.isfinite(value):
            raise ValueError('nonfinite native score')
        return value


class EventLog:
    """Complete bounded log: exceeding a guard fails; never silently truncates."""
    def __init__(self, path, max_events=2_000_000, max_bytes=32 * 1024**2):
        self.path, self.max_events, self.max_bytes = Path(path), max_events, max_bytes
        self.count, self.digest = 0, hashlib.sha256()
        self.failed = False
        self.raw = self.path.open('xb')
        self.stream = gzip.GzipFile(filename='', mode='wb', fileobj=self.raw, mtime=0)

    def __call__(self, event):
        if self.count >= self.max_events:
            self.failed = True
            raise RuntimeError('complete event-log guard exceeded')
        raw = (json.dumps(event, separators=(',', ':'), allow_nan=False) + '\n').encode()
        self.digest.update(raw)
        self.stream.write(raw)
        self.count += 1
        if self.raw.tell() > self.max_bytes:
            self.failed = True
            raise RuntimeError('compressed event-log guard exceeded')

    def close(self):
        self.stream.close()
        self.raw.close()
        if self.path.stat().st_size > self.max_bytes:
            raise RuntimeError('compressed event-log guard exceeded')
        return {'file': self.path.name, 'sha256': checksum(self.path),
                'events': self.count, 'event_stream_sha256': self.digest.hexdigest(),
                'complete': not self.failed}


def replay_search(neighbors, score, *, entry, entry_layer, k, ef, rescue=None,
                  distance_budget=None, emit=lambda event: None, native_event=None,
                  max_distances=1_000_000):
    """Native greedy descent + beam, optionally deferring ONE evicted candidate.

    Rescue triggers only for a previously admitted, unexpanded layer-0 candidate.
    After the ordinary beam exhausts, expand that candidate once outside the beam,
    then resume ordinary search. It is not inserted as a result or scored again.
    All scores, including repeated upper/entry scores, count toward the budget.
    """
    if k < 1 or ef < k or (distance_budget is not None and distance_budget < 2):
        raise ValueError('invalid search depth, effort or work budget')
    distances = expansions = 0
    upper_events = []

    def evaluate(label):
        nonlocal distances
        if distances >= max_distances:
            raise RuntimeError('distance guard exceeded')
        value = score(label)
        distances += 1
        return value

    def event(layer, source, target, distance, accepted):
        row = (layer, source, target, distance, accepted)
        if native_event is not None:
            native_event(row)
        emit(['score', *row])

    current = entry
    current_distance = evaluate(current)
    emit(['entry', current, current_distance])
    for layer in range(entry_layer, 0, -1):
        improved = True
        while improved:
            improved = False
            origin = current
            expansions += 1
            emit(['expand', layer, origin, False])
            adjacent = list(map(int, neighbors(origin, layer)))
            emit(['neighbors', layer, origin, adjacent])
            for other in adjacent:
                if distance_budget is not None and distances >= distance_budget:
                    raise ValueError('work budget does not cover upper-layer route')
                value = evaluate(other)
                accepted = value < current_distance
                event(layer, origin, other, value, accepted)
                upper_events.append((layer, origin, other, value, accepted))
                if accepted:
                    current, current_distance, improved = other, value, True
    if distance_budget is not None and distances >= distance_budget:
        raise ValueError('work budget does not cover layer-0 initialization')
    start = current
    initial = evaluate(start)  # native search scores its layer-0 entry again.
    emit(['layer0_entry', start, initial])
    items, values = [[initial, start, False]], [initial]
    visited, expanded = {start}, set()
    cursor = 0
    evicted_unexpanded = []
    deferred = None
    rescue_triggered = rescue_expanded = False
    budget_stopped = False
    while cursor < len(items) or deferred is not None:
        if distance_budget is not None and distances >= distance_budget:
            budget_stopped = True
            break
        forced = cursor >= len(items)
        if forced:
            origin = deferred
            deferred = None
            rescue_expanded = True
        else:
            item = items[cursor]
            origin = item[1]
            item[2] = True
            while cursor < len(items) and items[cursor][2]:
                cursor += 1
        if origin in expanded:
            raise ValueError('duplicate layer-0 expansion')
        expanded.add(origin)
        expansions += 1
        emit(['expand', 0, origin, forced])
        pending = []
        adjacent = list(map(int, neighbors(origin, 0)))
        emit(['neighbors', 0, origin, adjacent])
        for other in adjacent:
            if other not in visited:
                visited.add(other)
                pending.append(other)
        for other in pending:
            if distance_budget is not None and distances >= distance_budget:
                budget_stopped = True
                break
            value = evaluate(other)
            accepted = len(items) < ef or value < values[-1]
            event(0, origin, other, value, accepted)
            if accepted:
                position = bisect_right(values, value)
                values.insert(position, value)
                items.insert(position, [value, other, False])
                if len(items) > ef:
                    removed = items.pop()
                    values.pop()
                    emit(['evict', removed[1], removed[0], removed[2], other])
                    if not removed[2]:
                        evicted_unexpanded.append(removed[1])
                        if removed[1] == rescue:
                            deferred = rescue
                            rescue_triggered = True
                            emit(['defer', rescue])
                if position < cursor:
                    cursor = position
        if budget_stopped:
            break
    result = [(item[1], item[0]) for item in items[:k]]
    if len(result) != k:
        raise ValueError('search cannot produce requested result depth')
    return {'results': result, 'distance_computations': distances, 'nodes_expanded': expansions,
            'layer0_expansions': len(expanded), 'layer0_start': start, 'upper_events': upper_events,
            'evicted_unexpanded': evicted_unexpanded, 'rescue_label': rescue,
            'rescue_triggered': rescue_triggered, 'rescue_expanded': rescue_expanded,
            'distance_budget': distance_budget, 'budget_stopped': budget_stopped,
            'ef': ef}


def native_parity(index, query, k, ef, *, emit=lambda event: None):
    """Fail before intervention unless the entire native search is reproduced."""
    trace = index.trace(query, k=k, ef=ef)
    events = iter(trace['events'])
    def check(row):
        if next(events, None) != row:
            raise ValueError('native event parity failed')
    scorer = NativeScorer(index, query)
    result = replay_search(index.neighbors, scorer, entry=trace['entry_label'],
        entry_layer=trace['entry_layer'], k=k, ef=ef, emit=emit, native_event=check)
    if next(events, None) is not None:
        raise ValueError('native event replay ended early')
    for key in ('results', 'distance_computations', 'nodes_expanded'):
        if result[key] != trace[key]:
            raise ValueError(f'native {key} parity failed')
    labels, scores = index.search(np.asarray(query).reshape(1, -1), k=k, ef=ef,
        rerank=0, rerank_z=0, patience=0, threads=1)
    if result['results'] != list(zip(map(int, labels[0]), map(float, scores[0]))):
        raise ValueError('ordinary native search parity failed')
    result.update(native_parity=True, entry_label=trace['entry_label'], entry_layer=trace['entry_layer'],
                  native_binary_sha256=scorer.binary_sha256)
    return result
