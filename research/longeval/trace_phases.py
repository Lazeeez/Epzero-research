"""Streaming audit of first discovery of exact neighbors in complete native traces.

These are retrospective oracle measurements, not deployable stopping signals.
Only scored edges are recorded; absent edges and full graph connectivity are unknown.
"""
import gzip
import hashlib
import json
import math


def scan_trace(path, native, exact_labels):
    """Return exact-neighbor first-score calls and independently check trace accounting."""
    if native['events_truncated'] or native['events_saved'] != native['events_total']:
        raise ValueError('complete trace required')
    exact = set(exact_labels)
    entry = native['entry_label']
    start = entry
    upper = 0
    lower = False
    total = 0
    seen = set()
    any_seen = {entry}
    first = {}
    any_first = {entry: 1} if entry in exact else {}
    h = hashlib.sha256()
    with gzip.open(path, 'rb') as stream:
        header = json.loads(next(stream))
        if (header['entry_label'] != entry or header['entry_layer'] != native['entry_layer']
                or header['event_fields'] != ['layer', 'from', 'to', 'distance', 'accepted']):
            raise ValueError('trace header differs')
        for encoded in stream:
            h.update(encoded)
            level, source, target, distance, accepted = json.loads(encoded)
            if level < 0 or not math.isfinite(distance):
                raise ValueError('invalid trace event')
            total += 1
            any_seen.update((source, target))
            if level > 0:
                if lower:
                    raise ValueError('upper event after layer zero')
                upper += 1
                if target in exact:
                    any_first.setdefault(target, upper + 1)
                if accepted:
                    start = target
            else:
                if not lower:
                    lower = True
                    if source != start or start != native['layer0_start_label']:
                        raise ValueError('layer zero start differs')
                    seen.add(start)
                    if start in exact:
                        first[start] = upper + 2
                if source not in seen or target in seen:
                    raise ValueError('invalid layer zero discovery order')
                seen.add(target)
                if target in exact:
                    first[target] = total + 2
                    any_first.setdefault(target, total + 2)
    if not lower:
        seen.add(start)
        if start in exact:
            first[start] = upper + 2
    if (total != native['events_total'] or total + 2 != native['distance_computations']
            or h.hexdigest() != native['all_events_sha256']):
        raise ValueError('trace hash or distance accounting differs')
    if (len(first) != native['exact_neighbors_encountered_layer0']
            or len(any_seen & exact) != native['exact_neighbors_encountered_any_layer']):
        raise ValueError('exact encounter counts differ')
    calls = sorted(first.values())
    return {'event_count': total, 'upper_distance_calls': upper + 1,
            'layer0_initial_call': upper + 2, 'distance_computations': total + 2,
            'first_exact_any_layer_call': min(any_first.values(), default=None),
            'milestones': {str(n): calls[n-1] if len(calls) >= n else None
                           for n in (1, 50, 95, 100)},
            'exact_first_layer0_calls': first,
            'layer0_encountered': len(first)}
