"""Calibration-only path-witness repair prototype; no recall/novelty guarantee."""
from collections import defaultdict


class Trace:
    """Consume complete replay events; first-discovery parents give witnessed paths."""
    def __init__(self):
        self.start = None
        self.scored = {}
        self.parents = {}
        self.expanded = set()
        self.adjacency = {}

    def __call__(self, event):
        if event[0] == 'layer0_entry':
            self.start = event[1]
            self.scored[event[1]] = event[2]
        elif event[0] == 'score' and event[1] == 0:
            _, _, source, target, distance, _ = event
            if target in self.scored:
                raise ValueError('duplicate layer-0 score')
            self.scored[target] = distance
            self.parents[target] = source
        elif event[0] == 'expand' and event[1] == 0:
            self.expanded.add(event[2])
        elif event[0] == 'neighbors' and event[1] == 0:
            self.adjacency[event[2]] = tuple(event[3])

    def path(self, destination):
        path, seen = [destination], {destination}
        while path[-1] != self.start:
            parent = self.parents[path[-1]]
            if parent in seen:
                raise ValueError('cycle in witness parents')
            if path[-1] not in self.adjacency[parent]:
                raise ValueError('unwitnessed parent edge')
            path.append(parent)
            seen.add(parent)
        return list(reversed(path))


def propose(query_id, low, high, low_result, exact, *, target=.95):
    """At most one proposal; caller must supply a calibration query only."""
    if low.start != high.start or low.start is None:
        raise ValueError('requires the same layer-0 start')
    k = len(low_result['results'])
    if len(exact) != k or len(set(exact)) != k or not 0 < target <= 1:
        raise ValueError('invalid exact reference or target')
    truth = set(exact)
    returned = {label for label, _ in low_result['results']}
    if len(returned & truth) / k >= target:
        return None
    threshold = low_result['results'][-1][1]
    for endpoint in high.scored:  # insertion order is high-search discovery order
        if endpoint not in truth or endpoint in returned:
            continue
        path = high.path(endpoint)
        for source_at in range(len(path) - 3, -1, -1):
            source = path[source_at]
            if source not in low.expanded:
                continue
            adjacent = set(low.adjacency[source])
            for dest_at in range(source_at + 2, len(path)):
                destination = path[dest_at]
                if (destination in low.scored or destination in adjacent
                        or high.scored[destination] >= threshold):
                    continue
                return {'query_id': query_id, 'source': source, 'destination': destination,
                        'endpoint': endpoint, 'path': path, 'source_at': source_at,
                        'destination_at': dest_at, 'skipped_nodes': dest_at - source_at - 1,
                        'destination_in_exact': destination in truth,
                        'low_recall': len(returned & truth) / k,
                        'low_threshold': threshold,
                        'destination_distance': high.scored[destination]}
    return None


def select(proposals, calibration_ids, *, edge_budget=32, per_source=2):
    """No evaluator callback: selection uses only declared calibration proposals."""
    if (type(edge_budget) is not int or type(per_source) is not int
            or edge_budget < 0 or per_source < 1):
        raise ValueError('invalid edge limits')
    allowed = set(calibration_ids)
    seen, grouped = set(), defaultdict(list)
    for row in proposals:
        if row['query_id'] not in allowed:
            raise ValueError('non-calibration query in repair fitting')
        if row['query_id'] in seen:
            raise ValueError('more than one proposal per query')
        seen.add(row['query_id'])
        if row['source'] == row['destination'] or row['skipped_nodes'] < 1:
            raise ValueError('invalid shortcut')
        grouped[(row['source'], row['destination'])].append(row)
    ranked = sorted(grouped, key=lambda edge: (-len(grouped[edge]),
        -sum(row['skipped_nodes'] for row in grouped[edge]), *edge))
    chosen, counts = [], defaultdict(int)
    for source, destination in ranked:
        if len(chosen) >= edge_budget:
            break
        if counts[source] >= per_source:
            continue
        counts[source] += 1
        evidence = sorted(grouped[(source, destination)], key=lambda row: row['query_id'])
        chosen.append({'source': source, 'destination': destination,
                       'support': len(evidence), 'evidence': evidence})
    return chosen


class Overlay:
    """Immutable base adjacency plus a fixed layer-0 overlay, no query features."""
    def __init__(self, neighbors, edges, live, *, per_source=2):
        self.base = neighbors
        allowed = set(live)
        extra = defaultdict(list)
        for source, destination in edges:
            if source not in allowed or destination not in allowed or source == destination:
                raise ValueError('invalid overlay endpoint')
            if destination in neighbors(source, 0) or destination in extra[source]:
                raise ValueError('duplicate overlay edge')
            extra[source].append(destination)
            if len(extra[source]) > per_source:
                raise ValueError('overlay degree budget exceeded')
        self.extra = {source: tuple(values) for source, values in extra.items()}
        self.edge_reads = 0

    def __call__(self, label, layer):
        added = self.extra.get(label, ()) if layer == 0 else ()
        self.edge_reads += len(added)
        return [*self.base(label, layer), *added]
