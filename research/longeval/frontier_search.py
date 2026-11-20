"""Exploratory overlap-guided frontier resumption; no efficacy or novelty claim.

Read-only float32/IP graph prototype, not a native latency implementation. The
policy sees scored distances, adjacency IDs and query-local visited state only.
Both rejected and evicted unexpanded candidates can enter a bounded reserve.
"""
from bisect import bisect_right
import math


def frontier_search(neighbors, score, *, entry, entry_layer, k, ef,
                    distance_budget=None, policy='overlap', reserve_capacity=256,
                    shortlist=8, max_resumptions=8, emit=lambda event: None,
                    max_distances=1_000_000):
    """Resume an exhausted beam from a bounded, previously scored frontier.

    ``closest`` selects its nearest reserve candidate. ``overlap`` examines the
    nearest ``shortlist`` and selects the largest fraction of currently unvisited
    neighbors; ties prefer smaller distance then earlier discovery. Adjacency
    lookahead never marks IDs visited or evaluates their vectors. Every adjacency
    access, inspected ID, expansion and distance call is counted. ``off`` must
    reproduce the ordinary Epzero replay before an experimental use.

    This heuristic has no recall guarantee. A node with many unvisited neighbors
    may lead away from the answer. A capped reserve may discard useful routes.
    """
    integer_options = (k, ef, entry_layer, reserve_capacity, shortlist,
                       max_resumptions, max_distances)
    if (any(not isinstance(v, int) or isinstance(v, bool) for v in integer_options)
            or k < 1 or ef < k or entry_layer < 0 or reserve_capacity < 0
            or shortlist < 1 or max_resumptions < 0 or max_distances < 2
            or policy not in ('off', 'closest', 'overlap')
            or (distance_budget is not None and
                (not isinstance(distance_budget, int) or isinstance(distance_budget, bool)
                 or distance_budget < 2))):
        raise ValueError('invalid frontier-search configuration')

    distances = expansions = reads = inspected = 0
    reserve_peak = lookahead_peak = 0
    upper_events, selections = [], []

    def evaluate(label):
        nonlocal distances
        if distances >= max_distances:
            raise RuntimeError('distance guard exceeded')
        value = float(score(label))
        distances += 1
        if not math.isfinite(value):
            raise ValueError('nonfinite distance')
        return value

    def read(label, layer, purpose):
        nonlocal reads, inspected
        adjacent = list(map(int, neighbors(label, layer)))
        reads += 1
        inspected += len(adjacent)
        emit(['adjacency', layer, label, purpose, adjacent])
        return adjacent

    def exhausted():
        return distance_budget is not None and distances >= distance_budget

    current, value = entry, evaluate(entry)
    emit(['entry', current, value])
    for layer in range(entry_layer, 0, -1):
        improved = True
        while improved:
            improved = False
            origin = current
            expansions += 1
            emit(['expand', layer, origin, False])
            for other in read(origin, layer, 'expand'):
                if exhausted():
                    raise ValueError('work budget does not cover upper-layer route')
                distance = evaluate(other)
                accepted = distance < value
                row = (layer, origin, other, distance, accepted)
                upper_events.append(row)
                emit(['score', *row])
                if accepted:
                    current, value, improved = other, distance, True
    if exhausted():
        raise ValueError('work budget does not cover layer-0 initialization')
    start = current
    initial = evaluate(start)
    emit(['layer0_entry', start, initial])
    items, values = [[initial, start, False]], [initial]
    visited, expanded = {start}, set()
    # Discovery order breaks distance ties without depending on document IDs.
    order = {start: 0}
    reserve = {}  # label -> (distance, discovery ordinal)
    cursor = 0
    budget_stopped = False

    def defer(label, distance):
        nonlocal reserve_peak
        if policy == 'off' or not reserve_capacity or label in expanded:
            return
        reserve[label] = (distance, order[label])
        if len(reserve) > reserve_capacity:
            worst = max(reserve, key=reserve.__getitem__)
            del reserve[worst]
        reserve_peak = max(reserve_peak, len(reserve))

    while True:
        if exhausted():
            budget_stopped = True
            break
        adjacent = None
        forced = cursor >= len(items)
        if forced:
            if not reserve or len(selections) >= max_resumptions:
                break
            candidates = sorted(reserve, key=reserve.__getitem__)
            if policy == 'closest':
                origin = candidates[0]
                evidence = []
            else:
                evidence, lists = [], {}
                for candidate in candidates[:shortlist]:
                    links = read(candidate, 0, 'lookahead')
                    lists[candidate] = links
                    unique = set(links)
                    unseen = len(unique - visited)
                    fraction = unseen / len(unique) if unique else 0.0
                    evidence.append((candidate, fraction, unseen, len(unique)))
                lookahead_peak = max(lookahead_peak, sum(map(len, lists.values())))
                origin = min(evidence, key=lambda row: (-row[1], *reserve[row[0]]))[0]
                adjacent = lists[origin]  # Charge the lookahead read once; reuse it.
            choice = {'label': origin, 'distance': reserve[origin][0],
                      'evidence': evidence, 'distance_computations_before': distances}
            selections.append(choice)
            emit(['resume', choice])
            del reserve[origin]
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
        if adjacent is None:
            adjacent = read(origin, 0, 'expand')
        pending = []
        for other in adjacent:
            if other not in visited:
                visited.add(other)
                pending.append(other)
        for other in pending:
            if exhausted():
                budget_stopped = True
                break
            distance = evaluate(other)
            order[other] = len(order)
            accepted = len(items) < ef or distance < values[-1]
            emit(['score', 0, origin, other, distance, accepted])
            if accepted:
                position = bisect_right(values, distance)
                values.insert(position, distance)
                items.insert(position, [distance, other, False])
                if len(items) > ef:
                    removed = items.pop()
                    values.pop()
                    if not removed[2]:
                        defer(removed[1], removed[0])
                if position < cursor:
                    cursor = position
            else:
                defer(other, distance)
        if budget_stopped:
            break
    results = [(item[1], item[0]) for item in items[:k]]
    if len(results) != k:
        raise ValueError('search cannot produce requested result depth')
    return {'results': results, 'distance_computations': distances,
            'nodes_expanded': expansions, 'layer0_expansions': len(expanded),
            'layer0_start': start, 'upper_events': upper_events,
            'adjacency_reads': reads, 'adjacency_ids_inspected': inspected,
            'reserve_peak': reserve_peak, 'lookahead_ids_peak': lookahead_peak,
            'selections': selections, 'distance_budget': distance_budget,
            'budget_stopped': budget_stopped, 'policy': policy, 'ef': ef}
