import gzip
import hashlib
import json

import pytest

from research.longeval.trace_phases import scan_trace


def fixture(tmp_path, events, **overrides):
    path = tmp_path / 'trace.gz'
    body = b''.join((json.dumps(e) + '\n').encode() for e in events)
    header = {'entry_label': 0, 'entry_layer': 1,
              'event_fields': ['layer', 'from', 'to', 'distance', 'accepted']}
    with gzip.open(path, 'wb') as stream:
        stream.write((json.dumps(header) + '\n').encode() + body)
    native = {'entry_label': 0, 'entry_layer': 1, 'events_truncated': 0,
              'events_saved': len(events), 'events_total': len(events),
              'distance_computations': len(events) + 2,
              'all_events_sha256': hashlib.sha256(body).hexdigest(),
              'layer0_start_label': 1, 'exact_neighbors_encountered_layer0': 2,
              'exact_neighbors_encountered_any_layer': 3}
    native.update(overrides)
    return path, native


def test_distinguishes_upper_encounter_from_layer_zero_and_counts_entry_rescore(tmp_path):
    path, native = fixture(tmp_path, [(1, 0, 1, .3, True), (1, 1, 2, .4, False),
                                    (0, 1, 3, .2, True), (0, 3, 4, .1, True)])
    result = scan_trace(path, native, {2, 3, 4})
    assert result['first_exact_any_layer_call'] == 3
    assert result['exact_first_layer0_calls'] == {3: 5, 4: 6}
    assert result['milestones'] == {'1': 5, '50': None, '95': None, '100': None}


def test_initial_exact_and_no_neighbors(tmp_path):
    path, native = fixture(tmp_path, [], layer0_start_label=0,
                          exact_neighbors_encountered_layer0=1,
                          exact_neighbors_encountered_any_layer=1)
    result = scan_trace(path, native, {0})
    assert result['first_exact_any_layer_call'] == 1
    assert result['milestones']['1'] == 2


@pytest.mark.parametrize('change,match', [({'events_truncated': 1}, 'complete'),
    ({'distance_computations': 99}, 'accounting'),
    ({'all_events_sha256': 'wrong'}, 'hash'),
    ({'exact_neighbors_encountered_layer0': 99}, 'encounter')])
def test_rejects_incomplete_corrupt_or_inconsistent_evidence(tmp_path, change, match):
    path, native = fixture(tmp_path, [(1, 0, 1, .3, True), (0, 1, 3, .2, True)],
                          exact_neighbors_encountered_layer0=1,
                          exact_neighbors_encountered_any_layer=1)
    native.update(change)
    with pytest.raises(ValueError, match=match):
        scan_trace(path, native, {3})


def test_rejects_duplicate_discovery(tmp_path):
    path, native = fixture(tmp_path, [(1, 0, 1, .3, True),
                                    (0, 1, 3, .2, True), (0, 1, 3, .2, True)])
    with pytest.raises(ValueError, match='discovery'):
        scan_trace(path, native, {3})
