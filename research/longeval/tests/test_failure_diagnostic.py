import copy
import gzip
import json
from types import SimpleNamespace

import pytest

from research.longeval import failure_diagnostic as diagnostic
from research.longeval.trajectory import trajectory_core
from research.longeval.tests.test_trajectory import small_problem


def test_native_trace_sweeps_preserve_original_trajectory(tmp_path):
    specs,queries,refs,config = small_problem(tmp_path)
    baseline = trajectory_core(specs,queries,refs,tmp_path/'baseline',config,
        kind='epzero',mode='incremental',seed=11,provenance_record={})
    # Real baseline receipts are loaded from JSON, including tuple-to-list conversion.
    observer = diagnostic.DiagnosticObserver(json.loads(json.dumps(baseline)),tmp_path/'diagnostic',tmp_path/'work',
        targets=queries['development'].ids[:2],efforts=(10,20,40),event_cap=3)
    result = trajectory_core(specs,queries,refs,tmp_path/'reproduced',config,
        kind='epzero',mode='incremental',seed=11,provenance_record={},snapshot_observer=observer)
    for a,b in zip(baseline['snapshots'],result['snapshots']):
        assert a['state'] == b['state']
        assert a['runs'] == b['runs']
        folder = tmp_path/'diagnostic'/f"snapshot-{a['snapshot']}"
        r = json.loads((folder/'report.json').read_text())
        assert r['status']=='completed' and r['baseline_reproduced'] and r['state_unchanged']
        assert len(r['sweeps'])==6 and len(r['traces'])==12
        assert len(r['trace_controls'])==2
        assert any(t['events_truncated']>0 for t in r['traces'])
        for trace in r['traces']:
            with gzip.open(folder/trace['trace_prefix_file'],'rt') as stream:
                lines = stream.readlines()
            assert len(lines)==1+trace['events_saved']
            assert trace['events_saved']<=3


@pytest.mark.parametrize('corruption', ['state','ranking'])
def test_reproduction_mismatch_blocks_diagnostics(tmp_path,corruption):
    specs,queries,refs,config = small_problem(tmp_path)
    baseline = trajectory_core(specs,queries,refs,tmp_path/'baseline',config,
        kind='epzero',mode='incremental',seed=11,provenance_record={})
    altered = copy.deepcopy(baseline)
    if corruption=='state':
        altered['snapshots'][0]['state']['topology_sha256']='0'*64
    else:
        altered['snapshots'][0]['runs']['frozen']['sha256']='0'*64
    observer = diagnostic.DiagnosticObserver(altered,tmp_path/'diagnostic',tmp_path/'work',
        targets=queries['development'].ids[:2],efforts=(10,20))
    with pytest.raises(ValueError,match='reproduction failed'):
        trajectory_core(specs,queries,refs,tmp_path/'reproduced',config,
            kind='epzero',mode='incremental',seed=11,provenance_record={},snapshot_observer=observer)
    assert not list((tmp_path/'diagnostic').iterdir())
    assert json.loads((tmp_path/'reproduced/report.json').read_text())['status']=='failed'


def test_directed_reachability_distinguishes_unvisited_from_unreachable():
    # Global label 1 is not live; the graph has two directed components and a cycle.
    adjacency={0:[2],2:[0,3],3:[],4:[3]}
    class Graph:
        def neighbors(self,label,layer):
            assert layer==0
            return adjacency[label]
    backend=SimpleNamespace(ids=('a','absent','b','c','d'),labels={'a':0,'b':2,'c':3,'d':4},
        live={'a','b','c','d'},index=Graph())
    references={'q':[('b',.9),('d',.8)]}
    result=diagnostic.directed_reachability(backend,0,references)
    assert result['reachable_count']==3 and result['edges_examined']==3
    assert result['queries']['q']=={'reachable_exact_count':1,'unreachable_exact_ids':['d']}
    reverse=diagnostic.directed_reachability(backend,4,references)
    assert reverse['reachable_count']==2
    adjacency[3]=[1]
    with pytest.raises(ValueError,match='nonlive'):
        diagnostic.directed_reachability(backend,0,references)


def test_trace_controls_are_stable_disjoint_and_development_only():
    ids=['a','b','c','d','e','f']
    chosen=diagnostic.trace_selection(ids,('a','b'))
    assert chosen==diagnostic.trace_selection(list(reversed(ids)),('a','b'))
    assert len(chosen)==len(set(chosen))==4 and set(chosen)<=set(ids)
    with pytest.raises(ValueError,match='development'):
        diagnostic.trace_selection(ids,('outside','a'))
