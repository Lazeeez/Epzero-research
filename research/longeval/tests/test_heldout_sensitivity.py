"""The sensitivity analyses: the fast scorer, the seed bootstrap and the label-noise model."""
import json

import numpy as np
import pytest

from research.longeval import heldout_sensitivity as sens
from research.longeval.evaluation import evaluate


def test_fast_ndcg_matches_the_frozen_evaluator():
    rng = np.random.default_rng(3)
    qrels = {f'q{i}': {f'd{j}': int(rng.integers(0, 3)) for j in range(30) if rng.random() < .5} for i in range(20)}
    run = {q: [(f'd{j}', 1.0 - j / 100) for j in rng.permutation(40)[:15]] for q in qrels}
    frozen = evaluate(run, qrels, list(qrels), population='explicit', ndcg_k=10, recall_k=100)
    for q in qrels:
        # The run lists are deliberately not in score order: the scorer must rank them as the evaluator does.
        fast = sens.ndcg10(sens.ranked(run[q]), qrels[q], sens.ideal_dcg(qrels[q]))
        assert fast == pytest.approx(frozen['per_query'][q]['nDCG@10'], abs=1e-12)
    assert sens.ndcg10(['a'], {}, 0.0) == 0.0


def test_seed_bootstrap_widens_when_seeds_disagree():
    rng = np.random.default_rng(1)
    agree = rng.normal(-.005, .02, (300, 1)) + np.zeros((1, 3))
    disagree = agree + np.array([[0.0, -.01, 0.0]])
    a = sens.seed_bootstrap({'x': agree}, draws=2000)['x']
    b = sens.seed_bootstrap({'x': disagree}, draws=2000)['x']
    width = lambda r: r['interval_95_queries_and_seeds'][1] - r['interval_95_queries_and_seeds'][0]
    assert a['seed_spread'] == pytest.approx(0.0, abs=1e-12) and b['seed_spread'] == pytest.approx(.01)
    assert width(b) > width(a)


def test_repeat_model_rows_are_distributions(tmp_path):
    report = {'between_repeats': [{'condition': 'provider_default', 'exact_agreement': .9,
                                   'confusion_rows_first_columns_second': [[9, 1, 0, 0], [1, 8, 1, 0], [0, 1, 9, 0], [0, 0, 0, 0]]},
                                  {'condition': 'temperature_0', 'exact_agreement': 1.0,
                                   'confusion_rows_first_columns_second': [[1, 0, 0, 0]] * 4}]}
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(report))
    matrix, agreement = sens.repeat_model(path)
    assert matrix.shape == (3, 3) and np.allclose(matrix.sum(axis=1), 1) and agreement == .9
    assert matrix[1].tolist() == pytest.approx([.1, .8, .1])
