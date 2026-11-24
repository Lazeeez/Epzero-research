import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from research.longeval import bge_encoder


def test_queries_get_the_instruction_and_documents_do_not():
    assert bge_encoder.query_text('graph search') == 'Represent this sentence for searching relevant passages: graph search'


def test_verify_model_checks_size_and_identity(tmp_path):
    raw = b'{"a": 1}'
    small = {'config.json': (len(raw), hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest())}
    (tmp_path / 'config.json').write_bytes(raw)
    assert bge_encoder.verify_model(tmp_path, small)['files']['config.json']['bytes'] == len(raw)
    (tmp_path / 'config.json').write_bytes(b'{"a": 2}')
    with pytest.raises(ValueError):
        bge_encoder.verify_model(tmp_path, small)


def test_verification_sample_is_deterministic_and_sorted():
    ids = [f'doc{i}' for i in range(5000)]
    rows = bge_encoder.verification_rows(ids, 100)
    assert len(rows) == 100 and (np.diff(rows) > 0).all()
    assert (rows == bge_encoder.verification_rows(list(ids), 100)).all()


def test_min_cosine():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert bge_encoder.min_cosine(a, a * 3) == pytest.approx(1.0)
    assert bge_encoder.min_cosine(a, np.array([[0.0, 1.0], [0.0, 1.0]])) == pytest.approx(0.0)


MODEL = os.environ.get('LONGEVAL_BGE_MODEL')


@pytest.mark.skipif(not MODEL, reason='set LONGEVAL_BGE_MODEL to a verified model folder')
def test_encoder_end_to_end_on_the_pinned_model():
    bge_encoder.verify_model(Path(MODEL))
    encoder = bge_encoder.Encoder(MODEL, 'cpu')
    docs = ['Approximate nearest neighbour search with hierarchical navigable small world graphs.',
            'A study of crop yields under drought in semi-arid regions.',
            'Graph-based vector indexes and their recall under insertion order.']
    vectors = encoder.encode(docs, batch_size=2)
    assert vectors.shape == (3, bge_encoder.DIM) and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    single = np.stack([encoder.encode([d], 1)[0] for d in docs])
    assert bge_encoder.min_cosine(vectors, single) >= bge_encoder.VERIFY_MIN_COSINE      # padding does not change vectors
    query = encoder.encode([bge_encoder.query_text('vector search graph index')])[0]
    scores = vectors @ query
    assert scores[1] < min(scores[0], scores[2])                                      # the off-topic document ranks last
    long = 'word ' * 2000
    assert encoder.token_counts([long])[0] > bge_encoder.MAX_LENGTH                   # counted before truncation
    assert np.isfinite(encoder.encode([long])).all()                                  # encoded after right truncation
    seen = []
    encoder.encode(docs, batch_size=2, progress=seen.append)
    assert seen == [2, 3]                                                             # called after each batch
    full = encoder.token_counts(docs + [long])
    batches = []
    capped = encoder.token_counts(docs + [long], cap=bge_encoder.MAX_LENGTH + 1, batch=2, progress=batches.append)
    assert batches == [2, 4] and np.array_equal(capped, np.minimum(full, bge_encoder.MAX_LENGTH + 1))
    assert (capped > bge_encoder.MAX_LENGTH).tolist() == (full > bge_encoder.MAX_LENGTH).tolist()   # truncation exact


def test_a_gpu_run_far_too_slow_after_the_probe_stops():
    probe = bge_encoder.PROBE_DOCUMENTS
    assert bge_encoder.throughput_check('cuda', probe - 1, 36000) < 1                 # too early to judge
    assert bge_encoder.throughput_check('cuda', probe, 1) == probe
    assert bge_encoder.throughput_check('cpu', probe, 3600) > 0                       # CPU runs are slow by design
    with pytest.raises(ValueError, match='not running on the GPU'):
        bge_encoder.throughput_check('cuda', probe, probe / (bge_encoder.MIN_GPU_DOCUMENTS_PER_SECOND - 1))
