"""Sufficient local inner-product progress bound; not a retrieval guarantee."""
import math
import numpy as np


def edge_certificate(center, source, destination, radius):
    center, source, destination = (np.asarray(x, dtype=np.float64) for x in (center, source, destination))
    if (center.ndim != 1 or not center.size or center.shape != source.shape or center.shape != destination.shape
            or not all(np.all(np.isfinite(x)) for x in (center, source, destination))
            or not math.isfinite(radius) or radius < 0):
        raise ValueError('invalid finite vectors or radius')
    delta = destination - source
    norm = float(np.linalg.norm(delta))
    margin = float(center @ delta)
    lower = margin - radius * norm
    if not all(map(math.isfinite, (norm, margin, lower))):
        raise ValueError('certificate arithmetic overflow')
    return {'center_progress': margin, 'edge_length': norm, 'radius': float(radius),
            'lower_progress': lower, 'certified_positive_progress': lower > 1e-12,
            'positive_radius_supremum': max(0., margin / norm) if norm else 0.}
