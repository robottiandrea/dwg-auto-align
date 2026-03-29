"""Matching robusto POINT con RANSAC per similarità 2D."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Tuple

import numpy as np

from transform import (
    SimilarityTransform2D,
    estimate_from_two_pairs,
    estimate_similarity_least_squares,
    rms_error,
)


@dataclass(slots=True)
class MatchResult:
    transform: SimilarityTransform2D
    inlier_src_idx: np.ndarray
    inlier_dst_idx: np.ndarray
    rms: float
    confidence: float
    tolerance: float


logger = logging.getLogger(__name__)


def _best_unique_matches(transformed_src: np.ndarray, dst: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trova nearest-neighbor entro tolleranza con unicità lato dst."""
    if len(transformed_src) == 0 or len(dst) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    diff = transformed_src[:, None, :] - dst[None, :, :]
    d2 = np.sum(diff * diff, axis=2)

    nearest_dst = np.argmin(d2, axis=1)
    nearest_d2 = d2[np.arange(len(transformed_src)), nearest_dst]

    candidates = [
        (i, int(j), float(nearest_d2[i]))
        for i, j in enumerate(nearest_dst)
        if nearest_d2[i] <= tol * tol
    ]
    candidates.sort(key=lambda x: x[2])

    used_src: set[int] = set()
    used_dst: set[int] = set()
    m_src: List[int] = []
    m_dst: List[int] = []
    m_dist: List[float] = []

    for i, j, d in candidates:
        if i in used_src or j in used_dst:
            continue
        used_src.add(i)
        used_dst.add(j)
        m_src.append(i)
        m_dst.append(j)
        m_dist.append(np.sqrt(d))

    return np.asarray(m_src, dtype=int), np.asarray(m_dst, dtype=int), np.asarray(m_dist, dtype=float)


def _compute_confidence(num_inliers: int, n_src: int, n_dst: int, rms: float, tol: float) -> float:
    if n_src == 0 or n_dst == 0:
        return 0.0
    coverage = num_inliers / n_src
    purity = min(1.0, num_inliers / max(1, n_dst * 0.25))
    rms_score = max(0.0, 1.0 - (rms / max(tol, 1e-9)))
    return float(max(0.0, min(1.0, 0.55 * coverage + 0.25 * purity + 0.20 * rms_score)))


def find_similarity_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    tolerance: float,
    max_iterations: int,
    random_seed: int = 42,
) -> MatchResult:
    """Trova similarità robusta quando src è sottoinsieme parziale di dst."""
    if len(src_points) < 2 or len(dst_points) < 2:
        raise ValueError("Servono almeno 2 POINT in ciascun DWG.")

    rng = np.random.default_rng(random_seed)

    best_inliers = 0
    best_result: MatchResult | None = None

    for _ in range(max_iterations):
        i1, i2 = rng.choice(len(src_points), size=2, replace=False)
        j1, j2 = rng.choice(len(dst_points), size=2, replace=False)

        tr = estimate_from_two_pairs(src_points[i1], src_points[i2], dst_points[j1], dst_points[j2])
        if tr is None:
            continue

        transformed = tr.apply(src_points)
        ms, md, _ = _best_unique_matches(transformed, dst_points, tolerance)
        n_in = len(ms)

        if n_in > best_inliers:
            if n_in < 2:
                continue
            refined = estimate_similarity_least_squares(src_points[ms], dst_points[md])
            ref_transformed = refined.apply(src_points)
            r_ms, r_md, _ = _best_unique_matches(ref_transformed, dst_points, tolerance)
            if len(r_ms) < 2:
                continue
            final_refined = estimate_similarity_least_squares(src_points[r_ms], dst_points[r_md])
            final_pts = final_refined.apply(src_points[r_ms])
            final_rms = rms_error(final_pts, dst_points[r_md])
            conf = _compute_confidence(len(r_ms), len(src_points), len(dst_points), final_rms, tolerance)

            best_inliers = len(r_ms)
            best_result = MatchResult(
                transform=final_refined,
                inlier_src_idx=r_ms,
                inlier_dst_idx=r_md,
                rms=final_rms,
                confidence=conf,
                tolerance=tolerance,
            )

    if best_result is None:
        raise RuntimeError("Impossibile stimare una trasformazione robusta.")

    logger.info(
        "Best model: inliers=%s/%s, rms=%.4f, confidence=%.3f",
        len(best_result.inlier_src_idx),
        len(src_points),
        best_result.rms,
        best_result.confidence,
    )
    return best_result


def validate_result(result: MatchResult, min_confidence: float) -> tuple[bool, str]:
    """Valida risultato secondo soglia confidenza."""
    if result.confidence >= min_confidence:
        return True, "OK"
    return False, "Allineamento automatico incerto: verificare il risultato"
