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
    distance_check_max_abs_error: float


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


def _pairwise_distance_error(src_corr: np.ndarray, dst_corr: np.ndarray, scale: float) -> float:
    """Errore assoluto max sui rapporti di distanza tra coppie di inlier."""
    n = len(src_corr)
    if n < 2:
        return float("inf")

    max_abs = 0.0
    for i in range(n - 1):
        ds = np.linalg.norm(src_corr[i + 1 :] - src_corr[i], axis=1)
        dd = np.linalg.norm(dst_corr[i + 1 :] - dst_corr[i], axis=1)
        err = np.abs(dd - scale * ds)
        if len(err):
            local_max = float(np.max(err))
            if local_max > max_abs:
                max_abs = local_max
    return max_abs


def _compute_confidence(num_inliers: int, n_src: int, n_dst: int, rms: float, tol: float, dist_err: float, dist_eps: float) -> float:
    if n_src == 0 or n_dst == 0:
        return 0.0
    coverage = num_inliers / n_src
    purity = min(1.0, num_inliers / max(1, n_dst * 0.25))
    rms_score = max(0.0, 1.0 - (rms / max(tol, 1e-9)))
    dist_score = max(0.0, 1.0 - (dist_err / max(dist_eps, 1e-12)))
    return float(max(0.0, min(1.0, 0.45 * coverage + 0.20 * purity + 0.15 * rms_score + 0.20 * dist_score)))


def _build_pairs(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ii, jj, dd = [], [], []
    for i in range(len(pts) - 1):
        delta = pts[i + 1 :] - pts[i]
        dist = np.linalg.norm(delta, axis=1)
        for k, d in enumerate(dist):
            ii.append(i)
            jj.append(i + 1 + k)
            dd.append(float(d))
    return np.asarray(ii, dtype=int), np.asarray(jj, dtype=int), np.asarray(dd, dtype=float)


def _allowed_map_scales(allowed_project_scale_factors: list[float]) -> set[float]:
    # progetto -> rilievo
    return {1.0 / s for s in allowed_project_scale_factors if s > 0}


def find_similarity_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    tolerance: float,
    max_iterations: int,
    allowed_project_scale_factors: list[float],
    distance_error_epsilon: float,
    random_seed: int = 42,
) -> MatchResult:
    """Trova similarità robusta quando src è sottoinsieme parziale di dst."""
    if len(src_points) < 2 or len(dst_points) < 2:
        raise ValueError("Servono almeno 2 POINT in ciascun DWG.")

    rng = np.random.default_rng(random_seed)
    allowed_scales = _allowed_map_scales(allowed_project_scale_factors)

    si, sj, sd = _build_pairs(src_points)
    di, dj, dd = _build_pairs(dst_points)
    if len(sd) == 0 or len(dd) == 0:
        raise RuntimeError("Impossibile costruire coppie di punti valide.")

    ratio_tol = max(distance_error_epsilon, tolerance * 0.02)
    candidate_pairs: list[tuple[int, int]] = []

    for a_idx, dsrc in enumerate(sd):
        if dsrc <= 1e-12:
            continue
        for b_idx, ddst in enumerate(dd):
            ratio = ddst / dsrc
            if any(abs(ratio - s) <= ratio_tol for s in allowed_scales):
                candidate_pairs.append((a_idx, b_idx))

    if not candidate_pairs:
        raise RuntimeError("Nessuna coppia compatibile con le scale vincolate [0.5, 2, 5, 10].")

    best_inliers = 0
    best_result: MatchResult | None = None

    effective_iters = min(max_iterations, len(candidate_pairs))
    sampled_idx = rng.choice(len(candidate_pairs), size=effective_iters, replace=False)

    for idx in sampled_idx:
        a_idx, b_idx = candidate_pairs[idx]
        i1, i2 = si[a_idx], sj[a_idx]
        j1, j2 = di[b_idx], dj[b_idx]

        tr = estimate_from_two_pairs(src_points[i1], src_points[i2], dst_points[j1], dst_points[j2])
        if tr is None:
            continue

        if min(abs(tr.scale - s) for s in allowed_scales) > ratio_tol:
            continue

        transformed = tr.apply(src_points)
        ms, md, _ = _best_unique_matches(transformed, dst_points, tolerance)

        if len(ms) <= best_inliers or len(ms) < 2:
            continue

        refined = estimate_similarity_least_squares(src_points[ms], dst_points[md])
        if min(abs(refined.scale - s) for s in allowed_scales) > ratio_tol:
            continue

        ref_transformed = refined.apply(src_points)
        r_ms, r_md, _ = _best_unique_matches(ref_transformed, dst_points, tolerance)
        if len(r_ms) < 2:
            continue

        final_refined = estimate_similarity_least_squares(src_points[r_ms], dst_points[r_md])
        if min(abs(final_refined.scale - s) for s in allowed_scales) > ratio_tol:
            continue

        final_pts = final_refined.apply(src_points[r_ms])
        final_rms = rms_error(final_pts, dst_points[r_md])
        dist_err = _pairwise_distance_error(src_points[r_ms], dst_points[r_md], final_refined.scale)
        conf = _compute_confidence(
            len(r_ms),
            len(src_points),
            len(dst_points),
            final_rms,
            tolerance,
            dist_err,
            distance_error_epsilon,
        )

        best_inliers = len(r_ms)
        best_result = MatchResult(
            transform=final_refined,
            inlier_src_idx=r_ms,
            inlier_dst_idx=r_md,
            rms=final_rms,
            confidence=conf,
            tolerance=tolerance,
            distance_check_max_abs_error=dist_err,
        )

    if best_result is None:
        raise RuntimeError("Impossibile stimare una trasformazione robusta rispettando i vincoli di scala.")

    logger.info(
        "Best model: inliers=%s/%s, rms=%.6f, distance_err=%.9f, confidence=%.3f",
        len(best_result.inlier_src_idx),
        len(src_points),
        best_result.rms,
        best_result.distance_check_max_abs_error,
        best_result.confidence,
    )
    return best_result


def validate_result(result: MatchResult, min_confidence: float, distance_error_epsilon: float) -> tuple[bool, str]:
    """Valida risultato secondo soglia confidenza + controllo distanze."""
    if result.distance_check_max_abs_error > distance_error_epsilon:
        return False, (
            "Controllo distanze fallito: errore assoluto massimo "
            f"{result.distance_check_max_abs_error:.9f} > {distance_error_epsilon:.9f}"
        )

    if result.confidence >= min_confidence:
        return True, "OK"
    return False, "Allineamento automatico incerto: verificare il risultato"
