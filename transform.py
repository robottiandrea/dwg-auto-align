"""Stima e applicazione trasformazioni di similarità 2D."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class SimilarityTransform2D:
    """x' = s * R(theta) * x + t"""

    scale: float
    theta_rad: float
    tx: float
    ty: float

    def apply(self, pts: np.ndarray) -> np.ndarray:
        c = math.cos(self.theta_rad)
        s = math.sin(self.theta_rad)
        r = np.array([[c, -s], [s, c]], dtype=float)
        return (self.scale * (pts @ r.T)) + np.array([self.tx, self.ty])

    @property
    def theta_deg(self) -> float:
        return math.degrees(self.theta_rad)


def estimate_from_two_pairs(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> SimilarityTransform2D | None:
    """Stima trasformazione da due coppie corrispondenti."""
    va = a2 - a1
    vb = b2 - b1
    da = float(np.linalg.norm(va))
    db = float(np.linalg.norm(vb))
    if da < 1e-9 or db < 1e-9:
        return None

    scale = db / da
    ang_a = math.atan2(va[1], va[0])
    ang_b = math.atan2(vb[1], vb[0])
    theta = ang_b - ang_a

    c = math.cos(theta)
    s = math.sin(theta)
    ra1 = scale * np.array([c * a1[0] - s * a1[1], s * a1[0] + c * a1[1]])
    t = b1 - ra1
    return SimilarityTransform2D(scale=scale, theta_rad=theta, tx=float(t[0]), ty=float(t[1]))


def estimate_similarity_least_squares(src: np.ndarray, dst: np.ndarray) -> SimilarityTransform2D:
    """Stima LS (Umeyama 2D) di similarità con corrispondenze note."""
    if len(src) != len(dst) or len(src) < 2:
        raise ValueError("Servono almeno 2 corrispondenze src/dst.")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = (dst_c.T @ src_c) / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[1, 1] = -1

    r = u @ s_mat @ vt
    var_src = np.mean(np.sum(src_c**2, axis=1))
    scale = float(np.trace(np.diag(d) @ s_mat) / var_src)

    t = dst_mean - scale * (r @ src_mean)
    theta = math.atan2(r[1, 0], r[0, 0])
    return SimilarityTransform2D(scale=scale, theta_rad=theta, tx=float(t[0]), ty=float(t[1]))


def rms_error(a: np.ndarray, b: np.ndarray) -> float:
    """Errore RMS tra due insiemi di punti corrispondenti."""
    if len(a) == 0:
        return float("inf")
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def as_tuple(tr: SimilarityTransform2D) -> Tuple[float, float, float, float]:
    return tr.scale, tr.theta_deg, tr.tx, tr.ty
