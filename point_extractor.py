"""Estrazione POINT 2D da DXF."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import ezdxf
import numpy as np


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PointSet:
    points_xy: np.ndarray
    count: int


def extract_points_from_dxf(dxf_path: str | Path) -> PointSet:
    """Estrae coordinate XY da entità POINT del modelspace."""
    path = Path(dxf_path)
    if not path.exists():
        raise FileNotFoundError(f"DXF non trovato: {path}")

    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    pts = []
    for p in msp.query("POINT"):
        loc = p.dxf.location
        pts.append((float(loc.x), float(loc.y)))

    arr = np.asarray(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)
    logger.info("Estratti %d POINT da %s", len(arr), path)
    return PointSet(points_xy=arr, count=len(arr))
