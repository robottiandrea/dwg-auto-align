"""Applicazione della trasformazione a tutte le entità del DWG progetto."""
from __future__ import annotations

from pathlib import Path
import logging
import math

import ezdxf
from ezdxf.math import Matrix44
from ezdxf import transform as ezdxf_transform

from transform import SimilarityTransform2D


logger = logging.getLogger(__name__)


def _matrix_from_similarity(tr: SimilarityTransform2D) -> Matrix44:
    c = math.cos(tr.theta_rad) * tr.scale
    s = math.sin(tr.theta_rad) * tr.scale
    # Matrice affine 4x4 in convenzione ezdxf
    return Matrix44([
        c, s, 0.0, 0.0,
        -s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        tr.tx, tr.ty, 0.0, 1.0,
    ])


def transform_project_dxf(input_dxf: str | Path, output_dxf: str | Path, tr: SimilarityTransform2D) -> tuple[int, int]:
    """Trasforma tutte le entità modelspace e salva DXF."""
    doc = ezdxf.readfile(str(input_dxf))
    msp = doc.modelspace()

    matrix = _matrix_from_similarity(tr)
    log = ezdxf_transform.inplace(list(msp), matrix)

    unsupported = 0
    transformed = 0
    for entry in log:
        if entry.error:
            unsupported += 1
            logger.warning("Entità non trasformata: %s - %s", entry.entity.dxftype(), entry.error)
        else:
            transformed += 1

    doc.saveas(str(output_dxf))
    logger.info("DXF trasformato salvato in %s (ok=%d, unsupported=%d)", output_dxf, transformed, unsupported)
    return transformed, unsupported
