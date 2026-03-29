"""Generazione report JSON/TXT."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from matcher import MatchResult


@dataclass(slots=True)
class AlignmentReport:
    scale: float
    rotation_deg: float
    translation_x: float
    translation_y: float
    survey_points: int
    project_points: int
    valid_matches: int
    rms_error: float
    tolerance: float
    confidence: float
    warning: str | None = None
    transformed_entities: int = 0
    unsupported_entities: int = 0


def build_report(
    result: MatchResult,
    survey_points: int,
    project_points: int,
    transformed_entities: int,
    unsupported_entities: int,
    warning: str | None = None,
) -> AlignmentReport:
    tr = result.transform
    return AlignmentReport(
        scale=tr.scale,
        rotation_deg=tr.theta_deg,
        translation_x=tr.tx,
        translation_y=tr.ty,
        survey_points=survey_points,
        project_points=project_points,
        valid_matches=len(result.inlier_src_idx),
        rms_error=result.rms,
        tolerance=result.tolerance,
        confidence=result.confidence,
        warning=warning,
        transformed_entities=transformed_entities,
        unsupported_entities=unsupported_entities,
    )


def write_report(report: AlignmentReport, output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{stem}_report.json"
    txt_path = out / f"{stem}_report.txt"

    payload = asdict(report)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "DWG Auto Align - Report",
        "=" * 40,
        f"Scale: {report.scale:.8f}",
        f"Rotation (deg): {report.rotation_deg:.6f}",
        f"Translation X: {report.translation_x:.6f}",
        f"Translation Y: {report.translation_y:.6f}",
        f"POINT rilievo: {report.survey_points}",
        f"POINT progetto: {report.project_points}",
        f"Match validi: {report.valid_matches}",
        f"RMS error: {report.rms_error:.6f}",
        f"Tolleranza: {report.tolerance}",
        f"Confidenza: {report.confidence:.3f}",
        f"Entità trasformate: {report.transformed_entities}",
        f"Entità non supportate: {report.unsupported_entities}",
    ]
    if report.warning:
        lines.append(f"WARNING: {report.warning}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, txt_path
