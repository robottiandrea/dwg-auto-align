"""I/O DWG con conversione temporanea via ODA File Converter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import shutil
import subprocess
import tempfile


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConversionArtifacts:
    work_dir: Path
    survey_dxf: Path
    project_dxf: Path


def _run_oda(oda_exe: str, input_dir: Path, output_dir: Path, out_ver: str = "ACAD2018", out_type: str = "DXF") -> None:
    if not Path(oda_exe).exists():
        raise FileNotFoundError(f"ODA File Converter non trovato: {oda_exe}")

    cmd = [
        oda_exe,
        str(input_dir),
        str(output_dir),
        out_ver,
        out_type,
        "0",
        "1",
    ]
    logger.info("Eseguo ODA: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ODA conversion failed: {proc.stderr.strip() or proc.stdout.strip()}")


def convert_dwg_pair_to_temp_dxf(oda_exe: str, survey_dwg: str | Path, project_dwg: str | Path) -> ConversionArtifacts:
    """Converte i due DWG in DXF temporanei."""
    s_path = Path(survey_dwg).resolve()
    p_path = Path(project_dwg).resolve()
    if not s_path.exists() or not p_path.exists():
        raise FileNotFoundError("Uno o più DWG input non esistono.")

    work = Path(tempfile.mkdtemp(prefix="dwg_auto_align_"))
    in_dir = work / "in"
    out_dir = work / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(s_path, in_dir / s_path.name)
    shutil.copy2(p_path, in_dir / p_path.name)

    _run_oda(oda_exe, in_dir, out_dir, out_type="DXF")

    survey_dxf = out_dir / f"{s_path.stem}.dxf"
    project_dxf = out_dir / f"{p_path.stem}.dxf"
    if not survey_dxf.exists() or not project_dxf.exists():
        raise RuntimeError("Conversione DWG->DXF incompleta.")

    return ConversionArtifacts(work_dir=work, survey_dxf=survey_dxf, project_dxf=project_dxf)


def convert_dxf_to_dwg(oda_exe: str, input_dxf: str | Path, output_dir: str | Path, output_stem: str) -> Path:
    """Converte un DXF in DWG nel path finale."""
    dxf_path = Path(input_dxf).resolve()
    if not dxf_path.exists():
        raise FileNotFoundError(f"DXF input non trovato: {dxf_path}")

    work = Path(tempfile.mkdtemp(prefix="dwg_auto_align_out_"))
    in_dir = work / "in"
    out_dir = work / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    staged = in_dir / f"{output_stem}.dxf"
    shutil.copy2(dxf_path, staged)

    _run_oda(oda_exe, in_dir, out_dir, out_type="DWG")

    produced = out_dir / f"{output_stem}.dwg"
    if not produced.exists():
        raise RuntimeError("Conversione DXF->DWG fallita.")

    out_folder = Path(output_dir)
    out_folder.mkdir(parents=True, exist_ok=True)
    final = out_folder / produced.name
    shutil.copy2(produced, final)
    return final
