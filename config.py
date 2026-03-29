"""Configurazione centrale dell'MVP."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os


@dataclass(slots=True)
class AppConfig:
    """Parametri globali dell'applicazione."""

    oda_converter_path: str = r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe"
    matching_tolerance: float = 0.5
    ransac_max_iterations: int = 3000
    min_confidence: float = 0.55
    low_confidence_warn_only: bool = True
    # Scale del progetto rispetto al rilievo (vincolo dominio)
    allowed_project_scale_factors: list[float] = field(default_factory=lambda: [0.5, 2.0, 5.0, 10.0])
    # Controllo invarianti distanza sui match finali (quasi 0 errore)
    distance_error_epsilon: float = 1e-6


DEFAULT_CONFIG_PATH = Path("config.local.json")


def load_config(path: str | Path | None = None) -> AppConfig:
    """Carica config da JSON locale se presente, altrimenti default."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg = AppConfig()

    env_oda = os.getenv("ODA_FILE_CONVERTER")
    if env_oda:
        cfg.oda_converter_path = env_oda

    if not cfg_path.exists():
        return cfg

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
