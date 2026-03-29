"""Entry point CLI/GUI per allineamento automatico DWG."""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from apply_transform import transform_project_dxf
from cad_io import convert_dwg_pair_to_temp_dxf, convert_dxf_to_dwg
from config import AppConfig, load_config
from matcher import find_similarity_transform, validate_result
from point_extractor import extract_points_from_dxf
from report import build_report, write_report
from transform import SimilarityTransform2D


def setup_logging(output_dir: Path | None = None) -> Path:
    log_dir = output_dir or Path(tempfile.gettempdir())
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "dwg_auto_align.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def run_analysis(
    survey_dwg: str,
    project_dwg: str,
    tolerance: float,
    cfg: AppConfig,
) -> dict:
    """Esegue solo analisi/matching senza scrivere output DWG."""
    artifacts = convert_dwg_pair_to_temp_dxf(cfg.oda_converter_path, survey_dwg, project_dwg)
    survey_pts = extract_points_from_dxf(artifacts.survey_dxf)
    project_pts = extract_points_from_dxf(artifacts.project_dxf)

    result = find_similarity_transform(
        src_points=project_pts.points_xy,
        dst_points=survey_pts.points_xy,
        tolerance=tolerance,
        max_iterations=cfg.ransac_max_iterations,
        allowed_project_scale_factors=cfg.allowed_project_scale_factors,
        distance_error_epsilon=cfg.distance_error_epsilon,
    )

    valid, msg = validate_result(result, cfg.min_confidence, cfg.distance_error_epsilon)
    return {
        "warning": None if valid else msg,
        "valid": valid,
        "transform": {
            "scale": result.transform.scale,
            "rotation_deg": result.transform.theta_deg,
            "tx": result.transform.tx,
            "ty": result.transform.ty,
        },
        "matches": len(result.inlier_src_idx),
        "survey_points": survey_pts.count,
        "project_points": project_pts.count,
        "rms": result.rms,
        "distance_check_max_abs_error": result.distance_check_max_abs_error,
    }


def run_alignment(
    survey_dwg: str,
    project_dwg: str,
    output_dir: str,
    tolerance: float,
    cfg: AppConfig,
    force_save: bool = False,
) -> dict:
    logger = logging.getLogger(__name__)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = convert_dwg_pair_to_temp_dxf(cfg.oda_converter_path, survey_dwg, project_dwg)

    survey_pts = extract_points_from_dxf(artifacts.survey_dxf)
    project_pts = extract_points_from_dxf(artifacts.project_dxf)

    result = find_similarity_transform(
        src_points=project_pts.points_xy,
        dst_points=survey_pts.points_xy,
        tolerance=tolerance,
        max_iterations=cfg.ransac_max_iterations,
        allowed_project_scale_factors=cfg.allowed_project_scale_factors,
        distance_error_epsilon=cfg.distance_error_epsilon,
    )

    valid, msg = validate_result(result, cfg.min_confidence, cfg.distance_error_epsilon)
    if not valid and not force_save and not cfg.low_confidence_warn_only:
        raise RuntimeError(msg)

    aligned_dxf = artifacts.work_dir / f"{Path(project_dwg).stem}_aligned.dxf"
    transformed_entities, unsupported_entities = transform_project_dxf(
        artifacts.project_dxf,
        aligned_dxf,
        result.transform,
    )

    output_stem = f"{Path(project_dwg).stem}_aligned"
    out_dwg = convert_dxf_to_dwg(cfg.oda_converter_path, aligned_dxf, out_dir, output_stem)

    warning = None if valid else msg
    rep = build_report(
        result,
        survey_points=survey_pts.count,
        project_points=project_pts.count,
        transformed_entities=transformed_entities,
        unsupported_entities=unsupported_entities,
        warning=warning,
    )
    report_json, report_txt = write_report(rep, out_dir, output_stem)

    logger.info("Allineamento completato: %s", out_dwg)
    return {
        "output_dwg": str(out_dwg),
        "report_json": str(report_json),
        "report_txt": str(report_txt),
        "warning": warning,
        "transform": {
            "scale": result.transform.scale,
            "rotation_deg": result.transform.theta_deg,
            "tx": result.transform.tx,
            "ty": result.transform.ty,
        },
    }


def run_demo(noise_sigma: float = 0.03) -> None:
    """Demo sintetica per validare stima trasformazione."""
    rng = np.random.default_rng(7)
    survey = rng.uniform(-50, 50, size=(300, 2))

    gt = SimilarityTransform2D(scale=0.2, theta_rad=np.deg2rad(23.5), tx=350.0, ty=-180.0)
    idx = rng.choice(len(survey), size=90, replace=False)
    project_clean = survey[idx]

    c = np.cos(-gt.theta_rad)
    s = np.sin(-gt.theta_rad)
    r_inv = np.array([[c, -s], [s, c]])
    project_local = ((project_clean - np.array([gt.tx, gt.ty])) @ r_inv.T) / gt.scale
    project_local += rng.normal(0.0, noise_sigma, size=project_local.shape)

    result = find_similarity_transform(
        src_points=project_local,
        dst_points=survey,
        tolerance=0.35,
        max_iterations=3000,
        allowed_project_scale_factors=[0.5, 2.0, 5.0, 10.0],
        distance_error_epsilon=1e-5,
    )

    print("=== DEMO ===")
    print(f"GT scale={gt.scale:.6f} | est={result.transform.scale:.6f}")
    print(f"GT rot={np.rad2deg(gt.theta_rad):.6f} | est={result.transform.theta_deg:.6f}")
    print(f"GT tx={gt.tx:.6f} | est={result.transform.tx:.6f}")
    print(f"GT ty={gt.ty:.6f} | est={result.transform.ty:.6f}")
    print(f"Inliers={len(result.inlier_src_idx)} RMS={result.rms:.6f} confidence={result.confidence:.3f}")


def launch_gui(cfg: AppConfig) -> None:
    root = tk.Tk()
    root.title("DWG Auto Align MVP")
    root.geometry("760x340")

    survey_var = tk.StringVar()
    project_var = tk.StringVar()
    out_var = tk.StringVar(value=str(Path.cwd() / "output"))
    tol_var = tk.StringVar(value=str(cfg.matching_tolerance))
    status_var = tk.StringVar(value="Pronto")

    task_queue: queue.Queue[tuple[str, object]] = queue.Queue()
    worker_running = {"value": False}
    logger = logging.getLogger(__name__)

    def set_controls_enabled(enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        analyze_btn.config(state=state)
        align_btn.config(state=state)

    def pick_file(var: tk.StringVar):
        path = filedialog.askopenfilename(filetypes=[("DWG files", "*.dwg")])
        if path:
            var.set(path)

    def pick_dir(var: tk.StringVar):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def validate_inputs(require_out: bool = False) -> tuple[str, str, str, float]:
        survey = survey_var.get().strip()
        project = project_var.get().strip()
        out_dir = out_var.get().strip()
        if not survey or not project:
            raise ValueError("Seleziona DWG rilievo e DWG progetto.")
        if require_out and not out_dir:
            raise ValueError("Seleziona cartella output.")
        tolerance = float(tol_var.get())
        return survey, project, out_dir, tolerance

    def start_worker(task_name: str, fn) -> None:
        if worker_running["value"]:
            messagebox.showwarning("Operazione in corso", "Attendere la fine dell'elaborazione corrente.")
            return

        worker_running["value"] = True
        set_controls_enabled(False)
        status_var.set(f"{task_name} in corso...")

        def target():
            try:
                result = fn()
                task_queue.put(("ok", (task_name, result)))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Errore durante %s", task_name)
                task_queue.put(("err", (task_name, str(exc))))

        threading.Thread(target=target, daemon=True).start()

    def poll_queue():
        try:
            while True:
                status, payload = task_queue.get_nowait()
                worker_running["value"] = False
                set_controls_enabled(True)

                if status == "ok":
                    task_name, res = payload
                    status_var.set(f"{task_name} completata")
                    if task_name == "Analisi":
                        msg = (
                            "Analisi completata.\n"
                            f"Match validi: {res['matches']}\n"
                            f"RMS: {res['rms']:.6f}\n"
                            f"Max errore distanze: {res['distance_check_max_abs_error']:.9f}"
                        )
                        if res.get("warning"):
                            msg += f"\n\n{res['warning']}"
                        messagebox.showinfo("Analisi", msg)
                    else:
                        msg = f"Salvato:\n{res['output_dwg']}\n\nReport:\n{res['report_json']}\n{res['report_txt']}"
                        if res.get("warning"):
                            msg += f"\n\n{res['warning']}"
                        messagebox.showinfo("Completato", msg)
                else:
                    task_name, err = payload
                    status_var.set(f"{task_name} fallita")
                    messagebox.showerror("Errore", err)
        except queue.Empty:
            pass
        root.after(200, poll_queue)

    def analyze_only():
        try:
            survey, project, _, tolerance = validate_inputs(require_out=False)
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("Errore input", str(ex))
            return

        start_worker("Analisi", lambda: run_analysis(survey, project, tolerance, cfg))

    def align_and_save():
        try:
            survey, project, out_dir, tolerance = validate_inputs(require_out=True)
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("Errore input", str(ex))
            return

        start_worker("Allineamento", lambda: run_alignment(survey, project, out_dir, tolerance, cfg, force_save=True))

    labels = [
        ("DWG rilievo", survey_var),
        ("DWG tavola progetto", project_var),
        ("Cartella output", out_var),
    ]

    for i, (txt, var) in enumerate(labels):
        tk.Label(root, text=txt).grid(row=i, column=0, sticky="w", padx=8, pady=8)
        tk.Entry(root, textvariable=var, width=72).grid(row=i, column=1, padx=8)
        if i < 2:
            tk.Button(root, text="Sfoglia", command=lambda v=var: pick_file(v)).grid(row=i, column=2, padx=8)
        else:
            tk.Button(root, text="Sfoglia", command=lambda v=var: pick_dir(v)).grid(row=i, column=2, padx=8)

    tk.Label(root, text="Tolleranza").grid(row=3, column=0, sticky="w", padx=8, pady=8)
    tk.Entry(root, textvariable=tol_var, width=16).grid(row=3, column=1, sticky="w", padx=8)

    analyze_btn = tk.Button(root, text="Analizza", command=analyze_only, bg="#f5f5cc")
    analyze_btn.grid(row=4, column=1, sticky="w", padx=8, pady=16)

    align_btn = tk.Button(root, text="Allinea e salva", command=align_and_save, bg="#ccf5d8")
    align_btn.grid(row=4, column=1, sticky="e", padx=8, pady=16)

    tk.Label(root, textvariable=status_var, anchor="w", fg="#333").grid(row=5, column=0, columnspan=3, sticky="we", padx=8, pady=8)

    root.after(200, poll_queue)
    root.mainloop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MVP allineamento automatico DWG->DWG")
    p.add_argument("--survey", help="DWG rilievo georiferito")
    p.add_argument("--project", help="DWG tavola progetto da allineare")
    p.add_argument("--out", help="Cartella output")
    p.add_argument("--tolerance", type=float, help="Tolleranza matching")
    p.add_argument("--force-save", action="store_true", help="Salva anche con bassa confidenza")
    p.add_argument("--demo", action="store_true", help="Esegue demo sintetica")
    p.add_argument("--nogui", action="store_true", help="Forza modalità CLI")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    log_path = setup_logging(Path(args.out) if args.out else None)
    logging.getLogger(__name__).info("Log file: %s", log_path)

    if args.demo:
        run_demo()
        return

    if args.nogui or (args.survey and args.project and args.out):
        if not (args.survey and args.project and args.out):
            raise SystemExit("In CLI servono --survey --project --out")
        tolerance = args.tolerance if args.tolerance is not None else cfg.matching_tolerance
        res = run_alignment(args.survey, args.project, args.out, tolerance, cfg, force_save=args.force_save)
        print("Completato:")
        print(res)
    else:
        launch_gui(cfg)


if __name__ == "__main__":
    main()
