"""Verifica empíricamente cp_wf_coverage_80 >= 0.80 con --cp-walkforward.

PREREQUISITO: el worktree-agent-ace8b389 debe estar mergeado a main.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent


def check_flag_available() -> None:
    """Falla con mensaje claro si --cp-walkforward no está en main.py."""
    src = (PROJECT_DIR / "main.py").read_text()
    if "cp_walkforward" not in src and "cp-walkforward" not in src:
        print("ERROR: --cp-walkforward no encontrado en main.py")
        print("Mergea el worktree primero:")
        print("  git merge worktree-agent-ace8b389")
        sys.exit(1)


def print_gpu_info() -> None:
    """Muestra información de la GPU disponible."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU : {torch.cuda.get_device_name(0)}")
            print(f"SMs : {props.multi_processor_count}")
            print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
            if props.multi_processor_count < 40:
                print(
                    "NOTA: <40 SMs → torch.compile degrada a eager (comportamiento esperado)"
                )
        else:
            print("ADVERTENCIA: CUDA no disponible — ejecución en CPU")
    except ImportError:
        print("torch no importable")


def run_training() -> None:
    """Lanza el run canónico NVDA con --cp-walkforward."""
    cmd = [
        sys.executable,
        "main.py",
        "--csv",
        "data/NVDA_features.csv",
        "--targets",
        "Close",
        "--seq-len",
        "96",
        "--pred-len",
        "20",
        "--batch-size",
        "64",
        "--splits",
        "4",
        "--return-transform",
        "log_return",
        "--metric-space",
        "returns",
        "--gradient-clip-norm",
        "0.5",
        "--seed",
        "42",
        "--cp-walkforward",
        "--save-results",
        "--no-show",
    ]
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, check=False)
    if result.returncode != 0:
        print(f"ADVERTENCIA: main.py terminó con código {result.returncode}")


def report_results() -> None:
    """Extrae y evalúa cp_wf_coverage_80 del run_manifest más reciente."""
    results_dir = PROJECT_DIR / "results"
    manifests = sorted(
        results_dir.glob("run_manifest_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        print("No se encontró run_manifest en results/")
        return

    manifest: dict = json.loads(manifests[0].read_text())
    # Las métricas están en manifest["metrics"], no en raíz del manifiesto
    metrics: dict = manifest.get("metrics", manifest)
    cp_keys = {k: v for k, v in metrics.items() if "cp_wf" in str(k)}

    if not cp_keys:
        print("No se encontraron métricas cp_wf_ en el manifiesto")
        print("Claves raíz:", list(manifest.keys()))
        print("Claves metrics:", list(metrics.keys())[:20])
        return

    print("\n=== Métricas CP Enfoque 1 (walk-forward) ===")
    for k, v in cp_keys.items():
        print(f"  {k}: {v}")

    cov = cp_keys.get("cp_wf_coverage_80")
    folds_cal = cp_keys.get("cp_wf_folds_calibrated", "?")
    if cov is not None:
        objetivo = 0.80
        ok = float(cov) >= objetivo
        status = "OBJETIVO CUMPLIDO ✓" if ok else f"BAJO OBJETIVO (target={objetivo})"
        print(f"\n  cp_wf_coverage_80      = {float(cov):.4f}  [{status}]")
        print(f"  cp_wf_folds_calibrated = {folds_cal}")


def main() -> None:
    check_flag_available()
    print("=== GPU ===")
    print_gpu_info()
    print("\n=== Run NVDA --cp-walkforward ===")
    run_training()
    report_results()


if __name__ == "__main__":
    main()
