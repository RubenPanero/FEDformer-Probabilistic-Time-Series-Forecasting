"""Descarga y verifica especialistas publicados como GitHub Release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = PROJECT_DIR / "checkpoints" / "artifacts.manifest.json"
DEFAULT_REGISTRY_PATH = PROJECT_DIR / "checkpoints" / "model_registry.json"


def parse_args() -> argparse.Namespace:
    """Define la CLI del descargador de especialistas."""
    parser = argparse.ArgumentParser(
        description=(
            "Descarga checkpoints y preprocessing artifacts desde GitHub Releases, "
            "valida sha256 y deja los archivos instalados bajo checkpoints/."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Ruta al manifest versionado con URLs y checksums.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Ruta al model_registry.json versionado.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Subset de tickers a descargar. Por defecto descarga todos.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Descarga todos los especialistas publicados en el manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe archivos locales aunque ya existan.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Carga un JSON desde disco."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    """Calcula SHA256 de un archivo."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(url: str, destination: Path) -> Path:
    """Descarga un asset a disco."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    return destination


def verify_download(path: Path, expected_sha256: str, expected_size: int) -> None:
    """Verifica tamaño y hash del archivo descargado."""
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Tamaño inesperado para {path.name}: {actual_size} != {expected_size}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise ValueError(
            f"SHA256 inesperado para {path.name}: {actual_sha} != {expected_sha256}"
        )


def resolve_repo_path(path_str: str) -> Path:
    """Resuelve una ruta relativa al root del repo."""
    return (PROJECT_DIR / Path(path_str)).resolve()


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extrae un ZIP asegurando que no escape del directorio destino."""
    destination.mkdir(parents=True, exist_ok=True)
    dest_root = destination.resolve()
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(dest_root)):
                raise ValueError(f"ZIP contiene una ruta insegura: {member.filename}")
        archive.extractall(destination)


def validate_installed_files(base_dir: Path, required_files: list[str]) -> None:
    """Verifica que existan los archivos esperados tras la extracción."""
    for relative in required_files:
        target = base_dir / relative
        if not target.exists():
            raise FileNotFoundError(
                f"Falta archivo requerido tras extracción: {target}"
            )


def validate_registry_consistency(
    ticker: str,
    registry: dict,
    checkpoint_path: Path,
    preprocessing_dir: Path,
) -> None:
    """Comprueba que el registry siga apuntando a los artefactos instalados."""
    entry = registry.get("specialists", {}).get(ticker)
    if entry is None:
        raise KeyError(f"Ticker '{ticker}' no encontrado en model_registry.json")
    expected_checkpoint = resolve_repo_path(entry["checkpoint"])
    expected_preprocessing = resolve_repo_path(
        entry.get("data", {}).get("preprocessing_artifacts", "")
    )
    if checkpoint_path.resolve() != expected_checkpoint:
        raise ValueError(
            f"Checkpoint instalado en ruta distinta a la registrada: "
            f"{checkpoint_path} != {expected_checkpoint}"
        )
    if preprocessing_dir.resolve() != expected_preprocessing:
        raise ValueError(
            f"Preprocessing instalado en ruta distinta a la registrada: "
            f"{preprocessing_dir} != {expected_preprocessing}"
        )


def fetch_ticker(
    ticker: str,
    artifact_info: dict,
    registry: dict,
    force: bool = False,
) -> None:
    """Descarga e instala los assets de un ticker."""
    checkpoint_meta = artifact_info["checkpoint"]
    preprocessing_meta = artifact_info["preprocessing"]

    checkpoint_path = resolve_repo_path(checkpoint_meta["install_path"])
    preprocessing_dir = resolve_repo_path(preprocessing_meta["install_dir"])

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        checkpoint_download = tmp_dir / checkpoint_meta["asset_name"]
        preprocessing_download = tmp_dir / preprocessing_meta["asset_name"]

        print(f"DOWNLOAD {ticker}: {checkpoint_meta['url']}")
        download_asset(checkpoint_meta["url"], checkpoint_download)
        verify_download(
            checkpoint_download,
            checkpoint_meta["sha256"],
            checkpoint_meta["size_bytes"],
        )

        print(f"DOWNLOAD {ticker}: {preprocessing_meta['url']}")
        download_asset(preprocessing_meta["url"], preprocessing_download)
        verify_download(
            preprocessing_download,
            preprocessing_meta["sha256"],
            preprocessing_meta["size_bytes"],
        )

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if force or not checkpoint_path.exists():
            shutil.copy2(checkpoint_download, checkpoint_path)

        if force and preprocessing_dir.exists():
            shutil.rmtree(preprocessing_dir)
        safe_extract_zip(preprocessing_download, preprocessing_dir)
        validate_installed_files(
            preprocessing_dir,
            preprocessing_meta.get("required_files", []),
        )
        validate_registry_consistency(
            ticker, registry, checkpoint_path, preprocessing_dir
        )

    print(f"OK {ticker}: artifacts installed and verified")


def select_tickers(args: argparse.Namespace, manifest: dict) -> list[str]:
    """Resuelve la lista final de tickers a descargar."""
    available = list(manifest.get("artifacts", {}).keys())
    if args.all:
        return available
    if args.tickers:
        return args.tickers
    return available


def main() -> int:
    """Punto de entrada del descargador."""
    args = parse_args()
    manifest = load_json(args.manifest)
    registry = load_json(args.registry)
    tickers = select_tickers(args, manifest)

    for ticker in tickers:
        if ticker not in manifest.get("artifacts", {}):
            raise KeyError(f"Ticker '{ticker}' no encontrado en el manifest")
        fetch_ticker(
            ticker=ticker,
            artifact_info=manifest["artifacts"][ticker],
            registry=registry,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
