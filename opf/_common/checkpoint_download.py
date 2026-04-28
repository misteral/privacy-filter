from __future__ import annotations

from pathlib import Path
import shutil
import sys
from typing import Final

from .constants import DEFAULT_MODEL_ENV_VAR, DEFAULT_MODEL_PATH


DEFAULT_HF_MODEL_REPO: Final[str] = "openai/privacy-filter"


def _checkpoint_override_message() -> str:
    return (
        "pass --checkpoint /path/to/checkpoint, or set "
        f"{DEFAULT_MODEL_ENV_VAR}=/path/to/checkpoint"
    )


def _validate_default_checkpoint(target: Path) -> None:
    if not target.is_dir():
        raise RuntimeError(
            f"Default OPF checkpoint path exists but is not a directory: {target}. "
            f"Fix or remove it, {_checkpoint_override_message()}."
        )
    config_path = target / "config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"Default OPF checkpoint at {target} is incomplete: missing config.json. "
            f"Fix or remove it, {_checkpoint_override_message()}."
        )
    if not any(path.is_file() for path in target.glob("*.safetensors")):
        raise RuntimeError(
            f"Default OPF checkpoint at {target} is incomplete: "
            "missing at least one .safetensors file. "
            f"Fix or remove it, {_checkpoint_override_message()}."
        )


def _is_valid_default_checkpoint(target: Path) -> bool:
    try:
        _validate_default_checkpoint(target)
    except RuntimeError:
        return False
    return True


def _reset_terminal_after_download() -> None:
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[0m\033[?25h\033[K\n")
    else:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _build_download_progress_class():
    from tqdm.auto import tqdm

    class OpfDownloadTqdm(tqdm):
        """Flush HuggingFace's final progress footer before OPF resumes output."""

        def set_description(self, desc=None, refresh=True):
            super().set_description(desc=desc, refresh=refresh)
            if desc == "Download complete":
                self.refresh()
                self.close()
                sys.stderr.flush()

    return OpfDownloadTqdm


def _print_download_complete(target: Path) -> None:
    print(f"Download complete: {target}", file=sys.stderr, flush=True)
    print("------", file=sys.stderr, flush=True)
    sys.stderr.flush()
    sys.stdout.flush()


def _promote_original_subtree(target: Path) -> None:
    original_dir = target / "original"
    if not original_dir.is_dir():
        raise RuntimeError(
            f"Downloaded checkpoint is missing expected subtree: {original_dir}"
        )

    for path in original_dir.iterdir():
        destination = target / path.name
        if destination.exists():
            raise RuntimeError(
                "Cannot promote downloaded checkpoint file because destination "
                f"already exists: {destination}"
            )
        shutil.move(str(path), str(destination))
    original_dir.rmdir()


def ensure_default_checkpoint() -> str:
    """Ensure the first-use default checkpoint exists and return its path.

    HuggingFace downloads may be interrupted, leaving ``target/original`` and
    ``target/.cache`` behind. Treat that state as resumable instead of forcing
    the user to manually delete the partial directory.
    """
    target = DEFAULT_MODEL_PATH.expanduser()
    if target.exists():
        if _is_valid_default_checkpoint(target):
            return str(target)
        if not target.is_dir():
            _validate_default_checkpoint(target)
    missing_message = "not found" if not target.exists() else "incomplete"

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"Default OPF checkpoint was {missing_message} at {target}, and "
            "huggingface_hub is not installed. Install HuggingFace support with "
            "`pip install huggingface_hub`, "
            f"{_checkpoint_override_message()}."
        ) from exc

    try:
        print(
            "Default OPF checkpoint "
            f"{missing_message} at {target}. Downloading/resuming from "
            f"HuggingFace repo {DEFAULT_HF_MODEL_REPO!r} to {target}.",
            file=sys.stderr,
            flush=True,
        )
        try:
            snapshot_download(
                repo_id=DEFAULT_HF_MODEL_REPO,
                local_dir=str(target),
                tqdm_class=_build_download_progress_class(),
                allow_patterns=["original/*"],
            )
            if not _is_valid_default_checkpoint(target):
                _promote_original_subtree(target)
        finally:
            _reset_terminal_after_download()
    except Exception as exc:
        raise RuntimeError(
            "Failed to download default OPF checkpoint from HuggingFace repo "
            f"{DEFAULT_HF_MODEL_REPO!r} to {target}: {exc}"
        ) from exc

    _validate_default_checkpoint(target)
    _print_download_complete(target)
    return str(target)
