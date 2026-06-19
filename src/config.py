"""Centralised runtime configuration for the GraphRAG pipeline.

Every tunable value — host/port, model names, thresholds, directories — is read
from the ``.env`` file at import time and exposed as a typed attribute on the
module-level :data:`config` singleton. No model name or threshold is ever
hardcoded elsewhere in the codebase; all such values flow through this module so
the system stays model-agnostic and domain-agnostic.

Call :func:`config.validate` (or :func:`validate_or_exit`) at application
startup to fail loudly if FalkorDB, Ollama, or any required model is unreachable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from the project root (parent of src/). Explicit path so the values
# resolve regardless of the current working directory the app is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class ConfigValidationError(RuntimeError):
    """Raised when a required service or model is unavailable at startup."""


def _require(name: str) -> str:
    """Return a mandatory environment variable or raise a clear error.

    Args:
        name: The environment variable key.

    Returns:
        The variable's value.

    Raises:
        ConfigValidationError: If the variable is unset or empty.
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ConfigValidationError(
            f"Required environment variable '{name}' is missing from .env. "
            f"Add it to {_PROJECT_ROOT / '.env'} and retry."
        )
    return value.strip()


def _env_float(name: str, default: float) -> float:
    """Return an environment variable parsed as a float, or a default."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigValidationError(
            f"Environment variable '{name}'={raw!r} is not a valid float."
        ) from exc


def _env_int(name: str, default: int) -> int:
    """Return an environment variable parsed as an int, or a default."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigValidationError(
            f"Environment variable '{name}'={raw!r} is not a valid int."
        ) from exc


def _normalise_model(name: str) -> str:
    """Normalise an Ollama model tag so ``foo`` and ``foo:latest`` compare equal."""
    return name if ":" in name else f"{name}:latest"


@dataclass(frozen=True)
class Config:
    """Immutable, typed view over every runtime configuration value."""

    # --- FalkorDB ---
    falkordb_host: str
    falkordb_port: int

    # --- Ollama ---
    ollama_base_url: str
    ollama_llm_model: str
    ollama_vision_model: str
    ollama_embed_model: str

    # --- Re-ranking ---
    cross_encoder_model: str

    # --- Entity extraction ---
    gliner_model: str

    # --- Thresholds ---
    splink_merge_threshold: float
    adversarial_score_threshold: float
    gliner_confidence_threshold: float
    chunk_max_tokens: int

    # --- Directories (absolute paths, created on load) ---
    figures_dir: Path
    uploads_dir: Path
    outputs_dir: Path

    # Fixed embedding dimensionality the graph vector index is built around.
    embedding_dim: int = 768

    @property
    def ollama_models(self) -> tuple[str, str, str]:
        """The three Ollama models that must be present for the pipeline to run."""
        return (self.ollama_llm_model, self.ollama_vision_model, self.ollama_embed_model)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> dict[str, bool]:
        """Verify every external dependency is reachable and report status.

        Checks, in order: FalkorDB TCP reachability, the Ollama HTTP API, and the
        presence of all three configured Ollama models.

        Returns:
            A mapping of human-readable check name to pass/fail boolean.

        Raises:
            ConfigValidationError: If any check fails, with a specific message
                naming the failing dependency and how to fix it.
        """
        results: dict[str, bool] = {}
        errors: list[str] = []

        # FalkorDB ----------------------------------------------------------
        try:
            from falkordb import FalkorDB

            db = FalkorDB(host=self.falkordb_host, port=self.falkordb_port)
            db.connection.ping()
            results["falkordb"] = True
            logger.info("FalkorDB reachable at %s:%s", self.falkordb_host, self.falkordb_port)
        except Exception as exc:  # noqa: BLE001 - surface any connection failure
            results["falkordb"] = False
            errors.append(
                f"FalkorDB unreachable at {self.falkordb_host}:{self.falkordb_port} "
                f"({exc}). Start it with: "
                f"docker run -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest"
            )

        # Ollama API --------------------------------------------------------
        available: set[str] = set()
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            available = {
                _normalise_model(m["name"]) for m in resp.json().get("models", [])
            }
            results["ollama"] = True
            logger.info("Ollama reachable at %s (%d models)", self.ollama_base_url, len(available))
        except Exception as exc:  # noqa: BLE001
            results["ollama"] = False
            errors.append(
                f"Ollama unreachable at {self.ollama_base_url} ({exc}). "
                f"Start it with: ollama serve"
            )

        # Individual models (only meaningful if Ollama responded) -----------
        for label, model in (
            ("ollama_llm_model", self.ollama_llm_model),
            ("ollama_vision_model", self.ollama_vision_model),
            ("ollama_embed_model", self.ollama_embed_model),
        ):
            key = f"model:{model}"
            if not results.get("ollama"):
                results[key] = False
                continue
            present = _normalise_model(model) in available
            results[key] = present
            if not present:
                errors.append(
                    f"Configured {label}='{model}' is not pulled in Ollama. "
                    f"Pull it with: ollama pull {model}"
                )

        if errors:
            raise ConfigValidationError(
                "Startup validation failed:\n  - " + "\n  - ".join(errors)
            )

        logger.info("All startup checks passed.")
        return results


def _load() -> Config:
    """Construct the :class:`Config` singleton from environment variables."""
    figures_dir = (_PROJECT_ROOT / _require("FIGURES_DIR")).resolve()
    uploads_dir = (_PROJECT_ROOT / _require("UPLOADS_DIR")).resolve()
    outputs_dir = (_PROJECT_ROOT / os.getenv("OUTPUTS_DIR", "data/outputs")).resolve()
    for d in (figures_dir, uploads_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    return Config(
        falkordb_host=_require("FALKORDB_HOST"),
        falkordb_port=_env_int("FALKORDB_PORT", 6379),
        ollama_base_url=_require("OLLAMA_BASE_URL").rstrip("/"),
        ollama_llm_model=_require("OLLAMA_LLM_MODEL"),
        ollama_vision_model=_require("OLLAMA_VISION_MODEL"),
        ollama_embed_model=_require("OLLAMA_EMBED_MODEL"),
        cross_encoder_model=_require("CROSS_ENCODER_MODEL"),
        gliner_model=_require("GLINER_MODEL"),
        splink_merge_threshold=_env_float("SPLINK_MERGE_THRESHOLD", 0.85),
        adversarial_score_threshold=_env_float("ADVERSARIAL_SCORE_THRESHOLD", 0.3),
        gliner_confidence_threshold=_env_float("GLINER_CONFIDENCE_THRESHOLD", 0.7),
        chunk_max_tokens=_env_int("CHUNK_MAX_TOKENS", 512),
        figures_dir=figures_dir,
        uploads_dir=uploads_dir,
        outputs_dir=outputs_dir,
    )


# Module-level singleton imported everywhere as ``from src.config import config``.
config: Config = _load()


def validate_or_exit() -> None:
    """Run :meth:`Config.validate` and exit the process loudly on failure."""
    import sys

    try:
        config.validate()
    except ConfigValidationError as exc:
        logger.error("%s", exc)
        print(f"\n[CONFIG ERROR]\n{exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("Loaded configuration:")
    for key, value in vars(config).items():
        print(f"  {key:32s} = {value}")
    print("\nRunning startup validation...")
    status = config.validate()
    for check, ok in status.items():
        print(f"  {'✅' if ok else '❌'} {check}")
    print("\nAll checks passed.")
