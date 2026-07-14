"""Lightweight runtime configuration for dnafiber.

Settings can be overridden, in increasing order of precedence:

1. Built-in defaults (below).
2. A TOML config file, located at ``$DNAFIBER_CONFIG`` if set, otherwise
   ``~/.config/dnafiber/config.toml``.
3. Environment variables (highest precedence), e.g. ``DNAFIBER_CZI_DETECT_MOSAIC``.

Example config file::

    [czi]
    detect_mosaic = false

The config is read once and cached. Call :func:`reload_config` to re-read it
(useful in notebooks/tests after changing the environment or file).
"""

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # tomllib is stdlib on Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.10
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "czi": {
        # If False, CziFile is opened with detectmosaic=False. This is required
        # for some newer microscope exports that otherwise return all-zero
        # images. Expert users acquiring genuine multi-tile mosaics may want to
        # set this back to True.
        "detect_mosaic": False,
    },
}


@dataclass(frozen=True)
class Config:
    czi_detect_mosaic: bool = DEFAULTS["czi"]["detect_mosaic"]


def _config_path() -> Path:
    env = os.environ.get("DNAFIBER_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "dnafiber" / "config.toml"


def _load_file() -> dict:
    path = _config_path()
    if not path.is_file():
        return {}
    if tomllib is None:
        warnings.warn(
            f"Found config file at {path} but no TOML parser is available "
            "(install 'tomli' on Python < 3.11). Ignoring it.",
            stacklevel=2,
        )
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:  # pragma: no cover - malformed config
        warnings.warn(f"Failed to read config file {path}: {e}", stacklevel=2)
        return {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the cached runtime configuration."""
    file_cfg = _load_file()
    czi = {**DEFAULTS["czi"], **file_cfg.get("czi", {})}

    return Config(
        czi_detect_mosaic=_env_bool(
            "DNAFIBER_CZI_DETECT_MOSAIC", bool(czi["detect_mosaic"])
        ),
    )


def reload_config() -> Config:
    """Clear the cache and re-read configuration from file/environment."""
    get_config.cache_clear()
    return get_config()
