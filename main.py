"""MkDocs Macros hook that reads heat/core/version.py and sets env.variables['version'] and ['release'] for documentation builds."""

from pathlib import Path
import importlib.util


def define_env(env):
    """Populate MkDocs Macros variables 'version' and 'release' from heat/core/version.py; reads major, minor, micro and optional extension, then sets env.variables accordingly for use in templates."""
    version_path = Path(__file__).parent / "heat" / "core" / "version.py"
    spec = importlib.util.spec_from_file_location("heat_core_version", version_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    major = getattr(mod, "major")
    minor = getattr(mod, "minor")
    micro = getattr(mod, "micro")
    extension = getattr(mod, "extension", "")

    base = f"{major}.{minor}.{micro}"
    release = f"{base}-{extension}" if extension else base

    env.variables["version"] = base
    env.variables["release"] = release
