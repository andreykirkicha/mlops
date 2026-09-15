"""A wheel must deliver a compiled extension, not a source-path Python fallback."""
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sysconfig


def test_imported_native_extension_is_installed():
    assert importlib.util.find_spec("tensor_ops"), "Install the wheel before testing"
    core = importlib.import_module("tensor_ops._core")
    path = Path(core.__file__).resolve()
    assert any(str(path).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    locations = [Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")]
    assert any(path.is_relative_to(location) for location in locations), path
    assert callable(getattr(core, "mac", None))
