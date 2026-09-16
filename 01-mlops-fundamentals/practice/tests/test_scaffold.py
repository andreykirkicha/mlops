"""Provided infrastructure must work before students fill the three ML exercises."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import sklearn

from taxi_duration import model
from taxi_duration.data import FEATURES


def test_cli_help_does_not_require_completed_exercises(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(model.__file__).parents[1])}
    result = subprocess.run([sys.executable, "-m", "taxi_duration.cli", "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "train" in result.stdout and "predict" in result.stdout


def test_trusted_bundle_round_trip_and_no_overwrite(tmp_path):
    bundle = {"format_version": 1, "features": list(FEATURES),
              "sklearn_version": sklearn.__version__, "marker": "provided scaffold"}
    path = tmp_path / "own-model.pkl"
    model.save_model(bundle, path)
    assert model.load_model(path) == bundle
    with pytest.raises(FileExistsError):
        model.save_model(bundle, path)
