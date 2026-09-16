"""Wheel-check preconditions fail before creating an env or installing packages."""
from pathlib import Path
import shutil
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "check-wheel.sh"


def test_missing_inputs_stop_before_creating_receiver(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path), str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert str(tmp_path / "requirements-lock.txt") in result.stderr
    assert "Среда получателя:" not in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_default_paths_are_relative_to_script_not_current_directory(tmp_path):
    practice = tmp_path / "practice with spaces"
    (practice / "starter").mkdir(parents=True)
    (practice / "data").mkdir()
    (practice / "requirements-lock.txt").touch()
    (practice / "data/inference.csv").touch()
    script = practice / "check-wheel.sh"
    shutil.copyfile(SCRIPT, script)
    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
    expected = practice / "starter/dist/taxi_duration_course-0.1.0-py3-none-any.whl"
    assert str(expected) in result.stderr
    assert "Среда получателя:" not in result.stdout


def test_nonexistent_package_directory_is_reported(tmp_path):
    package = tmp_path / "missing package"
    result = subprocess.run(
        ["bash", str(SCRIPT), str(package), str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert str(package) in result.stderr
    assert "Среда получателя:" not in result.stdout
