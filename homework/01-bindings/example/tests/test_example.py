import importlib
import importlib.util


def test_add_from_installed_demo():
    assert importlib.util.find_spec("binding_demo"), "Build and install the example wheel first"
    demo = importlib.import_module("binding_demo")
    assert demo.add(2.0, -3.5) == -1.5
    assert demo.add(0.0, 0.0) == 0.0
