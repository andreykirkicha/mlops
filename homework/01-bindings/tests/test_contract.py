"""Public acceptance tests. Run against an installed wheel, never by adding src to sys.path."""
import importlib
import importlib.util

import numpy as np
import pytest


@pytest.fixture
def mac():
    assert importlib.util.find_spec("tensor_ops"), "Build and install your tensor_ops wheel first"
    package = importlib.import_module("tensor_ops")
    assert callable(getattr(package, "mac", None)), "Export tensor_ops.mac(a, b, c)"
    return package.mac


def inputs(shape=(2, 3, 4)):
    a = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    return a, np.full(shape, 0.5), np.full(shape, -2.0)


def test_hand_calculated_values(mac):
    a = np.array([[[1.0, -2.0, 0.0]]])
    b = np.array([[[3.0, 4.0, -9.0]]])
    c = np.array([[[0.5, 1.0, -2.0]]])
    np.testing.assert_allclose(mac(a, b, c), [[[3.5, -7.0, -2.0]]], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (4, 2, 7)])
def test_matches_numpy_with_output_contract(mac, shape):
    rng = np.random.default_rng(2026)
    a, b, c = [rng.normal(size=shape) for _ in range(3)]
    out = mac(a, b, c)
    assert isinstance(out, np.ndarray)
    assert out.shape == shape
    assert out.dtype == np.dtype("float64")
    assert out.flags.c_contiguous
    np.testing.assert_allclose(out, a * b + c, rtol=1e-12, atol=1e-12)


def test_inputs_are_unchanged_and_output_is_independent(mac):
    args = inputs()
    before = [x.copy() for x in args]
    out = mac(*args)
    for original, snapshot in zip(args, before):
        np.testing.assert_array_equal(original, snapshot)
        assert not np.shares_memory(original, out)
    out.fill(123.0)
    for original, snapshot in zip(args, before):
        np.testing.assert_array_equal(original, snapshot)


@pytest.mark.parametrize("shape", [(0, 3, 4), (2, 0, 4), (2, 3, 0)])
def test_empty_arrays_preserve_shape(mac, shape):
    out = mac(*inputs(shape))
    assert out.shape == shape and out.dtype == np.dtype("float64")
    assert out.size == 0


@pytest.mark.parametrize("position", [0, 1, 2])
def test_wrong_shape_is_not_broadcast(mac, position):
    args = list(inputs())
    args[position] = np.ones((1, 3, 4), dtype=np.float64)
    with pytest.raises(ValueError):
        mac(*args)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_equal_size_different_shape_is_rejected(mac, position):
    args = list(inputs())
    args[position] = args[position].reshape(4, 3, 2)
    with pytest.raises(ValueError):
        mac(*args)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_wrong_rank_is_rejected(mac, position):
    args = list(inputs())
    args[position] = np.ones((6, 4), dtype=np.float64)
    with pytest.raises(ValueError):
        mac(*args)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.bool_])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_wrong_dtype_is_not_silently_converted(mac, dtype, position):
    args = list(inputs())
    args[position] = args[position].astype(dtype)
    with pytest.raises(TypeError):
        mac(*args)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_non_native_byte_order_is_rejected(mac, position):
    args = list(inputs())
    args[position] = args[position].astype(np.dtype("float64").newbyteorder("S"))
    assert not args[position].dtype.isnative
    with pytest.raises(TypeError):
        mac(*args)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_python_list_is_not_silently_converted(mac, position):
    args = list(inputs())
    args[position] = args[position].tolist()
    with pytest.raises(TypeError):
        mac(*args)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_noncontiguous_view_is_rejected(mac, position):
    args = list(inputs())
    args[position] = np.arange(48.0).reshape(2, 3, 8)[:, :, ::2]
    assert args[position].shape == (2, 3, 4)
    assert not args[position].flags.c_contiguous
    with pytest.raises(ValueError):
        mac(*args)


def test_readonly_inputs_are_valid(mac):
    args = inputs()
    expected = args[0] * args[1] + args[2]
    for arg in args:
        arg.flags.writeable = False
    np.testing.assert_allclose(mac(*args), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_unaligned_buffer_is_rejected(mac, position):
    args = list(inputs())
    args[position] = np.ndarray((2, 3, 4), dtype=np.float64, buffer=bytearray(24 * 8 + 1), offset=1)
    assert args[position].flags.c_contiguous and not args[position].flags.aligned
    with pytest.raises(ValueError):
        mac(*args)
