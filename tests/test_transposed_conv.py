import numpy as np
import pytest

from dna_gan import encode_transposed_conv, decode_transposed_conv


def test_transposed_conv_roundtrip():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((8, 4, 3, 3))
    records = encode_transposed_conv(F, bits=8)
    assert len(records) == 8
    out = decode_transposed_conv(records, F.shape)
    assert out.shape == F.shape
    assert np.abs(F - out).max() < 0.05


def test_transposed_conv_mixed_ranges():
    rng = np.random.default_rng(1)
    chans = [rng.standard_normal((4, 3, 3)) * s for s in (0.01, 1000.0)]
    F = np.stack(chans, axis=0)
    out = decode_transposed_conv(encode_transposed_conv(F, bits=8), F.shape)
    rel = np.abs(F[1] - out[1]).max() / 1000.0
    assert rel < 0.05


def test_transposed_conv_rejects_2d():
    with pytest.raises(ValueError):
        encode_transposed_conv(np.zeros((4, 4)))


def test_decode_transposed_conv_rejects_bad_shape():
    rng = np.random.default_rng(2)
    F = rng.standard_normal((4, 4, 3, 3))
    records = encode_transposed_conv(F, bits=8)
    with pytest.raises(ValueError):
        decode_transposed_conv(records, (2, 4, 3, 3))
