# DNA-GAN

DNA-GAN is a small NumPy library for lossy quantization of weight arrays, with a helper for encoding and decoding transposed-convolution weights commonly stored by GAN generator layers. It provides a codec; it does not implement a GAN, model training, inference, or a training-data pipeline.

## Status and scope

The project is declared as version `0.1.0`. The repository contains one codec module and unit tests. The implementation is limited to matrix quantization and 4-D transposed-convolution weight arrays; it does not include a GAN architecture or model-level integration.

## Implemented functionality

- `encode(matrix, bits=8)` quantizes a non-empty 2-D matrix into a `GeneRecord`. Supported bit widths are 1 through 16.
- `decode(record)` reconstructs a matrix from a `GeneRecord`.
- `GeneRecord` stores the quantized codes and range/shape metadata, and provides `to_dict()`, `from_dict()`, and `nbytes()` helpers. `to_dict()` returns a Python dictionary containing a NumPy array; it is not a file serialization format.
- `encode_transposed_conv(filters, bits=8)` accepts a 4-D array in `(in_channels, out_channels, kernel_height, kernel_width)` order and creates one independently quantized record per input channel. `decode_transposed_conv(records, shape)` reconstructs the array.
- `codec_stats(matrix, bits=8)` reports an estimated size ratio and reconstruction error for a matrix.

The encoding uses the minimum and maximum of each encoded matrix or channel slice to map values to uniformly spaced integer codes. Decoding is therefore approximate and lossy in general.

## Installation

From the repository root, install the project and its runtime dependency:

```bash
python -m pip install .
```

To install the test dependencies as well, use:

```bash
python -m pip install -r requirements.txt
```

The project metadata requires Python 3.9 or newer and NumPy 1.24 or newer.

## Example

```python
import numpy as np
from dna_gan import (
    decode,
    decode_transposed_conv,
    encode,
    encode_transposed_conv,
)

weights = np.random.default_rng(0).standard_normal((8, 16))
record = encode(weights, bits=8)
restored_weights = decode(record)

# Layout: (in_channels, out_channels, kernel_height, kernel_width)
filters = np.random.default_rng(1).standard_normal((8, 4, 3, 3))
records = encode_transposed_conv(filters, bits=8)
restored_filters = decode_transposed_conv(records, filters.shape)
```

## Tests and CI

On the inspected repository commit, `python -m pytest -q` completed successfully: **16 tests passed**. GitHub Actions also reports successful CI and CodeQL runs for that commit. The CI workflow runs pytest, compilation, and an import/round-trip smoke check on Python 3.10, 3.11, and 3.12. These checks do not establish model-level quality or performance; the repository contains no benchmark results.

## Limitations

- Quantization is lossy. Reconstruction error depends on the value range and selected bit width.
- The transposed-convolution helper handles only the stated 4-D `(in_channels, out_channels, kernel_height, kernel_width)` layout. The general `encode` function accepts only non-empty 2-D matrices.
- `codec_stats()` assumes four bytes per original value when estimating the uncompressed size, and `GeneRecord.nbytes()` counts code-array bytes plus a fixed 16-byte metadata allowance. The reported ratio is an estimate, not the size of a serialized file.
- The repository does not provide a model, trainer, inference pipeline, file-format serializer, or benchmark suite.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for its terms and copyright notice.
