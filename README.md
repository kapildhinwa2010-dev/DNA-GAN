# DNA-GAN

[![CI](https://github.com/kapildhinwa2010-dev/DNA-GAN/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-GAN/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/DNA-GAN/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-GAN/actions/workflows/codeql.yml)

DNA quantized codec tuned for GAN generator weights: per-input-channel ConvTranspose encoding.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

DNA-GAN is the GAN-specialized member of the DNA codec family
(lineage: the private DNA repository). encode()/decode() quantize a
float matrix into a GeneRecord; the GAN-specific part is
encode_transposed_conv()/decode_transposed_conv(): ConvTranspose2d
weights (in_ch, out_ch, kh, kw) - the transpose of a regular conv -
are encoded per INPUT channel, each slice with its own scale.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from dna_gan import encode_transposed_conv, decode_transposed_conv

records = encode_transposed_conv(deconv_weight, bits=8)  # (in_ch, out_ch, kh, kw)
restored = decode_transposed_conv(records, deconv_weight.shape)
```

## API

- encode(matrix, bits) / decode(record) / codec_stats(matrix, bits)
- encode_transposed_conv(filters, bits) - one GeneRecord per input channel
- decode_transposed_conv(records, shape) - restores the 4-D tensor

## Honest results and limitations

- Lossy: error equals the quantization step of each input-channel
  slice.
- Only the (in_ch, out_ch, kh, kw) ConvTranspose layout; regular
  conv weights belong in DNA-CNN.
- "Ratio" in codec_stats() is float32-bytes / encoded-bytes
  ARITHMETIC - it counts stored codes, not a real serialized file.
- No per-layer fitting: the reserv.dean-part-2 log records that
  generated-weight reconstructions of trained models were not usable
  without per-layer fitting.

## License

MIT - see [LICENSE](LICENSE).
