# .toon Model Format Specification (v1)

The `.toon` format is a lightweight, single-file binary container for MoE Transformer models, optimized for low-resource CPU and iGPU (Vega 7) inference. It encapsulates the model's architecture (JSON), tokenizer (JSON/BPE), and quantized weights (INT8) into a predictable, memory-mappable structure.

## 1. Structure Overview

The file follows a strict sequential structure:

| Section | Data Type | Description |
| :--- | :--- | :--- |
| **MAGIC** | `b"TOON"` | 4-byte magic identifier |
| **VERSION** | `uint8` | Format version (currently 0x01) |
| **CONFIG_LEN** | `uint32` | Length of the JSON model configuration |
| **CONFIG_JSON** | `bytes` | UTF-8 encoded `SmallLLMConfig` parameters |
| **VOCAB_LEN** | `uint32` | Length of the tokenizer vocabulary data |
| **VOCAB_DATA** | `bytes` | UTF-8 encoded `tokenizer.json` |
| **TENSOR_COUNT**| `uint32` | Number of distinct tensor layers |
| **METADATA_LEN**| `uint32` | Length of the JSON tensor metadata |
| **METADATA_JSON**| `bytes` | UTF-8 list of tensor descriptions |
| **TENSOR_DATA** | `bytes` | Concatenated raw weight data (16-byte aligned) |

## 2. Tensor Metadata Structure

The `METADATA_JSON` section contains an array of objects, one for each weight matrix:

```json
[
  {
    "name": "layers.0.attention.wq.weight",
    "shape": [256, 256],
    "dtype": "qint8",
    "offset": 0,
    "length": 65536
  }
]
```

- **dtype**: Currently supports `qint8` (quantized INT8), `float16`, and `float32`.
- **offset**: Relative to the start of the **TENSOR_DATA** block.
- **length**: Total size of raw data in bytes for this specific tensor.

## 3. Storage Principles

- **Endianness**: Little-endian (standard for x86_64 and most modern ARM architectures).
- **Alignment**: All tensor data segments are 16-byte aligned to ensure optimal memory-mapped (mmap) reading performance.
- **Row-Major**: All weights are stored in standard C-style row-major order.
- **Weight Tying**: Shared embeddings (input/output) are stored once and referenced via metadata names if applicable.

## 4. Why .toon?

Standard formats like Safetensors or GGUF are robust but often include overhead for general-purpose features. The `.toon` format is designed to be a "thin" wrapper that requires minimal parsing logic, allowing a simple C++ or Python script to load and initialize a model in milliseconds.
