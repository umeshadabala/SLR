# Memory Accounting in Selective Layer Residency

## Core Metric Distinctions

To ensure scientific and research integrity, the application rigorously distinguishes between the following memory concepts:

| Metric | Typical Qwen2.5-1.5B Value | Definition |
|---|---|---|
| **Checkpoint Storage Size** | ~2.875 GB (BF16) | Total file size of `model.safetensors` on disk. |
| **Total Model Parameters** | 1,543,714,304 | All model weights including embeddings, layers, and norm. |
| **Total Transformer Parameters** | 1,310,339,072 | 28 transformer decoder layers (excluding embeddings). |
| **Active Layer Parameters** | 46,797,824 | Parameters in exactly one decoder layer (1/28th). |
| **Active Layer Storage** | 89.26 MB | Theoretical BF16 byte footprint of one layer ($46,797,824 \times 2$ bytes). |
| **Process RSS (Resident Set Size)** | ~1.8 – 2.9 GB | Actual physical RAM mapped to the Python process by the OS kernel. |
| **KV-Cache Memory** | $O(\text{seq\_len})$ | Key and value activations stored across decoding steps. |

## The "28× Reduction" Nuance
- The **28× reduction factor** specifically refers to the ratio of **active transformer layer parameters** ($46.8\text{M}$) compared to the **total transformer layer parameters** ($1.31\text{B}$).
- It does **not** represent a 28× reduction in total operating system process RSS. Total process memory includes:
  1. Python runtime and interpreter overhead (~60 MB).
  2. PyTorch framework internals and memory allocator caching (~200–400 MB).
  3. Resident non-layer weights (Token embedding table: ~291 MB in BF16).
  4. Operating system page cache and file memory mappings.
  5. KV-Cache activations.

## Research Rule
The application will **never** claim an 89.26 MB peak process RAM unless that specific operating system metric has been physically measured. Process RSS and active parameter counts are always presented as distinct metrics with explicit units.
