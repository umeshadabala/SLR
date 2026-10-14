# EXP-04 — Selective Residency Optimization Study

## Model

Qwen/Qwen2.5-1.5B-Instruct

- Parameters: 1,543,714,304
- Transformer layers: 28
- Hidden size: 1536
- Representation: BF16
- Device: CPU

## Research Objective

Evaluate low-resource transformer inference using selective layer
residency, storage-backed weight streaming, asynchronous loading,
and memory-bounded execution.

## Core Result

The complete 28-layer transformer was successfully executed while
keeping approximately 89.26 MB of transformer layer weights resident
at a time.

The transformer layer checkpoint data streamed during the complete
execution was approximately 2.4407 GB.

## Optimization Results

### Persistent safetensors handle

Validated successfully.

### Garbage-collection optimization

Removing per-layer garbage collection produced approximately:

4.53x speedup

while preserving the same one-layer residency constraint.

### Parameter prebinding

Negative result.

Approximately 4.64% slower than the dynamic resolver.

### Two-layer residency

Validated, but did not provide a reliable latency improvement over
one-layer residency.

### Asynchronous prefetch

Actual overlap between prefetch and computation was demonstrated.

### Storage/page-cache prefetch

Repeated testing did not demonstrate a reproducible latency benefit.

### End-to-end double buffering

Successfully executed all 28 layers with:

- One active model layer
- One staged layer
- Maximum weight data: approximately 178.52 MB
- Median end-to-end wall time: approximately 3.277 s

## Final Correctness

The selective-residency implementation was compared against a
conventionally loaded BF16 reference model.

Final results:

- Final hidden-state cosine similarity: 0.999999701977
- Relative L2 error: 0
- Maximum absolute hidden-state difference: 0
- Top-1 agreement: True
- Top-10 intersection: 10/10
- Final cleanup: PASS

## Experimental Philosophy

Both positive and negative optimization results are retained.

The objective is to determine which mechanisms actually improve
low-resource inference under memory constraints rather than selecting
only favorable measurements.

## Status

EXP-04 COMPLETE
