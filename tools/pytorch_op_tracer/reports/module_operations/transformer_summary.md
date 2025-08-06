# Transformer Modules - Summary

**Generated**: 2025-08-05 17:24:05

## Overview

This category contains 2 modules that form the transformer of UniAD.

## Module Summary

| Module | Description | Parameters | Memory | FLOPs |
|--------|-------------|------------|--------|-------|
| PerceptionTransformer | Main transformer for object queries and detection | 8.9M | ~35MB | 1.8 GFLOPs |
| MSDeformableAttention3D | Multi-scale deformable attention module for 3D feature aggregation | 0.5M | ~5MB | 0.2 GFLOPs |

## Common Operations in Category

| Operation | Total Count | Modules Using |
|-----------|-------------|---------------|
| Linear | 7 | 2 |
| LayerNorm | 3 | 1 |
| MultiheadAttention | 2 | 1 |
| ReLU | 1 | 1 |
| Softmax | 1 | 1 |
