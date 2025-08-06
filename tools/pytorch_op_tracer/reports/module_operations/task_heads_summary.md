# Task_Heads Modules - Summary

**Generated**: 2025-08-05 17:24:05

## Overview

This category contains 5 modules that form the task_heads of UniAD.

## Module Summary

| Module | Description | Parameters | Memory | FLOPs |
|--------|-------------|------------|--------|-------|
| TrackHead | 3D object detection and tracking head | 2.8M | ~50MB | 0.5 GFLOPs |
| SegHead | BEV segmentation head for lanes and drivable area | 0.8M | ~20MB | 0.3 GFLOPs |
| MotionHead | Multi-modal motion prediction for agents | 1.5M | ~30MB | 0.4 GFLOPs |
| OccHead | 3D occupancy and flow prediction | 2.1M | ~40MB | 0.8 GFLOPs |
| PlanningHead | Ego vehicle trajectory planning | 1.2M | ~25MB | 0.3 GFLOPs |

## Common Operations in Category

| Operation | Total Count | Modules Using |
|-----------|-------------|---------------|
| Linear | 10 | 3 |
| ReLU | 8 | 5 |
| Conv3d | 6 | 1 |
| Conv2d | 5 | 1 |
| LayerNorm | 2 | 1 |
| BatchNorm2d | 1 | 1 |
| LSTM | 1 | 1 |
| Softmax | 1 | 1 |
| GRU | 1 | 1 |
