# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

UniAD (Planning-oriented Autonomous Driving) is a unified framework that hierarchically integrates perception, prediction, and planning tasks for autonomous driving. Unlike traditional modular approaches, UniAD follows a planning-oriented philosophy where all tasks are designed to serve the ultimate goal of trajectory planning.

## Project Structure

```
UniAD/
├── projects/
│   ├── configs/                      # Configuration files
│   │   ├── _base_/                  # Base configs (datasets, runtime)
│   │   ├── stage1_track_map/        # Stage 1 perception configs
│   │   └── stage2_e2e/              # Stage 2 end-to-end configs
│   └── mmdet3d_plugin/
│       ├── core/                    # Core utilities
│       │   ├── bbox/                # Bounding box utilities
│       │   └── evaluation/          # Evaluation hooks
│       ├── datasets/                # Dataset handling
│       │   ├── eval_utils/          # Evaluation utilities
│       │   └── pipelines/           # Data processing pipelines
│       ├── losses/                  # Loss functions
│       ├── models/                  # Model components
│       │   ├── backbones/           # Backbone networks
│       │   ├── utils/               # Model utilities
│       │   └── opt/                 # Optimizers
│       └── uniad/                   # Main UniAD implementation
│           ├── apis/                # Training/testing APIs
│           ├── detectors/           # Main model classes
│           ├── dense_heads/         # Task-specific heads
│           └── modules/             # Transformer modules
├── tools/                           # Training/evaluation scripts
├── docs/                            # Documentation
└── ckpts/                          # Checkpoints directory
```

## Core Architecture

### Main Components

1. **Core Detector Classes**:
   - `UniADTrack` (projects/mmdet3d_plugin/uniad/detectors/uniad_track.py): Base perception module for tracking and mapping
   - `UniAD` (projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py): Full end-to-end model integrating all tasks

2. **Task Heads** (projects/mmdet3d_plugin/uniad/dense_heads/):
   - `BEVFormerTrackHead` (track_head.py): 3D object tracking with temporal modeling
   - `PansegformerHead` (panseg_head.py): BEV segmentation for lane/map understanding
   - `MotionHead` (motion_head.py): Multi-modal motion prediction
   - `OccHead` (occ_head.py): Occupancy flow prediction
   - `PlanningHeadSingleMode` (planning_head.py): Ego vehicle trajectory planning

3. **Key Modules**:
   - BEV Encoder: Transforms multi-view images to bird's-eye view features
   - Temporal Aggregation: Fuses information across multiple frames
   - Task Decoders: Specialized decoders for each downstream task

### Training Pipeline

UniAD employs a two-stage training strategy:

**Stage 1 - Perception Training**:
- Modules: Track + Map heads only
- Queue Length: 5 frames
- Focus: Stable BEV feature learning and object tracking
- Memory: ~50GB GPU (reducible to ~30GB with queue_length=3)

**Stage 2 - End-to-End Training**:
- Modules: All task heads
- Queue Length: 3 frames  
- Key: BEV encoder frozen from Stage 1
- Memory: ~17GB GPU

## Development Commands

### Training
```bash
# Stage 1 training (perception)
./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/base_track_map.py 8

# Stage 2 training (end-to-end)
./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/base_e2e.py 8
```

### Evaluation
```bash
# Evaluate stage 1 model
./tools/uniad_dist_eval.sh ./projects/configs/stage1_track_map/base_track_map.py ./ckpts/uniad_base_track_map.pth 8

# Evaluate stage 2 model
./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/base_e2e.py ./ckpts/uniad_base_e2e.pth 8
```

### Visualization
```bash
python ./tools/analysis_tools/visualize/run.py \
    --predroot /PATH/TO/RESULTS.pkl \
    --out_folder /PATH/TO/OUTPUT \
    --demo_video test_demo.avi \
    --project_to_cam True
```

### Testing a Single Configuration
```bash
# Test with specific GPU
CUDA_VISIBLE_DEVICES=0 python tools/test.py [config_file] [checkpoint] --eval bbox

# Distributed testing
python -m torch.distributed.launch --nproc_per_node=8 tools/test.py [config_file] [checkpoint] --launcher pytorch --eval bbox
```

## Key Technical Details

### Model Configuration
- **BEV Grid**: 200x200, resolution 0.512m per pixel
- **Point Cloud Range**: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
- **Feature Dimension**: 256
- **Temporal Aggregation**: 3-5 frames depending on stage

### Task Loss Weights (Stage 2)
- track: 1.0
- map: 1.0
- motion: 1.0
- occ: 1.0
- planning: 1.0

### Performance Targets
- **Stage 1**: AMOTA ~0.390 for tracking
- **Stage 2**: 
  - Motion: ~0.705 minADE
  - Occupancy: ~63.7% IoU
  - Planning: ~0.29% avg collision rate

## Important Notes

1. **GPU Memory Requirements**:
   - Stage 1: ~50GB (can reduce to ~30GB with queue_length=3)
   - Stage 2: ~17GB (BEV encoder frozen)

2. **Pretrained Weights**:
   - BEVFormer backbone: bevformer_r101_dcn_24ep.pth
   - Stage 1: uniad_base_track_map.pth
   - Stage 2: uniad_base_e2e.pth

3. **Dataset**: Uses nuScenes dataset with custom preprocessing for occupancy flow and planning labels

4. **Dependencies**: Requires specific versions - torch==1.9.1, mmcv-full==1.4.0, mmdet==2.14.0, mmdet3d==0.17.1

## Code Organization

### Configuration System
- Uses MMDetection3D's config system with inheritance
- Base configs define dataset, runtime settings
- Stage-specific configs override/extend base settings
- Key config parameters:
  - `point_cloud_range`: Defines BEV grid boundaries
  - `bev_h_`, `bev_w_`: BEV feature map dimensions
  - `queue_length`: Number of temporal frames
  - Loss weights for multi-task learning

### Data Pipeline
1. **Loading**: Multi-view images, annotations, temporal information
2. **Augmentation**: Photometric distortion, normalization
3. **Formatting**: Convert to model input format
4. **Temporal**: Handle frame sequences and future annotations

### Model Workflow
1. **Image Backbone**: Extract multi-scale features from cameras
2. **BEV Encoder**: Project features to bird's-eye view
3. **Temporal Fusion**: Aggregate features across frames
4. **Task Decoders**: Parallel decoding for each task
5. **Loss Computation**: Multi-task loss with task-specific weights

## Common Development Tasks

### Adding a New Task Head
1. Create new head class in `projects/mmdet3d_plugin/uniad/dense_heads/`
2. Register the head in `__init__.py`
3. Add head configuration to model config
4. Implement forward pass and loss computation
5. Update evaluation metrics if needed

### Modifying the BEV Encoder
- BEV encoder is in the perception transformer
- Can swap BEVFormer with other methods (LSS, etc.)
- Must provide `bev_embed` and `bev_pos` tensors
- Ensure shape compatibility: [bs, bev_h, bev_w, embed_dims]

### Debugging Training
1. Check data loading with visualization tools
2. Monitor loss curves for each task
3. Verify gradient flow through all modules
4. Use smaller batch size for debugging
5. Enable find_unused_parameters for module debugging

## Performance Optimization

### Memory Optimization
- Reduce `queue_length` to save GPU memory
- Use gradient checkpointing for transformer layers
- Freeze modules not being trained (e.g., BEV encoder in Stage 2)
- Adjust batch size and image resolution

### Training Speed
- Use mixed precision training (fp16)
- Enable cudnn benchmarking
- Optimize data loading with more workers
- Consider gradient accumulation for larger effective batch size

## Evaluation Metrics

### Task-Specific Metrics
- **Tracking**: AMOTA (Average Multi-Object Tracking Accuracy)
- **Mapping**: IoU for lane segmentation
- **Motion**: minADE (minimum Average Displacement Error)
- **Occupancy**: IoU for occupancy prediction
- **Planning**: Average collision rate, L2 error

### Evaluation Strategies
- Planning metrics have two interpretations:
  - UniAD: Point-wise evaluation at specific time
  - STP3: Average evaluation up to specific time
- Configure with `planning_evaluation_strategy` parameter

## Analysis Tools

### PyTorch Operation Tracer

UniAD includes a comprehensive PyTorch operation tracing tool for analyzing model execution, memory usage, and dataflow. Located at `tools/pytorch_op_tracer/`, this tool is specifically designed for UniAD's multi-task architecture.

#### Features
- **Operation Tracing**: Hooks into PyTorch modules to trace all forward operations
- **UniAD Task Head Analysis**: Specialized tracking for all 5 task heads (track, seg, motion, occ, planning)
- **Memory Profiling**: Critical for understanding UniAD's 30-50GB GPU memory requirements
- **Data Type Analysis**: Track fp32/fp16/int8 operations and mixed precision opportunities
- **BEV Feature Tracking**: Specialized analysis for BEV encoder/decoder operations
- **Temporal Analysis**: Visualize multi-frame temporal queue processing
- **Mermaid Visualization**: Generate clear dataflow diagrams with shape annotations

#### Basic Usage
```bash
# Trace UniAD Stage 2 model
python tools/pytorch_op_tracer/trace_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --stage 2 \
    --output trace_analysis.md

# Focus on specific task heads
python tools/pytorch_op_tracer/trace_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --task-heads track motion planning \
    --memory-profile \
    --output task_analysis.md

# Analyze BEV operations
python tools/pytorch_op_tracer/trace_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --bev-focus \
    --filter-ops BEVFormer BEVEncoder \
    --output bev_analysis.md

# Mixed precision analysis
python tools/pytorch_op_tracer/trace_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --mixed-precision fp16 \
    --dtype-memory-analysis \
    --output mixed_precision.md
```

#### Output
The tool generates comprehensive Markdown reports containing:
- Summary statistics (operations, memory usage, compute time)
- Task head memory/compute breakdown
- Dataflow visualization with tensor shapes (e.g., `[1,256,200,200]@fp16`)
- Memory heatmaps identifying bottlenecks
- Data type distribution and conversion analysis
- Mixed precision optimization recommendations

#### Installation
```bash
# Install as package
cd tools/pytorch_op_tracer
pip install -e .

# Or use directly
export PYTHONPATH=$PYTHONPATH:$(pwd)/tools/pytorch_op_tracer
```

This tool is invaluable for:
- Understanding memory bottlenecks in UniAD
- Optimizing mixed precision training
- Debugging shape mismatches
- Analyzing compute distribution across task heads
- Planning memory optimization strategies