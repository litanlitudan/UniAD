# Model Loading Comparison: mmdet3d vs Text-based Approach

Generated at: 2025-08-01

## Executive Summary

This report compares two approaches for loading UniAD models:
1. **Standard mmdet3d approach**: Using the built-in registry system
2. **Text-based approach**: Direct config parsing and model construction

## Comparison Table

| Feature | mmdet3d Registry | Text-based Loading |
|---------|-----------------|-------------------|
| **Dependencies** | Requires full mmdet3d installation | Can work with mock models |
| **Registry System** | Uses `@DETECTORS.register_module()` | No registry needed |
| **Flexibility** | Limited to registered components | Easily customizable |
| **Config Parsing** | Through mmcv Config class | Direct Python execution |
| **Error Handling** | Registry errors if component missing | Falls back to mock models |
| **Use Case** | Production training/inference | Analysis and debugging |

## Technical Implementation

### 1. Standard mmdet3d Approach

```python
from mmcv import Config
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint

def load_uniad_model(config_path, checkpoint_path=None, device='cuda'):
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, map_location=device)
    return model, cfg
```

**Pros:**
- Official approach with full feature support
- Integrated with training pipeline
- Validated component compatibility

**Cons:**
- Requires complete mmdet3d environment
- Registry errors if components not found
- Less transparent model construction

### 2. Text-based Loading Approach

```python
def load_uniad_model_from_text(config_path, checkpoint_path=None, device='cuda'):
    # Read and execute config file
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    config_namespace = {'dict': dict, 'True': True, 'False': False, 'None': None}
    exec(config_text, config_namespace)
    
    # Build model based on type
    model_cfg = config_namespace.get('model', None)
    model_type = model_cfg.get('type', '')
    
    if model_type == 'UniAD':
        model = _build_uniad_model(model_cfg)
    elif model_type == 'UniADTrack':
        model = _build_uniad_track_model(model_cfg)
    
    return model, config
```

**Pros:**
- No dependency on registry system
- Works without full mmdet3d installation
- Transparent model construction
- Supports mock models for testing

**Cons:**
- Mock models don't have full functionality
- Requires manual model building logic
- May not support all edge cases

## Model Architecture Comparison

### UniADTrack (Mock Model)
- **Components**: img_backbone, img_neck, bev_encoder, pts_bbox_head
- **Parameters**: ~17M (mock version)
- **Output**: Detection tensor

### UniAD Full (Mock Model)  
- **Components**: Base model + seg_head, motion_head, occ_head, planning_head
- **Parameters**: ~18M (mock version)
- **Output**: Dictionary with task outputs

## Key Differences in Practice

### 1. Error Handling

**mmdet3d approach:**
```
ImportError: cannot import name 'BEVFormerTrackHead' from 'mmdet3d.models.dense_heads'
```

**Text-based approach:**
```
# Automatically falls back to mock implementation
Warning: Using mock BEVFormerTrackHead
```

### 2. Config Variable Resolution

Both approaches handle config variables like `_dim_`, `bev_h_`, etc., but:
- mmdet3d uses Config class with special parsing
- Text-based uses direct Python execution

### 3. Model Inspection

Text-based approach makes it easier to:
- Understand model construction flow
- Debug configuration issues
- Prototype new architectures

## Use Case Recommendations

### Use mmdet3d Registry When:
- Training models for production
- Running inference on real data
- Need full feature compatibility
- Working with established pipelines

### Use Text-based Loading When:
- Analyzing model architecture
- Debugging configuration issues
- Working without full dependencies
- Prototyping new components
- Creating architecture visualizations

## Performance Considerations

1. **Loading Time**: Text-based approach is slightly faster (no registry lookup)
2. **Memory Usage**: Mock models use less memory than full implementations
3. **Forward Pass**: Mock models are faster but don't produce meaningful outputs

## Conclusion

The text-based loading approach complements the standard mmdet3d approach by providing:
- A lightweight alternative for analysis
- Better debugging capabilities
- Independence from the registry system
- Flexibility for custom modifications

While it cannot replace mmdet3d for actual training and inference, it's valuable for:
- Understanding model architecture
- Generating documentation
- Testing configurations
- Educational purposes

Both approaches have their place in the UniAD ecosystem, with the choice depending on the specific use case and requirements.