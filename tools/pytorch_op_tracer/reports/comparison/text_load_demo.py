#!/usr/bin/env python3
"""Demo of text-based model loading"""

import sys
import os

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from datetime import datetime

# Import the text-based loader
from tools.pytorch_op_tracer.utils.model_utils import load_uniad_model_from_text

def trace_simple_model(model, name):
    """Simple tracing without full infrastructure"""
    print(f"\n=== Tracing {name} ===")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # List modules
    print("\nModel structure:")
    for name, module in model.named_children():
        print(f"  {name}: {module.__class__.__name__}")
        if hasattr(module, 'out_channels'):
            print(f"    - out_channels: {module.out_channels}")
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            print(f"    - in_features: {module.in_features}, out_features: {module.out_features}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    if isinstance(output, dict):
        print("\nOutput structure (dict):")
        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.shape}")
    else:
        print(f"\nOutput shape: {output.shape}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'output_type': type(output).__name__
    }

def main():
    print("Text-based Model Loading Demo")
    print("=" * 60)
    print(f"Generated at: {datetime.now()}")
    
    # Test 1: Simple UniADTrack
    print("\n\n## Test 1: UniADTrack Model")
    
    config1 = """
model = dict(
    type='UniADTrack',
    num_classes=10,
    embed_dims=256,
    num_query=900,
    queue_length=3,
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
)
"""
    
    with open('temp_track_config.py', 'w') as f:
        f.write(config1)
    
    try:
        model1, cfg1 = load_uniad_model_from_text('temp_track_config.py', device='cpu')
        stats1 = trace_simple_model(model1, "UniADTrack")
    finally:
        if os.path.exists('temp_track_config.py'):
            os.unlink('temp_track_config.py')
    
    # Test 2: Full UniAD
    print("\n\n## Test 2: Full UniAD Model")
    
    config2 = """
model = dict(
    type='UniAD',
    num_classes=10,
    embed_dims=256,
    num_query=900,
    queue_length=3,
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    task_loss_weight=dict(
        track=1.0,
        map=1.0,
        motion=1.0,
        occ=1.0,
        planning=1.0
    ),
)
"""
    
    with open('temp_uniad_config.py', 'w') as f:
        f.write(config2)
    
    try:
        model2, cfg2 = load_uniad_model_from_text('temp_uniad_config.py', device='cpu')
        stats2 = trace_simple_model(model2, "UniAD")
    finally:
        if os.path.exists('temp_uniad_config.py'):
            os.unlink('temp_uniad_config.py')
    
    # Generate report
    report_path = os.path.join(os.path.dirname(__file__), 'text_load_demo_report.md')
    with open(report_path, 'w') as f:
        f.write("# Text-based Model Loading Demo Report\n\n")
        f.write(f"Generated at: {datetime.now()}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report demonstrates the text-based model loading functionality that allows ")
        f.write("parsing UniAD config files and constructing models without relying on mmdet3d's registry system.\n\n")
        
        f.write("## Model Statistics\n\n")
        f.write("### UniADTrack (Mock)\n")
        f.write(f"- Total parameters: {stats1['total_params']:,}\n")
        f.write(f"- Trainable parameters: {stats1['trainable_params']:,}\n")
        f.write(f"- Output type: {stats1['output_type']}\n\n")
        
        f.write("### UniAD Full (Mock)\n")
        f.write(f"- Total parameters: {stats2['total_params']:,}\n")
        f.write(f"- Trainable parameters: {stats2['trainable_params']:,}\n")
        f.write(f"- Output type: {stats2['output_type']}\n\n")
        
        f.write("## Key Features\n\n")
        f.write("1. **Config Parsing**: Executes Python config files to extract model configuration\n")
        f.write("2. **Flexible Construction**: Builds models based on config type without registry\n")
        f.write("3. **Mock Models**: Creates lightweight models for testing when dependencies are missing\n")
        f.write("4. **Forward Pass Support**: Mock models support forward passes for tracing\n\n")
        
        f.write("## Benefits\n\n")
        f.write("- No dependency on mmdet3d's global registry\n")
        f.write("- Works without full mmdet3d installation\n")
        f.write("- Useful for architecture analysis and debugging\n")
        f.write("- Supports config files with variables\n\n")
        
        f.write("## Usage\n\n")
        f.write("```python\n")
        f.write("from utils.model_utils import load_uniad_model_from_text\n\n")
        f.write("# Load model from config\n")
        f.write("model, cfg = load_uniad_model_from_text('path/to/config.py', device='cpu')\n\n")
        f.write("# Use the model\n")
        f.write("output = model(input_tensor)\n")
        f.write("```\n")
    
    print(f"\n\nReport generated: {report_path}")
    print("\nDone!")

if __name__ == '__main__':
    main()