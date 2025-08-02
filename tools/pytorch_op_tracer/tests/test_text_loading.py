#!/usr/bin/env python3
"""Test the text-based model loading functionality"""

import unittest
import torch
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_uniad_model_from_text


class TestTextLoading(unittest.TestCase):
    """Test text-based model loading without mmdet3d registry"""
    
    def test_simple_config_parsing(self):
        """Test parsing a simple config file"""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
# Simple test config
model = dict(
    type='UniADTrack',
    num_classes=10,
    embed_dims=256,
    num_query=900,
    queue_length=3,
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=4,
    ),
)
""")
            config_path = f.name
        
        try:
            # Test loading
            model, cfg = load_uniad_model_from_text(config_path, device='cpu')
            
            # Verify model is created
            self.assertIsNotNone(model)
            self.assertTrue(isinstance(model, torch.nn.Module))
            
            # Verify config is parsed
            self.assertEqual(cfg.model['type'], 'UniADTrack')
            self.assertEqual(cfg.model['num_classes'], 10)
            self.assertEqual(cfg.model['embed_dims'], 256)
            
        finally:
            # Clean up
            os.unlink(config_path)
    
    def test_uniad_full_config(self):
        """Test parsing a full UniAD config"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
# Full UniAD test config
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
    img_backbone=dict(
        type='ResNet',
        depth=50,
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
    ),
    pts_bbox_head=dict(
        type='BEVFormerTrackHead',
        num_query=900,
        num_classes=10,
    ),
    seg_head=dict(
        type='PansegformerHead',
        num_classes=4,
    ),
    motion_head=dict(
        type='MotionHead',
        predict_steps=12,
        predict_modes=6,
    ),
    occ_head=dict(
        type='OccHead',
        n_future=4,
    ),
    planning_head=dict(
        type='PlanningHeadSingleMode',
        planning_steps=6,
    ),
)
""")
            config_path = f.name
        
        try:
            # Test loading
            model, cfg = load_uniad_model_from_text(config_path, device='cpu')
            
            # Verify model is created
            self.assertIsNotNone(model)
            self.assertTrue(isinstance(model, torch.nn.Module))
            
            # Verify it's a UniAD model
            self.assertEqual(cfg.model['type'], 'UniAD')
            
            # Check task loss weights
            self.assertIn('task_loss_weight', cfg.model)
            self.assertEqual(cfg.model['task_loss_weight']['track'], 1.0)
            
        finally:
            # Clean up
            os.unlink(config_path)
    
    def test_mock_model_forward(self):
        """Test that mock models can do forward pass"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
model = dict(
    type='UniAD',
    num_classes=10,
    embed_dims=128,
)
""")
            config_path = f.name
        
        try:
            # Load model
            model, cfg = load_uniad_model_from_text(config_path, device='cpu')
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 32, 32)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            # Verify output
            self.assertIsNotNone(output)
            if isinstance(output, dict):
                # UniAD returns dict of outputs
                self.assertIn('track', output)
            else:
                # UniADTrack returns tensor
                self.assertTrue(isinstance(output, torch.Tensor))
            
        finally:
            # Clean up
            os.unlink(config_path)
    
    def test_config_with_variables(self):
        """Test parsing config with variables"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
# Config with variables
_dim_ = 256
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200

model = dict(
    type='UniADTrack',
    embed_dims=_dim_,
    num_query=900,
    bev_h=bev_h_,
    bev_w=bev_w_,
)
""")
            config_path = f.name
        
        try:
            # Test loading
            model, cfg = load_uniad_model_from_text(config_path, device='cpu')
            
            # Verify variables are resolved
            self.assertEqual(cfg.model['embed_dims'], 256)
            self.assertEqual(cfg.model['bev_h'], 200)
            self.assertEqual(cfg.model['bev_w'], 200)
            
            # Verify model is created
            self.assertIsNotNone(model)
            
        finally:
            # Clean up
            os.unlink(config_path)


if __name__ == '__main__':
    unittest.main()