"""Model utilities for UniAD"""

import torch
from typing import Optional, Tuple, Any, Dict, Union

# Try importing UniAD components
try:
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    MMDET3D_AVAILABLE = True
except ImportError:
    MMDET3D_AVAILABLE = False
    Config = None
    build_model = None
    load_checkpoint = None


def create_dummy_input(config: Optional[Any] = None, device: str = 'cuda') -> Union[Dict[str, Any], torch.Tensor]:
    """Create dummy input for model tracing"""
    batch_size = 1

    if MMDET3D_AVAILABLE and config is not None:
        # Create proper input dict for UniAD
        dummy_input = {
            'img': torch.randn(batch_size, 6, 3, 928, 1600).to(device),
            'img_metas': [[{
                'lidar2img': torch.eye(4).unsqueeze(0).repeat(6, 1, 1).numpy(),
                'can_bus': torch.zeros(18).numpy(),
                'scene_token': 'dummy_scene',
                'timestamp': 0.0
            }]]
        }
    else:
        # Fallback for testing without mmdet3d
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    return dummy_input


def load_uniad_model(config_path: str, checkpoint_path: Optional[str] = None, device: str = 'cuda') -> Tuple[torch.nn.Module, Any]:
    """Load UniAD model from config and checkpoint"""
    if not MMDET3D_AVAILABLE:
        raise ImportError("mmdet3d is not available. Please install it to use UniAD models.")

    # Assert that imports are available (helps with type checking)
    assert Config is not None, "Config should be available when MMDET3D_AVAILABLE is True"
    assert build_model is not None, "build_model should be available when MMDET3D_AVAILABLE is True"
    assert load_checkpoint is not None, "load_checkpoint should be available when MMDET3D_AVAILABLE is True"

    # Load config
    cfg = Config.fromfile(config_path)

    # Check if model config exists
    if not hasattr(cfg, 'model'):
        raise ValueError(f"No 'model' configuration found in {config_path}")

    # Build model
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Load checkpoint if provided
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, map_location=device)

    model = model.to(device)
    model.eval()

    return model, cfg