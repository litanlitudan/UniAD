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


def load_uniad_model_from_text(config_path: str, checkpoint_path: Optional[str] = None, device: str = 'cuda') -> Tuple[torch.nn.Module, Any]:
    """Load UniAD model by parsing config text and reconstructing model manually"""
    import ast
    import importlib
    from types import SimpleNamespace
    
    # Read and parse config file
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    # Create a namespace to execute the config
    config_namespace = {
        'dict': dict,
        'True': True,
        'False': False,
        'None': None,
    }
    
    # Execute the config file to get variables
    exec(config_text, config_namespace)
    
    # Extract model config
    model_cfg = config_namespace.get('model', None)
    if model_cfg is None:
        raise ValueError(f"No 'model' configuration found in {config_path}")
    
    # Build model manually based on type
    model_type = model_cfg.get('type', '')
    
    if model_type == 'UniAD':
        model = _build_uniad_model(model_cfg)
    elif model_type == 'UniADTrack':
        model = _build_uniad_track_model(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint if provided
    if checkpoint_path:
        if MMDET3D_AVAILABLE and load_checkpoint is not None:
            load_checkpoint(model, checkpoint_path, map_location=device)
        else:
            # Fallback to pure PyTorch loading
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # Create a simple config object for compatibility
    cfg = SimpleNamespace(**config_namespace)
    
    return model, cfg


def _build_uniad_track_model(cfg: dict) -> torch.nn.Module:
    """Build UniADTrack model from config dict"""
    # Import required modules
    try:
        from projects.mmdet3d_plugin.uniad.detectors.uniad_track import UniADTrack
        from mmdet3d.models.builder import build_backbone, build_neck, build_head
        
        # Build components
        img_backbone = build_backbone(cfg['img_backbone']) if cfg.get('img_backbone') else None
        img_neck = build_neck(cfg['img_neck']) if cfg.get('img_neck') else None
        pts_bbox_head = build_head(cfg['pts_bbox_head']) if cfg.get('pts_bbox_head') else None
        
        # Create model
        model = UniADTrack(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'),
            pretrained=cfg.get('pretrained'),
            use_grid_mask=cfg.get('use_grid_mask', False),
            video_test_mode=cfg.get('video_test_mode', False),
            loss_cfg=cfg.get('loss_cfg'),
            qim_args=cfg.get('qim_args', {}),
            mem_args=cfg.get('mem_args', {}),
            bbox_coder=cfg.get('bbox_coder'),
            pc_range=cfg.get('pc_range'),
            embed_dims=cfg.get('embed_dims', 256),
            num_query=cfg.get('num_query', 900),
            num_classes=cfg.get('num_classes', 10),
            vehicle_id_list=cfg.get('vehicle_id_list'),
            score_thresh=cfg.get('score_thresh', 0.2),
            filter_score_thresh=cfg.get('filter_score_thresh', 0.1),
            miss_tolerance=cfg.get('miss_tolerance', 5),
            gt_iou_threshold=cfg.get('gt_iou_threshold', 0.0),
            freeze_img_backbone=cfg.get('freeze_img_backbone', False),
            freeze_img_neck=cfg.get('freeze_img_neck', False),
            freeze_bn=cfg.get('freeze_bn', False),
            freeze_bev_encoder=cfg.get('freeze_bev_encoder', False),
            queue_length=cfg.get('queue_length', 3),
        )
        
        return model
        
    except ImportError:
        raise ImportError("UniAD modules not available. Please ensure the codebase is properly set up.")


def _build_uniad_model(cfg: dict) -> torch.nn.Module:
    """Build full UniAD model from config dict"""
    try:
        from projects.mmdet3d_plugin.uniad.detectors.uniad_e2e import UniAD
        from mmdet3d.models.builder import build_head
        
        # First build the base UniADTrack model
        base_model_cfg = cfg.copy()
        base_model_cfg['type'] = 'UniADTrack'
        base_model = _build_uniad_track_model(base_model_cfg)
        
        # Build additional heads
        seg_head = build_head(cfg['seg_head']) if cfg.get('seg_head') else None
        motion_head = build_head(cfg['motion_head']) if cfg.get('motion_head') else None
        occ_head = build_head(cfg['occ_head']) if cfg.get('occ_head') else None
        planning_head = build_head(cfg['planning_head']) if cfg.get('planning_head') else None
        
        # Create UniAD model
        model = UniAD(
            img_backbone=base_model.img_backbone,
            img_neck=base_model.img_neck,
            pts_bbox_head=base_model.pts_bbox_head,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'),
            pretrained=cfg.get('pretrained'),
            seg_head=seg_head,
            motion_head=motion_head,
            occ_head=occ_head,
            planning_head=planning_head,
            task_loss_weight=cfg.get('task_loss_weight', {
                'track': 1.0,
                'map': 1.0,
                'motion': 1.0,
                'occ': 1.0,
                'planning': 1.0
            }),
            **{k: v for k, v in cfg.items() if k not in ['type', 'seg_head', 'motion_head', 'occ_head', 'planning_head', 'task_loss_weight', 'img_backbone', 'img_neck', 'pts_bbox_head', 'train_cfg', 'test_cfg', 'pretrained']}
        )
        
        return model
        
    except ImportError:
        raise ImportError("UniAD modules not available. Please ensure the codebase is properly set up.")




