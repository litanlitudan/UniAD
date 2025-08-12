"""Configuration loader for visualization settings

This module handles loading and saving visualization configurations,
including user preferences, color schemes, and component settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict
import yaml

try:
    from ..core.visualization_config import InteractiveConfig
except (ImportError, ValueError):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.visualization_config import InteractiveConfig


class ConfigLoader:
    """Handles loading and saving of visualization configurations"""
    
    DEFAULT_CONFIG_DIR = Path.home() / '.uniad_tracer'
    DEFAULT_CONFIG_FILE = 'visualization_config.json'
    
    # Default configurations for different use cases
    DEFAULT_CONFIGS = {
        'default': {
            'enable_zoom_pan': True,
            'enable_search': True,
            'enable_filter': True,
            'enable_tooltips': True,
            'enable_highlighting': True,
            'max_nodes_visible': 100,
            'animation_duration_ms': 750,
            'color_scheme': 'default',
            'node_size_range': [20, 50],
            'edge_width_range': [1, 3],
            'layout_algorithm': 'force-directed',
            'enable_minimap': True,
            'enable_fullscreen': True,
            'enable_export': True,
            'export_formats': ['svg', 'png', 'json']
        },
        'performance': {
            'enable_zoom_pan': True,
            'enable_search': False,
            'enable_filter': True,
            'enable_tooltips': False,
            'enable_highlighting': True,
            'max_nodes_visible': 50,
            'animation_duration_ms': 0,
            'color_scheme': 'default',
            'node_size_range': [15, 30],
            'edge_width_range': [1, 2],
            'layout_algorithm': 'hierarchical',
            'enable_minimap': False,
            'enable_fullscreen': False,
            'enable_export': False,
            'export_formats': []
        },
        'presentation': {
            'enable_zoom_pan': True,
            'enable_search': False,
            'enable_filter': False,
            'enable_tooltips': True,
            'enable_highlighting': True,
            'max_nodes_visible': 75,
            'animation_duration_ms': 1000,
            'color_scheme': 'high-contrast',
            'node_size_range': [25, 60],
            'edge_width_range': [2, 4],
            'layout_algorithm': 'force-directed',
            'enable_minimap': True,
            'enable_fullscreen': True,
            'enable_export': True,
            'export_formats': ['svg', 'png']
        },
        'debug': {
            'enable_zoom_pan': True,
            'enable_search': True,
            'enable_filter': True,
            'enable_tooltips': True,
            'enable_highlighting': True,
            'max_nodes_visible': 500,
            'animation_duration_ms': 500,
            'color_scheme': 'default',
            'node_size_range': [15, 40],
            'edge_width_range': [1, 3],
            'layout_algorithm': 'force-directed',
            'enable_minimap': True,
            'enable_fullscreen': True,
            'enable_export': True,
            'export_formats': ['svg', 'png', 'json', 'html']
        }
    }
    
    # Color scheme definitions
    COLOR_SCHEMES = {
        'default': {
            'background': '#ffffff',
            'text': '#333333',
            'node_default': '#4a90e2',
            'node_hover': '#357abd',
            'edge_default': '#999999',
            'edge_highlight': '#ff6b35',
            'task_track': '#ff6b35',
            'task_seg': '#4ecdc4',
            'task_motion': '#45b7d1',
            'task_occ': '#95e77e',
            'task_planning': '#ffd93d',
            'bev': '#9b59b6'
        },
        'dark': {
            'background': '#1a1a1a',
            'text': '#ffffff',
            'node_default': '#61dafb',
            'node_hover': '#4fa8c5',
            'edge_default': '#666666',
            'edge_highlight': '#ff8c42',
            'task_track': '#ff8c42',
            'task_seg': '#6ee7d8',
            'task_motion': '#5cc7e0',
            'task_occ': '#a8f091',
            'task_planning': '#ffe066',
            'bev': '#b07cc6'
        },
        'colorblind': {
            'background': '#ffffff',
            'text': '#333333',
            'node_default': '#0173b2',
            'node_hover': '#005a8b',
            'edge_default': '#999999',
            'edge_highlight': '#de8f05',
            'task_track': '#de8f05',
            'task_seg': '#029e73',
            'task_motion': '#56b4e9',
            'task_occ': '#cc78bc',
            'task_planning': '#ece133',
            'bev': '#949494'
        },
        'high-contrast': {
            'background': '#000000',
            'text': '#ffffff',
            'node_default': '#00ff00',
            'node_hover': '#00cc00',
            'edge_default': '#808080',
            'edge_highlight': '#ff0000',
            'task_track': '#ff0000',
            'task_seg': '#00ffff',
            'task_motion': '#0080ff',
            'task_occ': '#ffff00',
            'task_planning': '#ff00ff',
            'bev': '#ffffff'
        }
    }
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration loader
        
        Args:
            config_dir: Optional directory for config files (default: ~/.uniad_tracer)
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self.config_file = self.config_dir / self.DEFAULT_CONFIG_FILE
        self._ensure_config_dir()
    
    def _ensure_config_dir(self):
        """Ensure configuration directory exists"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_visualization_config(self, config_path: Optional[str] = None,
                                 preset: Optional[str] = None) -> InteractiveConfig:
        """Load visualization configuration from file or preset
        
        Args:
            config_path: Optional path to config file
            preset: Optional preset name ('default', 'performance', 'presentation', 'debug')
            
        Returns:
            InteractiveConfig object with loaded settings
        """
        config_data = {}
        
        # 1. Start with default configuration
        config_data.update(self.DEFAULT_CONFIGS['default'])
        
        # 2. Apply preset if specified
        if preset and preset in self.DEFAULT_CONFIGS:
            config_data.update(self.DEFAULT_CONFIGS[preset])
        
        # 3. Load from default config file if it exists
        if self.config_file.exists() and not config_path:
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    config_data.update(user_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load default config: {e}")
        
        # 4. Load from specified config file if provided
        if config_path:
            config_data.update(self._load_config_file(config_path))
        
        # 5. Apply color scheme if specified
        if 'color_scheme' in config_data:
            config_data['colors'] = self.COLOR_SCHEMES.get(
                config_data['color_scheme'],
                self.COLOR_SCHEMES['default']
            )
        
        # Create and return InteractiveConfig
        return InteractiveConfig(**config_data)
    
    def _load_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format from extension
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def save_visualization_config(self, config: InteractiveConfig,
                                 config_path: Optional[str] = None) -> str:
        """Save visualization configuration to file
        
        Args:
            config: InteractiveConfig object to save
            config_path: Optional path to save to (default: user config file)
            
        Returns:
            Path where config was saved
        """
        save_path = Path(config_path) if config_path else self.config_file
        
        # Convert config to dictionary
        config_dict = asdict(config)
        
        # Remove color data (store only scheme name)
        if 'colors' in config_dict:
            del config_dict['colors']
        
        # Save based on file format
        if save_path.suffix in ['.yaml', '.yml']:
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            # Default to JSON
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        return str(save_path)
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from default config file
        
        Returns:
            Dictionary of user preferences
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return {}
    
    def save_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """Save user preferences to default config file
        
        Args:
            preferences: Dictionary of preferences to save
        """
        # Merge with existing preferences
        existing = self.get_user_preferences()
        existing.update(preferences)
        
        # Save to file
        with open(self.config_file, 'w') as f:
            json.dump(existing, f, indent=2)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        if self.config_file.exists():
            self.config_file.unlink()
    
    def list_presets(self) -> Dict[str, str]:
        """List available configuration presets
        
        Returns:
            Dictionary of preset names and descriptions
        """
        return {
            'default': 'Balanced configuration for general use',
            'performance': 'Optimized for large graphs with minimal features',
            'presentation': 'Enhanced visuals for presentations and demos',
            'debug': 'Full features enabled for debugging and analysis'
        }
    
    def list_color_schemes(self) -> Dict[str, str]:
        """List available color schemes
        
        Returns:
            Dictionary of color scheme names and descriptions
        """
        return {
            'default': 'Standard blue-based color scheme',
            'dark': 'Dark theme for reduced eye strain',
            'colorblind': 'Colorblind-friendly palette',
            'high-contrast': 'High contrast for accessibility'
        }
    
    def export_config(self, config: InteractiveConfig, format: str = 'json') -> str:
        """Export configuration as string
        
        Args:
            config: InteractiveConfig to export
            format: Export format ('json' or 'yaml')
            
        Returns:
            Configuration as formatted string
        """
        config_dict = asdict(config)
        
        if format == 'yaml':
            return yaml.dump(config_dict, default_flow_style=False)
        else:
            return json.dumps(config_dict, indent=2)
    
    def merge_configs(self, base_config: InteractiveConfig,
                     overrides: Dict[str, Any]) -> InteractiveConfig:
        """Merge configuration with overrides
        
        Args:
            base_config: Base InteractiveConfig
            overrides: Dictionary of values to override
            
        Returns:
            New InteractiveConfig with merged values
        """
        config_dict = asdict(base_config)
        config_dict.update(overrides)
        
        # Apply color scheme if changed
        if 'color_scheme' in overrides:
            config_dict['colors'] = self.COLOR_SCHEMES.get(
                overrides['color_scheme'],
                self.COLOR_SCHEMES['default']
            )
        
        return InteractiveConfig(**config_dict)


def load_visualization_config(config_path: Optional[str] = None,
                            preset: Optional[str] = None) -> InteractiveConfig:
    """Convenience function to load visualization configuration
    
    Args:
        config_path: Optional path to config file
        preset: Optional preset name
        
    Returns:
        InteractiveConfig object
    """
    loader = ConfigLoader()
    return loader.load_visualization_config(config_path, preset)


def save_visualization_config(config: InteractiveConfig,
                            config_path: Optional[str] = None) -> str:
    """Convenience function to save visualization configuration
    
    Args:
        config: InteractiveConfig to save
        config_path: Optional path to save to
        
    Returns:
        Path where config was saved
    """
    loader = ConfigLoader()
    return loader.save_visualization_config(config, config_path)