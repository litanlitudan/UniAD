"""Interactive visualization configuration for PyTorch Operation Tracer

This module provides configuration classes for interactive visualization features
including zoom, pan, search, filtering, and other UI interactions for the dataflow
visualization interface.
"""

from dataclasses import dataclass
from typing import Literal, Dict, Any


@dataclass
class InteractiveConfig:
    """Configuration for interactive dataflow visualization features
    
    This dataclass defines all interactive capabilities for the visualization interface,
    including navigation controls, search functionality, filtering options, and UI behaviors.
    Designed to support complex visualizations with hundreds or thousands of nodes while
    maintaining smooth performance.
    
    Attributes:
        enable_zoom: Enable zoom in/out functionality for detailed inspection
        enable_pan: Enable drag-to-pan navigation across large diagrams
        enable_search: Enable search functionality to find specific nodes/operations
        enable_filters: Enable filtering capabilities for focused analysis
        enable_tooltips: Enable hover tooltips with detailed node information
        enable_export: Enable export functionality (PNG, SVG, JSON)
        max_nodes_visible: Maximum nodes to display simultaneously for performance
        animation_duration_ms: Duration for smooth transitions and animations
        color_scheme: Visual color scheme for node categorization
    """
    
    # Core interaction features
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_search: bool = True
    enable_filters: bool = True
    enable_tooltips: bool = True
    enable_export: bool = True
    
    # Performance and display limits
    max_nodes_visible: int = 1000
    animation_duration_ms: int = 300
    
    # Visual styling
    color_scheme: Literal["memory", "operation", "temporal"] = "memory"
    
    def validate(self) -> bool:
        """Validate configuration parameters
        
        Returns:
            True if all parameters are valid, False otherwise
        """
        # Validate max_nodes_visible range
        if not (10 <= self.max_nodes_visible <= 10000):
            return False
        
        # Validate animation duration
        if not (0 <= self.animation_duration_ms <= 2000):
            return False
        
        # Validate color scheme
        valid_schemes = ["memory", "operation", "temporal"]
        if self.color_scheme not in valid_schemes:
            return False
        
        return True
    
    def get_performance_level(self) -> str:
        """Determine performance level based on max_nodes_visible
        
        Returns:
            Performance level: "high", "medium", or "low"
        """
        if self.max_nodes_visible >= 2000:
            return "high"
        elif self.max_nodes_visible >= 500:
            return "medium"
        else:
            return "low"
    
    def get_color_palette(self) -> Dict[str, str]:
        """Get color palette for the selected color scheme
        
        Returns:
            Dictionary mapping categories to hex colors
        """
        if self.color_scheme == "memory":
            return {
                "high_memory": "#ff4444",      # Red for high memory usage
                "medium_memory": "#ffaa00",    # Orange for medium memory
                "low_memory": "#44ff44",       # Green for low memory
                "no_memory": "#cccccc"         # Gray for no memory info
            }
        elif self.color_scheme == "operation":
            return {
                "conv": "#ff6b6b",             # Red for convolution operations
                "linear": "#4ecdc4",           # Teal for linear operations
                "attention": "#45b7d1",        # Blue for attention operations
                "activation": "#96ceb4",       # Light green for activations
                "normalization": "#feca57",    # Yellow for normalization
                "pooling": "#ff9ff3",          # Pink for pooling
                "other": "#95a5a6"             # Gray for other operations
            }
        elif self.color_scheme == "temporal":
            return {
                "current_frame": "#2ecc71",    # Green for current frame
                "past_frame": "#3498db",       # Blue for past frames
                "future_frame": "#9b59b6",     # Purple for future frames
                "temporal_fusion": "#e74c3c",  # Red for temporal fusion
                "static": "#34495e"            # Dark gray for non-temporal
            }
        
        # Default fallback
        return {"default": "#34495e"}
    
    def optimize_for_large_graphs(self):
        """Optimize settings for large graph visualization (>500 nodes)"""
        self.max_nodes_visible = 500
        self.animation_duration_ms = 150
        self.enable_tooltips = False  # Disable for performance
    
    def optimize_for_performance(self):
        """Optimize settings for maximum performance"""
        self.max_nodes_visible = 100
        self.animation_duration_ms = 0
        self.enable_tooltips = False
        self.enable_export = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'enable_zoom': self.enable_zoom,
            'enable_pan': self.enable_pan,
            'enable_search': self.enable_search,
            'enable_filters': self.enable_filters,
            'enable_tooltips': self.enable_tooltips,
            'enable_export': self.enable_export,
            'max_nodes_visible': self.max_nodes_visible,
            'animation_duration_ms': self.animation_duration_ms,
            'color_scheme': self.color_scheme,
            'performance_level': self.get_performance_level(),
            'color_palette': self.get_color_palette()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InteractiveConfig':
        """Create configuration from dictionary
        
        Args:
            data: Dictionary representation of configuration
            
        Returns:
            InteractiveConfig instance
        """
        config = cls()
        
        # Update fields if present in data
        for field_name in ['enable_zoom', 'enable_pan', 'enable_search', 
                          'enable_filters', 'enable_tooltips', 'enable_export']:
            if field_name in data:
                setattr(config, field_name, data[field_name])
        
        if 'max_nodes_visible' in data:
            config.max_nodes_visible = data['max_nodes_visible']
        if 'animation_duration_ms' in data:
            config.animation_duration_ms = data['animation_duration_ms']
        if 'color_scheme' in data:
            config.color_scheme = data['color_scheme']
        
        return config
    
    @classmethod
    def for_uniad_analysis(cls) -> 'InteractiveConfig':
        """Create optimized configuration for UniAD model analysis
        
        UniAD models are complex with 5 task heads and hierarchical architecture.
        This preset balances detail with performance for typical UniAD analysis.
        
        Returns:
            InteractiveConfig optimized for UniAD
        """
        return cls(
            enable_zoom=True,
            enable_pan=True,
            enable_search=True,
            enable_filters=True,
            enable_tooltips=True,
            enable_export=True,
            max_nodes_visible=800,  # Good balance for UniAD's complexity
            animation_duration_ms=250,
            color_scheme="memory"   # Memory is critical for UniAD (30-50GB)
        )
    
    @classmethod
    def for_large_scale_analysis(cls) -> 'InteractiveConfig':
        """Create configuration optimized for large-scale model analysis
        
        Returns:
            InteractiveConfig optimized for performance with large models
        """
        config = cls()
        config.optimize_for_large_graphs()
        return config