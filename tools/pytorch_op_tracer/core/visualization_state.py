"""Visualization state management for hierarchical views"""

from typing import Set, Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class VisualizationState:
    """Manages visualization state for hierarchical module views"""
    
    # Visualization mode
    visualization_mode: str = 'top-level'  # 'top-level', 'expanded', 'full'
    
    # Module expansion
    expanded_modules: Set[str] = field(default_factory=set)
    depth_limit: Optional[int] = None
    memory_threshold: Optional[float] = None  # MB
    
    # Shape display
    show_shapes: bool = True
    shape_format: str = 'full'  # 'full', 'compact', 'semantic'
    show_dtype: bool = True  # Show data type (fp32, fp16, etc.)
    
    # Memory display
    show_memory: bool = True
    memory_format: str = 'auto'  # 'auto', 'MB', 'GB'
    
    # Performance display
    show_timing: bool = False
    show_flops: bool = False
    
    # Filtering
    filter_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    
    # Shape transformation highlighting
    highlight_shape_changes: bool = False
    shape_change_threshold: float = 0.5  # Ratio of dimension change to highlight
    
    # Task-specific settings
    group_by_task: bool = True
    task_colors: Dict[str, str] = field(default_factory=lambda: {
        'track': '#ff9999',
        'seg': '#99ccff',
        'motion': '#9999ff',
        'occ': '#ffcc99',
        'planning': '#99ff99',
        'bev': '#ffccff',
        'backbone': '#e1e1e1'
    })
    
    def should_expand_module(self, module_name: str, module_info: Dict[str, Any]) -> bool:
        """Determine if a module should be expanded
        
        Args:
            module_name: Name of the module
            module_info: Information about the module (memory, operations, etc.)
            
        Returns:
            True if module should be expanded
        """
        # Full mode expands everything
        if self.visualization_mode == 'full':
            return True
        
        # Top-level mode expands nothing by default
        if self.visualization_mode == 'top-level' and module_name not in self.expanded_modules:
            return False
        
        # Check explicit expansion list
        if module_name in self.expanded_modules:
            return True
        
        # Check memory threshold
        if self.memory_threshold and module_info.get('memory', 0) > self.memory_threshold:
            return True
        
        # Check depth limit
        if self.depth_limit is not None:
            depth = len(module_name.split('.'))
            if depth <= self.depth_limit:
                return True
        
        return False
    
    def toggle_module_expansion(self, module_name: str):
        """Toggle expansion state of a module
        
        Args:
            module_name: Name of the module to toggle
        """
        if module_name in self.expanded_modules:
            self.expanded_modules.remove(module_name)
        else:
            self.expanded_modules.add(module_name)
    
    def expand_modules_by_pattern(self, pattern: str):
        """Add modules matching pattern to expanded set
        
        Args:
            pattern: Pattern to match (supports * wildcard)
        """
        # This would be used with ModuleHierarchyAnalyzer
        # to find matching modules
        pass
    
    def set_auto_expansion_by_memory(self, threshold_mb: float):
        """Set automatic expansion for modules using more than threshold memory
        
        Args:
            threshold_mb: Memory threshold in MB
        """
        self.memory_threshold = threshold_mb
    
    def format_memory(self, memory_mb: float) -> str:
        """Format memory value for display
        
        Args:
            memory_mb: Memory in MB
            
        Returns:
            Formatted string
        """
        if self.memory_format == 'GB' or (self.memory_format == 'auto' and memory_mb >= 1024):
            return f"{memory_mb / 1024:.1f}GB"
        else:
            return f"{memory_mb:.1f}MB"
    
    def should_filter_module(self, module_name: str) -> bool:
        """Check if module should be filtered out
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if module should be filtered out
        """
        import fnmatch
        
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(module_name, pattern):
                return True
        
        # If filter patterns exist, module must match at least one
        if self.filter_patterns:
            for pattern in self.filter_patterns:
                if fnmatch.fnmatch(module_name, pattern):
                    return False
            return True  # Didn't match any filter pattern
        
        return False  # No filtering
    
    def get_module_style(self, module_name: str, task_head: Optional[str] = None) -> Dict[str, str]:
        """Get visualization style for a module
        
        Args:
            module_name: Name of the module
            task_head: Associated task head (if any)
            
        Returns:
            Style dict for Mermaid diagram
        """
        style = {}
        
        # Apply task-specific colors
        if self.group_by_task and task_head and task_head in self.task_colors:
            style['fill'] = self.task_colors[task_head]
        elif 'bev' in module_name.lower():
            style['fill'] = self.task_colors.get('bev', '#ffccff')
        elif 'backbone' in module_name.lower():
            style['fill'] = self.task_colors.get('backbone', '#e1e1e1')
        
        # Add border style for expanded modules
        if module_name in self.expanded_modules:
            style['stroke'] = '#333'
            style['stroke-width'] = '3px'
        else:
            style['stroke'] = '#666'
            style['stroke-width'] = '2px'
        
        return style
    
    def reset(self):
        """Reset visualization state to defaults"""
        self.visualization_mode = 'top-level'
        self.expanded_modules.clear()
        self.depth_limit = None
        self.memory_threshold = None
        self.filter_patterns.clear()
        self.exclude_patterns.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization
        
        Returns:
            Dict representation of state
        """
        return {
            'visualization_mode': self.visualization_mode,
            'expanded_modules': list(self.expanded_modules),
            'depth_limit': self.depth_limit,
            'memory_threshold': self.memory_threshold,
            'show_shapes': self.show_shapes,
            'shape_format': self.shape_format,
            'show_memory': self.show_memory,
            'memory_format': self.memory_format,
            'show_timing': self.show_timing,
            'show_flops': self.show_flops,
            'filter_patterns': self.filter_patterns,
            'exclude_patterns': self.exclude_patterns,
            'highlight_shape_changes': self.highlight_shape_changes,
            'shape_change_threshold': self.shape_change_threshold,
            'group_by_task': self.group_by_task,
            'task_colors': self.task_colors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualizationState':
        """Create state from dictionary
        
        Args:
            data: Dict representation
            
        Returns:
            VisualizationState instance
        """
        state = cls()
        
        if 'visualization_mode' in data:
            state.visualization_mode = data['visualization_mode']
        if 'expanded_modules' in data:
            state.expanded_modules = set(data['expanded_modules'])
        if 'depth_limit' in data:
            state.depth_limit = data['depth_limit']
        if 'memory_threshold' in data:
            state.memory_threshold = data['memory_threshold']
        if 'show_shapes' in data:
            state.show_shapes = data['show_shapes']
        if 'shape_format' in data:
            state.shape_format = data['shape_format']
        if 'show_memory' in data:
            state.show_memory = data['show_memory']
        if 'memory_format' in data:
            state.memory_format = data['memory_format']
        if 'show_timing' in data:
            state.show_timing = data['show_timing']
        if 'show_flops' in data:
            state.show_flops = data['show_flops']
        if 'filter_patterns' in data:
            state.filter_patterns = data['filter_patterns']
        if 'exclude_patterns' in data:
            state.exclude_patterns = data['exclude_patterns']
        if 'highlight_shape_changes' in data:
            state.highlight_shape_changes = data['highlight_shape_changes']
        if 'shape_change_threshold' in data:
            state.shape_change_threshold = data['shape_change_threshold']
        if 'group_by_task' in data:
            state.group_by_task = data['group_by_task']
        if 'task_colors' in data:
            state.task_colors = data['task_colors']
        
        return state