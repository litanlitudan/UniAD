"""Module hierarchy analysis for hierarchical visualization"""

from typing import Dict, List, Optional, Set, Any
import torch
import torch.nn as nn


class ModuleHierarchyAnalyzer:
    """Analyzes module hierarchy for intelligent visualization"""
    
    def __init__(self, model: nn.Module):
        """Initialize analyzer with a PyTorch model
        
        Args:
            model: PyTorch model to analyze
        """
        self.model = model
        self.module_tree = self._build_module_tree()
        self.module_info_cache = {}
        
    def _build_module_tree(self) -> Dict[str, Any]:
        """Build hierarchical tree of modules
        
        Returns:
            Nested dict representing module hierarchy
        """
        tree = {}
        
        for name, module in self.model.named_modules():
            if not name:  # Skip root module
                continue
                
            parts = name.split('.')
            current = tree
            
            # Build nested structure
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {
                        '_module': None,
                        '_children': {},
                        '_full_name': '.'.join(parts[:i+1])
                    }
                current = current[part]['_children']
            
            # Store module reference at leaf
            parent = tree
            for part in parts[:-1]:
                parent = parent[part]['_children']
            parent[parts[-1]]['_module'] = module
            
        return tree
    
    def get_top_level_modules(self) -> List[str]:
        """Return list of top-level module names
        
        Returns:
            List of module names at the top level
        """
        return [name for name in self.module_tree.keys() 
                if not name.startswith('_')]
    
    def get_module_info(self, module_path: str) -> Dict[str, Any]:
        """Get aggregated info for a module
        
        Args:
            module_path: Dot-separated module path (e.g., 'backbone.layer1')
            
        Returns:
            Dict with module information including memory, operations, etc.
        """
        if module_path in self.module_info_cache:
            return self.module_info_cache[module_path]
        
        # Navigate to module in tree
        parts = module_path.split('.')
        current = self.module_tree
        
        try:
            for part in parts:
                current = current[part]['_children']
        except KeyError:
            return {
                'exists': False,
                'memory': 0,
                'flops': 0,
                'operations': 0,
                'has_children': False
            }
        
        # Get module reference
        module_ref = None
        parent = self.module_tree
        for part in parts[:-1]:
            parent = parent[part]['_children']
        if parts[-1] in parent:
            module_ref = parent[parts[-1]]['_module']
        
        # Calculate module info
        info = {
            'exists': True,
            'memory': self._estimate_module_memory(module_ref),
            'parameters': self._count_parameters(module_ref),
            'operations': len(list(module_ref.modules())) if module_ref else 0,
            'has_children': bool(current) if isinstance(current, dict) else False,
            'module_type': module_ref.__class__.__name__ if module_ref else None
        }
        
        self.module_info_cache[module_path] = info
        return info
    
    def get_module_children(self, module_path: str) -> List[str]:
        """Get immediate children of a module
        
        Args:
            module_path: Dot-separated module path
            
        Returns:
            List of child module names
        """
        parts = module_path.split('.') if module_path else []
        current = self.module_tree
        
        try:
            for part in parts:
                current = current[part]['_children']
        except KeyError:
            return []
        
        return [name for name in current.keys() 
                if not name.startswith('_')]
    
    def get_modules_by_pattern(self, pattern: str) -> List[str]:
        """Get modules matching a pattern
        
        Args:
            pattern: Pattern to match (supports * wildcard)
            
        Returns:
            List of module paths matching the pattern
        """
        import fnmatch
        
        matching_modules = []
        
        def traverse(node: Dict, current_path: str = ""):
            for name, info in node.items():
                if name.startswith('_'):
                    continue
                    
                full_path = f"{current_path}.{name}" if current_path else name
                
                # Check if matches pattern
                if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(full_path, pattern):
                    matching_modules.append(full_path)
                
                # Recurse into children
                if '_children' in info:
                    traverse(info['_children'], full_path)
        
        traverse(self.module_tree)
        return matching_modules
    
    def get_modules_by_memory_threshold(self, threshold_mb: float) -> List[str]:
        """Get modules using more memory than threshold
        
        Args:
            threshold_mb: Memory threshold in MB
            
        Returns:
            List of module paths exceeding threshold
        """
        heavy_modules = []
        
        def traverse(node: Dict, current_path: str = ""):
            for name, info in node.items():
                if name.startswith('_'):
                    continue
                    
                full_path = f"{current_path}.{name}" if current_path else name
                module_info = self.get_module_info(full_path)
                
                if module_info['memory'] > threshold_mb:
                    heavy_modules.append(full_path)
                
                # Recurse into children
                if '_children' in info:
                    traverse(info['_children'], full_path)
        
        traverse(self.module_tree)
        return heavy_modules
    
    def get_module_depth(self, module_path: str) -> int:
        """Get depth of a module in the hierarchy
        
        Args:
            module_path: Dot-separated module path
            
        Returns:
            Depth level (0 for top-level)
        """
        return len(module_path.split('.')) if module_path else 0
    
    def get_modules_at_depth(self, depth: int) -> List[str]:
        """Get all modules at a specific depth
        
        Args:
            depth: Target depth level
            
        Returns:
            List of module paths at the specified depth
        """
        modules_at_depth = []
        
        def traverse(node: Dict, current_path: str = "", current_depth: int = 0):
            if current_depth == depth:
                if current_path:
                    modules_at_depth.append(current_path)
                return
            
            for name, info in node.items():
                if name.startswith('_'):
                    continue
                    
                full_path = f"{current_path}.{name}" if current_path else name
                
                # Recurse into children
                if '_children' in info:
                    traverse(info['_children'], full_path, current_depth + 1)
        
        traverse(self.module_tree)
        return modules_at_depth
    
    def _estimate_module_memory(self, module: Optional[nn.Module]) -> float:
        """Estimate memory usage of a module in MB
        
        Args:
            module: PyTorch module
            
        Returns:
            Estimated memory in MB
        """
        if module is None:
            return 0.0
        
        total_bytes = 0
        
        # Count parameters
        for param in module.parameters():
            total_bytes += param.numel() * param.element_size()
        
        # Count buffers
        for buffer in module.buffers():
            total_bytes += buffer.numel() * buffer.element_size()
        
        return total_bytes / (1024 * 1024)
    
    def _count_parameters(self, module: Optional[nn.Module]) -> int:
        """Count total parameters in a module
        
        Args:
            module: PyTorch module
            
        Returns:
            Total number of parameters
        """
        if module is None:
            return 0
        
        return sum(p.numel() for p in module.parameters())
    
    def get_module_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the model hierarchy
        
        Returns:
            Dict with summary statistics
        """
        all_modules = []
        
        def traverse(node: Dict, current_path: str = ""):
            for name, info in node.items():
                if name.startswith('_'):
                    continue
                    
                full_path = f"{current_path}.{name}" if current_path else name
                all_modules.append(full_path)
                
                if '_children' in info:
                    traverse(info['_children'], full_path)
        
        traverse(self.module_tree)
        
        # Calculate statistics
        depths = [self.get_module_depth(m) for m in all_modules]
        
        return {
            'total_modules': len(all_modules),
            'max_depth': max(depths) if depths else 0,
            'top_level_modules': len(self.get_top_level_modules()),
            'module_distribution': {
                f'depth_{d}': len(self.get_modules_at_depth(d))
                for d in range(max(depths) + 1) if depths
            }
        }