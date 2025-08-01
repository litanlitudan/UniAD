"""Data type analyzer for PyTorch operations"""

from typing import Dict, List, Any, Optional
from collections import defaultdict

try:
    from ..core.data_structures import TraceNode, TensorInfo, PYTORCH_DTYPES, UNIAD_DTYPE_CONFIGS
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode, TensorInfo, PYTORCH_DTYPES, UNIAD_DTYPE_CONFIGS


class DtypeAnalyzer:
    """Analyzes data type usage and conversions in the model"""
    
    def __init__(self):
        self.dtype_counts = defaultdict(int)
        self.dtype_conversions = []
        self.module_dtypes = defaultdict(set)
        self.memory_by_dtype = defaultdict(float)
    
    def analyze_dtype_usage(self, trace_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Analyze data type usage across all operations
        
        Args:
            trace_nodes: List of traced operations
            
        Returns:
            Analysis results including dtype distribution, conversions, and memory impact
        """
        self._reset()
        
        for node in trace_nodes:
            self._analyze_node_dtypes(node)
        
        return {
            'dtype_distribution': dict(self.dtype_counts),
            'dtype_conversions': self.dtype_conversions[:50],  # Top 50 conversions
            'memory_by_dtype': dict(self.memory_by_dtype),
            'module_dtype_usage': self._get_module_dtype_summary(),
            'optimization_opportunities': self._identify_optimization_opportunities(),
            'mixed_precision_analysis': self._analyze_mixed_precision_potential()
        }
    
    def _reset(self):
        """Reset internal state"""
        self.dtype_counts.clear()
        self.dtype_conversions.clear()
        self.module_dtypes.clear()
        self.memory_by_dtype.clear()
    
    def _analyze_node_dtypes(self, node: TraceNode):
        """Analyze dtypes for a single node"""
        # Count input dtypes
        for tensor_info in node.input_shapes:
            if hasattr(tensor_info, 'dtype'):
                self.dtype_counts[tensor_info.dtype] += 1
                self.module_dtypes[node.module_path].add(tensor_info.dtype)
                
                # Calculate memory contribution
                memory_mb = tensor_info.memory_size() / (1024 * 1024)
                self.memory_by_dtype[tensor_info.dtype] += memory_mb
        
        # Check for dtype conversions
        if node.input_shapes and node.output_shapes:
            input_dtypes = {info.dtype for info in node.input_shapes if hasattr(info, 'dtype')}
            output_dtypes = {info.dtype for info in node.output_shapes if hasattr(info, 'dtype')}
            
            # Record conversions
            if input_dtypes and output_dtypes and input_dtypes != output_dtypes:
                for in_dtype in input_dtypes:
                    for out_dtype in output_dtypes:
                        if in_dtype != out_dtype:
                            self.dtype_conversions.append({
                                'module': node.module_path,
                                'operation': node.operation,
                                'from_dtype': in_dtype,
                                'to_dtype': out_dtype,
                                'memory_impact': node.memory_usage
                            })
    
    def _get_module_dtype_summary(self) -> Dict[str, List[str]]:
        """Get summary of dtypes used by each module"""
        summary = {}
        for module, dtypes in self.module_dtypes.items():
            # Group modules by major component
            component = module.split('.')[0] if '.' in module else module
            if component not in summary:
                summary[component] = []
            summary[component].extend(list(dtypes))
        
        # Deduplicate
        for component in summary:
            summary[component] = list(set(summary[component]))
        
        return summary
    
    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for dtype optimization"""
        opportunities = []
        
        # Check for unnecessary FP32 usage
        total_fp32_memory = self.memory_by_dtype.get('float32', 0) + self.memory_by_dtype.get('fp32', 0)
        if total_fp32_memory > 1000:  # More than 1GB in FP32
            opportunities.append({
                'type': 'mixed_precision',
                'description': f'Consider mixed precision: {total_fp32_memory:.1f}MB currently in FP32',
                'potential_savings': f'{total_fp32_memory * 0.5:.1f}MB with FP16/BF16'
            })
        
        # Check for quantization opportunities
        if self.dtype_counts.get('float32', 0) > 100:
            opportunities.append({
                'type': 'quantization',
                'description': 'Consider INT8 quantization for inference',
                'potential_savings': f'{total_fp32_memory * 0.75:.1f}MB reduction possible'
            })
        
        # Check for excessive conversions
        if len(self.dtype_conversions) > 10:
            opportunities.append({
                'type': 'conversion_reduction',
                'description': f'Found {len(self.dtype_conversions)} dtype conversions',
                'recommendation': 'Consider consistent dtype usage to reduce conversions'
            })
        
        return opportunities
    
    def _analyze_mixed_precision_potential(self) -> Dict[str, Any]:
        """Analyze potential for mixed precision training"""
        analysis = {
            'current_memory_mb': sum(self.memory_by_dtype.values()),
            'configurations': {}
        }
        
        # Calculate memory for each configuration
        for config_name, config in UNIAD_DTYPE_CONFIGS.items():
            if config_name == 'default':
                continue
                
            # Estimate memory savings
            estimated_memory = 0
            for component, target_dtype in config.items():
                # Simple estimation based on dtype reduction
                if 'backbone' in component.lower():
                    component_memory = self.memory_by_dtype.get('float32', 0) * 0.4
                elif 'bev' in component.lower():
                    component_memory = self.memory_by_dtype.get('float32', 0) * 0.3
                else:
                    component_memory = self.memory_by_dtype.get('float32', 0) * 0.3
                
                if 'int8' in target_dtype:
                    estimated_memory += component_memory * 0.25
                elif 'float16' in target_dtype or 'bfloat16' in target_dtype:
                    estimated_memory += component_memory * 0.5
                else:
                    estimated_memory += component_memory
            
            analysis['configurations'][config_name] = {
                'estimated_memory_mb': estimated_memory,
                'reduction_percent': (1 - estimated_memory / analysis['current_memory_mb']) * 100 if analysis['current_memory_mb'] > 0 else 0
            }
        
        return analysis
    
    def generate_dtype_report(self, trace_nodes: List[TraceNode]) -> str:
        """Generate a detailed dtype analysis report"""
        analysis = self.analyze_dtype_usage(trace_nodes)
        
        lines = ["## Data Type Analysis Report", ""]
        
        # Distribution
        lines.append("### Data Type Distribution")
        lines.append("| Data Type | Count | Memory (MB) |")
        lines.append("|-----------|-------|-------------|")
        for dtype, count in sorted(analysis['dtype_distribution'].items(), key=lambda x: x[1], reverse=True):
            memory = analysis['memory_by_dtype'].get(dtype, 0)
            lines.append(f"| {dtype} | {count} | {memory:.1f} |")
        lines.append("")
        
        # Conversions
        if analysis['dtype_conversions']:
            lines.append("### Data Type Conversions")
            lines.append("| Module | Operation | From → To | Memory Impact |")
            lines.append("|--------|-----------|-----------|---------------|")
            for conv in analysis['dtype_conversions'][:20]:
                module = conv['module'][:30]
                op = conv['operation'][:20]
                lines.append(f"| {module} | {op} | {conv['from_dtype']} → {conv['to_dtype']} | {conv['memory_impact']:.1f}MB |")
            lines.append("")
        
        # Optimization opportunities
        if analysis['optimization_opportunities']:
            lines.append("### Optimization Opportunities")
            for opp in analysis['optimization_opportunities']:
                lines.append(f"- **{opp['type']}**: {opp['description']}")
                if 'potential_savings' in opp:
                    lines.append(f"  - Potential savings: {opp['potential_savings']}")
                if 'recommendation' in opp:
                    lines.append(f"  - Recommendation: {opp['recommendation']}")
            lines.append("")
        
        # Mixed precision analysis
        mp_analysis = analysis['mixed_precision_analysis']
        lines.append("### Mixed Precision Potential")
        lines.append(f"Current memory usage: {mp_analysis['current_memory_mb']:.1f}MB")
        lines.append("")
        lines.append("| Configuration | Estimated Memory | Reduction |")
        lines.append("|---------------|-----------------|-----------|")
        for config, stats in mp_analysis['configurations'].items():
            lines.append(f"| {config} | {stats['estimated_memory_mb']:.1f}MB | {stats['reduction_percent']:.1f}% |")
        
        return '\n'.join(lines)