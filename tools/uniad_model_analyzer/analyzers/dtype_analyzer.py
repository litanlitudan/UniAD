"""
DType Analyzer for UniAD Model Analyzer.

This module analyzes data type usage in UniAD models, identifies mixed precision
opportunities, and provides recommendations for memory and performance optimization.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import torch

from core.data_structures import TraceNode, TaskHead, TensorShape


@dataclass
class DTypeProfile:
    """Profile for a specific data type usage."""
    
    dtype: torch.dtype
    operation_count: int = 0
    tensor_count: int = 0
    total_memory_bytes: int = 0
    
    # Shape statistics
    min_shape: Optional[Tuple[int, ...]] = None
    max_shape: Optional[Tuple[int, ...]] = None
    common_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    
    # Operation breakdown
    operations: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    modules: Set[str] = field(default_factory=set)
    
    # Performance metrics
    total_duration_ns: float = 0.0
    avg_operation_time_ns: float = 0.0
    
    # Conversion compatibility
    fp16_compatible: bool = False
    int8_compatible: bool = False
    bf16_compatible: bool = False
    
    def add_operation(self, node: TraceNode, shape: TensorShape) -> None:
        """Add an operation to this dtype profile."""
        self.operation_count += 1
        self.tensor_count += 1
        self.operations[node.name] += 1
        self.modules.add(node.module_path)
        self.total_duration_ns += node.duration
        
        # Update memory
        elements = np.prod(shape.shape) if shape.shape else 0
        self.total_memory_bytes += elements * self._get_dtype_size(shape.dtype)
        
        # Update shape statistics
        if self.min_shape is None or len(shape.shape) < len(self.min_shape):
            self.min_shape = shape.shape
        if self.max_shape is None or len(shape.shape) > len(self.max_shape):
            self.max_shape = shape.shape
        
        if shape.shape not in self.common_shapes:
            self.common_shapes.append(shape.shape)
    
    def finalize(self) -> None:
        """Finalize profile calculations."""
        if self.operation_count > 0:
            self.avg_operation_time_ns = self.total_duration_ns / self.operation_count
        
        # Keep only top 10 most common shapes
        if len(self.common_shapes) > 10:
            # Sort by frequency (would need actual counting for accuracy)
            self.common_shapes = self.common_shapes[:10]
    
    @staticmethod
    def _get_dtype_size(dtype: torch.dtype) -> int:
        """Get size in bytes for a dtype."""
        dtype_sizes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float64: 8,
            torch.int32: 4,
            torch.int64: 8,
            torch.int16: 2,
            torch.int8: 1,
            torch.uint8: 1,
            torch.bool: 1,
        }
        return dtype_sizes.get(dtype, 4)


@dataclass
class MixedPrecisionOpportunity:
    """Represents a mixed precision optimization opportunity."""
    
    module_path: str
    current_dtype: torch.dtype
    target_dtype: torch.dtype
    
    # Impact metrics
    memory_savings_mb: float
    speedup_factor: float
    accuracy_impact: str  # 'none', 'minimal', 'moderate', 'significant'
    
    # Implementation details
    conversion_method: str  # 'automatic', 'manual', 'amp'
    implementation_code: str
    
    # Risk assessment
    risk_level: str  # 'low', 'medium', 'high'
    considerations: List[str] = field(default_factory=list)


@dataclass
class QuantizationCandidate:
    """Candidate for quantization."""
    
    module_path: str
    operation_type: str
    current_dtype: torch.dtype
    
    # Quantization options
    int8_viable: bool = False
    int4_viable: bool = False
    dynamic_quantization: bool = False
    static_quantization: bool = False
    
    # Expected impact
    memory_reduction_percent: float = 0.0
    speedup_percent: float = 0.0
    accuracy_loss_percent: float = 0.0
    
    # Implementation
    quantization_scheme: str = ""
    calibration_required: bool = False


class DTypeAnalyzer:
    """
    Analyzer for data types in UniAD models.
    
    Identifies mixed precision opportunities, quantization candidates,
    and provides optimization recommendations.
    """
    
    def __init__(self):
        """Initialize the DType analyzer."""
        self.dtype_profiles: Dict[torch.dtype, DTypeProfile] = {}
        self.mixed_precision_opportunities: List[MixedPrecisionOpportunity] = []
        self.quantization_candidates: List[QuantizationCandidate] = []
        
        # Statistics
        self.total_tensors = 0
        self.total_memory_bytes = 0
        self.dtype_distribution: Dict[torch.dtype, float] = {}
        
        # Module-specific dtype usage
        self.module_dtypes: Dict[str, Set[torch.dtype]] = defaultdict(set)
        
        # Precision-sensitive operations
        self.precision_sensitive = [
            'loss', 'normalize', 'softmax', 'sigmoid',
            'tanh', 'exponential', 'logarithm'
        ]
        
        # Operations suitable for lower precision
        self.low_precision_suitable = [
            'conv', 'linear', 'matmul', 'bmm',
            'relu', 'maxpool', 'avgpool', 'concat'
        ]
        
        # Quantization-friendly operations
        self.quantization_friendly = [
            'conv2d', 'linear', 'relu', 'maxpool2d',
            'avgpool2d', 'batchnorm', 'add', 'mul'
        ]
    
    def analyze_dtypes(self, trace_data: List[TraceNode]) -> Dict[str, Any]:
        """
        Analyze data type usage in the model.
        
        Args:
            trace_data: List of trace nodes
            
        Returns:
            Comprehensive dtype analysis
        """
        # Profile dtypes
        self._profile_dtypes(trace_data)
        
        # Identify mixed precision opportunities
        self._identify_mixed_precision_opportunities(trace_data)
        
        # Identify quantization candidates
        self._identify_quantization_candidates(trace_data)
        
        # Calculate statistics
        self._calculate_statistics()
        
        # Generate analysis report
        analysis = {
            'summary': self._generate_summary(),
            'dtype_distribution': self._get_dtype_distribution(),
            'memory_breakdown': self._analyze_memory_by_dtype(),
            'mixed_precision_opportunities': self._format_mixed_precision_opportunities(),
            'quantization_candidates': self._format_quantization_candidates(),
            'optimization_potential': self._calculate_optimization_potential(),
            'recommendations': self._generate_recommendations(),
            'amp_compatibility': self._analyze_amp_compatibility(trace_data),
        }
        
        return analysis
    
    def analyze_precision_requirements(self, trace_data: List[TraceNode]) -> Dict[str, Any]:
        """
        Analyze precision requirements for different operations.
        
        Args:
            trace_data: List of trace nodes
            
        Returns:
            Precision requirement analysis
        """
        high_precision_ops = []
        low_precision_ops = []
        mixed_precision_ops = []
        
        for node in trace_data:
            op_lower = node.name.lower()
            
            # Check precision requirements
            if any(sensitive in op_lower for sensitive in self.precision_sensitive):
                high_precision_ops.append(node)
            elif any(suitable in op_lower for suitable in self.low_precision_suitable):
                low_precision_ops.append(node)
            else:
                mixed_precision_ops.append(node)
        
        return {
            'high_precision_required': len(high_precision_ops),
            'low_precision_suitable': len(low_precision_ops),
            'mixed_precision_flexible': len(mixed_precision_ops),
            'high_precision_modules': list(set(op.module_path for op in high_precision_ops)),
            'optimization_ratio': len(low_precision_ops) / len(trace_data) * 100 if trace_data else 0,
        }
    
    def suggest_dtype_strategy(self, trace_data: List[TraceNode]) -> Dict[str, Any]:
        """
        Suggest optimal dtype strategy for the model.
        
        Args:
            trace_data: List of trace nodes
            
        Returns:
            Dtype strategy recommendations
        """
        strategy = {
            'recommended_approach': '',
            'expected_memory_savings': 0.0,
            'expected_speedup': 0.0,
            'implementation_steps': [],
            'risk_assessment': '',
        }
        
        # Analyze current state
        fp32_count = sum(1 for p in self.dtype_profiles.values() if p.dtype == torch.float32)
        fp16_count = sum(1 for p in self.dtype_profiles.values() if p.dtype == torch.float16)
        
        # Determine strategy
        if fp32_count > fp16_count * 2:
            # Mostly FP32, good candidate for mixed precision
            strategy['recommended_approach'] = 'Automatic Mixed Precision (AMP)'
            strategy['expected_memory_savings'] = 40.0  # ~40% savings typical
            strategy['expected_speedup'] = 2.0  # ~2x speedup on modern GPUs
            strategy['implementation_steps'] = [
                "Enable torch.cuda.amp.autocast()",
                "Use GradScaler for gradient scaling",
                "Identify and exclude precision-sensitive operations",
                "Fine-tune loss scaling parameters",
            ]
            strategy['risk_assessment'] = 'low'
        elif fp16_count > 0:
            # Already using mixed precision
            strategy['recommended_approach'] = 'Optimize existing mixed precision'
            strategy['expected_memory_savings'] = 10.0
            strategy['expected_speedup'] = 1.2
            strategy['implementation_steps'] = [
                "Review FP32 operations for conversion potential",
                "Consider BF16 for better numerical stability",
                "Optimize memory layout for mixed precision",
            ]
            strategy['risk_assessment'] = 'low'
        else:
            # Consider quantization
            strategy['recommended_approach'] = 'Quantization-aware training'
            strategy['expected_memory_savings'] = 75.0  # INT8 saves 75% over FP32
            strategy['expected_speedup'] = 3.0
            strategy['implementation_steps'] = [
                "Profile model for quantization sensitivity",
                "Implement calibration dataset",
                "Apply dynamic or static quantization",
                "Fine-tune quantized model",
            ]
            strategy['risk_assessment'] = 'medium'
        
        return strategy
    
    def _profile_dtypes(self, trace_data: List[TraceNode]) -> None:
        """Profile data types in the trace."""
        for node in trace_data:
            # Process input tensors
            for shape in node.input_shapes:
                self._add_dtype_usage(node, shape, is_input=True)
            
            # Process output tensors
            for shape in node.output_shapes:
                self._add_dtype_usage(node, shape, is_input=False)
    
    def _add_dtype_usage(self, node: TraceNode, shape: TensorShape, is_input: bool) -> None:
        """Add dtype usage to profiles."""
        dtype = shape.dtype
        
        # Get or create profile
        if dtype not in self.dtype_profiles:
            self.dtype_profiles[dtype] = DTypeProfile(dtype=dtype)
        
        profile = self.dtype_profiles[dtype]
        profile.add_operation(node, shape)
        
        # Track module dtypes
        self.module_dtypes[node.module_path].add(dtype)
        
        # Update totals
        self.total_tensors += 1
        elements = np.prod(shape.shape) if shape.shape else 0
        self.total_memory_bytes += elements * profile._get_dtype_size(dtype)
        
        # Check compatibility
        self._check_precision_compatibility(profile, node)
    
    def _check_precision_compatibility(self, profile: DTypeProfile, node: TraceNode) -> None:
        """Check if dtype can be converted to lower precision."""
        op_lower = node.name.lower()
        
        # Check FP16 compatibility
        if profile.dtype == torch.float32:
            if any(suitable in op_lower for suitable in self.low_precision_suitable):
                profile.fp16_compatible = True
                profile.bf16_compatible = True
        
        # Check INT8 compatibility
        if profile.dtype in [torch.float32, torch.float16]:
            if any(friendly in op_lower for friendly in self.quantization_friendly):
                profile.int8_compatible = True
    
    def _identify_mixed_precision_opportunities(self, trace_data: List[TraceNode]) -> None:
        """Identify opportunities for mixed precision."""
        self.mixed_precision_opportunities = []
        
        # Group operations by module
        module_ops = defaultdict(list)
        for node in trace_data:
            module_ops[node.module_path].append(node)
        
        # Analyze each module
        for module_path, ops in module_ops.items():
            # Check if module uses FP32
            dtypes = self.module_dtypes.get(module_path, set())
            if torch.float32 in dtypes:
                # Check if suitable for FP16
                if self._is_fp16_suitable(ops):
                    opportunity = self._create_mixed_precision_opportunity(
                        module_path, ops, torch.float32, torch.float16
                    )
                    self.mixed_precision_opportunities.append(opportunity)
                # Check if suitable for BF16
                elif self._is_bf16_suitable(ops):
                    opportunity = self._create_mixed_precision_opportunity(
                        module_path, ops, torch.float32, torch.bfloat16
                    )
                    self.mixed_precision_opportunities.append(opportunity)
    
    def _is_fp16_suitable(self, ops: List[TraceNode]) -> bool:
        """Check if operations are suitable for FP16."""
        suitable_count = 0
        sensitive_count = 0
        
        for op in ops:
            op_lower = op.name.lower()
            if any(suitable in op_lower for suitable in self.low_precision_suitable):
                suitable_count += 1
            if any(sensitive in op_lower for sensitive in self.precision_sensitive):
                sensitive_count += 1
        
        # Suitable if mostly low-precision operations and few sensitive ones
        return suitable_count > len(ops) * 0.5 and sensitive_count < len(ops) * 0.1
    
    def _is_bf16_suitable(self, ops: List[TraceNode]) -> bool:
        """Check if operations are suitable for BF16."""
        # BF16 has better range than FP16, suitable for more operations
        for op in ops:
            op_lower = op.name.lower()
            if 'gradient' in op_lower or 'loss' in op_lower:
                return True  # BF16 is good for training
        return False
    
    def _create_mixed_precision_opportunity(self, 
                                           module_path: str,
                                           ops: List[TraceNode],
                                           current_dtype: torch.dtype,
                                           target_dtype: torch.dtype) -> MixedPrecisionOpportunity:
        """Create a mixed precision opportunity."""
        # Calculate memory savings
        current_memory = sum(op.cuda_memory_allocated for op in ops)
        dtype_ratio = 2 if target_dtype in [torch.float16, torch.bfloat16] else 1
        saved_memory = current_memory * (1 - 1/dtype_ratio) / (1024 * 1024)
        
        # Estimate speedup
        speedup = 1.5 if target_dtype == torch.float16 else 1.3
        
        # Determine implementation
        if target_dtype == torch.float16:
            method = 'amp'
            code = f"with torch.cuda.amp.autocast():\n    output = {module_path}(input)"
        else:
            method = 'manual'
            code = f"{module_path} = {module_path}.to(torch.{target_dtype})"
        
        opportunity = MixedPrecisionOpportunity(
            module_path=module_path,
            current_dtype=current_dtype,
            target_dtype=target_dtype,
            memory_savings_mb=saved_memory,
            speedup_factor=speedup,
            accuracy_impact='minimal',
            conversion_method=method,
            implementation_code=code,
            risk_level='low',
            considerations=[
                "Monitor gradient overflow",
                "Adjust loss scaling if needed",
                "Validate accuracy on validation set",
            ],
        )
        
        return opportunity
    
    def _identify_quantization_candidates(self, trace_data: List[TraceNode]) -> None:
        """Identify candidates for quantization."""
        self.quantization_candidates = []
        
        # Group by operation type
        op_groups = defaultdict(list)
        for node in trace_data:
            op_groups[node.name].append(node)
        
        # Check each operation type
        for op_type, nodes in op_groups.items():
            if self._is_quantization_friendly(op_type):
                for node in nodes[:5]:  # Limit to top 5 per type
                    candidate = self._create_quantization_candidate(node)
                    if candidate:
                        self.quantization_candidates.append(candidate)
    
    def _is_quantization_friendly(self, op_type: str) -> bool:
        """Check if operation type is quantization-friendly."""
        op_lower = op_type.lower()
        return any(friendly in op_lower for friendly in self.quantization_friendly)
    
    def _create_quantization_candidate(self, node: TraceNode) -> Optional[QuantizationCandidate]:
        """Create a quantization candidate."""
        # Get dominant dtype
        dtypes = [shape.dtype for shape in node.output_shapes]
        if not dtypes or dtypes[0] not in [torch.float32, torch.float16]:
            return None
        
        candidate = QuantizationCandidate(
            module_path=node.module_path,
            operation_type=node.name,
            current_dtype=dtypes[0],
        )
        
        # Determine quantization viability
        op_lower = node.name.lower()
        if 'conv' in op_lower or 'linear' in op_lower:
            candidate.int8_viable = True
            candidate.static_quantization = True
            candidate.memory_reduction_percent = 75.0
            candidate.speedup_percent = 200.0
            candidate.accuracy_loss_percent = 1.0
            candidate.quantization_scheme = "per_channel_symmetric"
            candidate.calibration_required = True
        elif 'relu' in op_lower or 'pool' in op_lower:
            candidate.int8_viable = True
            candidate.dynamic_quantization = True
            candidate.memory_reduction_percent = 50.0
            candidate.speedup_percent = 150.0
            candidate.accuracy_loss_percent = 0.5
            candidate.quantization_scheme = "per_tensor_affine"
            candidate.calibration_required = False
        
        return candidate if candidate.int8_viable else None
    
    def _calculate_statistics(self) -> None:
        """Calculate dtype distribution statistics."""
        if self.total_memory_bytes == 0:
            return
        
        self.dtype_distribution = {}
        for dtype, profile in self.dtype_profiles.items():
            self.dtype_distribution[dtype] = profile.total_memory_bytes / self.total_memory_bytes
        
        # Finalize profiles
        for profile in self.dtype_profiles.values():
            profile.finalize()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate dtype analysis summary."""
        return {
            'total_tensors': self.total_tensors,
            'total_memory_mb': self.total_memory_bytes / (1024 * 1024),
            'unique_dtypes': len(self.dtype_profiles),
            'dominant_dtype': self._get_dominant_dtype(),
            'mixed_precision_opportunities': len(self.mixed_precision_opportunities),
            'quantization_candidates': len(self.quantization_candidates),
        }
    
    def _get_dominant_dtype(self) -> str:
        """Get the dominant dtype by memory usage."""
        if not self.dtype_profiles:
            return "unknown"
        
        dominant = max(
            self.dtype_profiles.items(),
            key=lambda x: x[1].total_memory_bytes
        )
        return str(dominant[0])
    
    def _get_dtype_distribution(self) -> Dict[str, float]:
        """Get dtype distribution as percentages."""
        distribution = {}
        for dtype, percentage in self.dtype_distribution.items():
            distribution[str(dtype)] = round(percentage * 100, 2)
        return distribution
    
    def _analyze_memory_by_dtype(self) -> Dict[str, Dict[str, Any]]:
        """Analyze memory usage by dtype."""
        breakdown = {}
        
        for dtype, profile in self.dtype_profiles.items():
            breakdown[str(dtype)] = {
                'memory_mb': profile.total_memory_bytes / (1024 * 1024),
                'percentage': self.dtype_distribution.get(dtype, 0) * 100,
                'tensor_count': profile.tensor_count,
                'operation_count': profile.operation_count,
                'avg_tensor_size_kb': (
                    profile.total_memory_bytes / profile.tensor_count / 1024
                    if profile.tensor_count > 0 else 0
                ),
            }
        
        return breakdown
    
    def _format_mixed_precision_opportunities(self) -> List[Dict[str, Any]]:
        """Format mixed precision opportunities for output."""
        formatted = []
        
        # Sort by potential savings
        sorted_opps = sorted(
            self.mixed_precision_opportunities,
            key=lambda x: x.memory_savings_mb,
            reverse=True
        )
        
        for opp in sorted_opps[:10]:  # Top 10
            formatted.append({
                'module': opp.module_path,
                'current_dtype': str(opp.current_dtype),
                'target_dtype': str(opp.target_dtype),
                'memory_savings_mb': opp.memory_savings_mb,
                'speedup_factor': opp.speedup_factor,
                'accuracy_impact': opp.accuracy_impact,
                'implementation': opp.conversion_method,
                'code': opp.implementation_code,
                'risk': opp.risk_level,
            })
        
        return formatted
    
    def _format_quantization_candidates(self) -> List[Dict[str, Any]]:
        """Format quantization candidates for output."""
        formatted = []
        
        # Sort by potential impact
        sorted_candidates = sorted(
            self.quantization_candidates,
            key=lambda x: x.memory_reduction_percent,
            reverse=True
        )
        
        for candidate in sorted_candidates[:10]:  # Top 10
            formatted.append({
                'module': candidate.module_path,
                'operation': candidate.operation_type,
                'current_dtype': str(candidate.current_dtype),
                'int8_viable': candidate.int8_viable,
                'memory_reduction': candidate.memory_reduction_percent,
                'speedup': candidate.speedup_percent,
                'accuracy_loss': candidate.accuracy_loss_percent,
                'quantization_scheme': candidate.quantization_scheme,
                'calibration_required': candidate.calibration_required,
            })
        
        return formatted
    
    def _calculate_optimization_potential(self) -> Dict[str, float]:
        """Calculate overall optimization potential."""
        # Memory optimization potential
        fp32_memory = self.dtype_profiles.get(torch.float32, DTypeProfile(torch.float32)).total_memory_bytes
        potential_memory_savings = fp32_memory * 0.5  # Assume 50% savings with FP16
        
        # Speed optimization potential
        low_precision_ops = sum(
            1 for opp in self.mixed_precision_opportunities
            if opp.speedup_factor > 1.0
        )
        potential_speedup = 1.5 if low_precision_ops > 10 else 1.2
        
        # Quantization potential
        quant_candidates = len(self.quantization_candidates)
        quantization_savings = quant_candidates * 10 * 1024 * 1024  # 10MB per candidate
        
        return {
            'memory_savings_mb': (potential_memory_savings + quantization_savings) / (1024 * 1024),
            'speedup_factor': potential_speedup,
            'optimization_score': self._calculate_optimization_score(),
        }
    
    def _calculate_optimization_score(self) -> float:
        """Calculate optimization score (0-1)."""
        # Factors
        fp32_ratio = self.dtype_distribution.get(torch.float32, 0)
        opportunity_ratio = len(self.mixed_precision_opportunities) / max(1, len(self.module_dtypes))
        quant_ratio = len(self.quantization_candidates) / max(1, self.total_tensors) * 100
        
        # Score calculation
        score = (
            (1 - fp32_ratio) * 0.4 +  # Less FP32 is better
            opportunity_ratio * 0.4 +  # More opportunities is better
            min(1.0, quant_ratio) * 0.2  # Some quantization is good
        )
        
        return min(1.0, max(0.0, score))
    
    def _generate_recommendations(self) -> List[str]:
        """Generate dtype optimization recommendations."""
        recommendations = []
        
        # Check FP32 dominance
        if torch.float32 in self.dtype_distribution and self.dtype_distribution[torch.float32] > 0.7:
            recommendations.append(
                "Model is predominantly FP32. Enable Automatic Mixed Precision (AMP) for "
                "significant memory savings and speedup."
            )
        
        # Check for mixed precision opportunities
        if len(self.mixed_precision_opportunities) > 5:
            recommendations.append(
                f"Found {len(self.mixed_precision_opportunities)} modules suitable for "
                "mixed precision. Implement gradually starting with compute-intensive layers."
            )
        
        # Check for quantization
        if len(self.quantization_candidates) > 0:
            recommendations.append(
                f"Identified {len(self.quantization_candidates)} operations suitable for "
                "quantization. Consider INT8 quantization for inference optimization."
            )
        
        # Check for BF16 support
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            recommendations.append(
                "GPU supports BF16. Consider using BF16 instead of FP16 for better "
                "numerical stability in training."
            )
        
        # Memory-specific recommendations
        if self.total_memory_bytes > 10 * 1024**3:  # >10GB
            recommendations.append(
                "High memory usage detected. Prioritize dtype optimization for "
                "memory-intensive operations."
            )
        
        if not recommendations:
            recommendations.append("Model dtype usage appears well-optimized.")
        
        return recommendations
    
    def _analyze_amp_compatibility(self, trace_data: List[TraceNode]) -> Dict[str, Any]:
        """Analyze Automatic Mixed Precision compatibility."""
        compatible_ops = []
        incompatible_ops = []
        
        for node in trace_data:
            op_lower = node.name.lower()
            
            # Check AMP compatibility
            if any(suitable in op_lower for suitable in self.low_precision_suitable):
                compatible_ops.append(node)
            elif any(sensitive in op_lower for sensitive in self.precision_sensitive):
                incompatible_ops.append(node)
        
        total_ops = len(trace_data)
        compatibility_score = len(compatible_ops) / total_ops if total_ops > 0 else 0
        
        return {
            'compatible_operations': len(compatible_ops),
            'incompatible_operations': len(incompatible_ops),
            'compatibility_score': compatibility_score,
            'amp_ready': compatibility_score > 0.7,
            'blockers': [op.module_path for op in incompatible_ops[:5]],  # Top 5 blockers
            'implementation_difficulty': 'easy' if compatibility_score > 0.8 else 'medium',
        }
    
    def export_analysis(self) -> Dict[str, Any]:
        """Export complete dtype analysis."""
        return {
            'summary': self._generate_summary(),
            'distribution': self._get_dtype_distribution(),
            'memory_breakdown': self._analyze_memory_by_dtype(),
            'mixed_precision': self._format_mixed_precision_opportunities(),
            'quantization': self._format_quantization_candidates(),
            'optimization_potential': self._calculate_optimization_potential(),
            'recommendations': self._generate_recommendations(),
        }