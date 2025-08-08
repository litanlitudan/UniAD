#!/usr/bin/env python3
"""
Example usage of the HTMLDashboard generator.

This script demonstrates how to create interactive HTML dashboards
from UniAD model analysis results.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualizers.html_dashboard import HTMLDashboard


def create_sample_analysis_results():
    """Create sample analysis results for demonstration."""
    return {
        'memory_analysis': {
            'summary': {
                'peak_memory_gb': 42.5,
                'memory_efficiency': 0.78,
                'memory_utilization': 0.85,
                'status': 'warning'
            },
            'memory_timeline': [
                {'allocated_mb': 5000, 'freed_mb': 1000, 'delta_mb': 4000, 'timestamp_ms': 0},
                {'allocated_mb': 8000, 'freed_mb': 2000, 'delta_mb': 6000, 'timestamp_ms': 100},
                {'allocated_mb': 12000, 'freed_mb': 3000, 'delta_mb': 9000, 'timestamp_ms': 200},
                {'allocated_mb': 15000, 'freed_mb': 5000, 'delta_mb': 10000, 'timestamp_ms': 300},
                {'allocated_mb': 18000, 'freed_mb': 8000, 'delta_mb': 10000, 'timestamp_ms': 400},
            ],
            'task_head_distribution': {
                'track': {'memory_mb': 8500, 'percentage': 20},
                'segmentation': {'memory_mb': 17000, 'percentage': 40},
                'motion': {'memory_mb': 12750, 'percentage': 30},
                'occupancy': {'memory_mb': 4250, 'percentage': 10}
            },
            'optimization_suggestions': [
                {
                    'severity': 'critical',
                    'target': 'backbone.resnet.layer4',
                    'description': 'Enable gradient checkpointing for this large layer',
                    'expected_savings_mb': 3500,
                    'difficulty': 'easy'
                },
                {
                    'severity': 'high',
                    'target': 'bev_encoder.transformer',
                    'description': 'Use memory-efficient attention implementation',
                    'expected_savings_mb': 2200,
                    'difficulty': 'medium'
                }
            ]
        },
        'performance_analysis': {
            'summary': {
                'total_time_ms': 487.3,
                'ops_per_second': 856,
                'bottleneck_count': 5
            },
            'operation_timings': [
                {'operation': 'BEVEncoder', 'time_ms': 125.8, 'percentage': 25.8},
                {'operation': 'TemporalAggregation', 'time_ms': 89.4, 'percentage': 18.4},
                {'operation': 'TrackingHead', 'time_ms': 67.2, 'percentage': 13.8},
                {'operation': 'SegmentationHead', 'time_ms': 54.1, 'percentage': 11.1},
                {'operation': 'MotionHead', 'time_ms': 48.3, 'percentage': 9.9},
                {'operation': 'BackboneResNet', 'time_ms': 42.7, 'percentage': 8.8}
            ],
            'bottlenecks': [
                {'operation': 'MultiHeadAttention', 'severity': 0.92, 'time_ms': 78.5},
                {'operation': 'Conv3D_Large', 'severity': 0.85, 'time_ms': 65.2},
                {'operation': 'TransformerBlock', 'severity': 0.74, 'time_ms': 52.1},
                {'operation': 'PositionalEncoding', 'severity': 0.68, 'time_ms': 31.8}
            ]
        },
        'task_head_analysis': {
            'track': {
                'memory_usage': {'total_allocated_mb': 8500},
                'performance': {'total_time_ms': 67.2},
                'operations': {'total_ops': 245}
            },
            'segmentation': {
                'memory_usage': {'total_allocated_mb': 17000},
                'performance': {'total_time_ms': 54.1},
                'operations': {'total_ops': 180}
            },
            'motion': {
                'memory_usage': {'total_allocated_mb': 12750},
                'performance': {'total_time_ms': 48.3},
                'operations': {'total_ops': 165}
            },
            'occupancy': {
                'memory_usage': {'total_allocated_mb': 4250},
                'performance': {'total_time_ms': 28.7},
                'operations': {'total_ops': 95}
            },
            'planning': {
                'memory_usage': {'total_allocated_mb': 3200},
                'performance': {'total_time_ms': 22.1},
                'operations': {'total_ops': 78}
            }
        },
        'optimization_analysis': {
            'memory_optimizations': [
                {'severity': 'critical', 'expected_savings_mb': 3500, 'category': 'Gradient Checkpointing'},
                {'severity': 'high', 'expected_savings_mb': 2200, 'category': 'Memory-Efficient Attention'},
                {'severity': 'medium', 'expected_savings_mb': 1800, 'category': 'Activation Recomputation'}
            ],
            'dtype_optimizations': [
                {'severity': 'critical', 'expected_savings_mb': 4800, 'category': 'Mixed Precision Training'},
                {'severity': 'high', 'expected_savings_mb': 3200, 'category': 'FP16 Inference'}
            ],
            'performance_optimizations': [
                {'severity': 'high', 'expected_savings_mb': 800, 'category': 'Operator Fusion'},
                {'severity': 'medium', 'expected_savings_mb': 600, 'category': 'Memory Layout'}
            ]
        },
        'temporal_analysis': {
            'queue_analysis': [
                {'memory_mb': 2400, 'frame': 0},
                {'memory_mb': 2650, 'frame': 1},
                {'memory_mb': 2520, 'frame': 2},
                {'memory_mb': 2780, 'frame': 3},
                {'memory_mb': 2330, 'frame': 4}
            ],
            'num_frames': 5
        },
        'dtype_analysis': {
            'dtype_distribution': {
                'float32': 58.5,
                'float16': 32.0,
                'int8': 7.5,
                'bool': 2.0
            },
            'mixed_precision_opportunities': [
                {
                    'module': 'backbone.conv_layers',
                    'current_dtype': 'float32',
                    'target_dtype': 'float16',
                    'memory_savings_mb': 2400,
                    'speedup_factor': 1.9,
                    'risk': 'low'
                },
                {
                    'module': 'bev_encoder.attention',
                    'current_dtype': 'float32',
                    'target_dtype': 'float16',
                    'memory_savings_mb': 1800,
                    'speedup_factor': 2.1,
                    'risk': 'medium'
                },
                {
                    'module': 'temporal_aggregator',
                    'current_dtype': 'float32',
                    'target_dtype': 'bfloat16',
                    'memory_savings_mb': 1200,
                    'speedup_factor': 1.6,
                    'risk': 'low'
                }
            ]
        }
    }


def main():
    """Generate example HTML dashboard."""
    print("🚀 Generating UniAD Model Analysis Dashboard Example...")
    
    # Create dashboard generator
    dashboard = HTMLDashboard(theme='light')
    
    # Create sample analysis results
    analysis_results = create_sample_analysis_results()
    
    # Generate dashboard
    output_path = "uniad_dashboard_example.html"
    result_path = dashboard.generate_dashboard(
        analysis_results=analysis_results,
        output_path=output_path,
        title="UniAD Model Analysis - Example Dashboard"
    )
    
    print(f"✅ Dashboard generated successfully!")
    print(f"📊 Output file: {result_path}")
    print(f"🌐 Open the file in your browser to view the interactive dashboard")
    
    # Print summary statistics
    print("\n📈 Dashboard Contents:")
    print(f"   • {len(dashboard.sections)} sections created")
    print(f"   • {len(dashboard.charts)} interactive charts")
    print(f"   • Memory analysis with {len(analysis_results['memory_analysis']['memory_timeline'])} timeline points")
    print(f"   • Performance analysis of {len(analysis_results['performance_analysis']['operation_timings'])} operations")
    print(f"   • {len(analysis_results['task_head_analysis'])} task heads analyzed")
    print(f"   • {len(analysis_results['optimization_analysis']['memory_optimizations'])} optimization suggestions")


if __name__ == "__main__":
    main()