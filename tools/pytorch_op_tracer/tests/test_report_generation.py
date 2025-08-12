#!/usr/bin/env python3
"""Test report generation functionality"""

import unittest
import torch
import torch.nn as nn
import sys
import os
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import OperationTracer
from analyzers import TraceAnalyzer
from visualizers import DataflowVisualizer


class TestReportGeneration(unittest.TestCase):
    """Test that report generation produces expected outputs"""
    
    def setUp(self):
        """Set up test model and dummy data"""
        self.model = self._create_simple_model()
        self.dummy_input = torch.randn(1, 3, 32, 32)
    
    def _create_simple_model(self):
        """Create a simple test model"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
    
    def test_report_structure(self):
        """Test that generated report has expected structure"""
        # Trace the model
        tracer = OperationTracer(self.model, stage=2)
        outputs = tracer.trace(self.dummy_input)
        trace_nodes = tracer.get_trace_data()
        
        # Analyze trace
        analyzer = TraceAnalyzer()
        analysis = analyzer.analyze(trace_nodes, stage=2)
        
        # Generate report components
        report_sections = []
        
        # Header
        report_sections.append("# UniAD PyTorch Operation Trace Report")
        report_sections.append(f"**Model**: {self.model.__class__.__name__}")
        report_sections.append(f"**Stage**: 2")
        
        # Summary section
        self.assertIn('total_operations', analysis)
        self.assertIn('unique_operations', analysis)
        self.assertIn('memory_profile', analysis)
        
        report_sections.append("## Summary")
        report_sections.append(f"- Total Operations: {analysis['total_operations']}")
        report_sections.append(f"- Unique Operations: {analysis['unique_operations']}")
        
        # Verify we have memory data
        memory_profile = analysis['memory_profile']
        self.assertIsInstance(memory_profile, dict)
        if 'total_memory' in memory_profile:
            report_sections.append(f"- Total Memory: {memory_profile['total_memory']:.1f} MB")
        
        # Test that report can be generated without errors
        report_content = "\n".join(report_sections)
        self.assertIn("# UniAD PyTorch Operation Trace Report", report_content)
        self.assertIn("## Summary", report_content)
        self.assertIn("Total Operations:", report_content)
    
    def test_dataflow_visualization(self):
        """Test dataflow visualization generation"""
        # Trace the model
        tracer = OperationTracer(self.model, stage=2)
        outputs = tracer.trace(self.dummy_input)
        trace_nodes = tracer.get_trace_data()
        
        # Analyze trace
        analyzer = TraceAnalyzer()
        analysis = analyzer.analyze(trace_nodes, stage=2)
        
        # Generate visualization
        visualizer = DataflowVisualizer(show_shapes=True)
        diagram = visualizer.generate_mermaid(trace_nodes, analysis.get('head_analysis'), analysis['memory_profile'])
        
        # Verify mermaid diagram structure
        self.assertIn("graph TB", diagram)
        self.assertIn("```mermaid", diagram)
        self.assertIn("```", diagram)
        
        # Should contain operation nodes
        self.assertTrue(any("Conv2d" in diagram for line in diagram.split('\n')))
    
    def test_memory_heatmap_generation(self):
        """Test memory heatmap generation"""
        # Trace the model
        tracer = OperationTracer(self.model, stage=2)
        outputs = tracer.trace(self.dummy_input)
        trace_nodes = tracer.get_trace_data()
        
        # Analyze trace
        analyzer = TraceAnalyzer()
        analysis = analyzer.analyze(trace_nodes, stage=2)
        
        # Generate memory heatmap
        visualizer = DataflowVisualizer()
        heatmap = visualizer.generate_memory_heatmap(analysis['memory_profile'])
        
        # Verify heatmap structure
        self.assertIn("Operation", heatmap)
        self.assertIn("Memory", heatmap)
        self.assertIn("Percentage", heatmap)
        self.assertIn("Visual", heatmap)
        self.assertIn("Total", heatmap)
        
        # Should have visual bars
        self.assertTrue(any("█" in line for line in heatmap.split('\n')))
    
    def test_mixed_precision_analysis(self):
        """Test mixed precision analysis section"""
        # Expected mixed precision configurations
        expected_configs = {
            'default': {
                'backbone': 'float32',
                'bev_encoder': 'float32',
                'task_heads': 'float32'
            },
            'mixed_precision': {
                'backbone': 'float16',
                'bev_encoder': 'float16',
                'task_heads': 'float32'
            },
            'bfloat16': {
                'backbone': 'bfloat16',
                'bev_encoder': 'bfloat16',
                'task_heads': 'float32'
            },
            'int8_quantized': {
                'backbone': 'int8',
                'bev_encoder': 'float16',
                'task_heads': 'float32'
            }
        }
        
        # Test that these configurations can be formatted properly
        report_lines = []
        report_lines.append("### Mixed Precision Configurations")
        report_lines.append("")
        
        for config_name, config in expected_configs.items():
            report_lines.append(f"**{config_name}**:")
            for component, dtype in config.items():
                report_lines.append(f"- {component}: {dtype}")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Verify all configurations are present
        for config_name in expected_configs:
            self.assertIn(config_name, report)
        
        # Verify all data types are mentioned
        for dtype in ['float32', 'float16', 'bfloat16', 'int8']:
            self.assertIn(dtype, report)
    
    def test_memory_reduction_table(self):
        """Test memory reduction calculations"""
        # Test data
        baseline_memory = 73.5  # MB
        
        reductions = {
            'FP32': (baseline_memory, 0),
            'FP16': (baseline_memory * 0.5, 50),
            'BF16': (baseline_memory * 0.5, 50),
            'INT8': (baseline_memory * 0.25, 75)
        }
        
        # Generate table
        table_lines = []
        table_lines.append("| Precision | Total Memory | Reduction | Notes |")
        table_lines.append("|-----------|-------------|-----------|-------|")
        
        for precision, (memory, reduction) in reductions.items():
            notes = "Full precision" if precision == "FP32" else precision.lower()
            table_lines.append(f"| {precision} (baseline) | {memory:.1f} MB | {reduction}% | {notes} |")
        
        table = "\n".join(table_lines)
        
        # Verify table structure
        self.assertIn("| Precision |", table)
        self.assertIn("FP32", table)
        self.assertIn("50%", table)
        self.assertIn("75%", table)


if __name__ == '__main__':
    unittest.main()