"""
Unit tests for HTMLDashboard.

Tests the HTML dashboard generation functionality including charts, heatmaps,
and interactive dashboard features.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualizers.html_dashboard import HTMLDashboard, ChartData, DashboardSection


class TestHTMLDashboard(unittest.TestCase):
    """Test suite for HTMLDashboard."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dashboard = HTMLDashboard()
        self.sample_analysis_results = self._create_sample_analysis_results()
    
    def _create_sample_analysis_results(self):
        """Create sample analysis results for testing."""
        return {
            'memory_analysis': {
                'summary': {
                    'peak_memory_gb': 35.2,
                    'memory_efficiency': 0.82,
                    'memory_utilization': 0.75
                },
                'memory_timeline': [
                    {'allocated_mb': 1000, 'freed_mb': 500, 'delta_mb': 500},
                    {'allocated_mb': 1500, 'freed_mb': 200, 'delta_mb': 1300},
                    {'allocated_mb': 2000, 'freed_mb': 800, 'delta_mb': 1200},
                ],
                'task_head_distribution': {
                    'track': {'memory_mb': 5000, 'percentage': 25},
                    'segmentation': {'memory_mb': 8000, 'percentage': 40},
                    'motion': {'memory_mb': 7000, 'percentage': 35}
                },
                'optimization_suggestions': [
                    {
                        'severity': 'critical',
                        'target': 'backbone.conv1',
                        'description': 'Use gradient checkpointing',
                        'expected_savings_mb': 2000,
                        'difficulty': 'medium'
                    }
                ]
            },
            'performance_analysis': {
                'summary': {
                    'total_time_ms': 250.5,
                    'ops_per_second': 1200
                },
                'operation_timings': [
                    {'operation': 'Conv2d', 'time_ms': 45.2},
                    {'operation': 'BatchNorm', 'time_ms': 12.8},
                    {'operation': 'ReLU', 'time_ms': 5.1},
                ],
                'bottlenecks': [
                    {'operation': 'Conv2d_large', 'severity': 0.9, 'time_ms': 85.6},
                    {'operation': 'Attention', 'severity': 0.7, 'time_ms': 32.1},
                ]
            },
            'task_head_analysis': {
                'track': {
                    'memory_usage': {'total_allocated_mb': 5000},
                    'performance': {'total_time_ms': 50.2},
                    'operations': {'total_ops': 150}
                },
                'segmentation': {
                    'memory_usage': {'total_allocated_mb': 8000}, 
                    'performance': {'total_time_ms': 80.5},
                    'operations': {'total_ops': 200}
                }
            },
            'optimization_analysis': {
                'memory_optimizations': [
                    {'severity': 'high', 'expected_savings_mb': 1500},
                    {'severity': 'medium', 'expected_savings_mb': 800}
                ],
                'dtype_optimizations': [
                    {'severity': 'critical', 'expected_savings_mb': 3000},
                ],
                'performance_optimizations': [
                    {'severity': 'medium', 'expected_savings_mb': 500}
                ]
            },
            'temporal_analysis': {
                'queue_analysis': [
                    {'memory_mb': 1200},
                    {'memory_mb': 1350},
                    {'memory_mb': 1100}
                ]
            },
            'dtype_analysis': {
                'dtype_distribution': {
                    'float32': 65.5,
                    'float16': 25.0,
                    'int8': 9.5
                },
                'mixed_precision_opportunities': [
                    {
                        'module': 'backbone.layer1',
                        'current_dtype': 'float32',
                        'target_dtype': 'float16',
                        'memory_savings_mb': 1200,
                        'speedup_factor': 1.8,
                        'risk': 'low'
                    }
                ]
            }
        }
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        dashboard = HTMLDashboard(theme='dark')
        
        self.assertEqual(dashboard.theme, 'dark')
        self.assertEqual(len(dashboard.sections), 0)
        self.assertEqual(len(dashboard.charts), 0)
    
    def test_generate_dashboard(self):
        """Test complete dashboard generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            result_path = self.dashboard.generate_dashboard(
                self.sample_analysis_results,
                temp_path,
                "Test Dashboard"
            )
            
            # Check file was created
            self.assertTrue(os.path.exists(result_path))
            self.assertEqual(result_path, temp_path)
            
            # Check content
            with open(temp_path, 'r') as f:
                content = f.read()
                
                self.assertIn("<!DOCTYPE html>", content)
                self.assertIn("Test Dashboard", content)
                self.assertIn("chart.js", content)
                self.assertIn("UniAD Model Analysis Dashboard", content)
                
                # Check for sections
                self.assertIn("Executive Summary", content)
                self.assertIn("Memory Analysis", content)
                self.assertIn("Performance Analysis", content)
                
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_create_summary_section(self):
        """Test summary section creation."""
        self.dashboard._create_summary_section(self.sample_analysis_results)
        
        # Check that summary section was created
        summary_sections = [s for s in self.dashboard.sections if s.title == "Executive Summary"]
        self.assertEqual(len(summary_sections), 1)
        
        summary_section = summary_sections[0]
        self.assertEqual(summary_section.section_type, 'summary')
        self.assertEqual(summary_section.order, 1)
        self.assertIn("metric-card", summary_section.content)
        self.assertIn("35.2 GB", summary_section.content)  # Peak memory
    
    def test_create_memory_section(self):
        """Test memory analysis section creation."""
        memory_analysis = self.sample_analysis_results['memory_analysis']
        self.dashboard._create_memory_section(memory_analysis)
        
        # Check memory section was created
        memory_sections = [s for s in self.dashboard.sections if s.title == "Memory Analysis"]
        self.assertEqual(len(memory_sections), 1)
        
        # Check charts were created
        timeline_charts = [c for c in self.dashboard.charts if c.container_id == 'memory-timeline-chart']
        self.assertEqual(len(timeline_charts), 1)
        
        timeline_chart = timeline_charts[0]
        self.assertEqual(timeline_chart.chart_type, 'line')
        self.assertIn('datasets', timeline_chart.data)
    
    def test_create_performance_section(self):
        """Test performance analysis section creation."""
        performance_analysis = self.sample_analysis_results['performance_analysis']
        self.dashboard._create_performance_section(performance_analysis)
        
        # Check performance section was created
        perf_sections = [s for s in self.dashboard.sections if s.title == "Performance Analysis"]
        self.assertEqual(len(perf_sections), 1)
        
        # Check chart was created
        timing_charts = [c for c in self.dashboard.charts if c.container_id == 'performance-timing-chart']
        self.assertEqual(len(timing_charts), 1)
        
        timing_chart = timing_charts[0]
        self.assertEqual(timing_chart.chart_type, 'bar')
        self.assertIn('datasets', timing_chart.data)
    
    def test_create_task_head_section(self):
        """Test task head analysis section creation."""
        task_head_analysis = self.sample_analysis_results['task_head_analysis']
        self.dashboard._create_task_head_section(task_head_analysis)
        
        # Check task head section was created
        head_sections = [s for s in self.dashboard.sections if s.title == "Task Head Analysis"]
        self.assertEqual(len(head_sections), 1)
        
        # Check scatter chart was created
        scatter_charts = [c for c in self.dashboard.charts if c.container_id == 'task-head-scatter-chart']
        self.assertEqual(len(scatter_charts), 1)
        
        scatter_chart = scatter_charts[0]
        self.assertEqual(scatter_chart.chart_type, 'scatter')
    
    def test_create_optimization_section(self):
        """Test optimization recommendations section creation."""
        optimization_analysis = self.sample_analysis_results['optimization_analysis']
        self.dashboard._create_optimization_section(optimization_analysis)
        
        # Check optimization section was created
        opt_sections = [s for s in self.dashboard.sections if s.title == "Optimization Opportunities"]
        self.assertEqual(len(opt_sections), 1)
        
        opt_section = opt_sections[0]
        self.assertIn("optimization-grid", opt_section.content)
        self.assertIn("Memory Optimization", opt_section.content)
        self.assertIn("Mixed Precision", opt_section.content)
    
    def test_chart_data_creation(self):
        """Test ChartData creation and structure."""
        chart_data = ChartData(
            chart_type='line',
            title='Test Chart',
            data={'labels': ['A', 'B'], 'datasets': []},
            config={'responsive': True},
            container_id='test-chart'
        )
        
        self.assertEqual(chart_data.chart_type, 'line')
        self.assertEqual(chart_data.title, 'Test Chart')
        self.assertEqual(chart_data.container_id, 'test-chart')
        self.assertIn('labels', chart_data.data)
        self.assertIn('responsive', chart_data.config)
    
    def test_dashboard_section_creation(self):
        """Test DashboardSection creation and ordering."""
        section1 = DashboardSection(
            title="Section 1",
            content="<div>Content 1</div>",
            section_type='chart',
            order=2
        )
        
        section2 = DashboardSection(
            title="Section 2", 
            content="<div>Content 2</div>",
            section_type='summary',
            order=1
        )
        
        sections = [section1, section2]
        sorted_sections = sorted(sections, key=lambda x: x.order)
        
        self.assertEqual(sorted_sections[0].title, "Section 2")
        self.assertEqual(sorted_sections[1].title, "Section 1")
    
    def test_summary_cards_generation(self):
        """Test summary cards HTML generation."""
        summary_data = {
            'peak_memory_gb': 32.5,
            'total_time_ms': 180.2,
            'active_task_heads': 5,
            'memory_efficiency': 0.85
        }
        
        cards_html = self.dashboard._create_summary_cards(summary_data)
        
        self.assertIn("summary-cards", cards_html)
        self.assertIn("32.5 GB", cards_html)
        self.assertIn("180.2 ms", cards_html)
        self.assertIn("metric-card", cards_html)
    
    def test_suggestions_table_generation(self):
        """Test suggestions table HTML generation."""
        suggestions = [
            {
                'severity': 'critical',
                'target': 'module.layer1',
                'description': 'Optimize this layer',
                'expected_savings_mb': 1500,
                'difficulty': 'medium'
            }
        ]
        
        table_html = self.dashboard._create_suggestions_table(suggestions, "Test Suggestions")
        
        self.assertIn("suggestions-table", table_html)
        self.assertIn("Test Suggestions", table_html)
        self.assertIn("module.layer1", table_html)
        self.assertIn("1500", table_html)
        self.assertIn("priority-badge", table_html)
    
    def test_task_head_table_generation(self):
        """Test task head table HTML generation."""
        head_data = [
            {'name': 'track', 'ops': 150, 'memory': 5000, 'time': 50.2},
            {'name': 'segmentation', 'ops': 200, 'memory': 8000, 'time': 80.5}
        ]
        
        table_html = self.dashboard._create_task_head_table(head_data)
        
        self.assertIn("task-head-table", table_html)
        self.assertIn("Track", table_html)
        self.assertIn("Segmentation", table_html)
        self.assertIn("5000", table_html)
        self.assertIn("efficiency-badge", table_html)
    
    def test_optimization_card_generation(self):
        """Test optimization card HTML generation."""
        optimizations = [
            {'severity': 'high', 'expected_savings_mb': 1200},
            {'severity': 'critical', 'expected_savings_mb': 2000}
        ]
        
        card_html = self.dashboard._create_optimization_card(
            "Memory Optimization",
            optimizations,
            "memory-icon",
            "primary"
        )
        
        self.assertIn("optimization-card primary", card_html)
        self.assertIn("Memory Optimization", card_html)
        self.assertIn("3200", card_html)  # Total savings
        self.assertIn("card-actions", card_html)
    
    def test_mixed_precision_table(self):
        """Test mixed precision opportunities table."""
        opportunities = [
            {
                'module': 'backbone.conv1',
                'current_dtype': 'float32',
                'target_dtype': 'float16',
                'memory_savings_mb': 1200,
                'speedup_factor': 1.8,
                'risk': 'low'
            }
        ]
        
        table_html = self.dashboard._create_mixed_precision_table(opportunities)
        
        self.assertIn("mixed-precision-table", table_html)
        self.assertIn("backbone.conv1", table_html)
        self.assertIn("float32", table_html)
        self.assertIn("float16", table_html)
        self.assertIn("1.8x", table_html)
        self.assertIn("risk-badge", table_html)
    
    def test_get_task_head_color(self):
        """Test task head color assignment."""
        track_color = self.dashboard._get_task_head_color('track')
        seg_color = self.dashboard._get_task_head_color('segmentation')
        unknown_color = self.dashboard._get_task_head_color('unknown')
        
        self.assertEqual(track_color, '#FF6384')
        self.assertEqual(seg_color, '#36A2EB')
        self.assertEqual(unknown_color, '#FF9F40')  # Default color
    
    def test_key_stats_extraction(self):
        """Test key statistics extraction."""
        self.dashboard._create_summary_section(self.sample_analysis_results)
        self.dashboard._create_memory_section(self.sample_analysis_results['memory_analysis'])
        
        stats = self.dashboard._extract_key_stats(self.sample_analysis_results)
        
        self.assertIn('timestamp', stats)
        self.assertIn('sections_count', stats)
        self.assertIn('charts_count', stats)
        self.assertEqual(stats['peak_memory_gb'], 35.2)
        self.assertEqual(stats['memory_efficiency'], 0.82)
    
    def test_empty_analysis_results(self):
        """Test dashboard generation with empty analysis results."""
        empty_results = {}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            result_path = self.dashboard.generate_dashboard(
                empty_results,
                temp_path,
                "Empty Dashboard"
            )
            
            # Should still generate valid HTML
            self.assertTrue(os.path.exists(result_path))
            
            with open(temp_path, 'r') as f:
                content = f.read()
                self.assertIn("<!DOCTYPE html>", content)
                self.assertIn("Empty Dashboard", content)
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_template_fallback(self):
        """Test fallback to default template when template file not found."""
        # Create dashboard with non-existent template path
        dashboard = HTMLDashboard()
        dashboard.template_path = Path("/non/existent/path.html")
        
        html = dashboard._generate_html("Test", {})
        
        # Should use default template
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Test", html)
        self.assertIn("chart.js", html)


if __name__ == '__main__':
    unittest.main()