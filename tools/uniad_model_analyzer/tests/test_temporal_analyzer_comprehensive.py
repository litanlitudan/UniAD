#!/usr/bin/env python3
"""
Comprehensive tests for the Temporal Analyzer module.

Tests edge cases, performance, and integration with UniAD's temporal patterns.
"""

import unittest
import torch
import numpy as np
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_structures import TraceNode, TensorShape
from analyzers.temporal_analyzer import (
    TemporalAnalyzer,
    TemporalFrameProfile,
    TemporalFlowPattern
)


class TestTemporalAnalyzerComprehensive(unittest.TestCase):
    """Comprehensive test cases for TemporalAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer_3 = TemporalAnalyzer(num_frames=3)
        self.analyzer_5 = TemporalAnalyzer(num_frames=5)
    
    def _create_realistic_trace(self, num_frames: int = 3) -> List[TraceNode]:
        """Create realistic trace data mimicking UniAD's temporal processing."""
        nodes = []
        node_id = 0
        
        # Stage 1: Per-frame feature extraction
        for frame in range(num_frames):
            # Image backbone processing per frame
            for layer in ['ResNet.layer1', 'ResNet.layer2', 'ResNet.layer3', 'ResNet.layer4']:
                node = TraceNode(
                    id=f"node_{node_id}",
                    name=layer.split('.')[-1],
                    module_path=f"img_backbone.{layer}",
                    temporal_frame=frame,
                    start_time=node_id * 1000000,
                    duration=2000000 if 'layer4' in layer else 1500000,
                    memory_allocated=0,
                    memory_freed=0,
                    cuda_memory_allocated=10 * 1024 * 1024,  # 10MB
                    cuda_memory_freed=0,
                )
                nodes.append(node)
                node_id += 1
        
        # Stage 2: Temporal self-attention
        for layer_idx in range(6):  # 6 layers of temporal attention
            node = TraceNode(
                id=f"node_{node_id}",
                name="TemporalSelfAttention",
                module_path=f"temporal_encoder.layers.{layer_idx}.self_attn",
                temporal_frame=None,  # Processes all frames
                start_time=node_id * 1000000,
                duration=3000000,  # 3ms - attention is expensive
                memory_allocated=0,
                memory_freed=0,
                cuda_memory_allocated=20 * 1024 * 1024,  # 20MB per layer
                cuda_memory_freed=5 * 1024 * 1024,  # Free some memory
            )
            nodes.append(node)
            node_id += 1
        
        # Stage 3: Temporal aggregation
        node = TraceNode(
            id=f"node_{node_id}",
            name="TemporalAggregation",
            module_path="temporal_encoder.aggregator",
            temporal_frame=None,
            start_time=node_id * 1000000,
            duration=1500000,
            memory_allocated=0,
            memory_freed=0,
            cuda_memory_allocated=15 * 1024 * 1024,
            cuda_memory_freed=10 * 1024 * 1024,
        )
        nodes.append(node)
        node_id += 1
        
        # Stage 4: BEV transformation after temporal fusion
        for module in ['bev_embed', 'positional_encoding', 'bev_encoder']:
            node = TraceNode(
                id=f"node_{node_id}",
                name=module,
                module_path=f"pts_bbox_head.{module}",
                temporal_frame=None,
                start_time=node_id * 1000000,
                duration=2500000,
                memory_allocated=0,
                memory_freed=0,
                cuda_memory_allocated=25 * 1024 * 1024,
                cuda_memory_freed=0,
                bev_operation=True,
            )
            nodes.append(node)
            node_id += 1
        
        return nodes
    
    def test_frame_distribution_analysis(self):
        """Test analysis of operation distribution across frames."""
        trace = self._create_realistic_trace(num_frames=3)
        analysis = self.analyzer_3.analyze_temporal_flow(trace)
        
        # Check frame distribution
        frame_profiles = analysis['frame_profiles']
        self.assertEqual(len(frame_profiles), 3)
        
        # Verify operations are distributed across frames
        total_ops = sum(fp['operation_count'] for fp in frame_profiles)
        self.assertGreater(total_ops, 0)
        
        # Check that each frame has operations
        for fp in frame_profiles:
            if fp['frame_index'] < 3:  # First 3 frames should have ops
                self.assertGreater(fp['operation_count'], 0)
    
    def test_attention_pattern_detection(self):
        """Test detection of attention patterns in temporal flow."""
        trace = self._create_realistic_trace(num_frames=3)
        analysis = self.analyzer_3.analyze_temporal_flow(trace)
        
        # Check attention analysis
        attention_analysis = analysis['attention_analysis']
        if 'no_attention_found' not in attention_analysis:
            self.assertIn('total_attention_ops', attention_analysis)
            self.assertIn('attention_memory_mb', attention_analysis)
            self.assertIn('attention_time_ms', attention_analysis)
            
            # Verify attention operations were detected
            self.assertGreater(attention_analysis['total_attention_ops'], 0)
    
    def test_memory_scaling_with_frames(self):
        """Test memory scaling analysis with different frame counts."""
        # Test 3-frame configuration
        trace_3 = self._create_realistic_trace(num_frames=3)
        analysis_3 = self.analyzer_3.analyze_temporal_flow(trace_3)
        memory_3 = analysis_3['memory_scaling']
        
        # Test 5-frame configuration
        trace_5 = self._create_realistic_trace(num_frames=5)
        analysis_5 = self.analyzer_5.analyze_temporal_flow(trace_5)
        memory_5 = analysis_5['memory_scaling']
        
        # 5-frame should use more memory
        if 'total_memory_mb' in memory_3 and 'total_memory_mb' in memory_5:
            self.assertGreater(memory_5['total_memory_mb'], memory_3['total_memory_mb'])
    
    def test_temporal_efficiency_calculation(self):
        """Test temporal efficiency score calculation."""
        trace = self._create_realistic_trace(num_frames=3)
        analysis = self.analyzer_3.analyze_temporal_flow(trace)
        
        efficiency = analysis['temporal_efficiency']
        
        # Efficiency should be between 0 and 1
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)
    
    def test_queue_memory_recommendations(self):
        """Test memory optimization recommendations for different queue lengths."""
        # Test recommendations for 3-frame queue
        queue_3 = self.analyzer_3.analyze_queue_memory(3)
        self.assertIn('optimization_suggestions', queue_3)
        self.assertIsInstance(queue_3['optimization_suggestions'], list)
        
        # Test recommendations for 5-frame queue
        queue_5 = self.analyzer_5.analyze_queue_memory(5)
        self.assertIn('optimization_suggestions', queue_5)
        
        # 5-frame queue should have more suggestions
        self.assertGreaterEqual(
            len(queue_5['optimization_suggestions']),
            len(queue_3['optimization_suggestions'])
        )
    
    def test_temporal_pattern_classification(self):
        """Test classification of temporal flow patterns."""
        trace = self._create_realistic_trace(num_frames=3)
        
        # Add some specific patterns
        # Add recurrent pattern
        trace.append(TraceNode(
            id="recurrent_1",
            name="LSTM",
            module_path="temporal.lstm",
            temporal_frame=None,
            start_time=100000000,
            duration=1000000,
            memory_allocated=0,
            memory_freed=0,
            cuda_memory_allocated=5 * 1024 * 1024,
            cuda_memory_freed=0,
        ))
        
        # Add aggregation pattern
        trace.append(TraceNode(
            id="aggregate_1",
            name="Concat",
            module_path="temporal.aggregate.concat",
            temporal_frame=None,
            start_time=101000000,
            duration=500000,
            memory_allocated=0,
            memory_freed=0,
            cuda_memory_allocated=3 * 1024 * 1024,
            cuda_memory_freed=0,
        ))
        
        analysis = self.analyzer_3.analyze_temporal_flow(trace)
        patterns = analysis['temporal_patterns']
        
        # Check that patterns were detected
        self.assertIsInstance(patterns, dict)
        
        # Check for specific pattern types if they exist
        pattern_types = set(patterns.keys())
        possible_types = {'attention', 'aggregation', 'recurrent', 'transformation'}
        self.assertTrue(pattern_types.issubset(possible_types))
    
    def test_frame_to_frame_relationships(self):
        """Test analysis of relationships between temporal frames."""
        # Create trace with explicit frame relationships
        trace = []
        
        # Create parent-child relationships across frames
        for frame in range(3):
            parent_id = f"frame_{frame}_parent"
            parent = TraceNode(
                id=parent_id,
                name="ParentOp",
                module_path=f"temporal.frame_{frame}",
                temporal_frame=frame,
                start_time=frame * 1000000,
                duration=500000,
                memory_allocated=0,
                memory_freed=0,
                cuda_memory_allocated=1024 * 1024,
                cuda_memory_freed=0,
            )
            trace.append(parent)
            
            # Add child in next frame
            if frame < 2:
                child = TraceNode(
                    id=f"frame_{frame+1}_child",
                    name="ChildOp",
                    module_path=f"temporal.frame_{frame+1}",
                    temporal_frame=frame + 1,
                    parent=parent_id,
                    start_time=(frame + 1) * 1000000,
                    duration=500000,
                    memory_allocated=0,
                    memory_freed=0,
                    cuda_memory_allocated=1024 * 1024,
                    cuda_memory_freed=0,
                )
                trace.append(child)
        
        analysis = self.analyzer_3.analyze_temporal_flow(trace)
        
        # Verify analysis completed without errors
        self.assertIn('frame_profiles', analysis)
        self.assertIn('temporal_patterns', analysis)
    
    def test_export_analysis_completeness(self):
        """Test that export_analysis returns all expected data."""
        trace = self._create_realistic_trace(num_frames=3)
        self.analyzer_3.analyze_temporal_flow(trace)
        
        export_data = self.analyzer_3.export_analysis()
        
        # Check all expected keys are present
        expected_keys = [
            'num_frames',
            'frame_profiles',
            'temporal_patterns',
            'queue_memory_analysis',
            'attention_visualization'
        ]
        
        for key in expected_keys:
            self.assertIn(key, export_data)
        
        # Verify data types
        self.assertIsInstance(export_data['num_frames'], int)
        self.assertIsInstance(export_data['frame_profiles'], list)
        self.assertIsInstance(export_data['temporal_patterns'], dict)
        self.assertIsInstance(export_data['queue_memory_analysis'], dict)
        self.assertIsInstance(export_data['attention_visualization'], dict)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty trace
        empty_analysis = self.analyzer_3.analyze_temporal_flow([])
        self.assertIsInstance(empty_analysis, dict)
        
        # Test with single frame
        analyzer_1 = TemporalAnalyzer(num_frames=1)
        single_frame_trace = self._create_realistic_trace(num_frames=1)
        analysis_1 = analyzer_1.analyze_temporal_flow(single_frame_trace)
        self.assertEqual(analysis_1['num_frames'], 1)
        
        # Test with very large frame count
        analyzer_10 = TemporalAnalyzer(num_frames=10)
        queue_10 = analyzer_10.analyze_queue_memory(10)
        self.assertIn('optimization_suggestions', queue_10)
    
    def test_performance_metrics(self):
        """Test that analysis completes within reasonable time."""
        import time
        
        # Create a large trace
        large_trace = []
        for _ in range(100):  # 100 iterations
            large_trace.extend(self._create_realistic_trace(num_frames=5))
        
        start_time = time.time()
        analysis = self.analyzer_5.analyze_temporal_flow(large_trace)
        elapsed_time = time.time() - start_time
        
        # Analysis should complete within 1 second for 100x trace
        self.assertLess(elapsed_time, 1.0)
        
        # Verify analysis is still valid
        self.assertIn('frame_profiles', analysis)
        self.assertIn('temporal_efficiency', analysis)


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTemporalAnalyzerComprehensive))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())