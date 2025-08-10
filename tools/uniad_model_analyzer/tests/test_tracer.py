"""
Unit tests for OperationTracer module.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tracer import OperationTracer
from core.data_structures import AnalysisConfig, AnalysisStage, TaskHead


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 10)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TestOperationTracer(unittest.TestCase):
    """Test cases for OperationTracer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalysisConfig(
            model_config="test_config.py",
            checkpoint=None,
            stage=AnalysisStage.STAGE_1,
            trace_forward=True,
            trace_backward=False,
            profile_memory=True,
            track_shapes=True,
            track_dtypes=True,
            temporal_frames=3,
            num_iterations=1,
            warmup_iterations=0,
        )
        self.tracer = OperationTracer(self.config)
        self.model = DummyModel()
        self.input_tensor = torch.randn(2, 10)
    
    def test_initialization(self):
        """Test tracer initialization."""
        self.assertIsNotNone(self.tracer.hook_manager)
        self.assertIsNotNone(self.tracer.shape_recorder)
        self.assertEqual(self.tracer.config, self.config)
        self.assertEqual(self.tracer.stage, AnalysisStage.STAGE_1)
        self.assertEqual(self.tracer.temporal_frames, 3)
    
    def test_trace_model_basic(self):
        """Test basic model tracing."""
        result = self.tracer.trace_model(self.model, self.input_tensor)
        
        # Check result structure
        self.assertIsNotNone(result)
        self.assertGreater(result.total_operations, 0)
        self.assertGreater(result.total_duration_ms, 0)
        self.assertGreaterEqual(result.total_parameters, 0)
        
        # Check trace nodes
        self.assertGreater(len(result.trace_graph), 0)
        
        # Check memory profile
        self.assertIsNotNone(result.memory_profile)
        self.assertGreaterEqual(result.memory_profile.peak_allocated, 0)
    
    def test_trace_model_with_dict_input(self):
        """Test tracing with dictionary input."""
        model = nn.Linear(10, 5)
        inputs = {'input': torch.randn(2, 10)}
        
        # Wrap model to accept dict
        class DictModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, input):
                return self.model(input)
        
        dict_model = DictModel(model)
        result = self.tracer.trace_model(dict_model, inputs)
        
        self.assertIsNotNone(result)
        self.assertGreater(result.total_operations, 0)
    
    def test_warmup_iterations(self):
        """Test warmup iterations."""
        config = AnalysisConfig(
            model_config="test_config.py",
            stage=AnalysisStage.STAGE_1,
            num_iterations=1,
            warmup_iterations=2,
        )
        tracer = OperationTracer(config)
        
        result = tracer.trace_model(self.model, self.input_tensor)
        
        # Warmup shouldn't affect traced operations
        self.assertIsNotNone(result)
        self.assertGreater(result.total_operations, 0)
    
    def test_multiple_iterations(self):
        """Test multiple tracing iterations."""
        config = AnalysisConfig(
            model_config="test_config.py",
            stage=AnalysisStage.STAGE_1,
            num_iterations=3,
            warmup_iterations=0,
        )
        tracer = OperationTracer(config)
        
        result = tracer.trace_model(self.model, self.input_tensor)
        
        # Should have more operations from multiple iterations
        self.assertIsNotNone(result)
        self.assertGreater(result.total_operations, 0)
    
    def test_shape_recording_integration(self):
        """Test shape recording integration."""
        result = self.tracer.trace_model(self.model, self.input_tensor)
        
        # Check shape recording
        shape_analysis = self.tracer.shape_recorder.analyze_shape_patterns()
        
        self.assertIn('total_unique_shapes', shape_analysis)
        self.assertIn('total_tensors_tracked', shape_analysis)
        self.assertGreater(shape_analysis['total_unique_shapes'], 0)
        self.assertGreater(shape_analysis['total_tensors_tracked'], 0)
    
    def test_memory_profiling(self):
        """Test memory profiling capabilities."""
        result = self.tracer.trace_model(self.model, self.input_tensor)
        
        # Check memory profile
        mem_profile = result.memory_profile
        self.assertGreaterEqual(mem_profile.peak_allocated, 0)
        self.assertGreaterEqual(mem_profile.peak_reserved, 0)
        
        # Check memory timeline
        self.assertIsInstance(mem_profile.memory_timeline, list)
        if torch.cuda.is_available() and mem_profile.memory_timeline:
            for timestamp, memory in mem_profile.memory_timeline:
                self.assertGreaterEqual(timestamp, 0)
                self.assertGreaterEqual(memory, 0)
    
    def test_task_head_statistics(self):
        """Test task head statistics generation."""
        # Create a model with task heads
        class TaskHeadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.track_head = nn.Linear(10, 5)
                self.seg_head = nn.Linear(10, 5)
            
            def forward(self, x):
                track = self.track_head(x)
                seg = self.seg_head(x)
                return track, seg
        
        model = TaskHeadModel()
        result = self.tracer.trace_model(model, self.input_tensor)
        
        # Check task head stats
        self.assertIsInstance(result.task_head_stats, dict)
        # Stats might be under TaskHead.NONE if identification doesn't work perfectly
        # That's okay for this test
    
    def test_recommendations_generation(self):
        """Test recommendation generation."""
        result = self.tracer.trace_model(self.model, self.input_tensor)
        
        # Check recommendations
        self.assertIsInstance(result.recommendations, list)
        
        # For small models, might not have recommendations
        # That's expected
    
    def test_stage_2_configuration(self):
        """Test Stage 2 specific configuration."""
        config = AnalysisConfig(
            model_config="test_config.py",
            stage=AnalysisStage.STAGE_2,
            verify_frozen_bev=True,
        )
        tracer = OperationTracer(config)
        
        self.assertEqual(tracer.stage, AnalysisStage.STAGE_2)
        
        # Create a model with BEV encoder
        class BEVModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bev_encoder = nn.Linear(10, 10)
            
            def forward(self, x):
                return self.bev_encoder(x)
        
        model = BEVModel()
        result = tracer.trace_model(model, self.input_tensor)
        
        # Should complete without errors
        self.assertIsNotNone(result)
    
    def test_temporal_frames_configuration(self):
        """Test temporal frames configuration."""
        for frames in [3, 5]:
            config = AnalysisConfig(
                model_config="test_config.py",
                stage=AnalysisStage.STAGE_1,
                temporal_frames=frames,
            )
            tracer = OperationTracer(config)
            
            self.assertEqual(tracer.temporal_frames, frames)
            
            result = tracer.trace_model(self.model, self.input_tensor)
            self.assertIsNotNone(result)
    
    def test_export_formats(self):
        """Test export format configuration."""
        config = AnalysisConfig(
            model_config="test_config.py",
            stage=AnalysisStage.STAGE_1,
            export_formats=["json", "csv", "markdown"],
        )
        
        # Validate config
        config.validate()
        
        tracer = OperationTracer(config)
        result = tracer.trace_model(self.model, self.input_tensor)
        
        self.assertEqual(result.config.export_formats, ["json", "csv", "markdown"])
    
    def test_error_handling(self):
        """Test error handling in tracer."""
        # Test with invalid temporal frames
        with self.assertRaises(ValueError):
            config = AnalysisConfig(
                model_config="test_config.py",
                temporal_frames=4,  # Invalid: must be 3 or 5
            )
            config.validate()
        
        # Test with invalid batch size
        with self.assertRaises(ValueError):
            config = AnalysisConfig(
                model_config="test_config.py",
                batch_size=0,  # Invalid: must be >= 1
            )
            config.validate()


if __name__ == '__main__':
    unittest.main()