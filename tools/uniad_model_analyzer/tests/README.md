# UniAD Model Analyzer - Test Suite

This directory contains all test files for the UniAD Model Analyzer.

## Test Organization

All test files have been consolidated into this single `tests/` directory for better organization and maintainability.

### Core Module Tests
- `test_hook_manager.py` - Tests for PyTorch hook-based operation tracing
- `test_shape_recorder.py` - Tests for tensor shape recording and transformation tracking
- `test_tracer.py` - Tests for the main operation tracer

### Analyzer Module Tests
- `test_analyzers.py` - Comprehensive tests for all analyzer modules:
  - MultiHeadAnalyzer (5 task heads)
  - TemporalAnalyzer (3-5 frame aggregation)
  - BEVAnalyzer (Bird's Eye View operations)
  - MemoryProfiler (GPU memory optimization)
  - DTypeAnalyzer (Mixed precision analysis)

### Integration Tests
- `test_current_implementation.py` - Tests current implementation with all components
- `test_integrated_tracer.py` - Tests integrated tracer with shape recorder and hook manager

## Running Tests

### Run all tests
```bash
python -m pytest tests/
```

### Run specific test file
```bash
python -m pytest tests/test_analyzers.py
```

### Run with verbose output
```bash
python -m pytest tests/ -v
```

### Run specific test class or method
```bash
python -m pytest tests/test_analyzers.py::TestMultiHeadAnalyzer
python -m pytest tests/test_analyzers.py::TestMultiHeadAnalyzer::test_analyze_task_head
```

### Run integration tests directly
```bash
python tests/test_current_implementation.py
python tests/test_integrated_tracer.py
```

## Test Coverage

- **Total Tests**: 68
- **Pass Rate**: 100%
- **Modules Covered**:
  - Core data structures
  - Hook manager
  - Shape recorder
  - Operation tracer
  - All 5 analyzer modules
  - Mermaid visualizer

## Test Helpers

The test suite includes helper functions for creating mock data:
- `create_tensor_shape()` - Creates TensorShape objects with proper memory calculation
- Mock trace nodes for testing analyzers
- Sample models simulating UniAD architecture

## Notes

- All tests use relative imports from the parent directory
- Tests include both unit tests and integration tests
- Warning about `UserWarning` for backward hooks is expected and can be ignored
- Tests verify UniAD-specific features like task heads, BEV operations, and temporal aggregation