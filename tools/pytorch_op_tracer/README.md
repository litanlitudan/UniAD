# PyTorch Operation Tracer

A comprehensive tool for tracing PyTorch operations with specialized support for UniAD's multi-task autonomous driving framework.

## 📋 Quick Links

- **[📖 Complete Documentation](INDEX.md)** - Full documentation with examples and guides
- **[🚀 Quick Start](#quick-start)** - Get started immediately
- **[⚙️ Installation](#installation)** - Installation options
- **[🐍 Python API](#python-api)** - Programmatic usage

## ✨ Key Features

- **🔍 Operation Tracing**: Hooks into PyTorch modules to trace all forward operations
- **🚗 UniAD-Specific Support**: Specialized tracking for all 5 task heads (track, seg, motion, occ, planning)  
- **💾 Memory Profiling**: Critical for UniAD's 30-50GB GPU memory requirements
- **🎯 Data Type Analysis**: Comprehensive dtype tracking (fp32, fp16, bf16, int8) with memory impact
- **⏱️ Temporal Analysis**: Visualize multi-frame temporal queue processing (3-5 frames)
- **🗺️ BEV Feature Tracking**: Specialized analysis for BEV encoder/decoder operations
- **📊 Mermaid Visualization**: Generate clear dataflow diagrams with shape and dtype annotations
- **🔄 Stage-Aware**: Different handling for Stage 1 (perception) vs Stage 2 (end-to-end)
- **⚡ Mixed Precision**: Analyze and optimize mixed precision configurations

## ⚙️ Installation

```bash
# Clone and install
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD/tools/pytorch_op_tracer
pip install -e ".[uniad]"  # Full installation with UniAD support
```

## 🚀 Quick Start

```bash
# Trace UniAD Stage 2 model
pytorch-trace --config projects/configs/stage2_e2e/base_e2e.py \
              --checkpoint ckpts/uniad_base_e2e.pth \
              --output analysis.md

# Test without UniAD dependencies
pytorch-trace --test-mode --output test.md
```

## 🐍 Python API

```python
from pytorch_op_tracer import OperationTracer, TraceAnalyzer, DataflowVisualizer

# Trace operations
tracer = OperationTracer(model, stage=2)
trace_nodes = tracer.trace(inputs)

# Analyze results
analyzer = TraceAnalyzer()
analysis = analyzer.analyze(trace_nodes)

# Generate visualization
visualizer = DataflowVisualizer()
diagram = visualizer.generate_mermaid(trace_nodes, analysis)
```

## 📖 Documentation

For complete documentation including:
- **Full CLI Reference** with all options and examples
- **Architecture Guide** with detailed component descriptions  
- **Usage Examples** for common scenarios
- **Troubleshooting Guide** for common issues
- **Developer Guide** for contributing

**👉 See [INDEX.md](INDEX.md) for comprehensive documentation**

## 🚀 Examples

```bash
# Memory analysis for optimization
pytorch-trace --config projects/configs/stage2_e2e/base_e2e.py \
              --memory-profile --dtype-memory-analysis \
              --output memory_analysis.md

# BEV operations deep dive  
pytorch-trace --config projects/configs/stage2_e2e/base_e2e.py \
              --bev-focus --visualize-temporal \
              --output bev_analysis.md

# Mixed precision analysis
pytorch-trace --config projects/configs/stage2_e2e/base_e2e.py \
              --mixed-precision fp16 --track-dtype \
              --output mixed_precision.md
```

## 📊 UniAD Insights

- **Memory Patterns**: Stage 1 (~50GB) vs Stage 2 (~17GB) analysis
- **Task Head Analysis**: Track interactions across all 5 heads (track, seg, motion, occ, planning)
- **Temporal Processing**: Multi-frame queue optimization (3-5 frames)
- **BEV Operations**: Spatial transformation and feature flow analysis
- **Mixed Precision**: Optimize memory usage with dtype analysis

## 🤝 Contributing

```bash
# Development setup
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD/tools/pytorch_op_tracer
pip install -e ".[dev]"
pytest tests/  # Run tests
```

---

*Part of the [UniAD](https://github.com/OpenDriveLab/UniAD) autonomous driving framework*