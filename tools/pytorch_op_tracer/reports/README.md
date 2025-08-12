# PyTorch Operation Tracer Reports

This directory contains all analysis reports generated for the UniAD model.

## Report Structure

```
reports/
├── UniAD_Master_Analysis_Report.md          # Main consolidated report (START HERE)
├── module_operations/                       # Basic operation analysis
│   ├── UniAD_Complete_Operations_Analysis.md
│   └── [category]/[module]_operations.md
├── enhanced_module_operations/              # Enhanced analysis with hardware decomposition
│   ├── UniAD_Enhanced_Operations_Analysis.md
│   └── [category]/[module]_enhanced.md
├── comparison/                              # Model loading approach comparisons
│   ├── model_loading_comparison.md
│   ├── architecture_visualization.md
│   └── UniAD_[variant]_mock_analysis.md
└── enhanced_profiling_architecture.md       # Profiling system documentation
```

## Quick Start

1. **Main Report**: Start with `UniAD_Master_Analysis_Report.md` for a complete overview
2. **Detailed Analysis**: Explore subdirectories for specific module analyses
3. **Hardware Insights**: Check enhanced reports for operation decomposition

## Key Findings

- **Memory Usage**: 30-50GB GPU memory required
- **Compute Patterns**: GEMM operations dominate (>70%)
- **Optimization**: Mixed precision and kernel fusion opportunities

## Regenerating Reports

```bash
# Complete workflow
python ../regenerate_all_reports.py

# Individual components
python ../generate_module_op_analysis.py
python ../generate_enhanced_module_reports.py
python ../generate_comparison_standalone.py
```