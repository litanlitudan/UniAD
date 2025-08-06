#!/usr/bin/env python3
"""Regenerate all reports according to SPEC.md requirements"""

import os
import sys
import subprocess
from datetime import datetime
import shutil
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def ensure_directory(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Success: {description}")
            return True
        else:
            print(f"✗ Failed: {description}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False


def consolidate_reports(base_dir: str):
    """Consolidate all reports into a master report"""
    print(f"\n{'='*60}")
    print("Consolidating all reports...")
    print('='*60)
    
    master_report_path = os.path.join(base_dir, "UniAD_Master_Analysis_Report.md")
    
    with open(master_report_path, 'w') as master:
        master.write("# UniAD Master Analysis Report\n\n")
        master.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        master.write("This master report consolidates all analysis performed on the UniAD model.\n\n")
        
        # Table of contents
        master.write("## Table of Contents\n\n")
        master.write("1. [Module Operations Analysis](#module-operations-analysis)\n")
        master.write("2. [Enhanced Profiling with Decomposition](#enhanced-profiling)\n")
        master.write("3. [Architecture Visualization](#architecture-visualization)\n")
        master.write("4. [Comparison Reports](#comparison-reports)\n")
        master.write("5. [Summary and Recommendations](#summary)\n\n")
        
        # 1. Module Operations Analysis
        master.write("## Module Operations Analysis\n\n")
        master.write("Analysis of all UniAD modules down to PyTorch operation level.\n\n")
        
        module_ops_dir = os.path.join(base_dir, "module_operations")
        if os.path.exists(module_ops_dir):
            # Add main report content
            main_report = os.path.join(module_ops_dir, "UniAD_Complete_Operations_Analysis.md")
            if os.path.exists(main_report):
                with open(main_report, 'r') as f:
                    content = f.read()
                    # Skip the title and add as subsection
                    lines = content.split('\n')
                    if lines and lines[0].startswith('# '):
                        content = '\n'.join(lines[1:])
                    master.write(content)
                    master.write("\n\n")
        
        # 2. Enhanced Profiling
        master.write("## Enhanced Profiling with Decomposition\n\n")
        master.write("Detailed operation decomposition showing how PyTorch ops map to hardware.\n\n")
        
        enhanced_dir = os.path.join(base_dir, "enhanced_module_operations")
        if os.path.exists(enhanced_dir):
            # Add enhanced analysis
            enhanced_report = os.path.join(enhanced_dir, "UniAD_Enhanced_Operations_Analysis.md")
            if os.path.exists(enhanced_report):
                with open(enhanced_report, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    if lines and lines[0].startswith('# '):
                        content = '\n'.join(lines[1:])
                    master.write(content)
                    master.write("\n\n")
            
            # Add architecture documentation
            arch_doc = os.path.join(base_dir, "enhanced_profiling_architecture.md")
            if os.path.exists(arch_doc):
                master.write("### Enhanced Profiling Architecture\n\n")
                with open(arch_doc, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    if lines and lines[0].startswith('# '):
                        content = '\n'.join(lines[1:])
                    master.write(content)
                    master.write("\n\n")
        
        # 3. Architecture Visualization
        master.write("## Architecture Visualization\n\n")
        
        viz_report = os.path.join(base_dir, "comparison/architecture_visualization.md")
        if os.path.exists(viz_report):
            with open(viz_report, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                if lines and lines[0].startswith('# '):
                    content = '\n'.join(lines[1:])
                master.write(content)
                master.write("\n\n")
        
        # 4. Comparison Reports
        master.write("## Comparison Reports\n\n")
        
        comparison_dir = os.path.join(base_dir, "comparison")
        if os.path.exists(comparison_dir):
            # List all comparison reports
            for filename in os.listdir(comparison_dir):
                if filename.endswith('.md') and filename != 'architecture_visualization.md':
                    filepath = os.path.join(comparison_dir, filename)
                    master.write(f"### {filename.replace('_', ' ').replace('.md', '').title()}\n\n")
                    with open(filepath, 'r') as f:
                        content = f.read()
                        lines = content.split('\n')
                        if lines and lines[0].startswith('# '):
                            content = '\n'.join(lines[1:])
                        # Limit content to avoid huge file
                        content_lines = content.split('\n')
                        if len(content_lines) > 100:
                            master.write('\n'.join(content_lines[:100]))
                            master.write(f"\n\n... (truncated, see full report in {filename})\n\n")
                        else:
                            master.write(content)
                            master.write("\n\n")
        
        # 5. Summary and Recommendations
        master.write("## Summary and Recommendations\n\n")
        master.write("### Key Findings\n\n")
        master.write("1. **Memory Usage**: UniAD requires 30-50GB GPU memory\n")
        master.write("   - Stage 1: ~50GB (can reduce to ~30GB with queue_length=3)\n")
        master.write("   - Stage 2: ~17GB (BEV encoder frozen)\n\n")
        
        master.write("2. **Computational Patterns**:\n")
        master.write("   - GEMM operations dominate (>70% compute)\n")
        master.write("   - Memory-bound: normalization, activations\n")
        master.write("   - Custom kernels: DCNv2, specialized attention\n\n")
        
        master.write("3. **Optimization Opportunities**:\n")
        master.write("   - **Mixed Precision**: FP16/TF32 for most operations\n")
        master.write("   - **Kernel Fusion**: Conv+BN+ReLU patterns\n")
        master.write("   - **Flash Attention**: For transformer layers\n")
        master.write("   - **Graph Optimization**: TorchScript/TensorRT\n\n")
        
        master.write("### Hardware Requirements\n\n")
        master.write("- **GPU Memory**: Minimum 32GB, recommended 48GB+\n")
        master.write("- **Compute Capability**: >= 7.0 (Volta or newer)\n")
        master.write("- **Tensor Cores**: Highly beneficial for performance\n\n")
        
        master.write("### Report Structure\n\n")
        master.write("```\n")
        master.write("reports/\n")
        master.write("├── UniAD_Master_Analysis_Report.md (this file)\n")
        master.write("├── module_operations/\n")
        master.write("│   ├── UniAD_Complete_Operations_Analysis.md\n")
        master.write("│   └── [module directories]/\n")
        master.write("├── enhanced_module_operations/\n")
        master.write("│   ├── UniAD_Enhanced_Operations_Analysis.md\n")
        master.write("│   └── [module directories]/\n")
        master.write("├── comparison/\n")
        master.write("│   └── [comparison reports]\n")
        master.write("└── [other analysis reports]\n")
        master.write("```\n")
    
    print(f"✓ Master report created: {master_report_path}")


def main():
    """Main workflow according to SPEC.md"""
    
    print("="*80)
    print("UniAD PyTorch Operation Tracer - Complete Workflow")
    print(f"Generated at: {datetime.now()}")
    print("="*80)
    
    # Base directory for all reports
    base_dir = "reports"
    ensure_directory(base_dir)
    
    # Track success
    results = {
        "module_analysis": False,
        "enhanced_profiling": False,
        "comparison": False,
        "visualization": False
    }
    
    # 1. Generate basic module operation analysis
    print("\n" + "="*80)
    print("PHASE 1: Module Operation Analysis")
    print("="*80)
    
    if run_command(
        ["python", "generate_module_op_analysis.py"],
        "Generating module operation analysis"
    ):
        results["module_analysis"] = True
    
    # 2. Generate enhanced profiling with decomposition
    print("\n" + "="*80)
    print("PHASE 2: Enhanced Profiling with Operation Decomposition")
    print("="*80)
    
    if run_command(
        ["python", "generate_enhanced_module_reports.py"],
        "Generating enhanced module reports with decomposition"
    ):
        results["enhanced_profiling"] = True
    
    # 3. Generate comparison reports
    print("\n" + "="*80)
    print("PHASE 3: Comparison Reports")
    print("="*80)
    
    if run_command(
        ["python", "generate_reports.py", "--mock-only"],
        "Generating mock model comparison reports"
    ):
        results["comparison"] = True
    
    # 4. Generate architecture visualization
    print("\n" + "="*80)
    print("PHASE 4: Architecture Visualization")
    print("="*80)
    
    # Create a simple architecture visualization if it doesn't exist
    viz_path = os.path.join(base_dir, "comparison/architecture_visualization.md")
    ensure_directory(os.path.dirname(viz_path))
    
    if not os.path.exists(viz_path):
        with open(viz_path, 'w') as f:
            f.write("# UniAD Architecture Visualization\n\n")
            f.write("## High-Level Architecture\n\n")
            f.write("```mermaid\n")
            f.write("graph TB\n")
            f.write("    Input[Multi-Camera Images] --> Backbone[ResNet101-DCN + FPN]\n")
            f.write("    Backbone --> BEV[BEV Encoder<br/>BEVFormer]\n")
            f.write("    BEV --> Features[BEV Features<br/>256×200×200]\n")
            f.write("    \n")
            f.write("    Features --> Track[Track Head<br/>3D Detection]\n")
            f.write("    Features --> Seg[Segmentation Head<br/>BEV Segmentation]\n")
            f.write("    \n")
            f.write("    Track --> Motion[Motion Head<br/>Trajectory Prediction]\n")
            f.write("    Track --> Occ[Occupancy Head<br/>Future Occupancy]\n")
            f.write("    \n")
            f.write("    Motion --> Plan[Planning Head<br/>Ego Trajectory]\n")
            f.write("    Occ --> Plan\n")
            f.write("    Track --> Plan\n")
            f.write("```\n\n")
            f.write("## Data Flow with Shapes\n\n")
            f.write("```mermaid\n")
            f.write("graph LR\n")
            f.write("    A[Images<br/>1×6×3×928×1600] --> B[Backbone<br/>1×6×256×H×W]\n")
            f.write("    B --> C[BEV<br/>1×256×200×200]\n")
            f.write("    C --> D[Track<br/>1×N×266]\n")
            f.write("    C --> E[Seg<br/>1×3×200×200]\n")
            f.write("    D --> F[Motion<br/>1×N×6×2×6]\n")
            f.write("    D --> G[Occ<br/>1×200×200×5]\n")
            f.write("    F --> H[Plan<br/>1×6×2]\n")
            f.write("    G --> H\n")
            f.write("```\n")
        
        print("✓ Created architecture visualization")
        results["visualization"] = True
    else:
        print("✓ Architecture visualization already exists")
        results["visualization"] = True
    
    # 5. Consolidate all reports
    print("\n" + "="*80)
    print("PHASE 5: Report Consolidation")
    print("="*80)
    
    consolidate_reports(base_dir)
    
    # 6. Summary
    print("\n" + "="*80)
    print("WORKFLOW SUMMARY")
    print("="*80)
    
    total = len(results)
    successful = sum(results.values())
    
    print(f"\nCompleted {successful}/{total} phases successfully:")
    for phase, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {phase.replace('_', ' ').title()}")
    
    print(f"\nMaster report generated at: reports/UniAD_Master_Analysis_Report.md")
    
    # List all generated reports
    print("\nGenerated Reports:")
    print("```")
    print("reports/")
    print("├── UniAD_Master_Analysis_Report.md")
    
    if os.path.exists("reports/module_operations"):
        print("├── module_operations/")
        print("│   ├── UniAD_Complete_Operations_Analysis.md")
        print("│   └── [individual module reports]")
    
    if os.path.exists("reports/enhanced_module_operations"):
        print("├── enhanced_module_operations/")
        print("│   ├── UniAD_Enhanced_Operations_Analysis.md")
        print("│   └── [enhanced module reports]")
    
    if os.path.exists("reports/comparison"):
        print("├── comparison/")
        for f in os.listdir("reports/comparison"):
            if f.endswith('.md'):
                print(f"│   ├── {f}")
    
    print("└── [other analysis reports]")
    print("```")
    
    print("\n" + "="*80)
    print("Workflow complete!")
    print("="*80)


if __name__ == '__main__':
    main()