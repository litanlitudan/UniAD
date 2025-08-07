#!/usr/bin/env python3
"""
Master report generation script for PyTorch Operation Tracer

This script consolidates all report generation functionality and replaces
the multiple individual generation scripts.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def ensure_directory(path: str):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def run_trace_command(config: Optional[str] = None, 
                     checkpoint: Optional[str] = None,
                     output: Optional[str] = None,
                     stage: int = 2,
                     test_mode: bool = False,
                     extra_args: Optional[List[str]] = None) -> bool:
    """Run pytorch-trace command with specified parameters"""
    
    cmd = ["python", "trace_ops.py"]
    
    if test_mode:
        cmd.extend(["--test-mode"])
    else:
        if config:
            cmd.extend(["--config", config])
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
    
    cmd.extend(["--stage", str(stage)])
    
    if output:
        cmd.extend(["--output", output])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode == 0:
            print(f"✓ Successfully generated: {output}")
            return True
        else:
            print(f"✗ Error generating {output}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Exception running command: {e}")
        return False

def generate_basic_reports(reports_dir: str):
    """Generate basic operation analysis reports"""
    print("\n" + "="*60)
    print("GENERATING BASIC REPORTS")
    print("="*60)
    
    ensure_directory(reports_dir)
    
    # Master analysis report
    success = run_trace_command(
        output=os.path.join(reports_dir, "UniAD_Master_Analysis_Report.md"),
        test_mode=True,
        extra_args=["--memory-profile", "--show-shapes", "--shape-format", "semantic"]
    )
    
    if success:
        print("✓ Master analysis report generated")
    else:
        print("✗ Failed to generate master analysis report")

def generate_enhanced_reports(reports_dir: str):
    """Generate enhanced analysis reports with profiling"""
    print("\n" + "="*60)
    print("GENERATING ENHANCED REPORTS")
    print("="*60)
    
    enhanced_dir = os.path.join(reports_dir, "enhanced_module_operations")
    ensure_directory(enhanced_dir)
    
    success = run_trace_command(
        output=os.path.join(enhanced_dir, "UniAD_Enhanced_Operations_Analysis.md"),
        test_mode=True,
        extra_args=[
            "--memory-profile",
            "--bev-focus",
            "--visualize-temporal",
            "--track-shape-changes",
            "--dtype-memory-analysis"
        ]
    )
    
    if success:
        print("✓ Enhanced analysis reports generated")
    else:
        print("✗ Failed to generate enhanced analysis reports")

def generate_comparison_reports(reports_dir: str):
    """Generate comparison and architecture reports"""
    print("\n" + "="*60)
    print("GENERATING COMPARISON REPORTS")
    print("="*60)
    
    comparison_dir = os.path.join(reports_dir, "comparison")
    ensure_directory(comparison_dir)
    
    # Model architecture comparison
    success = run_trace_command(
        output=os.path.join(comparison_dir, "architecture_visualization.md"),
        test_mode=True,
        extra_args=[
            "--visualization-mode", "expanded",
            "--expand-modules", "BEVFormer", "TrackHead", "PlanningHead",
            "--show-shapes"
        ]
    )
    
    if success:
        print("✓ Comparison reports generated")
    else:
        print("✗ Failed to generate comparison reports")

def generate_all_reports(reports_dir: Optional[str] = None):
    """Generate all reports"""
    if reports_dir is None:
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    
    print(f"Generating all reports in: {reports_dir}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create main reports directory
    ensure_directory(reports_dir)
    
    # Generate different types of reports
    generate_basic_reports(reports_dir)
    generate_enhanced_reports(reports_dir)
    generate_comparison_reports(reports_dir)
    
    # Create index file
    index_content = f"""# PyTorch Operation Tracer Reports

Generated on: {timestamp}

## Available Reports

### Master Analysis
- [UniAD Master Analysis Report](UniAD_Master_Analysis_Report.md) - Complete overview

### Enhanced Analysis  
- [Enhanced Operations Analysis](enhanced_module_operations/UniAD_Enhanced_Operations_Analysis.md) - Hardware-level analysis

### Comparison Analysis
- [Architecture Visualization](comparison/architecture_visualization.md) - Model structure comparison

---

*Reports generated by PyTorch Operation Tracer v1.0*
"""
    
    with open(os.path.join(reports_dir, "README.md"), "w") as f:
        f.write(index_content)
    
    print(f"\n✓ All reports generated in: {reports_dir}")
    print(f"✓ Report index created: {os.path.join(reports_dir, 'README.md')}")

def main():
    parser = argparse.ArgumentParser(description="Generate PyTorch Operation Tracer reports")
    parser.add_argument("--output-dir", default=None, help="Output directory for reports")
    parser.add_argument("--type", choices=["basic", "enhanced", "comparison", "all"], 
                       default="all", help="Type of reports to generate")
    
    args = parser.parse_args()
    
    reports_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    
    if args.type == "basic":
        generate_basic_reports(reports_dir)
    elif args.type == "enhanced":
        generate_enhanced_reports(reports_dir)
    elif args.type == "comparison":
        generate_comparison_reports(reports_dir)
    else:
        generate_all_reports(reports_dir)

if __name__ == "__main__":
    main()