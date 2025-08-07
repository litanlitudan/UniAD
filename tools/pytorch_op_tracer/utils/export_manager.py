"""Export manager for converting visualizations to various formats

This module provides comprehensive export functionality for PyTorch operation tracer
visualizations, supporting multiple output formats including SVG, PNG, HTML, and Mermaid
diagrams. It leverages existing visualizers and provides UniAD-specific export features.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    # Import PIL for image exports
    from PIL import Image, ImageDraw
except (ImportError, ValueError):
    Image = ImageDraw = None

# Handle imports with fallback
try:
    from ..visualizers.mermaid_visualizer import DataflowVisualizer
    from ..visualizers.interactive_visualizer import InteractiveVisualizer
    from ..visualizers.memory_timeline import MemoryTimelineVisualizer, MemoryTimelineData
    from ..visualizers.task_head_comparator import TaskHeadComparator
    from ..core.data_structures import TraceNode
    from ..core.visualization_config import InteractiveConfig
    from ..web.template_engine import HTMLTemplateEngine, TemplateConfig, VisualizationData
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from visualizers.mermaid_visualizer import DataflowVisualizer
    from visualizers.interactive_visualizer import InteractiveVisualizer
    from visualizers.memory_timeline import MemoryTimelineVisualizer, MemoryTimelineData
    from visualizers.task_head_comparator import TaskHeadComparator
    from core.data_structures import TraceNode
    from core.visualization_config import InteractiveConfig
    from web.template_engine import HTMLTemplateEngine, TemplateConfig, VisualizationData


class ExportManager:
    """
    Comprehensive export manager for PyTorch operation tracer visualizations.
    
    Supports multiple export formats:
    - SVG: Scalable vector graphics from D3.js visualizations
    - PNG: Raster images using Pillow or browser rendering
    - HTML: Interactive standalone files with embedded JavaScript using HTMLTemplateEngine
    - Mermaid: Markdown-compatible diagrams for documentation
    - Reports: Comprehensive analysis reports with statistics and recommendations
    
    HTML Export Features:
    - Multiple visualization types: dataflow, memory timeline, dashboard, comparison
    - Interactive D3.js-based visualizations with zoom, pan, and filtering
    - Standalone HTML files with embedded CSS and JavaScript (no external dependencies)
    - Content Security Policy support for enhanced security
    - Responsive design with customizable themes (default, dark, UniAD)
    - UniAD-specific task head coloring and layout
    - Memory usage visualization and analysis
    - Comprehensive tooltips with operation details and tensor shapes
    
    Security Features:
    - Path sanitization to prevent directory traversal attacks
    - HTML escaping and data sanitization
    - Content Security Policy headers
    - Safe template variable substitution
    
    Additional Features:
    - UniAD-specific task head analysis and visualization
    - Memory optimization recommendations
    - Stage 1 vs Stage 2 comparison support
    - Batch export operations
    - Graceful fallback mechanisms
    """
    
    def __init__(self, 
                 output_dir: Optional[str] = None,
                 dpi: int = 300,
                 max_nodes: int = 50,
                 stage: int = 2,
                 template_config: Optional[TemplateConfig] = None):
        """
        Initialize the export manager.
        
        Args:
            output_dir: Base output directory for exports (defaults to current directory)
            dpi: DPI for PNG exports
            max_nodes: Maximum nodes to include in visualizations
            stage: UniAD training stage (1 or 2)
            template_config: HTML template engine configuration
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.dpi = dpi
        self.max_nodes = max_nodes
        self.stage = stage
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizers
        self.dataflow_visualizer = DataflowVisualizer(
            max_nodes=max_nodes,
            show_shapes=True,
            shape_format='semantic'
        )
        
        self.interactive_visualizer = InteractiveVisualizer(
            config=InteractiveConfig.for_uniad_analysis()
        )
        
        self.memory_visualizer = MemoryTimelineVisualizer()
        
        # Initialize task head comparator for version comparisons
        self.task_head_comparator = TaskHeadComparator()
        
        # Initialize HTML template engine
        self.template_config = template_config or TemplateConfig(
            enable_csp=True,
            enable_caching=True,
            theme="uniad",
            include_d3js=True,
            include_chartjs=True,
            compression_level="moderate"
        )
        self.html_template_engine = HTMLTemplateEngine(self.template_config)
        
        # Check for local CSS/JS files for standalone operation
        self._check_local_assets()
        
        # File format handlers
        self.format_handlers = {
            'svg': self._export_svg,
            'png': self._export_png,
            'html': self._export_html,
            'mermaid': self._export_mermaid,
            'md': self._export_markdown_report
        }
    
    def export_svg(self, visualization_data: Dict[str, Any], path: str) -> None:
        """
        Export visualization as SVG format.
        
        Args:
            visualization_data: Structured visualization data
            path: Output file path
            
        Raises:
            ValueError: If path is invalid or data is malformed
            IOError: If file cannot be written
        """
        safe_path = self._sanitize_path(path)
        self._export_svg(visualization_data, safe_path)
    
    def export_png(self, visualization_data: Dict[str, Any], path: str, dpi: Optional[int] = None) -> None:
        """
        Export visualization as PNG format.
        
        Args:
            visualization_data: Structured visualization data
            path: Output file path
            dpi: DPI for image export (defaults to instance DPI)
            
        Raises:
            ValueError: If path is invalid or data is malformed
            IOError: If file cannot be written
            ImportError: If PIL is not available
        """
        safe_path = self._sanitize_path(path)
        export_dpi = dpi if dpi is not None else self.dpi
        self._export_png(visualization_data, safe_path, export_dpi)
    
    def export_mermaid(self, trace_nodes: List[TraceNode], path: str, 
                      head_analysis: Optional[Dict[str, Any]] = None,
                      memory_profile: Optional[Dict[str, Any]] = None) -> None:
        """
        Export Mermaid diagram for Markdown documentation.
        
        Args:
            trace_nodes: List of traced operations
            path: Output file path
            head_analysis: Optional task head analysis results
            memory_profile: Optional memory usage profile
            
        Raises:
            ValueError: If path is invalid or nodes are empty
            IOError: If file cannot be written
        """
        safe_path = self._sanitize_path(path)
        
        # Provide default analysis if not provided
        if head_analysis is None:
            head_analysis = self._create_default_head_analysis(trace_nodes)
        if memory_profile is None:
            memory_profile = self._create_default_memory_profile(trace_nodes)
        
        mermaid_content = self.dataflow_visualizer.generate_mermaid(
            trace_nodes, head_analysis, memory_profile
        )
        
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(mermaid_content)
    
    def export_report(self, analysis_data: Dict[str, Any], path: str, 
                     include_recommendations: bool = True,
                     format: str = 'markdown',
                     include_key_findings: bool = True) -> None:
        """
        Export comprehensive analysis report with summary statistics, key findings, and optimization recommendations.
        
        This method implements Requirement 5.2 by generating comprehensive analysis reports
        that help developers understand and optimize their UniAD models. Reports include:
        - Summary statistics (total operations, memory usage, compute time)
        - Key findings (bottlenecks, inefficiencies, performance patterns)
        - Optimization recommendations based on profiling data
        - Task head breakdown for UniAD (tracking, segmentation, motion, occupancy, planning)
        - Stage-specific analysis (Stage 1 vs Stage 2)
        
        Args:
            analysis_data: Complete analysis results including:
                - trace_nodes: List of traced operations with memory and compute metrics
                - head_analysis: Task head analysis results with per-head breakdowns
                - memory_profile: Memory usage profile with peak usage and top consumers
                - memory_timeline: Optional memory timeline data for temporal analysis
                - stage_comparison: Optional stage comparison data (Stage 1 vs Stage 2)
            path: Output file path for the report
            include_recommendations: Whether to include optimization recommendations (default: True)
            format: Report format - 'markdown' for .md files, 'html' for interactive reports
            include_key_findings: Whether to include automated key findings analysis (default: True)
            
        Returns:
            None - Report is written to the specified file path
            
        Raises:
            ValueError: If path is invalid, data is incomplete, or format is unsupported
            IOError: If file cannot be written due to permissions or disk space
        """
        safe_path = self._sanitize_path(path)
        
        # Validate analysis data
        if not self._validate_analysis_data(analysis_data):
            raise ValueError("Analysis data is incomplete or invalid for report generation")
        
        # Generate report based on format
        if format.lower() == 'markdown':
            self._export_comprehensive_report(analysis_data, safe_path, include_recommendations, include_key_findings)
        elif format.lower() == 'html':
            self._export_html_report(analysis_data, safe_path, include_recommendations, include_key_findings)
        else:
            raise ValueError(f"Unsupported report format: {format}. Supported formats: 'markdown', 'html'")
    
    def export_interactive_html(self, trace_nodes: List[TraceNode], path: str,
                               head_analysis: Optional[Dict[str, Any]] = None,
                               memory_profile: Optional[Dict[str, Any]] = None) -> None:
        """
        Export interactive HTML visualization.
        
        Args:
            trace_nodes: List of traced operations
            path: Output file path
            head_analysis: Optional task head analysis results
            memory_profile: Optional memory usage profile
            
        Raises:
            ValueError: If path is invalid or nodes are empty
            IOError: If file cannot be written
        """
        safe_path = self._sanitize_path(path)
        
        # Provide default analysis if not provided
        if head_analysis is None:
            head_analysis = self._create_default_head_analysis(trace_nodes)
        if memory_profile is None:
            memory_profile = self._create_default_memory_profile(trace_nodes)
        
        html_content = self.interactive_visualizer.generate_interactive_html(
            trace_nodes, head_analysis, memory_profile
        )
        
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def export_memory_timeline(self, timeline_data: MemoryTimelineData, path: str,
                              format: str = 'html') -> None:
        """
        Export memory timeline visualization.
        
        Args:
            timeline_data: Memory timeline data
            path: Output file path
            format: Export format ('html' or 'md')
            
        Raises:
            ValueError: If path is invalid or format is unsupported
            IOError: If file cannot be written
        """
        safe_path = self._sanitize_path(path)
        
        if format.lower() == 'html':
            html_content = self.memory_visualizer.generate_interactive_html(timeline_data)
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        elif format.lower() == 'md':
            markdown_content = self.memory_visualizer.generate_markdown_report(timeline_data)
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
        else:
            raise ValueError(f"Unsupported memory timeline format: {format}")
    
    def export_batch(self, analysis_data: Dict[str, Any], base_name: str,
                    formats: List[str]) -> Dict[str, str]:
        """
        Export multiple formats in a batch operation.
        
        Args:
            analysis_data: Complete analysis results
            base_name: Base filename (without extension)
            formats: List of formats to export ('svg', 'png', 'html', 'mermaid', 'md')
            
        Returns:
            Dictionary mapping format to output file path
            
        Raises:
            ValueError: If any format is unsupported
        """
        results = {}
        
        for format_type in formats:
            if format_type not in self.format_handlers:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            # Generate appropriate filename
            filename = f"{base_name}.{format_type}"
            output_path = self.output_dir / filename
            
            try:
                if format_type == 'mermaid':
                    self.export_mermaid(
                        analysis_data['trace_nodes'],
                        str(output_path),
                        analysis_data.get('head_analysis'),
                        analysis_data.get('memory_profile')
                    )
                elif format_type == 'html':
                    self.export_interactive_html(
                        analysis_data['trace_nodes'],
                        str(output_path),
                        analysis_data.get('head_analysis'),
                        analysis_data.get('memory_profile')
                    )
                elif format_type in ['md', 'svg', 'png']:
                    handler = self.format_handlers[format_type]
                    handler(analysis_data, str(output_path))
                
                results[format_type] = str(output_path)
                
            except Exception as e:
                print(f"Warning: Failed to export {format_type} format: {e}")
                continue
        
        return results
    
    def export_html_dataflow(self, trace_nodes: List[TraceNode], path: str,
                            head_analysis: Optional[Dict[str, Any]] = None,
                            memory_profile: Optional[Dict[str, Any]] = None,
                            title: Optional[str] = None) -> None:
        """
        Export interactive dataflow HTML visualization using HTMLTemplateEngine.
        
        Args:
            trace_nodes: List of traced operations
            path: Output file path
            head_analysis: Optional task head analysis results
            memory_profile: Optional memory usage profile
            title: Optional custom title (defaults to UniAD Stage X Dataflow)
        """
        safe_path = self._sanitize_path(path)
        
        # Provide default analysis if not provided
        if head_analysis is None:
            head_analysis = self._create_default_head_analysis(trace_nodes)
        if memory_profile is None:
            memory_profile = self._create_default_memory_profile(trace_nodes)
        
        # Create visualization data structure for HTMLTemplateEngine
        visualization_data = {
            'trace_nodes': trace_nodes,
            'head_analysis': head_analysis,
            'memory_profile': memory_profile,
            'visualization_type': 'dataflow'
        }
        
        # Override title if provided
        if title:
            visualization_data['custom_title'] = title
        
        self._export_html(visualization_data, safe_path)
    
    def export_html_memory_timeline(self, memory_timeline: MemoryTimelineData, path: str,
                                   title: Optional[str] = None) -> None:
        """
        Export interactive memory timeline HTML visualization using HTMLTemplateEngine.
        
        Args:
            memory_timeline: Memory timeline data
            path: Output file path
            title: Optional custom title (defaults to UniAD Stage X Memory Timeline)
        """
        safe_path = self._sanitize_path(path)
        
        # Create visualization data structure for HTMLTemplateEngine
        visualization_data = {
            'trace_nodes': [],
            'head_analysis': {},
            'memory_profile': {},
            'memory_timeline': memory_timeline,
            'visualization_type': 'memory'
        }
        
        # Override title if provided
        if title:
            visualization_data['custom_title'] = title
        
        self._export_html(visualization_data, safe_path)
    
    def export_html_dashboard(self, trace_nodes: List[TraceNode], path: str,
                             head_analysis: Optional[Dict[str, Any]] = None,
                             memory_profile: Optional[Dict[str, Any]] = None,
                             memory_timeline: Optional[MemoryTimelineData] = None,
                             title: Optional[str] = None) -> None:
        """
        Export interactive dashboard HTML visualization with multiple views.
        
        Args:
            trace_nodes: List of traced operations
            path: Output file path
            head_analysis: Optional task head analysis results
            memory_profile: Optional memory usage profile
            memory_timeline: Optional memory timeline data
            title: Optional custom title (defaults to UniAD Stage X Dashboard)
        """
        safe_path = self._sanitize_path(path)
        
        # Provide default analysis if not provided
        if head_analysis is None:
            head_analysis = self._create_default_head_analysis(trace_nodes)
        if memory_profile is None:
            memory_profile = self._create_default_memory_profile(trace_nodes)
        
        # Create visualization data structure for HTMLTemplateEngine
        visualization_data = {
            'trace_nodes': trace_nodes,
            'head_analysis': head_analysis,
            'memory_profile': memory_profile,
            'memory_timeline': memory_timeline,
            'visualization_type': 'dashboard'
        }
        
        # Override title if provided
        if title:
            visualization_data['custom_title'] = title
        
        self._export_html(visualization_data, safe_path)
    
    def export_html_comparison(self, models_data: List[Dict[str, Any]], path: str,
                              title: Optional[str] = None) -> None:
        """
        Export model comparison HTML visualization.
        
        Args:
            models_data: List of model data dictionaries, each containing:
                - trace_nodes: List of traced operations
                - head_analysis: Task head analysis results  
                - memory_profile: Memory usage profile
                - model_name: Name of the model
            path: Output file path
            title: Optional custom title (defaults to UniAD Model Comparison)
        """
        safe_path = self._sanitize_path(path)
        
        # Create comparison visualization data
        visualization_data = {
            'trace_nodes': [],
            'head_analysis': {},
            'memory_profile': {},
            'models_data': models_data,
            'visualization_type': 'comparison'
        }
        
        # Override title if provided
        if title:
            visualization_data['custom_title'] = title
        
        self._export_html(visualization_data, safe_path)
    
    def _sanitize_path(self, path: str) -> str:
        """
        Sanitize file path to prevent directory traversal attacks.
        
        Args:
            path: Input file path
            
        Returns:
            Sanitized path string
            
        Raises:
            ValueError: If path is invalid or contains dangerous patterns
        """
        if not path or not isinstance(path, str):
            raise ValueError("Path must be a non-empty string")
        
        # Remove any path traversal attempts
        clean_path = os.path.normpath(path)
        
        # Check for dangerous patterns
        if '..' in clean_path or clean_path.startswith('/'):
            raise ValueError("Path contains invalid characters or directory traversal")
        
        # Ensure path is within output directory if it's relative
        if not os.path.isabs(clean_path):
            clean_path = str(self.output_dir / clean_path)
        
        # Final validation
        try:
            Path(clean_path).resolve()
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid file path: {e}")
        
        return clean_path
    
    def _export_svg(self, visualization_data: Dict[str, Any], path: str) -> None:
        """Export SVG format (placeholder implementation)"""
        # This would typically convert D3.js visualizations to SVG
        # For now, we'll create a basic SVG representation
        trace_nodes = visualization_data.get('trace_nodes', [])
        
        svg_content = self._generate_basic_svg(trace_nodes)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
    
    def _export_png(self, visualization_data: Dict[str, Any], path: str, dpi: int) -> None:
        """Export PNG format using PIL or alternative method"""
        if Image is None or ImageDraw is None:
            raise ImportError("PIL (Pillow) is required for PNG export. Install with: pip install Pillow")
        
        # For now, create a placeholder PNG
        # In a full implementation, this would render the actual visualization
        img = Image.new('RGB', (1200, 800), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw title
        draw.text((50, 50), f"UniAD Stage {self.stage} Visualization", fill='black')
        
        # Add some basic visualization elements
        trace_nodes = visualization_data.get('trace_nodes', [])
        if trace_nodes:
            draw.text((50, 100), f"Total Operations: {len(trace_nodes)}", fill='black')
            
            # Draw simple node representation
            y_pos = 150
            for i, node in enumerate(trace_nodes[:10]):  # Show first 10 nodes
                text = f"{i+1}. {getattr(node, 'operation', 'unknown')} ({getattr(node, 'memory_usage', 0.0):.1f}MB)"
                draw.text((50, y_pos), text, fill='blue')
                y_pos += 30
        
        img.save(path, dpi=(dpi, dpi))
    
    def _export_html(self, visualization_data: Dict[str, Any], path: str) -> None:
        """Export interactive HTML format using HTMLTemplateEngine"""
        trace_nodes = visualization_data.get('trace_nodes', [])
        head_analysis = visualization_data.get('head_analysis', {})
        memory_profile = visualization_data.get('memory_profile', {})
        memory_timeline = visualization_data.get('memory_timeline')
        visualization_type = visualization_data.get('visualization_type', 'dataflow')
        
        try:
            # Create visualization data using HTMLTemplateEngine
            viz_data = self._create_html_visualization_data(
                visualization_type, trace_nodes, head_analysis, memory_profile, memory_timeline, visualization_data
            )
            
            # Override title if custom title provided
            if 'custom_title' in visualization_data:
                viz_data.title = visualization_data['custom_title']
            
            # Render HTML using template engine
            html_content = self.html_template_engine.render_template(viz_data, visualization_type)
            
            # Export the HTML file
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
        except Exception as e:
            # Fallback to legacy method if HTMLTemplateEngine fails
            print(f"Warning: HTMLTemplateEngine failed ({e}), falling back to legacy method")
            html_content = self.interactive_visualizer.generate_interactive_html(
                trace_nodes, head_analysis, memory_profile
            )
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html_content)
    
    def _export_mermaid(self, visualization_data: Dict[str, Any], path: str) -> None:
        """Export Mermaid diagram format"""
        trace_nodes = visualization_data.get('trace_nodes', [])
        head_analysis = visualization_data.get('head_analysis', {})
        memory_profile = visualization_data.get('memory_profile', {})
        
        mermaid_content = self.dataflow_visualizer.generate_mermaid(
            trace_nodes, head_analysis, memory_profile
        )
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(mermaid_content)
    
    def _export_markdown_report(self, visualization_data: Dict[str, Any], path: str) -> None:
        """Export comprehensive Markdown report"""
        report_content = self._generate_comprehensive_markdown_report(visualization_data)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def _export_comprehensive_report(self, analysis_data: Dict[str, Any], path: str,
                                   include_recommendations: bool = True,
                                   include_key_findings: bool = True) -> None:
        """Export comprehensive analysis report with all data in Markdown format"""
        report_content = self._generate_comprehensive_markdown_report(
            analysis_data, include_recommendations, include_key_findings
        )
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def _export_html_report(self, analysis_data: Dict[str, Any], path: str,
                           include_recommendations: bool = True,
                           include_key_findings: bool = True) -> None:
        """Export comprehensive analysis report as interactive HTML"""
        try:
            # Create visualization data for HTML template engine
            visualization_data = {
                'trace_nodes': analysis_data.get('trace_nodes', []),
                'head_analysis': analysis_data.get('head_analysis', {}),
                'memory_profile': analysis_data.get('memory_profile', {}),
                'memory_timeline': analysis_data.get('memory_timeline'),
                'visualization_type': 'report',
                'include_recommendations': include_recommendations,
                'include_key_findings': include_key_findings,
                'custom_title': f'UniAD Stage {self.stage} Analysis Report'
            }
            
            self._export_html(visualization_data, path)
            
        except Exception as e:
            # Fallback to markdown-based HTML if template engine fails
            print(f"Warning: HTML report generation failed ({e}), falling back to markdown-based HTML")
            markdown_content = self._generate_comprehensive_markdown_report(
                analysis_data, include_recommendations, include_key_findings
            )
            
            # Convert markdown to basic HTML
            html_content = self._markdown_to_html(markdown_content)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html_content)
    
    def _generate_comprehensive_markdown_report(self, analysis_data: Dict[str, Any],
                                              include_recommendations: bool = True,
                                              include_key_findings: bool = True) -> str:
        """Generate comprehensive Markdown report with all analysis results, key findings, and recommendations"""
        from datetime import datetime
        
        report = []
        
        # Header with metadata
        report.append(f"# UniAD PyTorch Operation Analysis Report - Stage {self.stage}")
        report.append("")
        report.append("**Generated by**: PyTorch Operation Tracer v1.0.0")
        report.append(f"**Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("**Model Framework**: UniAD (Planning-oriented Autonomous Driving)")
        report.append(f"**Training Stage**: Stage {self.stage} {'(Perception)' if self.stage == 1 else '(End-to-End)'}")
        report.append("")
        report.append("---")
        report.append("")
        
        # Executive Summary with enhanced metrics
        report.append("## Executive Summary")
        report.append("")
        
        trace_nodes = analysis_data.get('trace_nodes', [])
        head_analysis = analysis_data.get('head_analysis', {})
        memory_profile = analysis_data.get('memory_profile', {})
        
        if trace_nodes:
            total_memory = sum(getattr(node, 'memory_usage', 0.0) for node in trace_nodes)
            total_compute = sum(getattr(node, 'compute_time', 0.0) for node in trace_nodes)
            peak_memory = memory_profile.get('peak_memory_mb', max((getattr(node, 'memory_usage', 0.0) for node in trace_nodes), default=0.0) if trace_nodes else 0.0)
            
            # Calculate performance metrics
            memory_efficiency = (total_memory / len(trace_nodes)) if trace_nodes else 0
            compute_efficiency = (total_compute / len(trace_nodes)) if trace_nodes else 0
            
            # Memory budget analysis (Stage 1: ~30GB, Stage 2: ~17GB)
            expected_memory = 17000 if self.stage == 2 else 30000
            memory_utilization = (total_memory / expected_memory) * 100
            
            report.append("### Core Metrics")
            report.append("")
            report.append(f"- **Total Operations**: {len(trace_nodes):,}")
            report.append(f"- **Total Memory Usage**: {total_memory:.1f} MB ({memory_utilization:.1f}% of expected budget)")
            report.append(f"- **Peak Memory Usage**: {peak_memory:.1f} MB")
            report.append(f"- **Total Compute Time**: {total_compute:.1f} ms")
            report.append(f"- **Memory Efficiency**: {memory_efficiency:.1f} MB per operation")
            report.append(f"- **Compute Efficiency**: {compute_efficiency:.1f} ms per operation")
            report.append("")
            
            # Task head summary
            if head_analysis and head_analysis.get('memory_by_head'):
                task_heads = list(head_analysis['memory_by_head'].keys())
                active_heads = [head for head, memory in head_analysis['memory_by_head'].items() if memory > 0]
                report.append(f"- **Active Task Heads**: {len(active_heads)}/{len(task_heads)} ({', '.join(active_heads)})")
                report.append("")
            
            # Performance status
            status_icon = "✅" if memory_utilization < 120 else "⚠️" if memory_utilization < 150 else "❌"
            status_text = "Optimal" if memory_utilization < 120 else "High" if memory_utilization < 150 else "Critical"
            report.append(f"### Performance Status: {status_icon} {status_text}")
            report.append("")
        else:
            report.append("⚠️ **No trace data available for analysis**")
            report.append("")
        
        # Task Head Analysis
        if head_analysis and head_analysis.get('memory_by_head'):
            report.append("## Task Head Analysis")
            report.append("")
            
            memory_by_head = head_analysis['memory_by_head']
            compute_by_head = head_analysis.get('compute_by_head', {})
            
            report.append("| Task Head | Memory (MB) | Compute (ms) | Percentage |")
            report.append("|-----------|-------------|--------------|------------|")
            
            total_head_memory = sum(memory_by_head.values())
            for head, memory in sorted(memory_by_head.items(), key=lambda x: x[1], reverse=True):
                compute = compute_by_head.get(head, 0)
                percentage = (memory / total_head_memory * 100) if total_head_memory > 0 else 0
                report.append(f"| {head} | {memory:.1f} | {compute:.1f} | {percentage:.1f}% |")
            
            report.append("")
        
        # Memory Profile
        if memory_profile:
            report.append("## Memory Profile")
            report.append("")
            
            total_memory = memory_profile.get('total_memory_mb', 0)
            peak_memory = memory_profile.get('peak_memory_mb', 0)
            
            report.append(f"- **Total Memory**: {total_memory:.1f} MB")
            report.append(f"- **Peak Memory**: {peak_memory:.1f} MB")
            
            # Top memory consumers
            top_consumers = memory_profile.get('top_consumers', [])
            if top_consumers:
                report.append("")
                report.append("### Top Memory Consumers")
                report.append("")
                
                for i, consumer in enumerate(top_consumers[:10], 1):
                    op = consumer.get('operation', 'Unknown')
                    memory = consumer.get('memory_mb', 0)
                    percentage = consumer.get('percentage', 0)
                    report.append(f"{i}. **{op}**: {memory:.1f} MB ({percentage:.1f}%)")
            
            report.append("")
        
        # Operation Breakdown
        if trace_nodes:
            report.append("## Operation Breakdown")
            report.append("")
            
            # Group by operation type
            op_types = {}
            for node in trace_nodes:
                operation = getattr(node, 'operation', 'unknown')
                op_type = operation.split('_')[0] if '_' in operation else operation
                if op_type not in op_types:
                    op_types[op_type] = {'count': 0, 'memory': 0, 'compute': 0}
                op_types[op_type]['count'] += 1
                op_types[op_type]['memory'] += getattr(node, 'memory_usage', 0.0)
                op_types[op_type]['compute'] += getattr(node, 'compute_time', 0.0)
            
            report.append("| Operation Type | Count | Total Memory (MB) | Total Compute (ms) |")
            report.append("|---------------|-------|-------------------|--------------------|")
            
            for op_type, stats in sorted(op_types.items(), key=lambda x: x[1]['memory'], reverse=True):
                report.append(f"| {op_type} | {stats['count']} | {stats['memory']:.1f} | {stats['compute']:.1f} |")
            
            report.append("")
        
        # Shape Transformations
        if trace_nodes:
            shape_transforms = [node for node in trace_nodes if node.shape_transform]
            if shape_transforms:
                report.append("## Shape Transformations")
                report.append("")
                
                transform_types = {}
                for node in shape_transforms:
                    transform = node.shape_transform
                    if transform not in transform_types:
                        transform_types[transform] = 0
                    transform_types[transform] += 1
                
                report.append(f"- **Total Shape Transformations**: {len(shape_transforms)}")
                for transform, count in sorted(transform_types.items(), key=lambda x: x[1], reverse=True):
                    report.append(f"- **{transform}**: {count} operations")
                
                report.append("")
        
        # Memory Timeline Analysis (if available)
        memory_timeline = analysis_data.get('memory_timeline')
        if memory_timeline and isinstance(memory_timeline, MemoryTimelineData):
            timeline_report = self.memory_visualizer.generate_markdown_report(memory_timeline)
            report.append(timeline_report)
            report.append("")
        
        # UniAD-Specific Analysis
        report.append("## UniAD-Specific Analysis")
        report.append("")
        
        # BEV operations
        bev_ops = [node for node in trace_nodes if getattr(node, 'is_bev_operation', False)]
        if bev_ops:
            bev_memory = sum(getattr(node, 'memory_usage', 0.0) for node in bev_ops)
            report.append(f"- **BEV Operations**: {len(bev_ops)} ({bev_memory:.1f} MB)")
        
        # Frozen operations
        frozen_ops = [node for node in trace_nodes if getattr(node, 'is_frozen', False)]
        if frozen_ops:
            frozen_memory = sum(getattr(node, 'memory_usage', 0.0) for node in frozen_ops)
            report.append(f"- **Frozen Operations**: {len(frozen_ops)} ({frozen_memory:.1f} MB)")
        
        # Temporal operations
        temporal_ops = [node for node in trace_nodes if getattr(node, 'temporal_index', None) is not None]
        if temporal_ops:
            temporal_memory = sum(getattr(node, 'memory_usage', 0.0) for node in temporal_ops)
            report.append(f"- **Temporal Operations**: {len(temporal_ops)} ({temporal_memory:.1f} MB)")
        
        report.append("")
        
        # Key Findings Section (new requirement)
        if include_key_findings and trace_nodes:
            report.append("## Key Findings")
            report.append("")
            
            findings = self._generate_key_findings(analysis_data)
            for category, finding_list in findings.items():
                if finding_list:
                    report.append(f"### {category}")
                    report.append("")
                    for finding in finding_list:
                        report.append(f"- {finding}")
                    report.append("")
        
        # Enhanced Optimization Recommendations
        if include_recommendations:
            report.append("## Optimization Recommendations")
            report.append("")
            
            recommendations = self._generate_optimization_recommendations(analysis_data)
            
            # Group recommendations by priority
            critical_recs = [rec for rec in recommendations if '❌' in rec or 'Critical' in rec]
            high_recs = [rec for rec in recommendations if '⚠️' in rec or 'High' in rec]
            medium_recs = [rec for rec in recommendations if rec not in critical_recs and rec not in high_recs]
            
            if critical_recs:
                report.append("### 🚨 Critical Priority")
                report.append("")
                for i, rec in enumerate(critical_recs, 1):
                    report.append(f"{i}. {rec}")
                report.append("")
            
            if high_recs:
                report.append("### ⚠️ High Priority")
                report.append("")
                for i, rec in enumerate(high_recs, 1):
                    report.append(f"{i}. {rec}")
                report.append("")
            
            if medium_recs:
                report.append("### 💡 General Optimizations")
                report.append("")
                for i, rec in enumerate(medium_recs, 1):
                    report.append(f"{i}. {rec}")
                report.append("")
        
        # Visualization Links (if generated)
        report.append("## Generated Visualizations")
        report.append("")
        report.append("The following visualizations have been generated alongside this report:")
        report.append("")
        report.append("- **Interactive HTML**: Browse operations with filtering and drill-down")
        report.append("- **Mermaid Diagram**: Architecture overview for documentation")
        report.append("- **Memory Timeline**: Memory usage patterns over time")
        report.append("- **PNG Export**: Static visualization for presentations")
        report.append("")
        
        # Footer
        report.append("---")
        report.append("")
        report.append(f"*Report generated by PyTorch Operation Tracer for UniAD Stage {self.stage}*")
        
        return "\n".join(report)
    
    def _generate_optimization_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis data"""
        recommendations = []
        
        trace_nodes = analysis_data.get('trace_nodes', [])
        head_analysis = analysis_data.get('head_analysis', {})
        
        if not trace_nodes:
            return ["No trace data available for analysis"]
        
        # Memory-based recommendations with priority indicators
        total_memory = sum(getattr(node, 'memory_usage', 0.0) for node in trace_nodes)
        expected_memory = 17000 if self.stage == 2 else 30000  # Expected for each stage
        memory_ratio = total_memory / expected_memory
        
        if memory_ratio > 2.0:
            recommendations.append(
                f"🚨 **Critical Memory Usage**: Current usage ({total_memory:.1f} MB) exceeds "
                f"expected Stage {self.stage} usage ({expected_memory} MB) by "
                f"{((memory_ratio-1)*100):.1f}%. Immediate action required: reduce batch size, "
                f"queue length, or enable gradient checkpointing."
            )
        elif memory_ratio > 1.5:
            recommendations.append(
                f"⚠️ **High Memory Usage**: Current usage ({total_memory:.1f} MB) exceeds "
                f"expected Stage {self.stage} usage ({expected_memory} MB) by "
                f"{((memory_ratio-1)*100):.1f}%. Consider reducing batch size "
                f"or queue length."
            )
        
        # Task head imbalance
        if head_analysis and head_analysis.get('memory_by_head'):
            memory_by_head = head_analysis['memory_by_head']
            active_heads = {k: v for k, v in memory_by_head.items() if v > 0}
            
            if active_heads:
                max_head_memory = max(active_heads.values())
                min_head_memory = min(active_heads.values()) if min(active_heads.values()) > 0 else 1
                
                if max_head_memory > min_head_memory * 5:
                    max_head = max(active_heads, key=lambda x: active_heads[x])
                    recommendations.append(
                        f"⚠️ **Task Head Imbalance**: {max_head} head uses significantly more memory "
                        f"({max_head_memory:.1f} MB) - {max_head_memory/min_head_memory:.1f}x more than smallest head. "
                        f"Consider optimizing {max_head} task head or rebalancing computational load."
                    )
        
        # Shape transformation issues
        shape_transforms = [node for node in trace_nodes if hasattr(node, 'shape_transform') and node.shape_transform]
        transform_ratio = len(shape_transforms) / len(trace_nodes) if trace_nodes else 0
        
        if transform_ratio > 0.5:
            recommendations.append(
                f"🚨 **Excessive Shape Transformations**: {len(shape_transforms)} operations "
                f"({transform_ratio*100:.1f}% of total) involve shape changes. This indicates "
                f"inefficient tensor layouts. Redesign data flow to minimize reshaping operations."
            )
        elif transform_ratio > 0.3:
            recommendations.append(
                f"⚠️ **High Shape Transformation Rate**: {len(shape_transforms)} operations "
                f"({transform_ratio*100:.1f}% of total) involve shape changes. Consider optimizing "
                f"tensor layouts to reduce reshaping overhead."
            )
        
        # Large single operations
        large_ops = [node for node in trace_nodes if getattr(node, 'memory_usage', 0.0) > 2000]
        very_large_ops = [node for node in trace_nodes if getattr(node, 'memory_usage', 0.0) > 5000]
        
        if very_large_ops:
            largest_op = max(very_large_ops, key=lambda x: x.memory_usage)
            recommendations.append(
                f"🚨 **Very Large Memory Operations**: {len(very_large_ops)} operations use >5GB each. "
                f"Largest: {largest_op.operation} ({largest_op.memory_usage:.1f} MB). "
                f"Critical: implement gradient checkpointing, operation splitting, or model parallelism."
            )
        elif large_ops:
            largest_op = max(large_ops, key=lambda x: x.memory_usage)
            recommendations.append(
                f"⚠️ **Large Memory Operations**: {len(large_ops)} operations use >2GB each. "
                f"Largest: {largest_op.operation} ({largest_op.memory_usage:.1f} MB). "
                f"Consider gradient checkpointing or operation splitting."
            )
        
        # Stage-specific recommendations
        if self.stage == 1:
            queue_ops = [node for node in trace_nodes if 'queue' in getattr(node, 'operation', 'unknown').lower()]
            if queue_ops:
                queue_memory = sum(getattr(node, 'memory_usage', 0.0) for node in queue_ops)
                if queue_memory > 15000:  # >15GB for queue operations
                    recommendations.append(
                        f"⚠️ **Stage 1 Queue Memory**: Queue operations use {queue_memory:.1f} MB. "
                        f"Reduce queue_length from 5 to 3 to save ~{queue_memory*0.4:.1f} MB. "
                        f"This will significantly reduce memory pressure."
                    )
                else:
                    recommendations.append(
                        f"💡 **Stage 1 Queue Optimization**: Queue operations use {queue_memory:.1f} MB. "
                        f"Consider reducing queue_length from 5 to 3 for {queue_memory*0.4:.1f} MB savings."
                    )
        elif self.stage == 2:
            frozen_ops = [node for node in trace_nodes if hasattr(node, 'is_frozen') and node.is_frozen]
            if not frozen_ops:
                recommendations.append(
                    "🚨 **Stage 2 BEV Freezing**: No frozen BEV encoder operations detected. "
                    "Critical: Ensure BEV encoder is frozen from Stage 1 checkpoint to reduce memory usage. "
                    "This is essential for Stage 2 training efficiency."
                )
            else:
                frozen_memory = sum(getattr(node, 'memory_usage', 0.0) for node in frozen_ops)
                recommendations.append(
                    f"💡 **Stage 2 Architecture**: BEV encoder properly frozen ({len(frozen_ops)} ops, "
                    f"{frozen_memory:.1f} MB). This enables efficient end-to-end training."
                )
        
        # BEV-specific recommendations
        bev_ops = [node for node in trace_nodes if getattr(node, 'is_bev_operation', False)]
        if bev_ops:
            bev_memory = sum(getattr(node, 'memory_usage', 0.0) for node in bev_ops)
            if bev_memory > 15000:  # >15GB for BEV operations
                recommendations.append(
                    f"🚨 **Critical BEV Memory Usage**: BEV operations use {bev_memory:.1f} MB. "
                    f"Immediate optimization needed: reduce BEV grid size (200x200 → 100x100), "
                    f"decrease feature dimensions, or implement BEV feature compression."
                )
            elif bev_memory > 8000:  # >8GB for BEV operations
                recommendations.append(
                    f"⚠️ **High BEV Memory Usage**: BEV operations use {bev_memory:.1f} MB. "
                    f"Consider using smaller BEV grid (e.g., 150x150 instead of 200x200) "
                    f"or reducing feature dimensions from 256 to 128."
                )
            else:
                recommendations.append(
                    f"💡 **BEV Memory Efficiency**: BEV operations use {bev_memory:.1f} MB "
                    f"({(bev_memory/total_memory)*100:.1f}% of total). Current usage appears optimal."
                )
        
        # Mixed precision recommendations
        fp32_ops = [node for node in trace_nodes if hasattr(node, 'dtype') and 'float32' in str(node.dtype)]
        fp32_ratio = len(fp32_ops) / len(trace_nodes) if trace_nodes else 0
        
        if fp32_ratio > 0.9:
            potential_savings = total_memory * 0.35  # Estimate 35% savings
            recommendations.append(
                f"⚠️ **Mixed Precision Opportunity**: {fp32_ratio*100:.1f}% operations use FP32. "
                f"Enable mixed precision training (FP16/BF16) to save ~{potential_savings:.1f} MB memory "
                f"({(potential_savings/total_memory)*100:.1f}% reduction). Use automatic mixed precision (AMP)."
            )
        elif fp32_ratio > 0.7:
            potential_savings = total_memory * 0.25  # Estimate 25% savings
            recommendations.append(
                f"💡 **Mixed Precision Consideration**: {fp32_ratio*100:.1f}% operations use FP32. "
                f"Consider mixed precision training to save ~{potential_savings:.1f} MB memory."
            )
        
        # Performance and efficiency recommendations
        compute_intensive_ops = [node for node in trace_nodes if getattr(node, 'compute_time', 0.0) > 100]  # >100ms
        if compute_intensive_ops:
            compute_memory = sum(getattr(node, 'memory_usage', 0.0) for node in compute_intensive_ops)
            recommendations.append(
                f"💡 **Compute Optimization**: {len(compute_intensive_ops)} operations take >100ms each, "
                f"using {compute_memory:.1f} MB total. Consider optimizing these compute bottlenecks "
                f"or implementing operator fusion."
            )
        
        # Data type analysis
        dtype_distribution = {}
        for node in trace_nodes:
            if hasattr(node, 'dtype'):
                dtype_str = str(node.dtype)
                dtype_distribution[dtype_str] = dtype_distribution.get(dtype_str, 0) + 1
        
        if dtype_distribution and len(dtype_distribution) > 2:
            recommendations.append(
                f"💡 **Data Type Consistency**: Multiple data types detected: {', '.join(dtype_distribution.keys())}. "
                f"Consider standardizing on FP16/BF16 for memory efficiency while maintaining FP32 for critical operations."
            )
        
        # Default recommendations if none found
        if not recommendations:
            recommendations.extend([
                f"💡 **Performance Status**: Current memory usage ({total_memory:.1f} MB) appears reasonable "
                f"for UniAD Stage {self.stage} ({(memory_ratio*100):.1f}% of expected budget).",
                "💡 **Monitoring**: Continue monitoring memory patterns during training. "
                "Use memory timeline visualization to identify bottlenecks.",
                "💡 **Optimization Opportunities**: Consider profiling with different batch sizes or "
                "queue lengths to find optimal configuration."
            ])
        
        return recommendations
    
    def _generate_key_findings(self, analysis_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate key findings from analysis data to identify bottlenecks and patterns"""
        findings = {
            "Performance Bottlenecks": [],
            "Memory Patterns": [],
            "Task Head Analysis": [],
            "Architecture Insights": [],
            "Efficiency Metrics": []
        }
        
        trace_nodes = analysis_data.get('trace_nodes', [])
        head_analysis = analysis_data.get('head_analysis', {})
        
        if not trace_nodes:
            findings["Performance Bottlenecks"].append("No trace data available for analysis")
            return findings
        
        # Performance Bottlenecks
        total_memory = sum(getattr(node, 'memory_usage', 0.0) for node in trace_nodes)
        total_compute = sum(getattr(node, 'compute_time', 0.0) for node in trace_nodes)
        
        # Identify top memory consumers
        top_memory_ops = sorted(trace_nodes, key=lambda x: x.memory_usage, reverse=True)[:5]
        if top_memory_ops and top_memory_ops[0].memory_usage > total_memory * 0.1:
            findings["Performance Bottlenecks"].append(
                f"Top memory consumer: {top_memory_ops[0].operation} uses {top_memory_ops[0].memory_usage:.1f} MB "
                f"({(top_memory_ops[0].memory_usage/total_memory)*100:.1f}% of total)"
            )
        
        # Identify compute bottlenecks
        top_compute_ops = sorted(trace_nodes, key=lambda x: x.compute_time, reverse=True)[:5]
        if top_compute_ops and top_compute_ops[0].compute_time > total_compute * 0.1:
            findings["Performance Bottlenecks"].append(
                f"Top compute consumer: {top_compute_ops[0].operation} takes {top_compute_ops[0].compute_time:.1f} ms "
                f"({(top_compute_ops[0].compute_time/total_compute)*100:.1f}% of total)"
            )
        
        # Memory Patterns
        expected_memory = 17000 if self.stage == 2 else 30000
        memory_ratio = total_memory / expected_memory
        
        if memory_ratio > 1.5:
            findings["Memory Patterns"].append(
                f"High memory usage detected: {total_memory:.1f} MB exceeds expected {expected_memory} MB by {((memory_ratio-1)*100):.1f}%"
            )
        elif memory_ratio < 0.7:
            findings["Memory Patterns"].append(
                f"Low memory utilization: {total_memory:.1f} MB is only {(memory_ratio*100):.1f}% of expected {expected_memory} MB"
            )
        
        # Identify memory growth patterns
        temporal_ops = [node for node in trace_nodes if hasattr(node, 'temporal_index') and node.temporal_index is not None]
        if temporal_ops:
            findings["Memory Patterns"].append(
                f"Temporal processing uses {sum(getattr(node, 'memory_usage', 0.0) for node in temporal_ops):.1f} MB across {len(temporal_ops)} operations"
            )
        
        # Task Head Analysis
        if head_analysis and head_analysis.get('memory_by_head'):
            memory_by_head = head_analysis['memory_by_head']
            
            # Find dominant task head
            if memory_by_head:
                max_head = max(memory_by_head, key=memory_by_head.get)
                max_memory = memory_by_head[max_head]
                total_head_memory = sum(memory_by_head.values())
                
                if max_memory > total_head_memory * 0.4:
                    findings["Task Head Analysis"].append(
                        f"Dominant task head: {max_head} consumes {max_memory:.1f} MB ({(max_memory/total_head_memory)*100:.1f}% of task head memory)"
                    )
                
                # Check for inactive heads
                inactive_heads = [head for head, memory in memory_by_head.items() if memory == 0]
                if inactive_heads:
                    findings["Task Head Analysis"].append(
                        f"ℹ️ Inactive task heads detected: {', '.join(inactive_heads)} (expected for Stage {self.stage})"
                    )
                
                # Balance analysis
                if memory_by_head:
                    memory_values = list(memory_by_head.values())
                    if max(memory_values) > min(memory_values) * 5 and min(memory_values) > 0:
                        findings["Task Head Analysis"].append(
                            f"⚠️ Task head imbalance: {max(memory_values)/min(memory_values):.1f}x difference between highest and lowest"
                        )
        
        # Architecture Insights
        bev_ops = [node for node in trace_nodes if getattr(node, 'is_bev_operation', False)]
        if bev_ops:
            bev_memory = sum(getattr(node, 'memory_usage', 0.0) for node in bev_ops)
            findings["Architecture Insights"].append(
                f"BEV operations: {len(bev_ops)} operations using {bev_memory:.1f} MB ({(bev_memory/total_memory)*100:.1f}% of total)"
            )
        
        frozen_ops = [node for node in trace_nodes if getattr(node, 'is_frozen', False)]
        if self.stage == 2:
            if frozen_ops:
                frozen_memory = sum(getattr(node, 'memory_usage', 0.0) for node in frozen_ops)
                findings["Architecture Insights"].append(
                    f"Frozen operations: {len(frozen_ops)} operations using {frozen_memory:.1f} MB (BEV encoder frozen as expected)"
                )
            else:
                findings["Architecture Insights"].append(
                    "⚠️ No frozen operations detected - BEV encoder should be frozen in Stage 2"
                )
        
        # Shape transformations
        shape_transforms = [node for node in trace_nodes if hasattr(node, 'shape_transform') and node.shape_transform]
        if shape_transforms:
            transform_ratio = len(shape_transforms) / len(trace_nodes)
            findings["Architecture Insights"].append(
                f"ℹ️ Shape transformations: {len(shape_transforms)} operations ({transform_ratio*100:.1f}% of total) involve tensor reshaping"
            )
        
        # Efficiency Metrics
        avg_memory_per_op = total_memory / len(trace_nodes)
        avg_compute_per_op = total_compute / len(trace_nodes)
        
        findings["Efficiency Metrics"].append(
            f"Average memory efficiency: {avg_memory_per_op:.1f} MB per operation"
        )
        findings["Efficiency Metrics"].append(
            f"Average compute efficiency: {avg_compute_per_op:.1f} ms per operation"
        )
        
        # Memory-to-compute ratio
        if total_compute > 0:
            memory_compute_ratio = total_memory / total_compute
            findings["Efficiency Metrics"].append(
                f"Memory-to-compute ratio: {memory_compute_ratio:.1f} MB per ms (higher values indicate memory-intensive operations)"
            )
        
        # Remove empty categories
        return {k: v for k, v in findings.items() if v}
    
    def _validate_analysis_data(self, analysis_data: Dict[str, Any]) -> bool:
        """Validate that analysis data contains required fields for report generation"""
        required_fields = ['trace_nodes']
        
        for field in required_fields:
            if field not in analysis_data:
                return False
        
        # Check if trace_nodes is a list
        trace_nodes = analysis_data.get('trace_nodes')
        if not isinstance(trace_nodes, list):
            return False
        
        # Basic validation of trace node structure
        if trace_nodes:
            sample_node = trace_nodes[0]
            if not hasattr(sample_node, 'operation') or not hasattr(sample_node, 'memory_usage'):
                return False
        
        return True
    
    def _markdown_to_html(self, markdown_content: str) -> str:
        """Convert markdown content to basic HTML for fallback report generation"""
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '    <meta charset="UTF-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'    <title>UniAD Stage {self.stage} Analysis Report</title>',
            '    <style>',
            '        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }',
            '        h1, h2, h3 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }',
            '        table { border-collapse: collapse; width: 100%; margin: 20px 0; }',
            '        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }',
            '        th { background-color: #f8f9fa; font-weight: bold; }',
            '        .status-optimal { color: #27ae60; font-weight: bold; }',
            '        .status-high { color: #f39c12; font-weight: bold; }',
            '        .status-critical { color: #e74c3c; font-weight: bold; }',
            '        .finding { background-color: #f8f9fa; padding: 10px; border-left: 4px solid #3498db; margin: 10px 0; }',
            '        .recommendation { background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }',
            '        pre, code { background-color: #f4f4f4; padding: 10px; border-radius: 4px; }',
            '        .summary-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }',
            '        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }',
            '    </style>',
            '</head>',
            '<body>'
        ]
        
        # Convert markdown to HTML (basic implementation)
        lines = markdown_content.split('\n')
        in_table = False
        in_code_block = False
        
        for line in lines:
            if line.startswith('```'):
                if in_code_block:
                    html_parts.append('</pre>')
                else:
                    html_parts.append('<pre><code>')
                in_code_block = not in_code_block
                continue
                
            if in_code_block:
                html_parts.append(line)
                continue
            
            if line.startswith('# '):
                html_parts.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                html_parts.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_parts.append(f'<h3>{line[4:]}</h3>')
            elif line.startswith('|') and '|' in line[1:]:
                if not in_table:
                    html_parts.append('<table>')
                    in_table = True
                
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if all(cell.startswith('-') or cell == '' for cell in cells):
                    continue  # Skip separator row
                
                if '**' in line:
                    html_parts.append('<tr>')
                    for cell in cells:
                        html_parts.append(f'<th>{cell.replace("**", "")}</th>')
                    html_parts.append('</tr>')
                else:
                    html_parts.append('<tr>')
                    for cell in cells:
                        html_parts.append(f'<td>{cell}</td>')
                    html_parts.append('</tr>')
            else:
                if in_table:
                    html_parts.append('</table>')
                    in_table = False
                
                if line.strip() == '':
                    html_parts.append('<br/>')
                elif line.startswith('- '):
                    html_parts.append(f'<li>{line[2:]}</li>')
                elif line.startswith('---'):
                    html_parts.append('<hr/>')
                else:
                    # Convert bold markdown
                    line = line.replace('**', '<strong>').replace('**', '</strong>')
                    html_parts.append(f'<p>{line}</p>')
        
        if in_table:
            html_parts.append('</table>')
        
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)
    
    def _create_default_head_analysis(self, trace_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Create default task head analysis if not provided"""
        memory_by_head = {}
        compute_by_head = {}
        
        for node in trace_nodes:
            task_head = getattr(node, 'task_head', None)
            if task_head:
                if task_head not in memory_by_head:
                    memory_by_head[task_head] = 0
                    compute_by_head[task_head] = 0
                
                memory_by_head[task_head] += getattr(node, 'memory_usage', 0.0)
                compute_by_head[task_head] += getattr(node, 'compute_time', 0.0)
        
        return {
            'memory_by_head': memory_by_head,
            'compute_by_head': compute_by_head,
            'total_heads': len(memory_by_head)
        }
    
    def _create_default_memory_profile(self, trace_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Create default memory profile if not provided"""
        if not trace_nodes:
            return {'total_memory_mb': 0, 'peak_memory_mb': 0, 'top_consumers': []}
        
        total_memory = sum(getattr(node, 'memory_usage', 0.0) for node in trace_nodes)
        peak_memory = max(getattr(node, 'memory_usage', 0.0) for node in trace_nodes)
        
        # Top consumers
        sorted_nodes = sorted(trace_nodes, key=lambda x: x.memory_usage, reverse=True)
        top_consumers = []
        for i, node in enumerate(sorted_nodes[:10]):
            percentage = (getattr(node, 'memory_usage', 0.0) / total_memory * 100) if total_memory > 0 else 0
            top_consumers.append({
                'operation': getattr(node, 'operation', 'unknown'),
                'memory_mb': getattr(node, 'memory_usage', 0.0),
                'percentage': percentage,
                'module_path': getattr(node, 'module_path', 'unknown')
            })
        
        return {
            'total_memory_mb': total_memory,
            'peak_memory_mb': peak_memory,
            'top_consumers': top_consumers,
            'operation_count': len(trace_nodes)
        }
    
    def _generate_basic_svg(self, trace_nodes: List[TraceNode]) -> str:
        """Generate basic SVG representation of operations"""
        if not trace_nodes:
            return """<?xml version="1.0" encoding="UTF-8"?>
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
    <text x="50" y="50" font-family="Arial" font-size="16">No trace data available</text>
</svg>"""
        
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">',
            f'<text x="50" y="30" font-family="Arial" font-size="18" font-weight="bold">UniAD Stage {self.stage} Operations</text>',
            f'<text x="50" y="55" font-family="Arial" font-size="14">Total Operations: {len(trace_nodes)}</text>'
        ]
        
        # Draw simple bars for top operations
        top_nodes = sorted(trace_nodes, key=lambda x: x.memory_usage, reverse=True)[:20]
        max_memory = max(getattr(node, 'memory_usage', 0.0) for node in top_nodes) if top_nodes else 1
        
        y_pos = 100
        bar_height = 25
        max_bar_width = 800
        
        for node in top_nodes:
            bar_width = (getattr(node, 'memory_usage', 0.0) / max_memory) * max_bar_width if max_memory > 0 else 0
            color = self._get_svg_color(getattr(node, 'task_head', 'default'))
            
            # Draw bar
            svg_parts.append(
                f'<rect x="50" y="{y_pos}" width="{bar_width}" height="{bar_height-2}" '
                f'fill="{color}" stroke="#333" stroke-width="1"/>'
            )
            
            # Draw label
            label = f"{getattr(node, 'operation', 'unknown')} ({getattr(node, 'memory_usage', 0.0):.1f}MB)"
            if len(label) > 50:
                label = label[:47] + "..."
            
            svg_parts.append(
                f'<text x="55" y="{y_pos + bar_height//2 + 5}" font-family="Arial" '
                f'font-size="12" fill="white">{label}</text>'
            )
            
            y_pos += bar_height + 5
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
    
    def _get_svg_color(self, task_head: Optional[str]) -> str:
        """Get SVG color for task head"""
        colors = {
            'track': '#ff6347',      # Tomato
            'seg': '#32cd32',        # Lime green
            'motion': '#4169e1',     # Royal blue
            'occ': '#ff1493',        # Deep pink
            'planning': '#9932cc',   # Dark violet
            'bev': '#ffa500',        # Orange
            None: '#696969'          # Dim gray
        }
        return colors.get(task_head, '#696969')
    
    def _create_html_visualization_data(self, visualization_type: str, 
                                       trace_nodes: List[TraceNode],
                                       head_analysis: Dict[str, Any],
                                       memory_profile: Dict[str, Any],
                                       memory_timeline: Optional[MemoryTimelineData],
                                       visualization_data: Optional[Dict[str, Any]] = None) -> VisualizationData:
        """
        Create VisualizationData for HTMLTemplateEngine.
        
        Args:
            visualization_type: Type of visualization (dataflow, memory, comparison, dashboard)
            trace_nodes: List of traced operations
            head_analysis: Task head analysis results
            memory_profile: Memory usage profile
            memory_timeline: Optional memory timeline data
            
        Returns:
            VisualizationData object for template engine
        """
        from datetime import datetime
        
        # Create title based on visualization type and stage
        titles = {
            'dataflow': f'UniAD Stage {self.stage} Dataflow Visualization',
            'memory': f'UniAD Stage {self.stage} Memory Timeline',
            'comparison': 'UniAD Model Comparison',
            'dashboard': f'UniAD Stage {self.stage} Analysis Dashboard'
        }
        title = titles.get(visualization_type, f'UniAD Stage {self.stage} Visualization')
        
        # Create visualization data based on type
        if visualization_type == 'dataflow':
            data = self._create_dataflow_data(trace_nodes, head_analysis, memory_profile)
        elif visualization_type == 'memory':
            data = self._create_memory_data(memory_timeline, memory_profile)
        elif visualization_type == 'comparison':
            # Check if models_data is provided for comparison
            models_data = visualization_data.get('models_data') if visualization_data else None
            if models_data:
                data = self._create_comparison_data_from_models(models_data)
            else:
                data = self._create_comparison_data(trace_nodes, head_analysis, memory_profile)
        elif visualization_type == 'dashboard':
            data = self._create_dashboard_data(trace_nodes, head_analysis, memory_profile, memory_timeline)
        else:
            # Default to dataflow
            data = self._create_dataflow_data(trace_nodes, head_analysis, memory_profile)
        
        # Create configuration based on InteractiveConfig
        config = {
            'max_nodes_visible': self.max_nodes,
            'color_scheme': 'operation',
            'show_shapes': True,
            'enable_zoom': True,
            'enable_pan': True,
            'enable_filters': True,
            'stage': self.stage
        }
        
        # Create metadata
        metadata = {
            'generator': 'PyTorch Operation Tracer',
            'version': '1.0.0',
            'framework': 'UniAD',
            'stage': self.stage,
            'visualization_type': visualization_type,
            'node_count': len(trace_nodes),
            'head_count': len(head_analysis.get('memory_by_head', {})),
            'total_memory_mb': memory_profile.get('total_memory_mb', 0),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        return VisualizationData(
            title=title,
            data=data,
            config=config,
            metadata=metadata
        )
    
    def _create_dataflow_data(self, trace_nodes: List[TraceNode], 
                             head_analysis: Dict[str, Any], 
                             memory_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create dataflow visualization data"""
        # Use InteractiveVisualizer to create structured data
        viz_data = self.interactive_visualizer.create_visualization_data(
            trace_nodes, head_analysis, memory_profile
        )
        
        return viz_data
    
    def _create_memory_data(self, memory_timeline: Optional[MemoryTimelineData],
                           memory_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create memory visualization data"""
        data = {
            'timeline_points': [],
            'memory_pressure': 0.5,
            'peak_memory_mb': memory_profile.get('peak_memory_mb', 0),
            'average_memory_mb': memory_profile.get('total_memory_mb', 0),
            'memory_stats': memory_profile
        }
        
        if memory_timeline:
            # Extract timeline points if available
            try:
                if hasattr(memory_timeline, 'timeline_points'):
                    data['timeline_points'] = memory_timeline.timeline_points
                if hasattr(memory_timeline, 'memory_pressure'):
                    data['memory_pressure'] = memory_timeline.memory_pressure
            except Exception as e:
                print(f"Warning: Could not extract memory timeline data: {e}")
        
        return data
    
    def _create_comparison_data(self, trace_nodes: List[TraceNode],
                               head_analysis: Dict[str, Any],
                               memory_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparison visualization data"""
        return {
            'models': [
                {
                    'id': f'uniad_stage_{self.stage}',
                    'name': f'UniAD Stage {self.stage}',
                    'metrics': {
                        'node_count': len(trace_nodes),
                        'memory_mb': memory_profile.get('total_memory_mb', 0),
                        'task_heads': list(head_analysis.get('memory_by_head', {}).keys())
                    }
                }
            ],
            'comparison_metrics': ['node_count', 'memory_mb', 'task_heads']
        }
    
    def _create_comparison_data_from_models(self, models_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison visualization data from multiple models"""
        models = []
        
        for i, model_data in enumerate(models_data):
            model_name = model_data.get('model_name', f'Model {i+1}')
            trace_nodes = model_data.get('trace_nodes', [])
            head_analysis = model_data.get('head_analysis', {})
            memory_profile = model_data.get('memory_profile', {})
            
            model_info = {
                'id': f'model_{i}',
                'name': model_name,
                'metrics': {
                    'node_count': len(trace_nodes),
                    'memory_mb': memory_profile.get('total_memory_mb', 0),
                    'task_heads': list(head_analysis.get('memory_by_head', {}).keys()),
                    'task_head_count': len(head_analysis.get('memory_by_head', {})),
                    'compute_ms': sum(getattr(node, 'compute_time', 0.0) for node in trace_nodes),
                    'peak_memory_mb': memory_profile.get('peak_memory_mb', 0)
                }
            }
            models.append(model_info)
        
        return {
            'models': models,
            'comparison_metrics': ['node_count', 'memory_mb', 'task_heads', 'compute_ms', 'peak_memory_mb']
        }
    
    def _create_dashboard_data(self, trace_nodes: List[TraceNode],
                              head_analysis: Dict[str, Any],
                              memory_profile: Dict[str, Any],
                              memory_timeline: Optional[MemoryTimelineData]) -> Dict[str, Any]:
        """Create dashboard visualization data"""
        return {
            'dataflow': self._create_dataflow_data(trace_nodes, head_analysis, memory_profile),
            'memory': self._create_memory_data(memory_timeline, memory_profile),
            'summary': {
                'total_nodes': len(trace_nodes),
                'total_memory_mb': memory_profile.get('total_memory_mb', 0),
                'task_head_count': len(head_analysis.get('memory_by_head', {})),
                'stage': self.stage
            }
        }
    
    def compare_model_versions(self, version_analysis_data: List[Dict[str, Any]], 
                             path: str,
                             comparison_title: Optional[str] = None,
                             format: str = 'markdown',
                             include_diff_visualizations: bool = True,
                             include_metrics_comparison: bool = True,
                             include_architectural_diff: bool = True) -> None:
        """
        Compare different model versions and generate diff visualizations highlighting changes.
        
        This method implements Requirement 5.3 by generating model version comparisons that show
        differences between two or more model versions, highlighting architectural changes and
        comparing memory and compute metrics. Supports UniAD Stage 1 vs Stage 2 comparisons.
        
        Args:
            version_analysis_data: List of analysis data dictionaries, each containing:
                - version_name: Name/identifier of the model version (e.g., "Stage 1", "v1.0.1")
                - trace_nodes: List of traced operations
                - head_analysis: Task head analysis results
                - memory_profile: Memory usage profile
                - version_metadata: Optional metadata about the version (checkpoint path, config, etc.)
            path: Output file path for the comparison report
            comparison_title: Optional custom title (defaults to "UniAD Model Version Comparison")
            format: Report format - 'markdown' for .md files, 'html' for interactive reports
            include_diff_visualizations: Whether to generate diff visualizations (default: True)
            include_metrics_comparison: Whether to include detailed metrics comparison (default: True)
            include_architectural_diff: Whether to show architectural differences (default: True)
            
        Returns:
            None - Comparison report is written to the specified file path
            
        Raises:
            ValueError: If path is invalid, data is incomplete, or format is unsupported
            IOError: If file cannot be written due to permissions or disk space
        """
        if len(version_analysis_data) < 2:
            raise ValueError("At least two model versions are required for comparison")
        
        safe_path = self._sanitize_path(path)
        
        # Validate analysis data for all versions
        for i, version_data in enumerate(version_analysis_data):
            if not self._validate_version_analysis_data(version_data):
                raise ValueError(f"Analysis data for version {i+1} is incomplete or invalid")
        
        # Generate comparison report based on format
        if format.lower() == 'markdown':
            self._export_version_comparison_report(
                version_analysis_data, safe_path, comparison_title,
                include_diff_visualizations, include_metrics_comparison, include_architectural_diff
            )
        elif format.lower() == 'html':
            self._export_version_comparison_html(
                version_analysis_data, safe_path, comparison_title,
                include_diff_visualizations, include_metrics_comparison, include_architectural_diff
            )
        else:
            raise ValueError(f"Unsupported comparison format: {format}. Supported formats: 'markdown', 'html'")

    def _validate_version_analysis_data(self, version_data: Dict[str, Any]) -> bool:
        """Validate that version analysis data contains required fields for comparison"""
        required_fields = ['version_name', 'trace_nodes']
        
        for field in required_fields:
            if field not in version_data:
                return False
        
        # Check if trace_nodes is a list
        trace_nodes = version_data.get('trace_nodes')
        if not isinstance(trace_nodes, list):
            return False
        
        # Basic validation of trace node structure
        if trace_nodes:
            sample_node = trace_nodes[0]
            if not hasattr(sample_node, 'operation') or not hasattr(sample_node, 'memory_usage'):
                return False
        
        return True

    def _export_version_comparison_report(self, version_analysis_data: List[Dict[str, Any]], 
                                        path: str,
                                        comparison_title: Optional[str],
                                        include_diff_visualizations: bool,
                                        include_metrics_comparison: bool,
                                        include_architectural_diff: bool) -> None:
        """Export model version comparison as Markdown report"""
        from datetime import datetime
        
        report = []
        
        # Header with metadata
        title = comparison_title or "UniAD Model Version Comparison"
        report.append(f"# {title}")
        report.append("")
        report.append("**Generated by**: PyTorch Operation Tracer v1.0.0")
        report.append(f"**Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("**Model Framework**: UniAD (Planning-oriented Autonomous Driving)")
        report.append(f"**Versions Compared**: {len(version_analysis_data)}")
        report.append("")
        report.append("---")
        report.append("")
        
        # Version Overview
        report.append("## Version Overview")
        report.append("")
        
        version_table = ["| Version | Total Operations | Memory (MB) | Compute (ms) | Active Task Heads |",
                        "|---------|------------------|-------------|--------------|-------------------|"]
        
        for version_data in version_analysis_data:
            version_name = version_data.get('version_name', 'Unknown')
            trace_nodes = version_data.get('trace_nodes', [])
            head_analysis = version_data.get('head_analysis', {})
            
            total_memory = sum(getattr(node, 'memory_usage', 0.0) for node in trace_nodes)
            total_compute = sum(getattr(node, 'compute_time', 0.0) for node in trace_nodes)
            
            active_heads = []
            if head_analysis and head_analysis.get('memory_by_head'):
                active_heads = [head for head, memory in head_analysis['memory_by_head'].items() if memory > 0]
            
            version_table.append(
                f"| {version_name} | {len(trace_nodes):,} | {total_memory:.1f} | {total_compute:.1f} | {', '.join(active_heads) if active_heads else 'None'} |"
            )
        
        report.extend(version_table)
        report.append("")
        
        # Metrics Comparison
        if include_metrics_comparison:
            report.append("## Detailed Metrics Comparison")
            report.append("")
            
            # Generate pairwise comparisons
            for i in range(len(version_analysis_data)):
                for j in range(i + 1, len(version_analysis_data)):
                    base_version = version_analysis_data[i]
                    compare_version = version_analysis_data[j]
                    
                    comparison_section = self._generate_version_comparison_section(
                        base_version, compare_version, include_architectural_diff
                    )
                    report.extend(comparison_section)
                    report.append("")
        
        # Diff Visualizations
        if include_diff_visualizations:
            report.append("## Diff Visualizations")
            report.append("")
            
            # Generate task head comparisons using TaskHeadComparator
            report.extend(self._generate_task_head_diff_visualizations(version_analysis_data))
            report.append("")
        
        # Architecture Evolution Analysis
        if include_architectural_diff:
            report.append("## Architecture Evolution Analysis")
            report.append("")
            
            evolution_analysis = self._analyze_architecture_evolution(version_analysis_data)
            report.extend(evolution_analysis)
            report.append("")
        
        # Summary and Recommendations
        report.append("## Summary and Recommendations")
        report.append("")
        
        recommendations = self._generate_version_comparison_recommendations(version_analysis_data)
        report.extend(recommendations)
        
        # Footer
        report.append("")
        report.append("---")
        report.append("")
        report.append("*Comparison report generated by PyTorch Operation Tracer*")
        
        # Write the report
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))

    def _export_version_comparison_html(self, version_analysis_data: List[Dict[str, Any]], 
                                      path: str,
                                      comparison_title: Optional[str],
                                      include_diff_visualizations: bool,
                                      include_metrics_comparison: bool,
                                      include_architectural_diff: bool) -> None:
        """Export model version comparison as interactive HTML"""
        try:
            # Use HTML comparison template
            models_data = []
            for version_data in version_analysis_data:
                models_data.append({
                    'model_name': version_data.get('version_name', 'Unknown'),
                    'trace_nodes': version_data.get('trace_nodes', []),
                    'head_analysis': version_data.get('head_analysis', {}),
                    'memory_profile': version_data.get('memory_profile', {}),
                    'version_metadata': version_data.get('version_metadata', {})
                })
            
            self.export_html_comparison(models_data, path, comparison_title)
            
        except Exception as e:
            # Fallback to markdown-based HTML
            print(f"Warning: HTML version comparison failed ({e}), falling back to markdown-based HTML")
            
            # Generate markdown content first
            markdown_path = path.replace('.html', '.md')
            self._export_version_comparison_report(
                version_analysis_data, markdown_path, comparison_title,
                include_diff_visualizations, include_metrics_comparison, include_architectural_diff
            )
            
            # Read and convert to HTML
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            html_content = self._markdown_to_html(markdown_content)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html_content)

    def _generate_version_comparison_section(self, base_version: Dict[str, Any], 
                                           compare_version: Dict[str, Any],
                                           include_architectural_diff: bool) -> List[str]:
        """Generate comparison section between two versions"""
        base_name = base_version.get('version_name', 'Version 1')
        compare_name = compare_version.get('version_name', 'Version 2')
        
        base_nodes = base_version.get('trace_nodes', [])
        compare_nodes = compare_version.get('trace_nodes', [])
        
        base_head_analysis = base_version.get('head_analysis', {})
        compare_head_analysis = compare_version.get('head_analysis', {})
        
        section = []
        section.append(f"### {base_name} vs {compare_name}")
        section.append("")
        
        # Calculate metrics differences
        base_memory = sum(getattr(node, 'memory_usage', 0.0) for node in base_nodes)
        compare_memory = sum(getattr(node, 'memory_usage', 0.0) for node in compare_nodes)
        memory_diff = compare_memory - base_memory
        memory_percent = (memory_diff / base_memory * 100) if base_memory > 0 else 0
        
        base_compute = sum(getattr(node, 'compute_time', 0.0) for node in base_nodes)
        compare_compute = sum(getattr(node, 'compute_time', 0.0) for node in compare_nodes)
        compute_diff = compare_compute - base_compute
        compute_percent = (compute_diff / base_compute * 100) if base_compute > 0 else 0
        
        operation_diff = len(compare_nodes) - len(base_nodes)
        operation_percent = (operation_diff / len(base_nodes) * 100) if base_nodes else 0
        
        # Performance comparison
        section.append("**Performance Metrics:**")
        section.append(f"- **Memory Usage**: {compare_memory:.1f} MB vs {base_memory:.1f} MB ({memory_diff:+.1f} MB, {memory_percent:+.1f}%)")
        section.append(f"- **Compute Time**: {compare_compute:.1f} ms vs {base_compute:.1f} ms ({compute_diff:+.1f} ms, {compute_percent:+.1f}%)")
        section.append(f"- **Operation Count**: {len(compare_nodes):,} vs {len(base_nodes):,} ({operation_diff:+,} operations, {operation_percent:+.1f}%)")
        section.append("")
        
        # Task head comparison
        base_heads = set(base_head_analysis.get('memory_by_head', {}).keys())
        compare_heads = set(compare_head_analysis.get('memory_by_head', {}).keys())
        
        common_heads = base_heads & compare_heads
        added_heads = compare_heads - base_heads
        removed_heads = base_heads - compare_heads
        
        section.append("**Task Head Changes:**")
        if added_heads:
            section.append(f"- **Added Heads**: {', '.join(added_heads)}")
        if removed_heads:
            section.append(f"- **Removed Heads**: {', '.join(removed_heads)}")
        if common_heads:
            section.append(f"- **Common Heads**: {', '.join(common_heads)} ({len(common_heads)} heads)")
        section.append("")
        
        # Memory breakdown by task head
        if common_heads and base_head_analysis.get('memory_by_head') and compare_head_analysis.get('memory_by_head'):
            section.append("**Task Head Memory Comparison:**")
            section.append("")
            section.append("| Task Head | Base Memory (MB) | Compare Memory (MB) | Difference (MB) | Change (%) |")
            section.append("|-----------|------------------|---------------------|-----------------|------------|")
            
            base_memory_by_head = base_head_analysis['memory_by_head']
            compare_memory_by_head = compare_head_analysis['memory_by_head']
            
            for head in sorted(common_heads):
                base_mem = base_memory_by_head.get(head, 0)
                compare_mem = compare_memory_by_head.get(head, 0)
                diff = compare_mem - base_mem
                percent = (diff / base_mem * 100) if base_mem > 0 else 0
                
                section.append(f"| {head} | {base_mem:.1f} | {compare_mem:.1f} | {diff:+.1f} | {percent:+.1f}% |")
            
            section.append("")
        
        # Architecture differences
        if include_architectural_diff:
            arch_diff = self._analyze_architectural_differences(base_nodes, compare_nodes)
            if arch_diff:
                section.append("**Architectural Differences:**")
                section.extend(arch_diff)
                section.append("")
        
        return section

    def _generate_task_head_diff_visualizations(self, version_analysis_data: List[Dict[str, Any]]) -> List[str]:
        """Generate diff visualizations using TaskHeadComparator"""
        diff_sections = []
        
        try:
            # Compare task heads across versions if they have common heads
            if len(version_analysis_data) >= 2:
                base_version = version_analysis_data[0]
                compare_version = version_analysis_data[1]
                
                base_head_analysis = base_version.get('head_analysis', {})
                compare_head_analysis = compare_version.get('head_analysis', {})
                
                base_heads = set(base_head_analysis.get('memory_by_head', {}).keys())
                compare_heads = set(compare_head_analysis.get('memory_by_head', {}).keys())
                common_heads = base_heads & compare_heads
                
                if len(common_heads) >= 2:
                    # Generate diff view for common heads
                    head_traces = {}
                    
                    # Prepare head traces for comparison
                    all_base_nodes = base_version.get('trace_nodes', [])
                    all_compare_nodes = compare_version.get('trace_nodes', [])
                    
                    for head in common_heads:
                        head_traces[f"{head}_{base_version.get('version_name', 'v1')}"] = [
                            node for node in all_base_nodes if getattr(node, 'task_head', None) == head
                        ]
                        head_traces[f"{head}_{compare_version.get('version_name', 'v2')}"] = [
                            node for node in all_compare_nodes if getattr(node, 'task_head', None) == head
                        ]
                    
                    diff_sections.append("### Task Head Diff Analysis")
                    diff_sections.append("")
                    
                    # Use TaskHeadComparator for detailed diff
                    for head in sorted(common_heads):
                        base_head_key = f"{head}_{base_version.get('version_name', 'v1')}"
                        compare_head_key = f"{head}_{compare_version.get('version_name', 'v2')}"
                        
                        try:
                            diff_view = self.task_head_comparator.generate_diff_view(
                                base_head_key, compare_head_key, head_traces
                            )
                            
                            if 'error' not in diff_view:
                                diff_sections.extend(self._format_task_head_diff_view(head, diff_view, 
                                                   base_version.get('version_name', 'v1'),
                                                   compare_version.get('version_name', 'v2')))
                            else:
                                diff_sections.append(f"**{head.upper()} Head**: Comparison data not available")
                        except Exception as e:
                            diff_sections.append(f"**{head.upper()} Head**: Error generating diff ({str(e)})")
                        
                        diff_sections.append("")
                else:
                    diff_sections.append("No common task heads found between versions for detailed diff analysis.")
            else:
                diff_sections.append("At least two versions required for diff visualization.")
        
        except Exception as e:
            diff_sections.append(f"Error generating task head diff visualizations: {str(e)}")
        
        return diff_sections

    def _format_task_head_diff_view(self, head_name: str, diff_view: Dict[str, Any], 
                                   base_version_name: str, compare_version_name: str) -> List[str]:
        """Format TaskHeadComparator diff view into markdown"""
        lines = []
        
        lines.append(f"#### {head_name.upper()} Head Comparison")
        lines.append("")
        
        detailed_analysis = diff_view.get('detailed_analysis', {})
        
        # Operation differences
        op_diff = detailed_analysis.get('operation_diff', {})
        if op_diff:
            lines.append("**Operation Changes:**")
            
            added_ops = op_diff.get('added_operations', [])
            removed_ops = op_diff.get('removed_operations', [])
            common_ops = op_diff.get('common_operations', [])
            
            if added_ops:
                lines.append(f"- **Added in {compare_version_name}**: {len(added_ops)} operations")
            if removed_ops:
                lines.append(f"- **Removed from {base_version_name}**: {len(removed_ops)} operations")
            if common_ops:
                lines.append(f"- **Common Operations**: {len(common_ops)} operations")
            lines.append("")
        
        # Memory breakdown
        memory_breakdown = detailed_analysis.get('memory_breakdown', {})
        if memory_breakdown:
            lines.append("**Memory Analysis:**")
            base_memory = memory_breakdown.get('base_head_memory', 0)
            compare_memory = memory_breakdown.get('compare_head_memory', 0)
            memory_diff = memory_breakdown.get('memory_diff_mb', 0)
            memory_percent = memory_breakdown.get('memory_diff_percent', 0)
            
            lines.append(f"- **Memory Change**: {base_memory:.1f} MB → {compare_memory:.1f} MB ({memory_diff:+.1f} MB, {memory_percent:+.1f}%)")
            lines.append("")
        
        # Compute breakdown
        compute_breakdown = detailed_analysis.get('compute_breakdown', {})
        if compute_breakdown:
            lines.append("**Compute Analysis:**")
            base_compute = compute_breakdown.get('base_head_compute', 0)
            compare_compute = compute_breakdown.get('compare_head_compute', 0)
            compute_diff = compute_breakdown.get('compute_diff_ms', 0)
            compute_percent = compute_breakdown.get('compute_diff_percent', 0)
            
            lines.append(f"- **Compute Change**: {base_compute:.1f} ms → {compare_compute:.1f} ms ({compute_diff:+.1f} ms, {compute_percent:+.1f}%)")
            lines.append("")
        
        return lines

    def _analyze_architecture_evolution(self, version_analysis_data: List[Dict[str, Any]]) -> List[str]:
        """Analyze how the architecture has evolved across versions"""
        evolution_lines = []
        
        if len(version_analysis_data) < 2:
            evolution_lines.append("At least two versions required for architecture evolution analysis.")
            return evolution_lines
        
        # Track changes across versions
        version_summaries = []
        for version_data in version_analysis_data:
            version_name = version_data.get('version_name', 'Unknown')
            trace_nodes = version_data.get('trace_nodes', [])
            head_analysis = version_data.get('head_analysis', {})
            
            # Calculate key metrics
            total_memory = sum(getattr(node, 'memory_usage', 0.0) for node in trace_nodes)
            total_compute = sum(getattr(node, 'compute_time', 0.0) for node in trace_nodes)
            
            # Categorize operations
            operation_types = {}
            for node in trace_nodes:
                op_type = getattr(node, 'operation', 'unknown').split('_')[0] if '_' in getattr(node, 'operation', 'unknown') else getattr(node, 'operation', 'unknown')
                operation_types[op_type] = operation_types.get(op_type, 0) + 1
            
            # BEV and temporal operations
            bev_ops = [node for node in trace_nodes if hasattr(node, 'is_bev_operation') and node.is_bev_operation]
            temporal_ops = [node for node in trace_nodes if hasattr(node, 'temporal_index') and node.temporal_index is not None]
            frozen_ops = [node for node in trace_nodes if hasattr(node, 'is_frozen') and node.is_frozen]
            
            active_heads = []
            if head_analysis and head_analysis.get('memory_by_head'):
                active_heads = [head for head, memory in head_analysis['memory_by_head'].items() if memory > 0]
            
            version_summaries.append({
                'name': version_name,
                'total_memory': total_memory,
                'total_compute': total_compute,
                'operation_count': len(trace_nodes),
                'operation_types': operation_types,
                'bev_ops': len(bev_ops),
                'temporal_ops': len(temporal_ops),
                'frozen_ops': len(frozen_ops),
                'active_heads': active_heads
            })
        
        # Analyze trends
        evolution_lines.append("### Architecture Evolution Trends")
        evolution_lines.append("")
        
        # Memory evolution
        evolution_lines.append("**Memory Usage Evolution:**")
        for i, vs in enumerate(version_summaries):
            trend_indicator = ""
            if i > 0:
                prev_memory = version_summaries[i-1]['total_memory']
                if vs['total_memory'] > prev_memory * 1.1:
                    trend_indicator = " 📈 (significant increase)"
                elif vs['total_memory'] < prev_memory * 0.9:
                    trend_indicator = " 📉 (significant decrease)"
                else:
                    trend_indicator = " ➡️ (stable)"
            
            evolution_lines.append(f"- **{vs['name']}**: {vs['total_memory']:.1f} MB{trend_indicator}")
        evolution_lines.append("")
        
        # Operation count evolution
        evolution_lines.append("**Operation Count Evolution:**")
        for i, vs in enumerate(version_summaries):
            trend_indicator = ""
            if i > 0:
                prev_count = version_summaries[i-1]['operation_count']
                if vs['operation_count'] > prev_count * 1.2:
                    trend_indicator = " 📈 (significant increase)"
                elif vs['operation_count'] < prev_count * 0.8:
                    trend_indicator = " 📉 (significant decrease)"
                else:
                    trend_indicator = " ➡️ (stable)"
            
            evolution_lines.append(f"- **{vs['name']}**: {vs['operation_count']:,} operations{trend_indicator}")
        evolution_lines.append("")
        
        # Task head activation evolution
        evolution_lines.append("**Task Head Activation Evolution:**")
        for vs in version_summaries:
            active_head_str = ', '.join(vs['active_heads']) if vs['active_heads'] else 'None'
            evolution_lines.append(f"- **{vs['name']}**: {active_head_str} ({len(vs['active_heads'])} heads)")
        evolution_lines.append("")
        
        # Architecture complexity analysis
        evolution_lines.append("**Architecture Complexity Analysis:**")
        for vs in version_summaries:
            complexity_indicators = []
            
            if vs['bev_ops'] > 0:
                complexity_indicators.append(f"{vs['bev_ops']} BEV operations")
            if vs['temporal_ops'] > 0:
                complexity_indicators.append(f"{vs['temporal_ops']} temporal operations")
            if vs['frozen_ops'] > 0:
                complexity_indicators.append(f"{vs['frozen_ops']} frozen operations")
            
            complexity_str = ', '.join(complexity_indicators) if complexity_indicators else 'No special operations detected'
            evolution_lines.append(f"- **{vs['name']}**: {complexity_str}")
        evolution_lines.append("")
        
        return evolution_lines

    def _analyze_architectural_differences(self, base_nodes: List[TraceNode], 
                                         compare_nodes: List[TraceNode]) -> List[str]:
        """Analyze architectural differences between two sets of nodes"""
        differences = []
        
        # Operation type differences
        base_op_types = set(getattr(node, 'operation', 'unknown').split('_')[0] if '_' in getattr(node, 'operation', 'unknown') else getattr(node, 'operation', 'unknown') for node in base_nodes)
        compare_op_types = set(getattr(node, 'operation', 'unknown').split('_')[0] if '_' in getattr(node, 'operation', 'unknown') else getattr(node, 'operation', 'unknown') for node in compare_nodes)
        
        added_types = compare_op_types - base_op_types
        removed_types = base_op_types - compare_op_types
        
        if added_types:
            differences.append(f"- **Added Operation Types**: {', '.join(sorted(added_types))}")
        if removed_types:
            differences.append(f"- **Removed Operation Types**: {', '.join(sorted(removed_types))}")
        
        # Module path differences
        base_modules = set(getattr(node, 'module_path', 'unknown') for node in base_nodes if hasattr(node, 'module_path'))
        compare_modules = set(getattr(node, 'module_path', 'unknown') for node in compare_nodes if hasattr(node, 'module_path'))
        
        added_modules = compare_modules - base_modules
        removed_modules = base_modules - compare_modules
        
        if added_modules:
            differences.append(f"- **Added Modules**: {len(added_modules)} new module paths")
        if removed_modules:
            differences.append(f"- **Removed Modules**: {len(removed_modules)} module paths removed")
        
        # Special operation analysis
        base_bev_count = len([node for node in base_nodes if hasattr(node, 'is_bev_operation') and node.is_bev_operation])
        compare_bev_count = len([node for node in compare_nodes if hasattr(node, 'is_bev_operation') and node.is_bev_operation])
        
        if base_bev_count != compare_bev_count:
            differences.append(f"- **BEV Operations**: {base_bev_count} → {compare_bev_count} ({compare_bev_count - base_bev_count:+} change)")
        
        base_frozen_count = len([node for node in base_nodes if hasattr(node, 'is_frozen') and node.is_frozen])
        compare_frozen_count = len([node for node in compare_nodes if hasattr(node, 'is_frozen') and node.is_frozen])
        
        if base_frozen_count != compare_frozen_count:
            differences.append(f"- **Frozen Operations**: {base_frozen_count} → {compare_frozen_count} ({compare_frozen_count - base_frozen_count:+} change)")
        
        return differences

    def _generate_version_comparison_recommendations(self, version_analysis_data: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on version comparisons"""
        recommendations = []
        
        if len(version_analysis_data) < 2:
            return ["At least two versions required for recommendations."]
        
        # Analyze trends across versions
        version_metrics = []
        for version_data in version_analysis_data:
            trace_nodes = version_data.get('trace_nodes', [])
            total_memory = sum(getattr(node, 'memory_usage', 0.0) for node in trace_nodes)
            total_compute = sum(getattr(node, 'compute_time', 0.0) for node in trace_nodes)
            operation_count = len(trace_nodes)
            
            version_metrics.append({
                'name': version_data.get('version_name', 'Unknown'),
                'memory': total_memory,
                'compute': total_compute,
                'operations': operation_count
            })
        
        # Memory trend analysis
        memory_trend = [vm['memory'] for vm in version_metrics]
        if len(memory_trend) >= 2:
            memory_growth = memory_trend[-1] / memory_trend[0] if memory_trend[0] > 0 else 1
            if memory_growth > 1.5:
                recommendations.append(
                    f"🚨 **Memory Growth Alert**: Memory usage increased by {(memory_growth-1)*100:.1f}% "
                    f"from {version_metrics[0]['name']} to {version_metrics[-1]['name']}. "
                    f"Consider memory optimization strategies."
                )
            elif memory_growth > 1.2:
                recommendations.append(
                    f"⚠️ **Memory Growth**: Memory usage increased by {(memory_growth-1)*100:.1f}%. "
                    f"Monitor memory usage trends."
                )
            elif memory_growth < 0.8:
                recommendations.append(
                    f"💡 **Memory Optimization Success**: Memory usage decreased by {(1-memory_growth)*100:.1f}%. "
                    f"Good optimization work!"
                )
        
        # Operation complexity analysis
        op_trend = [vm['operations'] for vm in version_metrics]
        if len(op_trend) >= 2:
            op_growth = op_trend[-1] / op_trend[0] if op_trend[0] > 0 else 1
            if op_growth > 2.0:
                recommendations.append(
                    f"⚠️ **Complexity Growth**: Operation count increased by {(op_growth-1)*100:.1f}%. "
                    f"Consider operation fusion or architectural simplification."
                )
        
        # Version-specific recommendations
        latest_version = version_analysis_data[-1]
        latest_head_analysis = latest_version.get('head_analysis', {})
        
        if latest_head_analysis and latest_head_analysis.get('memory_by_head'):
            memory_by_head = latest_head_analysis['memory_by_head']
            active_heads = [head for head, memory in memory_by_head.items() if memory > 0]
            
            # Stage-specific recommendations
            if len(active_heads) == 2:  # Likely Stage 1
                recommendations.append(
                    "💡 **Stage 1 Detected**: Consider transitioning to Stage 2 training "
                    "by freezing BEV encoder and enabling all task heads."
                )
            elif len(active_heads) >= 4:  # Likely Stage 2
                recommendations.append(
                    "💡 **Stage 2 Detected**: Monitor individual task head performance "
                    "and consider task-specific optimizations."
                )
        
        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "💡 **Version Comparison Complete**: All metrics appear stable across versions.",
                "💡 **Monitoring**: Continue tracking memory and compute trends for optimization opportunities.",
                "💡 **Documentation**: Consider documenting significant changes between versions for future reference."
            ])
        
        return recommendations

    def _check_local_assets(self) -> None:
        """Check if local CSS/JS assets are available for standalone HTML export"""
        try:
            # Check for local web assets
            web_dir = Path(__file__).parent.parent / 'web'
            css_file = web_dir / 'static' / 'css' / 'visualization.css'
            js_file = web_dir / 'static' / 'js' / 'interactive.js'
            memory_js_file = web_dir / 'static' / 'js' / 'memory_timeline.js'
            
            self.local_assets = {
                'css_available': css_file.exists(),
                'js_available': js_file.exists(),
                'memory_js_available': memory_js_file.exists(),
                'css_path': str(css_file) if css_file.exists() else None,
                'js_path': str(js_file) if js_file.exists() else None,
                'memory_js_path': str(memory_js_file) if memory_js_file.exists() else None
            }
            
            # Update template config to include local assets if available
            if self.local_assets['css_available']:
                with open(css_file, 'r', encoding='utf-8') as f:
                    self.template_config.custom_css = f.read()
            
            if self.local_assets['js_available']:
                with open(js_file, 'r', encoding='utf-8') as f:
                    custom_js = f.read()
                    if self.local_assets['memory_js_available']:
                        with open(memory_js_file, 'r', encoding='utf-8') as f:
                            custom_js += '\n' + f.read()
                    self.template_config.custom_js = custom_js
                    
        except Exception as e:
            print(f"Warning: Could not load local assets: {e}")
            self.local_assets = {
                'css_available': False,
                'js_available': False, 
                'memory_js_available': False
            }

# Re-export TemplateConfig for easier imports
__all__ = ['ExportManager', 'TemplateConfig']
