"""HTML Template Engine for Interactive UniAD Dataflow Visualizations

This module provides a comprehensive HTML template engine specifically designed for
UniAD model analysis visualizations, supporting multiple visualization types with
interactive JavaScript libraries and security features.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    # Try to import from the package structure
    from ..core.data_structures import TraceNode, TensorInfo
    from ..core.visualization_config import InteractiveConfig as CoreInteractiveConfig
except (ImportError, ValueError):
    # Fallback for standalone usage
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from core.data_structures import TraceNode, TensorInfo
        from core.visualization_config import InteractiveConfig as CoreInteractiveConfig
    except (ImportError, ValueError):
        # Create minimal stubs if dependencies not available
        TraceNode = None
        TensorInfo = None
        
        # Define CoreInteractiveConfig as a dynamic type to avoid import conflicts
        CoreInteractiveConfig = type('CoreInteractiveConfig', (), {
            '__init__': lambda self: setattr(self, 'max_nodes_visible', 50) or setattr(self, 'color_scheme', 'operation'),
            'for_uniad_analysis': classmethod(lambda cls: cls()),
            'to_dict': lambda self: {
                'max_nodes_visible': getattr(self, 'max_nodes_visible', 50),
                'color_scheme': getattr(self, 'color_scheme', 'operation')
            }
        })


@dataclass
class TemplateConfig:
    """Configuration for HTML template engine"""
    enable_csp: bool = True
    enable_caching: bool = True
    compression_level: str = "moderate"  # none, moderate, aggressive
    theme: str = "default"  # default, dark, light, uniad
    include_d3js: bool = True
    include_chartjs: bool = True
    include_plotlyjs: bool = False
    custom_css: Optional[str] = None
    custom_js: Optional[str] = None


@dataclass
class VisualizationData:
    """Structured data for template injection"""
    title: str
    data: Dict[str, Any]
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON with datetime handling"""
        data_copy = {
            'title': self.title,
            'data': self.data,
            'config': self.config,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else ''
        }
        return json.dumps(data_copy, indent=indent, default=str)


class HTMLTemplateEngine:
    """
    Advanced HTML template engine for UniAD dataflow visualizations.
    
    Features:
    - Multiple template types (dataflow, memory, comparison, etc.)
    - Interactive JavaScript library management (D3.js, Chart.js, Plotly)
    - Template caching for performance
    - Content Security Policy support
    - Responsive design with customizable themes
    - Data sanitization and validation
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        Initialize the HTML template engine.
        
        Args:
            config: Template engine configuration
        """
        self.config = config or TemplateConfig()
        self.template_cache = {}
        self.data_cache = {}
        
        # Initialize template directory
        self.template_dir = Path(__file__).parent / "templates"
        self.template_dir.mkdir(exist_ok=True)
        
        # CDN URLs for JavaScript libraries
        self.cdn_urls = {
            'd3js': 'https://d3js.org/d3.v7.min.js',
            'chartjs': 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.min.js',
            'plotlyjs': 'https://cdn.plot.ly/plotly-2.26.0.min.js'
        }
        
        # Initialize base templates if they don't exist
        self._ensure_base_templates()
    
    def render_template(self, data: VisualizationData, 
                       template_name: str = "dataflow") -> str:
        """
        Render HTML template with visualization data.
        
        Args:
            data: Structured visualization data
            template_name: Template type (dataflow, memory, comparison, etc.)
            
        Returns:
            Complete HTML string ready for display or export
        """
        # Validate input data
        self._validate_data(data)
        
        # Get template content
        template_content = self._get_template(template_name)
        
        # Inject visualization data
        html_with_data = self.inject_visualization_data(template_content, data)
        
        # Add interactive scripts
        html_with_scripts = self.add_interactive_scripts(html_with_data)
        
        # Apply security headers if enabled
        if self.config.enable_csp:
            html_with_scripts = self._add_security_headers(html_with_scripts)
        
        return html_with_scripts
    
    def inject_visualization_data(self, html: str, data: VisualizationData) -> str:
        """
        Inject visualization data into HTML template.
        
        Args:
            html: Base HTML template
            data: Visualization data to inject
            
        Returns:
            HTML with embedded data
        """
        # Sanitize data for safe injection
        sanitized_data = self._sanitize_data(data.data)
        sanitized_config = self._sanitize_data(data.config)
        
        # Create data injection object
        injection_data = {
            'TITLE': self._escape_html(data.title),
            'DATA_JSON': json.dumps(sanitized_data, indent=2),
            'CONFIG_JSON': json.dumps(sanitized_config, indent=2),
            'METADATA_JSON': json.dumps(data.metadata, indent=2),
            'TIMESTAMP': data.timestamp.strftime('%Y-%m-%d %H:%M:%S') if data.timestamp else '',
            'CSS_THEME': self._get_theme_css(),
            'CUSTOM_CSS': self.config.custom_css or '',
            'CUSTOM_JS': self.config.custom_js or ''
        }
        
        # Perform template substitution
        return self._substitute_template_variables(html, injection_data)
    
    def add_interactive_scripts(self, html: str) -> str:
        """
        Add interactive JavaScript libraries to HTML.
        
        Args:
            html: HTML content
            
        Returns:
            HTML with embedded JavaScript libraries
        """
        scripts = []
        
        # Add D3.js if enabled
        if self.config.include_d3js:
            scripts.append(f'<script src="{self.cdn_urls["d3js"]}"></script>')
        
        # Add Chart.js if enabled
        if self.config.include_chartjs:
            scripts.append(f'<script src="{self.cdn_urls["chartjs"]}"></script>')
        
        # Add Plotly.js if enabled
        if self.config.include_plotlyjs:
            scripts.append(f'<script src="{self.cdn_urls["plotlyjs"]}"></script>')
        
        # Add utility functions
        scripts.append(self._get_utility_scripts())
        
        # Inject scripts before closing head tag
        script_content = '\n    '.join(scripts)
        if '</head>' in html:
            html = html.replace('</head>', f'    {script_content}\n</head>')
        else:
            # If no head tag, add scripts at the beginning
            html = f"{script_content}\n{html}"
        
        return html
    
    def create_dataflow_template(self, **kwargs) -> str:
        """Create dataflow visualization template"""
        return self._get_base_template("dataflow", **kwargs)
    
    def create_memory_template(self, **kwargs) -> str:
        """Create memory timeline visualization template"""
        return self._get_base_template("memory", **kwargs)
    
    def create_comparison_template(self, **kwargs) -> str:
        """Create model comparison visualization template"""
        return self._get_base_template("comparison", **kwargs)
    
    def create_dashboard_template(self, **kwargs) -> str:
        """Create multi-view dashboard template"""
        return self._get_base_template("dashboard", **kwargs)
    
    def save_template(self, template_name: str, content: str):
        """
        Save custom template to disk.
        
        Args:
            template_name: Name of the template
            content: HTML template content
        """
        template_path = self.template_dir / f"{template_name}.html"
        template_path.write_text(content, encoding='utf-8')
        
        # Clear cache for this template
        if template_name in self.template_cache:
            del self.template_cache[template_name]
    
    def export_html(self, html_content: str, output_path: str, 
                   compress: Optional[bool] = None) -> str:
        """
        Export HTML content to file with optional compression.
        
        Args:
            html_content: Complete HTML content
            output_path: Output file path
            compress: Whether to compress HTML (uses config if None)
            
        Returns:
            Path to exported file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Apply compression if configured
        if compress or (compress is None and self.config.compression_level != "none"):
            html_content = self._compress_html(html_content)
        
        # Write to file
        output_file.write_text(html_content, encoding='utf-8')
        
        return str(output_file)
    
    def _get_template(self, template_name: str) -> str:
        """Get template content with caching"""
        if self.config.enable_caching and template_name in self.template_cache:
            return self.template_cache[template_name]
        
        template_path = self.template_dir / f"{template_name}.html"
        
        if template_path.exists():
            content = template_path.read_text(encoding='utf-8')
        else:
            # Generate default template
            content = self._get_base_template(template_name)
            # Save for future use
            self.save_template(template_name, content)
        
        if self.config.enable_caching:
            self.template_cache[template_name] = content
        
        return content
    
    def _get_base_template(self, template_type: str, **kwargs) -> str:
        """Generate base template for specified type"""
        templates = {
            "dataflow": self._create_dataflow_template,
            "memory": self._create_memory_template,
            "comparison": self._create_comparison_template,
            "dashboard": self._create_dashboard_template
        }
        
        if template_type in templates:
            return templates[template_type](**kwargs)
        else:
            # Default fallback template
            return self._create_default_template(**kwargs)
    
    def _create_dataflow_template(self, **kwargs) -> str:
        """Create interactive dataflow visualization template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}} - UniAD Dataflow Visualization</title>
    {{CSS_THEME}}
    <style>
        {{CUSTOM_CSS}}
        .dataflow-container {
            display: flex;
            height: 100vh;
            background-color: var(--bg-color);
        }
        .control-panel {
            width: 320px;
            background-color: var(--panel-bg);
            border-right: 1px solid var(--border-color);
            padding: 20px;
            overflow-y: auto;
            box-shadow: 2px 0 4px rgba(0,0,0,0.1);
        }
        .visualization-area {
            flex: 1;
            position: relative;
            overflow: hidden;
        }
        #dataflow-svg {
            width: 100%;
            height: 100%;
            background-color: var(--canvas-bg);
        }
        .filter-group {
            margin-bottom: 25px;
            padding: 15px;
            background-color: var(--card-bg);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        .filter-title {
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--text-primary);
        }
        .node {
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .node:hover {
            stroke-width: 3px;
            filter: brightness(1.1);
        }
        .edge {
            fill: none;
            stroke: var(--edge-color);
            stroke-width: 2px;
            marker-end: url(#arrowhead);
        }
        #tooltip {
            position: absolute;
            background-color: var(--tooltip-bg);
            color: var(--tooltip-text);
            padding: 12px;
            border-radius: 6px;
            font-size: 12px;
            max-width: 300px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            color: #666;
            z-index: 999;
        }
        .task-head-filter {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .task-chip {
            background-color: var(--chip-bg);
            color: var(--chip-text);
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .task-chip.active {
            background-color: var(--primary-color);
            color: white;
        }
    </style>
</head>
<body>
    <div class="dataflow-container">
        <div class="control-panel">
            <div class="filter-group">
                <div class="filter-title">Task Head Filters</div>
                <div id="task-head-filters" class="task-head-filter"></div>
            </div>
            <div class="filter-group">
                <div class="filter-title">Memory Threshold</div>
                <input type="range" id="memory-threshold" min="0" max="10000" value="0" style="width: 100%;">
                <span id="memory-threshold-value">0 MB</span>
            </div>
            <div class="filter-group">
                <div class="filter-title">Search</div>
                <input type="text" id="search-input" placeholder="Search operations..." style="width: 100%; padding: 8px; border: 1px solid var(--border-color); border-radius: 4px;">
            </div>
            <div class="filter-group">
                <div class="filter-title">View Controls</div>
                <button class="control-btn" onclick="expandAllNodes()">Expand All</button>
                <button class="control-btn" onclick="collapseAllNodes()">Collapse All</button>
                <button class="control-btn" onclick="resetZoom()">Reset Zoom</button>
                <button class="control-btn" onclick="exportVisualization()">Export PNG</button>
            </div>
            <div class="filter-group">
                <div class="filter-title">Statistics</div>
                <div id="summary-stats"></div>
            </div>
        </div>
        <div class="visualization-area">
            <div id="loading-overlay" class="loading-overlay" style="display: none;">
                Loading visualization...
            </div>
            <svg id="dataflow-svg"></svg>
            <div id="tooltip"></div>
        </div>
    </div>

    <script>
        // Embed configuration and data
        const CONFIG = {{CONFIG_JSON}};
        const DATA = {{DATA_JSON}};
        const METADATA = {{METADATA_JSON}};
        
        {{CUSTOM_JS}}
        
        // Initialize dataflow visualization
        class DataflowVisualization {
            constructor() {
                this.svg = d3.select("#dataflow-svg");
                this.width = window.innerWidth - 320;
                this.height = window.innerHeight;
                this.simulation = null;
                this.nodes = DATA.nodes || [];
                this.edges = DATA.edges || [];
                this.filteredNodes = [...this.nodes];
                this.filteredEdges = [...this.edges];
                
                this.init();
            }
            
            init() {
                this.setupSVG();
                this.setupFilters();
                this.setupSummary();
                this.renderVisualization();
                this.setupEventListeners();
            }
            
            setupSVG() {
                // Create main group with zoom behavior
                this.zoom = d3.zoom()
                    .scaleExtent([0.1, 3])
                    .on("zoom", (event) => {
                        this.mainGroup.attr("transform", event.transform);
                    });
                
                this.svg.call(this.zoom);
                this.mainGroup = this.svg.append("g");
                
                // Define arrowhead marker
                this.svg.append("defs").append("marker")
                    .attr("id", "arrowhead")
                    .attr("viewBox", "0 -5 10 10")
                    .attr("refX", 8)
                    .attr("refY", 0)
                    .attr("markerWidth", 6)
                    .attr("markerHeight", 6)
                    .attr("orient", "auto")
                    .append("path")
                    .attr("d", "M0,-5L10,0L0,5")
                    .attr("fill", "var(--edge-color)");
            }
            
            setupFilters() {
                // Task head filters
                const taskHeads = [...new Set(this.nodes.map(n => n.task_head).filter(Boolean))];
                const taskContainer = d3.select("#task-head-filters");
                
                taskHeads.forEach(taskHead => {
                    taskContainer.append("div")
                        .attr("class", "task-chip active")
                        .attr("data-task", taskHead)
                        .text(taskHead)
                        .on("click", (event) => {
                            const chip = d3.select(event.target);
                            chip.classed("active", !chip.classed("active"));
                            this.applyFilters();
                        });
                });
                
                // Memory threshold
                d3.select("#memory-threshold").on("input", (event) => {
                    const value = event.target.value;
                    d3.select("#memory-threshold-value").text(value + " MB");
                    this.applyFilters();
                });
                
                // Search input
                d3.select("#search-input").on("input", () => {
                    this.applyFilters();
                });
            }
            
            applyFilters() {
                const selectedTasks = [];
                d3.selectAll(".task-chip.active").each(function() {
                    selectedTasks.push(this.getAttribute("data-task"));
                });
                
                const memoryThreshold = +d3.select("#memory-threshold").node().value;
                const searchQuery = d3.select("#search-input").node().value.toLowerCase();
                
                this.filteredNodes = this.nodes.filter(node => {
                    // Task filter
                    if (selectedTasks.length > 0 && node.task_head && !selectedTasks.includes(node.task_head)) {
                        return false;
                    }
                    
                    // Memory filter
                    if (node.memory_mb < memoryThreshold) {
                        return false;
                    }
                    
                    // Search filter
                    if (searchQuery && !node.display_name.toLowerCase().includes(searchQuery)) {
                        return false;
                    }
                    
                    return true;
                });
                
                // Filter edges
                const nodeIds = new Set(this.filteredNodes.map(n => n.id));
                this.filteredEdges = this.edges.filter(edge => 
                    nodeIds.has(edge.source) && nodeIds.has(edge.target)
                );
                
                this.renderVisualization();
            }
            
            renderVisualization() {
                this.showLoading(true);
                
                // Clear existing elements
                this.mainGroup.selectAll("*").remove();
                
                if (this.filteredNodes.length === 0) {
                    this.showMessage("No nodes match current filters");
                    this.showLoading(false);
                    return;
                }
                
                // Create force simulation
                this.simulation = d3.forceSimulation(this.filteredNodes)
                    .force("link", d3.forceLink(this.filteredEdges)
                        .id(d => d.id)
                        .distance(100))
                    .force("charge", d3.forceManyBody().strength(-500))
                    .force("center", d3.forceCenter(this.width / 2, this.height / 2))
                    .force("collision", d3.forceCollide().radius(d => d.size + 10));
                
                // Render edges
                this.renderEdges();
                
                // Render nodes
                this.renderNodes();
                
                // Start simulation
                this.simulation.on("tick", () => this.updatePositions());
                
                setTimeout(() => this.showLoading(false), 500);
            }
            
            renderEdges() {
                this.edgeElements = this.mainGroup.append("g")
                    .attr("class", "edges")
                    .selectAll(".edge")
                    .data(this.filteredEdges)
                    .enter().append("line")
                    .attr("class", "edge");
            }
            
            renderNodes() {
                this.nodeElements = this.mainGroup.append("g")
                    .attr("class", "nodes")
                    .selectAll(".node-group")
                    .data(this.filteredNodes)
                    .enter().append("g")
                    .attr("class", "node-group")
                    .call(this.createDragBehavior());
                
                // Add circles
                this.nodeElements.append("circle")
                    .attr("class", "node")
                    .attr("r", d => d.size || 20)
                    .attr("fill", d => d.color || "#69b3a2")
                    .attr("stroke", "#fff")
                    .attr("stroke-width", 2)
                    .on("mouseover", (event, d) => this.showTooltip(event, d))
                    .on("mouseout", () => this.hideTooltip());
                
                // Add labels
                this.nodeElements.append("text")
                    .attr("class", "node-label")
                    .attr("text-anchor", "middle")
                    .attr("dy", "0.3em")
                    .text(d => this.truncateText(d.display_name, 15))
                    .style("font-size", "11px")
                    .style("fill", "var(--text-primary)")
                    .style("pointer-events", "none");
            }
            
            updatePositions() {
                this.edgeElements
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                this.nodeElements
                    .attr("transform", d => `translate(${d.x},${d.y})`);
            }
            
            createDragBehavior() {
                return d3.drag()
                    .on("start", (event, d) => {
                        if (!event.active) this.simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    })
                    .on("drag", (event, d) => {
                        d.fx = event.x;
                        d.fy = event.y;
                    })
                    .on("end", (event, d) => {
                        if (!event.active) this.simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    });
            }
            
            showTooltip(event, d) {
                const tooltip = d3.select("#tooltip");
                let content = `<strong>${d.display_name}</strong><br>`;
                
                if (d.memory_mb) {
                    content += `Memory: ${d.memory_mb.toFixed(1)} MB<br>`;
                }
                if (d.compute_ms) {
                    content += `Compute: ${d.compute_ms.toFixed(1)} ms<br>`;
                }
                if (d.task_head) {
                    content += `Task Head: ${d.task_head}<br>`;
                }
                
                tooltip.html(content)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px")
                    .style("opacity", 1);
            }
            
            hideTooltip() {
                d3.select("#tooltip").style("opacity", 0);
            }
            
            showLoading(show) {
                d3.select("#loading-overlay").style("display", show ? "flex" : "none");
            }
            
            showMessage(message) {
                this.mainGroup.append("text")
                    .attr("x", this.width / 2)
                    .attr("y", this.height / 2)
                    .attr("text-anchor", "middle")
                    .style("font-size", "18px")
                    .style("fill", "var(--text-secondary)")
                    .text(message);
            }
            
            truncateText(text, maxLength) {
                return text.length > maxLength ? text.substring(0, maxLength - 3) + "..." : text;
            }
            
            setupSummary() {
                const summary = DATA.summary || {};
                const container = d3.select("#summary-stats");
                
                const stats = [
                    { label: "Total Nodes", value: this.nodes.length },
                    { label: "Total Memory", value: (summary.total_memory_mb || 0).toFixed(1) + " MB" },
                    { label: "Task Heads", value: summary.task_head_count || 0 }
                ];
                
                stats.forEach(stat => {
                    const div = container.append("div")
                        .style("margin-bottom", "8px")
                        .style("font-size", "12px");
                    
                    div.append("div")
                        .style("font-weight", "500")
                        .style("color", "var(--text-secondary)")
                        .text(stat.label);
                    
                    div.append("div")
                        .style("font-weight", "600")
                        .style("color", "var(--text-primary)")
                        .text(stat.value);
                });
            }
            
            setupEventListeners() {
                window.addEventListener("resize", () => {
                    this.width = window.innerWidth - 320;
                    this.height = window.innerHeight;
                    this.svg.attr("width", this.width).attr("height", this.height);
                    if (this.simulation) {
                        this.simulation.force("center", d3.forceCenter(this.width / 2, this.height / 2));
                        this.simulation.restart();
                    }
                });
            }
        }
        
        // Global functions for control buttons
        function expandAllNodes() {
            console.log("Expand all nodes");
        }
        
        function collapseAllNodes() {
            console.log("Collapse all nodes");
        }
        
        function resetZoom() {
            viz.svg.transition().duration(750)
                .call(viz.zoom.transform, d3.zoomIdentity);
        }
        
        function exportVisualization() {
            console.log("Export visualization");
        }
        
        // Initialize visualization when DOM is ready
        let viz;
        document.addEventListener("DOMContentLoaded", () => {
            viz = new DataflowVisualization();
        });
    </script>
</body>
</html>'''
    
    def _create_memory_template(self, **kwargs) -> str:
        """Create memory timeline visualization template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}} - Memory Timeline</title>
    {{CSS_THEME}}
    <style>
        {{CUSTOM_CSS}}
        .memory-container {
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
            background-color: var(--bg-color);
        }
        .header {
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-container {
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: var(--card-bg);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--primary-color);
        }
        .stat-label {
            color: var(--text-secondary);
            font-size: 14px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="memory-container">
        <div class="header">
            <h1>{{TITLE}}</h1>
            <p>Memory usage timeline analysis</p>
            <small>Generated: {{TIMESTAMP}}</small>
        </div>
        
        <div class="stats-grid" id="stats-grid"></div>
        
        <div class="chart-container">
            <canvas id="memoryChart" width="400" height="150"></canvas>
        </div>
    </div>

    <script>
        const CONFIG = {{CONFIG_JSON}};
        const DATA = {{DATA_JSON}};
        
        {{CUSTOM_JS}}
        
        // Initialize memory visualization
        document.addEventListener("DOMContentLoaded", () => {
            initializeMemoryVisualization();
        });
        
        function initializeMemoryVisualization() {
            setupStats();
            setupChart();
        }
        
        function setupStats() {
            const statsGrid = document.getElementById("stats-grid");
            const timeline = DATA.timeline_points || [];
            
            if (timeline.length > 0) {
                const maxMemory = Math.max(...timeline.map(p => p[1]));
                const avgMemory = timeline.reduce((sum, p) => sum + p[1], 0) / timeline.length;
                
                const stats = [
                    { label: "Peak Memory", value: maxMemory.toFixed(1) + " MB" },
                    { label: "Average Memory", value: avgMemory.toFixed(1) + " MB" },
                    { label: "Memory Pressure", value: (DATA.memory_pressure * 100).toFixed(1) + "%" },
                    { label: "Data Points", value: timeline.length.toLocaleString() }
                ];
                
                stats.forEach(stat => {
                    const card = document.createElement("div");
                    card.className = "stat-card";
                    card.innerHTML = `
                        <div class="stat-value">${stat.value}</div>
                        <div class="stat-label">${stat.label}</div>
                    `;
                    statsGrid.appendChild(card);
                });
            }
        }
        
        function setupChart() {
            const ctx = document.getElementById("memoryChart").getContext("2d");
            const timeline = DATA.timeline_points || [];
            
            if (timeline.length === 0) {
                ctx.font = "16px Arial";
                ctx.fillStyle = "#666";
                ctx.textAlign = "center";
                ctx.fillText("No memory data available", ctx.canvas.width / 2, ctx.canvas.height / 2);
                return;
            }
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: timeline.map(p => `${p[0].toFixed(1)}ms`),
                    datasets: [{
                        label: 'Memory Usage (MB)',
                        data: timeline.map(p => p[1]),
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Memory Usage Over Time'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Time (ms)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Memory Usage (MB)'
                            }
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>'''
    
    def _create_comparison_template(self, **kwargs) -> str:
        """Create model comparison visualization template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}} - Model Comparison</title>
    {{CSS_THEME}}
    <style>
        {{CUSTOM_CSS}}
        .comparison-container {
            padding: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .comparison-panel {
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .model-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: var(--primary-color);
        }
    </style>
</head>
<body>
    <div class="comparison-container">
        <h1>{{TITLE}}</h1>
        <div class="comparison-grid" id="comparison-grid">
            <!-- Comparison panels will be populated by JavaScript -->
        </div>
    </div>

    <script>
        const CONFIG = {{CONFIG_JSON}};
        const DATA = {{DATA_JSON}};
        
        {{CUSTOM_JS}}
        
        document.addEventListener("DOMContentLoaded", () => {
            initializeComparison();
        });
        
        function initializeComparison() {
            const grid = document.getElementById("comparison-grid");
            const models = DATA.models || [];
            
            models.forEach(model => {
                const panel = document.createElement("div");
                panel.className = "comparison-panel";
                panel.innerHTML = `
                    <div class="model-title">${model.name}</div>
                    <canvas id="chart-${model.id}" width="350" height="200"></canvas>
                `;
                grid.appendChild(panel);
                
                // Create chart for each model
                setTimeout(() => createModelChart(model), 100);
            });
        }
        
        function createModelChart(model) {
            const ctx = document.getElementById(`chart-${model.id}`).getContext("2d");
            // Chart implementation would go here
        }
    </script>
</body>
</html>'''
    
    def _create_dashboard_template(self, **kwargs) -> str:
        """Create multi-view dashboard template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}} - UniAD Dashboard</title>
    {{CSS_THEME}}
    <style>
        {{CUSTOM_CSS}}
        .dashboard-container {
            display: grid;
            grid-template-areas: 
                "header header"
                "sidebar main";
            grid-template-columns: 300px 1fr;
            grid-template-rows: 80px 1fr;
            height: 100vh;
            gap: 0;
        }
        .dashboard-header {
            grid-area: header;
            background: var(--primary-color);
            color: white;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .dashboard-sidebar {
            grid-area: sidebar;
            background: var(--panel-bg);
            padding: 20px;
            overflow-y: auto;
        }
        .dashboard-main {
            grid-area: main;
            background: var(--bg-color);
            padding: 20px;
            overflow-y: auto;
        }
        .view-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .view-tab {
            padding: 8px 16px;
            background: var(--tab-bg);
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .view-tab.active {
            background: var(--primary-color);
            color: white;
        }
        .view-content {
            display: none;
        }
        .view-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1>{{TITLE}}</h1>
            <div>
                <span>Generated: {{TIMESTAMP}}</span>
            </div>
        </div>
        <div class="dashboard-sidebar">
            <div class="view-tabs">
                <button class="view-tab active" onclick="showView('dataflow')">Dataflow</button>
                <button class="view-tab" onclick="showView('memory')">Memory</button>
                <button class="view-tab" onclick="showView('analysis')">Analysis</button>
            </div>
        </div>
        <div class="dashboard-main">
            <div id="dataflow-view" class="view-content active">
                <h2>Dataflow Visualization</h2>
                <div id="dataflow-container"></div>
            </div>
            <div id="memory-view" class="view-content">
                <h2>Memory Timeline</h2>
                <div id="memory-container"></div>
            </div>
            <div id="analysis-view" class="view-content">
                <h2>Analysis Results</h2>
                <div id="analysis-container"></div>
            </div>
        </div>
    </div>

    <script>
        const CONFIG = {{CONFIG_JSON}};
        const DATA = {{DATA_JSON}};
        
        {{CUSTOM_JS}}
        
        function showView(viewName) {
            // Hide all views
            document.querySelectorAll('.view-content').forEach(view => {
                view.classList.remove('active');
            });
            
            // Hide all tabs
            document.querySelectorAll('.view-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected view
            document.getElementById(`${viewName}-view`).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>'''
    
    def _create_default_template(self, **kwargs) -> str:
        """Create default fallback template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    {{CSS_THEME}}
    <style>{{CUSTOM_CSS}}</style>
</head>
<body>
    <div class="container">
        <h1>{{TITLE}}</h1>
        <p>Generated: {{TIMESTAMP}}</p>
        <div id="visualization-area"></div>
    </div>
    
    <script>
        const CONFIG = {{CONFIG_JSON}};
        const DATA = {{DATA_JSON}};
        {{CUSTOM_JS}}
    </script>
</body>
</html>'''
    
    def _get_theme_css(self) -> str:
        """Get theme-specific CSS"""
        themes = {
            "default": '''
                <style>
                :root {
                    --bg-color: #f8f9fa;
                    --panel-bg: #ffffff;
                    --card-bg: #ffffff;
                    --text-primary: #212529;
                    --text-secondary: #6c757d;
                    --primary-color: #007bff;
                    --border-color: #dee2e6;
                    --canvas-bg: #ffffff;
                    --edge-color: #adb5bd;
                    --tooltip-bg: rgba(0, 0, 0, 0.9);
                    --tooltip-text: white;
                    --chip-bg: #e9ecef;
                    --chip-text: #495057;
                    --tab-bg: #f8f9fa;
                }
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; }
                .control-btn {
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 8px 12px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin: 4px 0;
                    width: 100%;
                    transition: background-color 0.2s;
                }
                .control-btn:hover { background: #0056b3; }
                </style>
            ''',
            "dark": '''
                <style>
                :root {
                    --bg-color: #1a1a1a;
                    --panel-bg: #2d2d2d;
                    --card-bg: #333333;
                    --text-primary: #ffffff;
                    --text-secondary: #cccccc;
                    --primary-color: #0d6efd;
                    --border-color: #444444;
                    --canvas-bg: #2d2d2d;
                    --edge-color: #666666;
                    --tooltip-bg: rgba(0, 0, 0, 0.95);
                    --tooltip-text: white;
                    --chip-bg: #444444;
                    --chip-text: #cccccc;
                    --tab-bg: #2d2d2d;
                }
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; color: var(--text-primary); }
                .control-btn {
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 8px 12px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin: 4px 0;
                    width: 100%;
                }
                </style>
            ''',
            "uniad": '''
                <style>
                :root {
                    --bg-color: #f0f4f8;
                    --panel-bg: #ffffff;
                    --card-bg: #ffffff;
                    --text-primary: #1a202c;
                    --text-secondary: #718096;
                    --primary-color: #3182ce;
                    --border-color: #e2e8f0;
                    --canvas-bg: #ffffff;
                    --edge-color: #a0aec0;
                    --tooltip-bg: rgba(26, 32, 44, 0.9);
                    --tooltip-text: white;
                    --chip-bg: #edf2f7;
                    --chip-text: #4a5568;
                    --tab-bg: #f7fafc;
                }
                body { font-family: 'Inter', 'Segoe UI', sans-serif; margin: 0; padding: 0; }
                .control-btn {
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 8px 12px;
                    border-radius: 6px;
                    cursor: pointer;
                    margin: 4px 0;
                    width: 100%;
                    font-weight: 500;
                }
                </style>
            '''
        }
        
        return themes.get(self.config.theme, themes["default"])
    
    def _get_utility_scripts(self) -> str:
        """Get utility JavaScript functions"""
        return '''
            <script>
                // Utility functions for all templates
                function downloadData(data, filename, type = 'application/json') {
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    a.click();
                    URL.revokeObjectURL(url);
                }
                
                function copyToClipboard(text) {
                    navigator.clipboard.writeText(text).then(() => {
                        console.log('Copied to clipboard');
                    });
                }
                
                function formatNumber(num, decimals = 2) {
                    return parseFloat(num).toFixed(decimals);
                }
                
                function formatBytes(bytes, decimals = 1) {
                    if (bytes === 0) return '0 B';
                    const k = 1024;
                    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
                    const i = Math.floor(Math.log(bytes) / Math.log(k));
                    return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
                }
            </script>
        '''
    
    def _validate_data(self, data: VisualizationData):
        """Validate input data for security and correctness"""
        if not isinstance(data.title, str):
            raise ValueError("Title must be a string")
        
        if not isinstance(data.data, dict):
            raise ValueError("Data must be a dictionary")
        
        if not isinstance(data.config, dict):
            raise ValueError("Config must be a dictionary")
        
        # Check for potentially dangerous content
        dangerous_patterns = ['<script', 'javascript:', 'onload=', 'onerror=']
        title_lower = data.title.lower()
        
        for pattern in dangerous_patterns:
            if pattern in title_lower:
                raise ValueError(f"Potentially dangerous content detected in title: {pattern}")
    
    def _sanitize_data(self, data: Any) -> Any:
        """Sanitize data for safe HTML injection"""
        if isinstance(data, dict):
            return {key: self._sanitize_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, str):
            # Basic HTML escaping
            return data.replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
        else:
            return data
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML characters"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))
    
    def _substitute_template_variables(self, html: str, variables: Dict[str, str]) -> str:
        """Substitute template variables in HTML"""
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            html = html.replace(placeholder, str(value))
        
        return html
    
    def _add_security_headers(self, html: str) -> str:
        """Add Content Security Policy headers"""
        csp_meta = '''<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline' https://d3js.org https://cdn.jsdelivr.net https://cdn.plot.ly; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self';">'''
        
        if '<head>' in html:
            html = html.replace('<head>', f'<head>\n    {csp_meta}')
        
        return html
    
    def _compress_html(self, html: str) -> str:
        """Compress HTML content based on configuration"""
        if self.config.compression_level == "none":
            return html
        
        # Basic compression: remove extra whitespace
        import re
        
        if self.config.compression_level == "moderate":
            # Remove leading/trailing whitespace from lines
            html = re.sub(r'^\s+', '', html, flags=re.MULTILINE)
            html = re.sub(r'\s+$', '', html, flags=re.MULTILINE)
            # Collapse multiple spaces
            html = re.sub(r' +', ' ', html)
        
        elif self.config.compression_level == "aggressive":
            # More aggressive compression
            html = re.sub(r'\s+', ' ', html)  # Collapse all whitespace
            html = re.sub(r'> <', '><', html)  # Remove spaces between tags
            html = html.strip()
        
        return html
    
    def _ensure_base_templates(self):
        """Ensure base template files exist"""
        base_templates = ["dataflow", "memory", "comparison", "dashboard"]
        
        for template_name in base_templates:
            template_path = self.template_dir / f"{template_name}.html"
            if not template_path.exists():
                content = self._get_base_template(template_name)
                template_path.write_text(content, encoding='utf-8')


# Convenience functions for common use cases
def create_dataflow_visualization(nodes: List[Dict], edges: List[Dict], 
                                title: str = "UniAD Dataflow Visualization",
                                config: Optional[TemplateConfig] = None) -> str:
    """
    Create a dataflow visualization HTML with the given nodes and edges.
    
    Args:
        nodes: List of node dictionaries with visualization data
        edges: List of edge dictionaries connecting nodes
        title: Visualization title
        config: Template configuration
        
    Returns:
        Complete HTML string for dataflow visualization
    """
    engine = HTMLTemplateEngine(config)
    
    vis_data = VisualizationData(
        title=title,
        data={
            'nodes': nodes,
            'edges': edges,
            'summary': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'total_memory_mb': sum(node.get('memory_mb', 0) for node in nodes),
                'task_head_count': len(set(node.get('task_head') for node in nodes if node.get('task_head')))
            }
        },
        config={
            'max_nodes_visible': len(nodes),
            'color_scheme': 'operation',
            'show_shapes': True
        },
        metadata={
            'generator': 'HTMLTemplateEngine',
            'version': '1.0.0',
            'visualization_type': 'dataflow'
        }
    )
    
    return engine.render_template(vis_data, "dataflow")


def create_memory_visualization(timeline_points: List[Tuple[float, float]],
                              title: str = "UniAD Memory Timeline",
                              config: Optional[TemplateConfig] = None) -> str:
    """
    Create a memory timeline visualization HTML.
    
    Args:
        timeline_points: List of (timestamp, memory_mb) tuples
        title: Visualization title
        config: Template configuration
        
    Returns:
        Complete HTML string for memory visualization
    """
    engine = HTMLTemplateEngine(config)
    
    vis_data = VisualizationData(
        title=title,
        data={
            'timeline_points': timeline_points,
            'memory_pressure': 0.5  # Placeholder
        },
        config={
            'warning_threshold_mb': 15000,
            'critical_threshold_mb': 25000
        },
        metadata={
            'generator': 'HTMLTemplateEngine',
            'version': '1.0.0',
            'visualization_type': 'memory'
        }
    )
    
    return engine.render_template(vis_data, "memory")