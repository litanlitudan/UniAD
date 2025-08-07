"""Memory timeline visualizer for tracking memory allocation/deallocation patterns

This module provides visualization capabilities for memory usage over time,
specifically designed for UniAD's multi-task architecture and memory profiling needs.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from ..core.memory_structures import MemoryTimelineEvent, MemoryTimelineAnalysis
    from ..core.data_structures import TraceNode
except (ImportError, ValueError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.memory_structures import MemoryTimelineEvent, MemoryTimelineAnalysis
    from core.data_structures import TraceNode


@dataclass
class MemoryPeak:
    """Represents a memory usage peak in the timeline"""
    timestamp: float
    peak_memory_mb: float
    operation: str
    module_path: str
    task_head: Optional[str] = None
    is_problematic: bool = False
    context_operations: List[str] = field(default_factory=list)  # Operations leading to this peak


@dataclass
class MemoryTimelineData:
    """Structured data for memory timeline visualization"""
    events: List[MemoryTimelineEvent]
    timeline_points: List[Tuple[float, float]]  # (timestamp, cumulative_memory)
    peaks: List[MemoryPeak]
    task_breakdown: Dict[str, List[Tuple[float, float]]]  # task -> timeline points
    memory_pressure: float
    problematic_operations: List[MemoryTimelineEvent]
    recommendations: List[str]


class MemoryTimelineVisualizer:
    """
    Generates interactive memory timeline visualizations for UniAD operations.
    
    This class provides comprehensive memory timeline analysis including:
    - Memory allocation/deallocation patterns over time
    - Identification of memory peaks and bottlenecks  
    - Task head memory breakdown for multi-task analysis
    - Threshold-based highlighting of problematic operations
    - Interactive timeline visualization with hover details
    """
    
    def __init__(self, 
                 stage: int = 2,
                 warning_threshold_mb: float = 15000,  # 15GB warning
                 critical_threshold_mb: float = 25000,  # 25GB critical
                 peak_detection_window: int = 5,  # Number of events to look for peaks
                 enable_task_breakdown: bool = True):
        """
        Initialize the memory timeline visualizer.
        
        Args:
            stage: UniAD training stage (1 or 2)
            warning_threshold_mb: Memory threshold for warnings (MB)
            critical_threshold_mb: Memory threshold for critical alerts (MB)
            peak_detection_window: Window size for peak detection
            enable_task_breakdown: Whether to generate task head breakdown
        """
        self.stage = stage
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.peak_detection_window = peak_detection_window
        self.enable_task_breakdown = enable_task_breakdown
        
        # Expected memory usage by stage
        self.expected_memory = {1: 30000, 2: 17000}  # MB
        
        # Color scheme for visualization
        self.colors = {
            'normal': '#2E8B57',      # Sea green
            'warning': '#FF8C00',     # Dark orange  
            'critical': '#DC143C',    # Crimson
            'peak': '#4169E1',        # Royal blue
            'task_track': '#FF6347',  # Tomato
            'task_seg': '#32CD32',    # Lime green
            'task_motion': '#4169E1', # Royal blue
            'task_occ': '#FF1493',    # Deep pink
            'task_planning': '#9932CC' # Dark violet
        }
    
    def generate_timeline(self, memory_events: List[MemoryTimelineEvent]) -> MemoryTimelineData:
        """
        Generate comprehensive timeline data from memory events.
        
        Args:
            memory_events: List of memory timeline events or TraceNode objects
            
        Returns:
            MemoryTimelineData containing all timeline information
        """
        if not memory_events:
            return MemoryTimelineData(
                events=[],
                timeline_points=[],
                peaks=[],
                task_breakdown={},
                memory_pressure=0.0,
                problematic_operations=[],
                recommendations=[]
            )
        
        # Convert TraceNode objects to MemoryTimelineEvent if needed
        converted_events = []
        cumulative_memory = 0.0
        
        for i, item in enumerate(memory_events):
            if isinstance(item, TraceNode):
                # Convert TraceNode to MemoryTimelineEvent
                cumulative_memory += item.memory_usage
                event = MemoryTimelineEvent(
                    timestamp=float(i),  # Use index as timestamp
                    operation=item.operation,
                    module_path=item.module_path,
                    memory_delta=item.memory_usage,
                    cumulative_memory=cumulative_memory,
                    task_head=item.task_head if hasattr(item, 'task_head') else None,
                    temporal_index=item.temporal_index if hasattr(item, 'temporal_index') else None,
                    is_bev_operation=item.is_bev_operation if hasattr(item, 'is_bev_operation') else False,
                    is_frozen_module=item.is_frozen if hasattr(item, 'is_frozen') else False,
                    compute_time=item.compute_time,
                    tensor_info={
                        'input_shapes': [info.to_dict() for info in item.input_shapes] if item.input_shapes else [],
                        'output_shapes': [info.to_dict() for info in item.output_shapes] if item.output_shapes else []
                    }
                )
                converted_events.append(event)
            else:
                # Already a MemoryTimelineEvent
                converted_events.append(item)
        
        # Sort events by timestamp
        sorted_events = sorted(converted_events, key=lambda x: x.timestamp)
        
        # Generate basic timeline points
        timeline_points = [
            (event.timestamp, event.cumulative_memory) 
            for event in sorted_events
        ]
        
        # Identify peaks
        peaks = self.identify_peaks(sorted_events)
        
        # Generate task breakdown if enabled
        task_breakdown = {}
        if self.enable_task_breakdown:
            task_breakdown = self._generate_task_breakdown(sorted_events)
        
        # Calculate memory pressure
        memory_pressure = self.calculate_memory_pressure(sorted_events)
        
        # Identify problematic operations
        problematic_operations = [
            event for event in sorted_events 
            if event.is_problematic(self.warning_threshold_mb, self.critical_threshold_mb)
        ]
        
        # Generate analysis and recommendations
        analysis = MemoryTimelineAnalysis(events=sorted_events)
        analysis.analyze()
        recommendations = analysis.get_recommendations()
        
        return MemoryTimelineData(
            events=sorted_events,
            timeline_points=timeline_points,
            peaks=peaks,
            task_breakdown=task_breakdown,
            memory_pressure=memory_pressure,
            problematic_operations=problematic_operations,
            recommendations=recommendations
        )
    
    def identify_peaks(self, events: List[MemoryTimelineEvent]) -> List[MemoryPeak]:
        """
        Identify memory usage peaks in the timeline.
        
        Args:
            events: Sorted list of memory events
            
        Returns:
            List of identified memory peaks
        """
        if len(events) < 3:
            return []
        
        peaks = []
        window = self.peak_detection_window
        
        for i in range(window, len(events) - window):
            current_memory = events[i].cumulative_memory
            
            # Check if this is a local maximum
            is_peak = True
            for j in range(max(0, i - window), min(len(events), i + window + 1)):
                if j != i and events[j].cumulative_memory >= current_memory:
                    is_peak = False
                    break
            
            if is_peak:
                # Get context operations (operations leading to this peak)
                context_ops = [
                    events[k].operation 
                    for k in range(max(0, i - 3), i + 1)
                ]
                
                peak = MemoryPeak(
                    timestamp=events[i].timestamp,
                    peak_memory_mb=current_memory,
                    operation=events[i].operation,
                    module_path=events[i].module_path,
                    task_head=events[i].task_head,
                    is_problematic=events[i].is_problematic(
                        self.warning_threshold_mb, 
                        self.critical_threshold_mb
                    ),
                    context_operations=context_ops
                )
                peaks.append(peak)
        
        # Sort peaks by memory usage (highest first)
        peaks.sort(key=lambda x: x.peak_memory_mb, reverse=True)
        
        return peaks
    
    def calculate_memory_pressure(self, events: List[MemoryTimelineEvent]) -> float:
        """
        Calculate overall memory pressure score (0.0 to 1.0).
        
        Higher scores indicate more memory pressure and potential issues.
        
        Args:
            events: List of memory events
            
        Returns:
            Memory pressure score between 0.0 and 1.0
        """
        if not events:
            return 0.0
        
        # Factors contributing to memory pressure
        max_memory = max(event.cumulative_memory for event in events)
        expected_max = self.expected_memory.get(self.stage, 17000)
        
        # 1. Peak memory vs expected (0.4 weight)
        peak_pressure = min(1.0, max_memory / expected_max) * 0.4
        
        # 2. Threshold violations (0.3 weight)
        violations = sum(1 for event in events if event.exceeds_threshold)
        violation_pressure = min(1.0, violations / len(events) * 5) * 0.3
        
        # 3. Memory growth rate (0.2 weight)
        if len(events) > 1:
            start_memory = events[0].cumulative_memory
            end_memory = events[-1].cumulative_memory
            time_span = events[-1].timestamp - events[0].timestamp
            
            if time_span > 0:
                growth_rate = (end_memory - start_memory) / (time_span / 1000.0)  # MB/s
                growth_pressure = min(1.0, growth_rate / 1000.0) * 0.2  # Normalize by 1GB/s
            else:
                growth_pressure = 0.0
        else:
            growth_pressure = 0.0
        
        # 4. Large single allocations (0.1 weight)
        large_allocs = sum(1 for event in events if abs(event.memory_delta) > 2000)  # >2GB
        alloc_pressure = min(1.0, large_allocs / len(events) * 10) * 0.1
        
        total_pressure = peak_pressure + violation_pressure + growth_pressure + alloc_pressure
        return min(1.0, total_pressure)
    
    def rank_operations_by_memory(self, events: List[MemoryTimelineEvent], 
                                  top_k: int = 10) -> List[Tuple[str, float, str]]:
        """
        Rank operations by memory consumption.
        
        Args:
            events: List of memory events
            top_k: Number of top operations to return
            
        Returns:
            List of (operation, memory_usage, task_head) tuples sorted by usage
        """
        # Aggregate memory usage by operation
        operation_memory = defaultdict(float)
        operation_task = {}
        
        for event in events:
            if event.memory_delta > 0:  # Only count allocations
                operation_memory[event.operation] += event.memory_delta
                operation_task[event.operation] = event.task_head or "unknown"
        
        # Sort by memory usage
        sorted_ops = sorted(
            operation_memory.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top K with task head information
        return [
            (op, memory, operation_task.get(op, "unknown")) 
            for op, memory in sorted_ops[:top_k]
        ]
    
    def generate_interactive_html(self, timeline_data: MemoryTimelineData, 
                                 output_path: str = "memory_timeline.html") -> str:
        """
        Generate an interactive HTML visualization of the memory timeline.
        
        Args:
            timeline_data: Timeline data to visualize
            output_path: Output file path for HTML
            
        Returns:
            Generated HTML content as string
        """
        # Prepare data for Chart.js
        chart_data = {
            'labels': [f"{point[0]:.1f}ms" for point in timeline_data.timeline_points],
            'datasets': []
        }
        
        # Main memory timeline
        chart_data['datasets'].append({
            'label': 'Memory Usage (MB)',
            'data': [point[1] for point in timeline_data.timeline_points],
            'borderColor': self.colors['normal'],
            'backgroundColor': self.colors['normal'] + '20',  # 20% opacity
            'fill': True,
            'tension': 0.1
        })
        
        # Add threshold lines
        if self.warning_threshold_mb:
            chart_data['datasets'].append({
                'label': 'Warning Threshold',
                'data': [self.warning_threshold_mb] * len(timeline_data.timeline_points),
                'borderColor': self.colors['warning'],
                'borderDash': [5, 5],
                'fill': False
            })
        
        if self.critical_threshold_mb:
            chart_data['datasets'].append({
                'label': 'Critical Threshold', 
                'data': [self.critical_threshold_mb] * len(timeline_data.timeline_points),
                'borderColor': self.colors['critical'],
                'borderDash': [10, 5],
                'fill': False
            })
        
        # Add task breakdown if available
        for task_head, points in timeline_data.task_breakdown.items():
            if points and task_head in self.colors:
                chart_data['datasets'].append({
                    'label': f'Task: {task_head}',
                    'data': [point[1] for point in points],
                    'borderColor': self.colors.get(f'task_{task_head}', '#666666'),
                    'fill': False,
                    'hidden': True  # Hidden by default
                })
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UniAD Memory Timeline - Stage {self.stage}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .pressure-high {{ color: #e74c3c; }}
        .pressure-medium {{ color: #f39c12; }}
        .pressure-low {{ color: #27ae60; }}
        .peaks-list {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .peak-item {{
            border-left: 4px solid #3498db;
            padding: 10px;
            margin: 10px 0;
            background: #f8f9fa;
        }}
        .peak-problematic {{
            border-left-color: #e74c3c;
            background: #fff5f5;
        }}
        .recommendations {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        .rec-item {{
            padding: 8px;
            margin: 5px 0;
            background: #e8f4fd;
            border-radius: 4px;
            border-left: 3px solid #3498db;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>UniAD Memory Timeline Analysis - Stage {self.stage}</h1>
            <p>Interactive visualization of memory usage patterns during model execution</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Peak Memory Usage</div>
                <div class="stat-value">{max((point[1] for point in timeline_data.timeline_points), default=0):.1f} MB</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Memory Pressure</div>
                <div class="stat-value pressure-{'high' if timeline_data.memory_pressure > 0.7 else 'medium' if timeline_data.memory_pressure > 0.4 else 'low'}">
                    {timeline_data.memory_pressure:.2f}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Problematic Operations</div>
                <div class="stat-value">{len(timeline_data.problematic_operations)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Memory Peaks</div>
                <div class="stat-value">{len(timeline_data.peaks)}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="memoryChart" width="400" height="100"></canvas>
        </div>
        
        <div class="peaks-list">
            <h3>Memory Peaks</h3>
            {self._generate_peaks_html(timeline_data.peaks)}
        </div>
        
        <div class="recommendations">
            <h3>Optimization Recommendations</h3>
            {self._generate_recommendations_html(timeline_data.recommendations)}
        </div>
    </div>

    <script>
        const ctx = document.getElementById('memoryChart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {json.dumps(chart_data)},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Memory Usage Over Time'
                    }},
                    legend: {{
                        display: true
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: 'Time (ms)'
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: 'Memory Usage (MB)'
                        }}
                    }}
                }},
                interaction: {{
                    intersect: false,
                    mode: 'index'
                }},
                elements: {{
                    point: {{
                        radius: 3
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        # Write to file if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        return html_content
    
    def generate_markdown_report(self, timeline_data: MemoryTimelineData) -> str:
        """
        Generate a markdown report of the memory timeline analysis.
        
        Args:
            timeline_data: Timeline data to report
            
        Returns:
            Markdown formatted report
        """
        report = []
        
        # Header
        report.append(f"# UniAD Memory Timeline Analysis - Stage {self.stage}")
        report.append("")
        
        # Summary statistics
        if timeline_data.timeline_points:
            max_memory = max(point[1] for point in timeline_data.timeline_points)
            avg_memory = sum(point[1] for point in timeline_data.timeline_points) / len(timeline_data.timeline_points)
            
            report.append("## Summary Statistics")
            report.append("")
            report.append(f"- **Peak Memory Usage**: {max_memory:.1f} MB")
            report.append(f"- **Average Memory Usage**: {avg_memory:.1f} MB") 
            report.append(f"- **Memory Pressure Score**: {timeline_data.memory_pressure:.2f}")
            report.append(f"- **Problematic Operations**: {len(timeline_data.problematic_operations)}")
            report.append(f"- **Memory Peaks Detected**: {len(timeline_data.peaks)}")
            report.append("")
        
        # Memory pressure assessment
        pressure_level = "High" if timeline_data.memory_pressure > 0.7 else "Medium" if timeline_data.memory_pressure > 0.4 else "Low"
        report.append(f"## Memory Pressure Assessment: {pressure_level}")
        report.append("")
        
        # Top memory peaks
        if timeline_data.peaks:
            report.append("## Top Memory Peaks")
            report.append("")
            for i, peak in enumerate(timeline_data.peaks[:5], 1):
                status = "🔴 PROBLEMATIC" if peak.is_problematic else "🟢 Normal"
                task_info = f" ({peak.task_head})" if peak.task_head else ""
                report.append(f"{i}. **{peak.operation}**{task_info} - {peak.peak_memory_mb:.1f} MB {status}")
                report.append(f"   - Time: {peak.timestamp:.1f}ms")
                report.append(f"   - Module: `{peak.module_path}`")
                if peak.context_operations:
                    report.append(f"   - Context: {' → '.join(peak.context_operations[-3:])}")
                report.append("")
        
        # Top memory consumers
        top_operations = self.rank_operations_by_memory(timeline_data.events)
        if top_operations:
            report.append("## Top Memory Consumers")
            report.append("")
            for i, (operation, memory, task_head) in enumerate(top_operations, 1):
                task_info = f" ({task_head})" if task_head != "unknown" else ""
                report.append(f"{i}. **{operation}**{task_info}: {memory:.1f} MB")
            report.append("")
        
        # Task breakdown
        if timeline_data.task_breakdown:
            report.append("## Task Head Memory Breakdown")
            report.append("")
            for task, points in timeline_data.task_breakdown.items():
                if points:
                    max_task_memory = max(point[1] for point in points)
                    report.append(f"- **{task}**: {max_task_memory:.1f} MB peak")
            report.append("")
        
        # Recommendations
        if timeline_data.recommendations:
            report.append("## Optimization Recommendations")
            report.append("")
            for i, rec in enumerate(timeline_data.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_task_breakdown(self, events: List[MemoryTimelineEvent]) -> Dict[str, List[Tuple[float, float]]]:
        """Generate memory timeline breakdown by task head"""
        task_timelines = defaultdict(list)
        
        for event in events:
            if event.task_head:
                task_timelines[event.task_head].append((event.timestamp, event.cumulative_memory))
        
        return dict(task_timelines)
    
    def _generate_peaks_html(self, peaks: List[MemoryPeak]) -> str:
        """Generate HTML for memory peaks list"""
        if not peaks:
            return "<p>No significant memory peaks detected.</p>"
        
        html_parts = []
        for peak in peaks[:10]:  # Show top 10 peaks
            css_class = "peak-item peak-problematic" if peak.is_problematic else "peak-item"
            task_info = f" ({peak.task_head})" if peak.task_head else ""
            
            html_parts.append(f'''
            <div class="{css_class}">
                <strong>{peak.operation}{task_info}</strong> - {peak.peak_memory_mb:.1f} MB
                <br>
                <small>Time: {peak.timestamp:.1f}ms | Module: {peak.module_path}</small>
            </div>
            ''')
        
        return "".join(html_parts)
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate HTML for recommendations list"""
        if not recommendations:
            return "<p>No specific recommendations at this time.</p>"
        
        html_parts = []
        for rec in recommendations:
            html_parts.append(f'<div class="rec-item">{rec}</div>')
        
        return "".join(html_parts)
    
    def segment_by_task_head(self, nodes: List[TraceNode]) -> Dict[str, Dict[str, Any]]:
        """Segment nodes by task head for analysis
        
        Args:
            nodes: List of trace nodes
            
        Returns:
            Dictionary with task head segmentation
        """
        segments = {}
        for task_head in ['track', 'seg', 'motion', 'occ', 'planning']:
            head_nodes = [n for n in nodes if hasattr(n, 'task_head') and n.task_head == task_head]
            if head_nodes:
                segments[task_head] = {
                    'nodes': head_nodes,
                    'total_memory': sum(n.memory_usage for n in head_nodes),
                    'avg_memory': sum(n.memory_usage for n in head_nodes) / len(head_nodes)
                }
        return segments
    
    def generate_memory_heatmap(self, nodes: List[TraceNode]) -> Dict[str, Any]:
        """Generate memory heatmap data
        
        Args:
            nodes: List of trace nodes
            
        Returns:
            Dictionary with heatmap data
        """
        return {
            'data': [[node.memory_usage for node in nodes]],
            'color_scale': ['low', 'medium', 'high']
        }
    
    def compare_timelines(self, timeline1: MemoryTimelineData, timeline2: MemoryTimelineData) -> Dict[str, Any]:
        """Compare two memory timelines
        
        Args:
            timeline1: First timeline
            timeline2: Second timeline
            
        Returns:
            Comparison results
        """
        total1 = sum(point[1] for point in timeline1.timeline_points)
        total2 = sum(point[1] for point in timeline2.timeline_points)
        
        return {
            'memory_difference': total1 - total2,
            'pressure_difference': timeline1.memory_pressure - timeline2.memory_pressure
        }


def create_memory_timeline_from_trace_nodes(trace_nodes: List[TraceNode], 
                                           stage: int = 2) -> List[MemoryTimelineEvent]:
    """
    Convert TraceNode objects to MemoryTimelineEvent objects for visualization.
    
    Args:
        trace_nodes: List of trace nodes from operation tracing
        stage: UniAD training stage
        
    Returns:
        List of memory timeline events
    """
    events = []
    cumulative_memory = 0.0
    
    for i, node in enumerate(trace_nodes):
        # Convert to memory timeline event
        cumulative_memory += node.memory_usage
        
        event = MemoryTimelineEvent(
            timestamp=float(i),  # Use index as timestamp if no actual timing
            operation=node.operation,
            module_path=node.module_path,
            memory_delta=node.memory_usage,
            cumulative_memory=cumulative_memory,
            task_head=node.task_head,
            temporal_index=node.temporal_index,
            is_bev_operation=node.is_bev_operation,
            is_frozen_module=node.is_frozen,
            compute_time=node.compute_time,
            tensor_info={
                'input_shapes': [info.to_dict() for info in node.input_shapes],
                'output_shapes': [info.to_dict() for info in node.output_shapes]
            }
        )
        
        events.append(event)
    
    return events