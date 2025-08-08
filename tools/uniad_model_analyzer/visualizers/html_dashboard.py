"""
HTML Dashboard Generator for UniAD Model Analyzer.

Creates interactive HTML dashboards with charts, heatmaps, and analysis summaries
for visualizing UniAD model performance and optimization opportunities.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime


@dataclass
class ChartData:
    """Represents data for a chart visualization."""
    
    chart_type: str  # 'line', 'bar', 'pie', 'heatmap', 'scatter'
    title: str
    data: Dict[str, Any]
    config: Dict[str, Any]
    container_id: str


@dataclass 
class DashboardSection:
    """Represents a section in the dashboard."""
    
    title: str
    content: str
    section_type: str  # 'summary', 'chart', 'table', 'text'
    order: int = 0


class HTMLDashboard:
    """
    Generates interactive HTML dashboards for UniAD model analysis.
    
    Features:
    - Interactive charts using Chart.js
    - Memory usage heatmaps
    - Performance metrics visualization
    - Task head analysis sections
    - Optimization recommendations
    - Responsive design
    """
    
    def __init__(self, theme: str = 'light'):
        """
        Initialize the HTML dashboard generator.
        
        Args:
            theme: Dashboard theme ('light', 'dark')
        """
        self.theme = theme
        self.sections: List[DashboardSection] = []
        self.charts: List[ChartData] = []
        
        # Load template
        self.template_path = Path(__file__).parent.parent / "templates" / "dashboard.html"
    
    def generate_dashboard(self, 
                         analysis_results: Dict[str, Any],
                         output_path: str,
                         title: str = "UniAD Model Analysis Dashboard") -> str:
        """
        Generate complete interactive dashboard.
        
        Args:
            analysis_results: Dictionary of analysis results from various analyzers
            output_path: Path to save the HTML dashboard
            title: Dashboard title
            
        Returns:
            Path to generated HTML file
        """
        self.sections = []
        self.charts = []
        
        # Generate dashboard sections
        self._create_summary_section(analysis_results)
        self._create_memory_section(analysis_results.get('memory_analysis', {}))
        self._create_performance_section(analysis_results.get('performance_analysis', {}))
        self._create_task_head_section(analysis_results.get('task_head_analysis', {}))
        self._create_optimization_section(analysis_results.get('optimization_analysis', {}))
        self._create_temporal_section(analysis_results.get('temporal_analysis', {}))
        self._create_dtype_section(analysis_results.get('dtype_analysis', {}))
        
        # Load and populate template
        dashboard_html = self._generate_html(title, analysis_results)
        
        # Write to file
        Path(output_path).write_text(dashboard_html, encoding='utf-8')
        
        return output_path
    
    def _create_summary_section(self, analysis_results: Dict[str, Any]) -> None:
        """Create dashboard summary section."""
        summary_data = {}
        
        # Extract key metrics
        if 'memory_analysis' in analysis_results:
            memory = analysis_results['memory_analysis'].get('summary', {})
            summary_data['peak_memory_gb'] = memory.get('peak_memory_gb', 0)
            summary_data['memory_efficiency'] = memory.get('memory_efficiency', 0)
        
        if 'performance_analysis' in analysis_results:
            perf = analysis_results['performance_analysis'].get('summary', {})
            summary_data['total_time_ms'] = perf.get('total_time_ms', 0)
            summary_data['ops_per_second'] = perf.get('ops_per_second', 0)
        
        if 'task_head_analysis' in analysis_results:
            heads = analysis_results['task_head_analysis']
            summary_data['active_task_heads'] = len([h for h in heads.keys() if h != 'summary'])
        
        # Create summary cards HTML
        summary_html = self._create_summary_cards(summary_data)
        
        section = DashboardSection(
            title="Executive Summary",
            content=summary_html,
            section_type='summary',
            order=1
        )
        self.sections.append(section)
    
    def _create_memory_section(self, memory_analysis: Dict[str, Any]) -> None:
        """Create memory analysis section."""
        if not memory_analysis:
            return
        
        # Memory timeline chart
        if 'memory_timeline' in memory_analysis:
            timeline = memory_analysis['memory_timeline']
            
            chart_data = ChartData(
                chart_type='line',
                title='Memory Usage Over Time',
                data={
                    'labels': [f"T{i}" for i in range(len(timeline))],
                    'datasets': [{
                        'label': 'Allocated Memory (MB)',
                        'data': [point.get('allocated_mb', 0) for point in timeline],
                        'borderColor': 'rgb(255, 99, 132)',
                        'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                        'tension': 0.4
                    }]
                },
                config={
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'top'},
                        'title': {'display': True, 'text': 'Memory Usage Timeline'}
                    },
                    'scales': {
                        'y': {
                            'beginAtZero': True,
                            'title': {'display': True, 'text': 'Memory (MB)'}
                        }
                    }
                },
                container_id='memory-timeline-chart'
            )
            self.charts.append(chart_data)
        
        # Memory distribution by task head
        if 'task_head_distribution' in memory_analysis:
            distribution = memory_analysis['task_head_distribution']
            
            chart_data = ChartData(
                chart_type='pie',
                title='Memory Distribution by Task Head',
                data={
                    'labels': list(distribution.keys()),
                    'datasets': [{
                        'data': [dist.get('memory_mb', 0) for dist in distribution.values()],
                        'backgroundColor': [
                            '#FF6384', '#36A2EB', '#FFCE56', 
                            '#4BC0C0', '#9966FF', '#FF9F40'
                        ]
                    }]
                },
                config={
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'right'},
                        'title': {'display': True, 'text': 'Memory by Task Head'}
                    }
                },
                container_id='memory-distribution-chart'
            )
            self.charts.append(chart_data)
        
        # Memory optimization suggestions table
        suggestions_html = ""
        if 'optimization_suggestions' in memory_analysis:
            suggestions = memory_analysis['optimization_suggestions']
            suggestions_html = self._create_suggestions_table(suggestions, "Memory Optimization")
        
        section = DashboardSection(
            title="Memory Analysis",
            content=f'<div class="charts-row"><canvas id="memory-timeline-chart"></canvas><canvas id="memory-distribution-chart"></canvas></div>{suggestions_html}',
            section_type='chart',
            order=2
        )
        self.sections.append(section)
    
    def _create_performance_section(self, performance_analysis: Dict[str, Any]) -> None:
        """Create performance analysis section."""
        if not performance_analysis:
            return
        
        # Operation timing chart
        if 'operation_timings' in performance_analysis:
            timings = performance_analysis['operation_timings'][:20]  # Top 20
            
            chart_data = ChartData(
                chart_type='bar',
                title='Top Operations by Execution Time',
                data={
                    'labels': [op.get('operation', f'Op {i}') for i, op in enumerate(timings)],
                    'datasets': [{
                        'label': 'Execution Time (ms)',
                        'data': [op.get('time_ms', 0) for op in timings],
                        'backgroundColor': 'rgba(54, 162, 235, 0.8)',
                        'borderColor': 'rgba(54, 162, 235, 1)',
                        'borderWidth': 1
                    }]
                },
                config={
                    'responsive': True,
                    'plugins': {
                        'legend': {'display': False},
                        'title': {'display': True, 'text': 'Operation Timing Analysis'}
                    },
                    'scales': {
                        'y': {
                            'beginAtZero': True,
                            'title': {'display': True, 'text': 'Time (ms)'}
                        },
                        'x': {
                            'ticks': {'maxRotation': 45}
                        }
                    }
                },
                container_id='performance-timing-chart'
            )
            self.charts.append(chart_data)
        
        # Performance heatmap
        perf_html = '<canvas id="performance-timing-chart"></canvas>'
        if 'bottlenecks' in performance_analysis:
            bottlenecks = performance_analysis['bottlenecks']
            perf_html += self._create_bottlenecks_heatmap(bottlenecks)
        
        section = DashboardSection(
            title="Performance Analysis",
            content=perf_html,
            section_type='chart',
            order=3
        )
        self.sections.append(section)
    
    def _create_task_head_section(self, task_head_analysis: Dict[str, Any]) -> None:
        """Create task head analysis section."""
        if not task_head_analysis:
            return
        
        # Task head comparison chart
        head_data = []
        for head_name, analysis in task_head_analysis.items():
            if head_name != 'summary' and isinstance(analysis, dict):
                memory_mb = analysis.get('memory_usage', {}).get('total_allocated_mb', 0)
                time_ms = analysis.get('performance', {}).get('total_time_ms', 0)
                ops_count = analysis.get('operations', {}).get('total_ops', 0)
                
                head_data.append({
                    'name': head_name,
                    'memory': memory_mb,
                    'time': time_ms,
                    'ops': ops_count
                })
        
        if head_data:
            chart_data = ChartData(
                chart_type='scatter',
                title='Task Head Performance vs Memory',
                data={
                    'datasets': [{
                        'label': head['name'],
                        'data': [{'x': head['memory'], 'y': head['time']}],
                        'backgroundColor': self._get_task_head_color(head['name']),
                        'pointRadius': max(5, head['ops'] / 10)
                    } for head in head_data]
                },
                config={
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'top'},
                        'title': {'display': True, 'text': 'Performance vs Memory by Task Head'}
                    },
                    'scales': {
                        'x': {
                            'title': {'display': True, 'text': 'Memory (MB)'},
                            'beginAtZero': True
                        },
                        'y': {
                            'title': {'display': True, 'text': 'Time (ms)'},
                            'beginAtZero': True
                        }
                    }
                },
                container_id='task-head-scatter-chart'
            )
            self.charts.append(chart_data)
        
        # Task head summary table
        table_html = self._create_task_head_table(head_data)
        
        section = DashboardSection(
            title="Task Head Analysis",
            content=f'<canvas id="task-head-scatter-chart"></canvas>{table_html}',
            section_type='chart',
            order=4
        )
        self.sections.append(section)
    
    def _create_optimization_section(self, optimization_analysis: Dict[str, Any]) -> None:
        """Create optimization recommendations section."""
        if not optimization_analysis:
            return
        
        optimization_html = '<div class="optimization-grid">'
        
        # Memory optimizations
        if 'memory_optimizations' in optimization_analysis:
            memory_opts = optimization_analysis['memory_optimizations']
            optimization_html += self._create_optimization_card(
                "Memory Optimization",
                memory_opts,
                "memory-icon",
                "primary"
            )
        
        # DType optimizations  
        if 'dtype_optimizations' in optimization_analysis:
            dtype_opts = optimization_analysis['dtype_optimizations']
            optimization_html += self._create_optimization_card(
                "Mixed Precision",
                dtype_opts,
                "precision-icon", 
                "success"
            )
        
        # Performance optimizations
        if 'performance_optimizations' in optimization_analysis:
            perf_opts = optimization_analysis['performance_optimizations']
            optimization_html += self._create_optimization_card(
                "Performance Tuning",
                perf_opts,
                "performance-icon",
                "warning"
            )
        
        optimization_html += '</div>'
        
        section = DashboardSection(
            title="Optimization Opportunities",
            content=optimization_html,
            section_type='text',
            order=5
        )
        self.sections.append(section)
    
    def _create_temporal_section(self, temporal_analysis: Dict[str, Any]) -> None:
        """Create temporal analysis section."""
        if not temporal_analysis:
            return
        
        # Temporal queue visualization
        temporal_html = ""
        if 'queue_analysis' in temporal_analysis:
            queue_data = temporal_analysis['queue_analysis']
            
            chart_data = ChartData(
                chart_type='line',
                title='Temporal Queue Memory Usage',
                data={
                    'labels': [f"Frame {i}" for i in range(len(queue_data))],
                    'datasets': [{
                        'label': 'Queue Memory (MB)',
                        'data': [frame.get('memory_mb', 0) for frame in queue_data],
                        'borderColor': 'rgb(75, 192, 192)',
                        'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                        'tension': 0.1
                    }]
                },
                config={
                    'responsive': True,
                    'plugins': {
                        'title': {'display': True, 'text': 'Temporal Processing Queue'}
                    }
                },
                container_id='temporal-queue-chart'
            )
            self.charts.append(chart_data)
            temporal_html = '<canvas id="temporal-queue-chart"></canvas>'
        
        if temporal_html:
            section = DashboardSection(
                title="Temporal Analysis",
                content=temporal_html,
                section_type='chart',
                order=6
            )
            self.sections.append(section)
    
    def _create_dtype_section(self, dtype_analysis: Dict[str, Any]) -> None:
        """Create data type analysis section."""
        if not dtype_analysis:
            return
        
        dtype_html = ""
        
        # DType distribution
        if 'dtype_distribution' in dtype_analysis:
            distribution = dtype_analysis['dtype_distribution']
            
            chart_data = ChartData(
                chart_type='pie',
                title='Data Type Distribution',
                data={
                    'labels': list(distribution.keys()),
                    'datasets': [{
                        'data': list(distribution.values()),
                        'backgroundColor': [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'
                        ]
                    }]
                },
                config={
                    'responsive': True,
                    'plugins': {
                        'legend': {'position': 'right'},
                        'title': {'display': True, 'text': 'Memory by Data Type'}
                    }
                },
                container_id='dtype-distribution-chart'
            )
            self.charts.append(chart_data)
            dtype_html = '<canvas id="dtype-distribution-chart"></canvas>'
        
        # Mixed precision opportunities
        if 'mixed_precision_opportunities' in dtype_analysis:
            opportunities = dtype_analysis['mixed_precision_opportunities']
            dtype_html += self._create_mixed_precision_table(opportunities)
        
        if dtype_html:
            section = DashboardSection(
                title="Data Type Analysis",
                content=dtype_html,
                section_type='chart',
                order=7
            )
            self.sections.append(section)
    
    def _generate_html(self, title: str, analysis_results: Dict[str, Any]) -> str:
        """Generate complete HTML dashboard."""
        # Read template
        try:
            template_content = self.template_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            # Create default template if not found
            template_content = self._get_default_template()
        
        # Generate sections HTML
        sections_html = ""
        for section in sorted(self.sections, key=lambda x: x.order):
            sections_html += f'''
            <section class="dashboard-section">
                <h2>{section.title}</h2>
                <div class="section-content">
                    {section.content}
                </div>
            </section>
            '''
        
        # Generate chart configurations
        chart_configs = {}
        for chart in self.charts:
            chart_configs[chart.container_id] = {
                'type': chart.chart_type,
                'data': chart.data,
                'options': chart.config
            }
        
        # Replace template placeholders
        html = template_content.replace('{{TITLE}}', title)
        html = html.replace('{{THEME}}', self.theme)
        html = html.replace('{{SECTIONS}}', sections_html)
        html = html.replace('{{CHART_CONFIGS}}', json.dumps(chart_configs, indent=2))
        html = html.replace('{{TIMESTAMP}}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Add summary statistics
        stats = self._extract_key_stats(analysis_results)
        html = html.replace('{{SUMMARY_STATS}}', json.dumps(stats, indent=2))
        
        return html
    
    def _create_summary_cards(self, summary_data: Dict[str, Any]) -> str:
        """Create summary metric cards."""
        cards_html = '<div class="summary-cards">'
        
        # Memory card
        if 'peak_memory_gb' in summary_data:
            memory_gb = summary_data['peak_memory_gb']
            memory_status = 'success' if memory_gb < 30 else 'warning' if memory_gb < 45 else 'danger'
            cards_html += f'''
            <div class="metric-card {memory_status}">
                <div class="metric-icon">💾</div>
                <div class="metric-value">{memory_gb:.1f} GB</div>
                <div class="metric-label">Peak Memory</div>
            </div>
            '''
        
        # Performance card
        if 'total_time_ms' in summary_data:
            time_ms = summary_data['total_time_ms']
            time_status = 'success' if time_ms < 100 else 'warning' if time_ms < 500 else 'danger'
            cards_html += f'''
            <div class="metric-card {time_status}">
                <div class="metric-icon">⚡</div>
                <div class="metric-value">{time_ms:.1f} ms</div>
                <div class="metric-label">Total Time</div>
            </div>
            '''
        
        # Task heads card
        if 'active_task_heads' in summary_data:
            heads_count = summary_data['active_task_heads']
            cards_html += f'''
            <div class="metric-card info">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">{heads_count}</div>
                <div class="metric-label">Task Heads</div>
            </div>
            '''
        
        # Efficiency card
        if 'memory_efficiency' in summary_data:
            efficiency = summary_data['memory_efficiency']
            eff_status = 'success' if efficiency > 0.8 else 'warning' if efficiency > 0.6 else 'danger'
            cards_html += f'''
            <div class="metric-card {eff_status}">
                <div class="metric-icon">📈</div>
                <div class="metric-value">{efficiency:.2f}</div>
                <div class="metric-label">Efficiency</div>
            </div>
            '''
        
        cards_html += '</div>'
        return cards_html
    
    def _create_suggestions_table(self, suggestions: List[Dict[str, Any]], title: str) -> str:
        """Create a table of optimization suggestions."""
        if not suggestions:
            return ""
        
        table_html = f'''
        <div class="table-container">
            <h3>{title} Suggestions</h3>
            <table class="suggestions-table">
                <thead>
                    <tr>
                        <th>Priority</th>
                        <th>Target</th>
                        <th>Description</th>
                        <th>Expected Savings</th>
                        <th>Difficulty</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for suggestion in suggestions[:10]:  # Top 10
            priority = suggestion.get('severity', 'medium')
            target = suggestion.get('target', 'Unknown')
            description = suggestion.get('description', '')
            savings = suggestion.get('expected_savings_mb', 0)
            difficulty = suggestion.get('difficulty', 'medium')
            
            priority_class = 'high' if priority == 'critical' else priority.lower()
            
            table_html += f'''
            <tr class="suggestion-row">
                <td><span class="priority-badge {priority_class}">{priority.upper()}</span></td>
                <td class="target-cell">{target}</td>
                <td class="description-cell">{description}</td>
                <td class="savings-cell">{savings:.1f} MB</td>
                <td class="difficulty-cell">{difficulty.title()}</td>
            </tr>
            '''
        
        table_html += '''
                </tbody>
            </table>
        </div>
        '''
        
        return table_html
    
    def _create_task_head_table(self, head_data: List[Dict[str, Any]]) -> str:
        """Create task head comparison table."""
        if not head_data:
            return ""
        
        table_html = '''
        <div class="table-container">
            <h3>Task Head Performance Summary</h3>
            <table class="task-head-table">
                <thead>
                    <tr>
                        <th>Task Head</th>
                        <th>Operations</th>
                        <th>Memory (MB)</th>
                        <th>Time (ms)</th>
                        <th>Efficiency</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for head in head_data:
            efficiency = head['ops'] / max(1, head['time']) * 1000  # ops per second
            eff_class = 'high' if efficiency > 100 else 'medium' if efficiency > 50 else 'low'
            
            table_html += f'''
            <tr class="task-head-row">
                <td class="head-name">{head['name'].title()}</td>
                <td class="ops-count">{head['ops']}</td>
                <td class="memory-usage">{head['memory']:.1f}</td>
                <td class="time-usage">{head['time']:.2f}</td>
                <td class="efficiency-cell"><span class="efficiency-badge {eff_class}">{efficiency:.1f}</span></td>
            </tr>
            '''
        
        table_html += '''
                </tbody>
            </table>
        </div>
        '''
        
        return table_html
    
    def _create_optimization_card(self, title: str, optimizations: List[Dict[str, Any]], 
                                icon: str, card_type: str) -> str:
        """Create an optimization opportunity card."""
        if not optimizations:
            return ""
        
        total_savings = sum(opt.get('expected_savings_mb', 0) for opt in optimizations)
        impact_count = len([opt for opt in optimizations if opt.get('severity') in ['critical', 'high']])
        
        return f'''
        <div class="optimization-card {card_type}">
            <div class="card-header">
                <div class="card-icon {icon}">🔧</div>
                <h3>{title}</h3>
            </div>
            <div class="card-content">
                <div class="stat-row">
                    <span class="stat-label">Opportunities:</span>
                    <span class="stat-value">{len(optimizations)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">High Impact:</span>
                    <span class="stat-value">{impact_count}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Savings:</span>
                    <span class="stat-value">{total_savings:.1f} MB</span>
                </div>
                <div class="card-actions">
                    <button class="btn-details" onclick="showOptimizationDetails('{title.lower().replace(' ', '_')}')">
                        View Details
                    </button>
                </div>
            </div>
        </div>
        '''
    
    def _create_mixed_precision_table(self, opportunities: List[Dict[str, Any]]) -> str:
        """Create mixed precision opportunities table."""
        if not opportunities:
            return ""
        
        table_html = '''
        <div class="table-container">
            <h3>Mixed Precision Opportunities</h3>
            <table class="mixed-precision-table">
                <thead>
                    <tr>
                        <th>Module</th>
                        <th>Current Type</th>
                        <th>Target Type</th>
                        <th>Memory Savings</th>
                        <th>Speedup</th>
                        <th>Risk</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for opp in opportunities[:15]:  # Top 15
            module = opp.get('module', 'Unknown')
            current = opp.get('current_dtype', 'float32')
            target = opp.get('target_dtype', 'float16')
            savings = opp.get('memory_savings_mb', 0)
            speedup = opp.get('speedup_factor', 1.0)
            risk = opp.get('risk', 'low')
            
            risk_class = 'high' if risk == 'high' else 'medium' if risk == 'medium' else 'low'
            
            table_html += f'''
            <tr class="precision-row">
                <td class="module-cell">{module}</td>
                <td class="dtype-cell">{current}</td>
                <td class="dtype-cell">{target}</td>
                <td class="savings-cell">{savings:.1f} MB</td>
                <td class="speedup-cell">{speedup:.1f}x</td>
                <td class="risk-cell"><span class="risk-badge {risk_class}">{risk.upper()}</span></td>
            </tr>
            '''
        
        table_html += '''
                </tbody>
            </table>
        </div>
        '''
        
        return table_html
    
    def _create_bottlenecks_heatmap(self, bottlenecks: List[Dict[str, Any]]) -> str:
        """Create performance bottlenecks heatmap."""
        if not bottlenecks:
            return ""
        
        # Simple text-based heatmap for now
        heatmap_html = '''
        <div class="bottlenecks-heatmap">
            <h3>Performance Bottlenecks</h3>
            <div class="heatmap-grid">
        '''
        
        for bottleneck in bottlenecks[:20]:
            operation = bottleneck.get('operation', 'Unknown')
            severity = bottleneck.get('severity', 0.5)
            time_ms = bottleneck.get('time_ms', 0)
            
            severity_class = 'critical' if severity > 0.8 else 'high' if severity > 0.6 else 'medium'
            
            heatmap_html += f'''
            <div class="heatmap-cell {severity_class}" title="{operation}: {time_ms:.2f}ms">
                <div class="cell-label">{operation}</div>
                <div class="cell-value">{time_ms:.1f}ms</div>
            </div>
            '''
        
        heatmap_html += '''
            </div>
        </div>
        '''
        
        return heatmap_html
    
    def _get_task_head_color(self, head_name: str) -> str:
        """Get color for task head."""
        colors = {
            'track': '#FF6384',
            'segmentation': '#36A2EB', 
            'motion': '#FFCE56',
            'occupancy': '#4BC0C0',
            'planning': '#9966FF'
        }
        return colors.get(head_name.lower(), '#FF9F40')
    
    def _extract_key_stats(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key statistics for JavaScript access."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'sections_count': len(self.sections),
            'charts_count': len(self.charts)
        }
        
        # Add memory stats
        if 'memory_analysis' in analysis_results:
            memory = analysis_results['memory_analysis']
            if 'summary' in memory:
                stats['peak_memory_gb'] = memory['summary'].get('peak_memory_gb', 0)
                stats['memory_efficiency'] = memory['summary'].get('memory_efficiency', 0)
        
        return stats
    
    def _get_default_template(self) -> str:
        """Get default HTML template if template file not found."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .dashboard-header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .dashboard-section {
            background: white;
            margin-bottom: 20px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            color: white;
        }
        .metric-card.success { background: #28a745; }
        .metric-card.warning { background: #ffc107; color: #333; }
        .metric-card.danger { background: #dc3545; }
        .metric-card.info { background: #17a2b8; }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .charts-row {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        canvas {
            max-height: 400px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .priority-badge, .efficiency-badge, .risk-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .priority-badge.critical, .priority-badge.high { background: #dc3545; color: white; }
        .priority-badge.medium { background: #ffc107; color: #333; }
        .priority-badge.low { background: #28a745; color: white; }
        .optimization-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .optimization-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
        }
        .heatmap-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
        }
        .heatmap-cell {
            padding: 15px;
            border-radius: 4px;
            text-align: center;
            color: white;
        }
        .heatmap-cell.critical { background: #dc3545; }
        .heatmap-cell.high { background: #fd7e14; }
        .heatmap-cell.medium { background: #ffc107; color: #333; }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>{{TITLE}}</h1>
        <p>Generated on {{TIMESTAMP}}</p>
    </div>

    {{SECTIONS}}

    <script>
        const chartConfigs = {{CHART_CONFIGS}};
        const summaryStats = {{SUMMARY_STATS}};
        
        // Initialize all charts
        Object.entries(chartConfigs).forEach(([containerId, config]) => {
            const ctx = document.getElementById(containerId);
            if (ctx) {
                new Chart(ctx, config);
            }
        });
        
        function showOptimizationDetails(optimizationType) {
            alert(`Detailed view for ${optimizationType} optimizations would open here.`);
        }
        
        console.log('Dashboard loaded with stats:', summaryStats);
    </script>
</body>
</html>'''