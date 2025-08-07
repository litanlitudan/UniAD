/**
 * Memory Timeline Visualization for UniAD PyTorch Operation Tracer
 * 
 * This module provides interactive memory timeline visualization capabilities
 * specifically designed for UniAD's multi-task architecture and memory profiling needs.
 * 
 * Features:
 * - Interactive memory timeline charts using Chart.js
 * - Allocation/deallocation pattern visualization
 * - Configurable warning and critical threshold highlighting
 * - Task head memory breakdown (track, seg, motion, occ, planning)
 * - Peak detection and problematic operation highlighting
 * - Zoom/pan functionality for large timelines
 * - Real-time data updates and export functionality
 * - Integration with MemoryTimelineVisualizer Python class
 * 
 * Usage:
 *   const timeline = new MemoryTimelineChart(canvasElement, data, options);
 *   timeline.render();
 */

class MemoryTimelineChart {
    constructor(canvas, data, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.rawData = data;
        this.options = this.mergeOptions(options);
        
        // Chart.js instance
        this.chart = null;
        
        // UniAD-specific configurations
        this.stage = this.options.stage || 2;
        this.taskHeads = ['track', 'seg', 'motion', 'occ', 'planning', 'bev'];
        this.taskColors = {
            track: '#FF6347',    // Tomato
            seg: '#32CD32',      // Lime green  
            motion: '#4169E1',   // Royal blue
            occ: '#FF1493',      // Deep pink
            planning: '#9932CC', // Dark violet
            bev: '#20B2AA'       // Light sea green
        };
        
        // Memory thresholds based on UniAD stages
        this.memoryThresholds = {
            1: { warning: 35000, critical: 45000 },  // Stage 1: 35GB/45GB
            2: { warning: 15000, critical: 25000 }   // Stage 2: 15GB/25GB  
        };
        
        // State management
        this.highlightedPeaks = new Set();
        this.activeFilters = new Set(this.taskHeads);
        this.syncedCursor = null;
        
        // UI elements
        this.tooltip = null;
        this.controlPanel = null;
        
        this.init();
    }
    
    /**
     * Initialize the memory timeline chart
     */
    init() {
        this.createTooltip();
        this.createControlPanel();
        this.setupEventHandlers();
        console.log('MemoryTimelineChart initialized for Stage', this.stage);
    }
    
    /**
     * Merge user options with defaults
     */
    mergeOptions(userOptions) {
        const defaults = {
            stage: 2,
            responsive: true,
            maintainAspectRatio: false,
            enableZoom: true,
            enablePan: true,
            enableTaskBreakdown: true,
            enableThresholds: true,
            enableExport: true,
            animationDuration: 750,
            peakDetectionWindow: 5,
            highlightProblematic: true,
            colors: {
                normal: '#2E8B57',      // Sea green
                warning: '#FF8C00',     // Dark orange
                critical: '#DC143C',    // Crimson
                peak: '#4169E1',        // Royal blue
                allocation: '#32CD32',   // Lime green
                deallocation: '#FF6B35' // Red orange
            }
        };
        
        return { ...defaults, ...userOptions };
    }
    
    /**
     * Create tooltip element for detailed information
     */
    createTooltip() {
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'memory-timeline-tooltip';
        this.tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 12px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            max-width: 300px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            display: none;
        `;
        document.body.appendChild(this.tooltip);
    }
    
    /**
     * Create control panel for timeline interaction
     */
    createControlPanel() {
        const container = this.canvas.parentElement;
        
        this.controlPanel = document.createElement('div');
        this.controlPanel.className = 'memory-timeline-controls';
        this.controlPanel.style.cssText = `
            margin-bottom: 15px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        `;
        
        this.controlPanel.innerHTML = `
            <div class="control-row" style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap;">
                <div class="task-filters">
                    <label style="font-weight: bold; margin-right: 8px;">Task Heads:</label>
                    ${this.taskHeads.map(task => `
                        <label style="margin-right: 12px;">
                            <input type="checkbox" value="${task}" checked 
                                   style="margin-right: 4px;">
                            <span style="color: ${this.taskColors[task]};">●</span> ${task}
                        </label>
                    `).join('')}
                </div>
                
                <div class="timeline-controls">
                    <button id="resetZoom" class="btn-control">Reset Zoom</button>
                    <button id="highlightPeaks" class="btn-control">Highlight Peaks</button>
                    <button id="exportData" class="btn-control">Export CSV</button>
                    <button id="exportImage" class="btn-control">Export PNG</button>
                </div>
                
                <div class="memory-stats" style="margin-left: auto; font-size: 12px; color: #666;">
                    <span id="memoryStats"></span>
                </div>
            </div>
        `;
        
        // Add CSS for buttons
        const style = document.createElement('style');
        style.textContent = `
            .btn-control {
                padding: 6px 12px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                margin-right: 6px;
            }
            .btn-control:hover {
                background: #0056b3;
            }
            .btn-control:disabled {
                background: #6c757d;
                cursor: not-allowed;
            }
        `;
        document.head.appendChild(style);
        
        container.insertBefore(this.controlPanel, this.canvas);
    }
    
    /**
     * Set up event handlers for controls and interactions
     */
    setupEventHandlers() {
        // Task head filter checkboxes
        const checkboxes = this.controlPanel.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const taskHead = e.target.value;
                if (e.target.checked) {
                    this.activeFilters.add(taskHead);
                } else {
                    this.activeFilters.delete(taskHead);
                }
                this.updateChart();
            });
        });
        
        // Control buttons
        const resetBtn = this.controlPanel.querySelector('#resetZoom');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetZoom());
        }
        
        const peaksBtn = this.controlPanel.querySelector('#highlightPeaks');
        if (peaksBtn) {
            peaksBtn.addEventListener('click', () => this.togglePeakHighlights());
        }
        
        const exportBtn = this.controlPanel.querySelector('#exportData');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportToCSV());
        }
        
        const imageBtn = this.controlPanel.querySelector('#exportImage');
        if (imageBtn) {
            imageBtn.addEventListener('click', () => this.exportToPNG());
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'r':
                        e.preventDefault();
                        this.resetZoom();
                        break;
                    case 'p':
                        e.preventDefault();
                        this.togglePeakHighlights();
                        break;
                    case 's':
                        e.preventDefault();
                        this.exportToCSV();
                        break;
                }
            }
        });
    }
    
    /**
     * Process raw data and prepare for Chart.js
     */
    processData() {
        if (!this.rawData || !this.rawData.events) {
            console.warn('No memory timeline data provided');
            return this.getEmptyChartData();
        }
        
        const events = this.rawData.events;
        const thresholds = this.memoryThresholds[this.stage];
        
        // Main memory timeline dataset
        const timelineData = {
            labels: events.map(event => `${event.timestamp.toFixed(1)}ms`),
            datasets: []
        };
        
        // Primary memory usage line
        timelineData.datasets.push({
            label: 'Memory Usage (MB)',
            data: events.map(event => event.cumulative_memory),
            borderColor: this.options.colors.normal,
            backgroundColor: this.options.colors.normal + '20', // 20% opacity
            fill: true,
            tension: 0.1,
            pointRadius: 2,
            pointHoverRadius: 6,
            borderWidth: 2
        });
        
        // Memory delta (allocation/deallocation) bars
        if (this.options.enableAllocationBars) {
            const allocationData = events.map(event => event.memory_delta > 0 ? event.memory_delta : null);
            const deallocationData = events.map(event => event.memory_delta < 0 ? Math.abs(event.memory_delta) : null);
            
            timelineData.datasets.push({
                label: 'Allocations',
                type: 'bar',
                data: allocationData,
                backgroundColor: this.options.colors.allocation + '60',
                borderColor: this.options.colors.allocation,
                yAxisID: 'delta',
                order: 2
            });
            
            timelineData.datasets.push({
                label: 'Deallocations',
                type: 'bar', 
                data: deallocationData,
                backgroundColor: this.options.colors.deallocation + '60',
                borderColor: this.options.colors.deallocation,
                yAxisID: 'delta',
                order: 2
            });
        }
        
        // Warning threshold line
        if (this.options.enableThresholds && thresholds.warning) {
            timelineData.datasets.push({
                label: 'Warning Threshold',
                data: new Array(events.length).fill(thresholds.warning),
                borderColor: this.options.colors.warning,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0,
                borderWidth: 2
            });
        }
        
        // Critical threshold line
        if (this.options.enableThresholds && thresholds.critical) {
            timelineData.datasets.push({
                label: 'Critical Threshold',
                data: new Array(events.length).fill(thresholds.critical),
                borderColor: this.options.colors.critical,
                borderDash: [10, 5],
                fill: false,
                pointRadius: 0,
                borderWidth: 2
            });
        }
        
        // Task head breakdown
        if (this.options.enableTaskBreakdown && this.rawData.task_breakdown) {
            this.addTaskBreakdownDatasets(timelineData, events);
        }
        
        // Highlight problematic operations
        if (this.options.highlightProblematic) {
            this.addProblematicOperationsDataset(timelineData, events, thresholds);
        }
        
        return timelineData;
    }
    
    /**
     * Add task head breakdown datasets
     */
    addTaskBreakdownDatasets(timelineData, events) {
        const taskData = {};
        
        // Initialize task data arrays
        this.taskHeads.forEach(task => {
            taskData[task] = new Array(events.length).fill(null);
        });
        
        // Populate task data from events
        events.forEach((event, index) => {
            if (event.task_head && this.activeFilters.has(event.task_head)) {
                taskData[event.task_head][index] = event.cumulative_memory;
            }
        });
        
        // Add datasets for each active task head
        this.taskHeads.forEach(task => {
            if (this.activeFilters.has(task)) {
                const hasData = taskData[task].some(value => value !== null);
                if (hasData) {
                    timelineData.datasets.push({
                        label: `Task: ${task}`,
                        data: taskData[task],
                        borderColor: this.taskColors[task],
                        backgroundColor: this.taskColors[task] + '15',
                        fill: false,
                        tension: 0.1,
                        pointRadius: 1,
                        borderWidth: 1,
                        hidden: false // Show by default, user can toggle
                    });
                }
            }
        });
    }
    
    /**
     * Add dataset highlighting problematic operations
     */
    addProblematicOperationsDataset(timelineData, events, thresholds) {
        const problematicData = events.map(event => {
            const isProblematic = event.cumulative_memory > thresholds.critical ||
                                  Math.abs(event.memory_delta) > 5000 || // >5GB single operation
                                  event.exceeds_threshold;
            
            return isProblematic ? event.cumulative_memory : null;
        });
        
        timelineData.datasets.push({
            label: 'Problematic Operations',
            data: problematicData,
            backgroundColor: this.options.colors.critical,
            borderColor: this.options.colors.critical,
            pointRadius: 8,
            pointHoverRadius: 12,
            showLine: false,
            order: 1 // Render on top
        });
    }
    
    /**
     * Get Chart.js configuration options
     */
    getChartOptions() {
        const thresholds = this.memoryThresholds[this.stage];
        
        return {
            responsive: this.options.responsive,
            maintainAspectRatio: this.options.maintainAspectRatio,
            animation: {
                duration: this.options.animationDuration
            },
            plugins: {
                title: {
                    display: true,
                    text: `UniAD Memory Timeline - Stage ${this.stage}`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        title: (tooltipItems) => {
                            const index = tooltipItems[0].dataIndex;
                            const event = this.rawData.events[index];
                            return `${event.operation} (${event.timestamp.toFixed(1)}ms)`;
                        },
                        afterTitle: (tooltipItems) => {
                            const index = tooltipItems[0].dataIndex;
                            const event = this.rawData.events[index];
                            return `Module: ${event.module_path}`;
                        },
                        label: (context) => {
                            const value = context.parsed.y;
                            if (value === null) return null;
                            
                            if (context.dataset.label.includes('Task:')) {
                                return `${context.dataset.label}: ${value.toFixed(1)} MB`;
                            }
                            if (context.dataset.label === 'Memory Usage (MB)') {
                                const index = context.dataIndex;
                                const event = this.rawData.events[index];
                                const delta = event.memory_delta > 0 ? `+${event.memory_delta.toFixed(1)}` : event.memory_delta.toFixed(1);
                                return `Memory: ${value.toFixed(1)} MB (Δ${delta} MB)`;
                            }
                            
                            return `${context.dataset.label}: ${value.toFixed(1)} MB`;
                        },
                        afterBody: (tooltipItems) => {
                            const index = tooltipItems[0].dataIndex;
                            const event = this.rawData.events[index];
                            
                            const lines = [];
                            if (event.task_head) {
                                lines.push(`Task Head: ${event.task_head}`);
                            }
                            if (event.is_bev_operation) {
                                lines.push('BEV Operation: Yes');
                            }
                            if (event.exceeds_threshold) {
                                lines.push('⚠️ Exceeds Threshold');
                            }
                            if (event.memory_delta && Math.abs(event.memory_delta) > 2000) {
                                lines.push(`⚡ Large Memory Change: ${event.memory_delta.toFixed(1)} MB`);
                            }
                            
                            return lines;
                        }
                    }
                },
                zoom: this.options.enableZoom ? {
                    zoom: {
                        wheel: {
                            enabled: true,
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'x',
                    },
                    pan: this.options.enablePan ? {
                        enabled: true,
                        mode: 'x',
                    } : {}
                } : {}
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time (ms)'
                    },
                    ticks: {
                        maxTicksLimit: 20
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Memory Usage (MB)'
                    },
                    beginAtZero: true,
                    suggestedMax: Math.max(thresholds.critical * 1.1, 
                                         this.rawData.events ? Math.max(...this.rawData.events.map(e => e.cumulative_memory)) * 1.1 : 30000)
                },
                delta: {
                    type: 'linear',
                    display: false,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false,
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            onHover: (event, activeElements) => {
                this.canvas.style.cursor = activeElements.length > 0 ? 'pointer' : 'default';
            },
            onClick: (event, activeElements) => {
                if (activeElements.length > 0) {
                    const element = activeElements[0];
                    const dataIndex = element.index;
                    this.handlePointClick(dataIndex);
                }
            }
        };
    }
    
    /**
     * Handle click on data points
     */
    handlePointClick(dataIndex) {
        if (!this.rawData.events[dataIndex]) return;
        
        const event = this.rawData.events[dataIndex];
        const thresholds = this.memoryThresholds[this.stage];
        
        // Show detailed information modal/panel
        this.showDetailedEventInfo(event, dataIndex, thresholds);
    }
    
    /**
     * Show detailed information about a memory event
     */
    showDetailedEventInfo(event, index, thresholds) {
        // Create or update detail panel
        let detailPanel = document.getElementById('memoryEventDetail');
        
        if (!detailPanel) {
            detailPanel = document.createElement('div');
            detailPanel.id = 'memoryEventDetail';
            detailPanel.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                border: 2px solid #007bff;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                z-index: 2000;
                max-width: 500px;
                max-height: 70vh;
                overflow-y: auto;
            `;
            document.body.appendChild(detailPanel);
        }
        
        const isProblematic = event.cumulative_memory > thresholds.critical ||
                              Math.abs(event.memory_delta) > 5000 ||
                              event.exceeds_threshold;
        
        detailPanel.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="margin: 0; color: ${isProblematic ? '#dc3545' : '#28a745'};">
                    ${isProblematic ? '🔴' : '🟢'} Memory Event #${index + 1}
                </h3>
                <button id="closeDetail" style="background: #6c757d; color: white; border: none; border-radius: 4px; padding: 5px 10px; cursor: pointer;">×</button>
            </div>
            
            <div class="event-details">
                <p><strong>Operation:</strong> ${event.operation}</p>
                <p><strong>Module:</strong> ${event.module_path}</p>
                <p><strong>Time:</strong> ${event.timestamp.toFixed(1)}ms</p>
                <p><strong>Memory Usage:</strong> ${event.cumulative_memory.toFixed(1)} MB</p>
                <p><strong>Memory Delta:</strong> ${event.memory_delta > 0 ? '+' : ''}${event.memory_delta.toFixed(1)} MB</p>
                
                ${event.task_head ? `<p><strong>Task Head:</strong> <span style="color: ${this.taskColors[event.task_head]};">●</span> ${event.task_head}</p>` : ''}
                ${event.temporal_index !== null ? `<p><strong>Temporal Index:</strong> ${event.temporal_index}</p>` : ''}
                ${event.is_bev_operation ? '<p><strong>BEV Operation:</strong> Yes</p>' : ''}
                ${event.is_frozen_module ? '<p><strong>Frozen Module:</strong> Yes</p>' : ''}
                
                <hr style="margin: 15px 0;">
                
                <h4>Analysis</h4>
                <p><strong>Event Type:</strong> ${event.event_type}</p>
                <p><strong>Exceeds Threshold:</strong> ${event.exceeds_threshold ? 'Yes ⚠️' : 'No'}</p>
                <p><strong>Problematic:</strong> ${isProblematic ? 'Yes 🔴' : 'No 🟢'}</p>
                
                ${event.compute_time ? `<p><strong>Compute Time:</strong> ${event.compute_time.toFixed(2)}ms</p>` : ''}
                ${event.gpu_utilization ? `<p><strong>GPU Utilization:</strong> ${event.gpu_utilization.toFixed(1)}%</p>` : ''}
                
                ${event.tensor_info ? `
                    <hr style="margin: 15px 0;">
                    <h4>Tensor Information</h4>
                    <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto;">${JSON.stringify(event.tensor_info, null, 2)}</pre>
                ` : ''}
            </div>
        `;
        
        // Add close handler
        detailPanel.querySelector('#closeDetail').addEventListener('click', () => {
            document.body.removeChild(detailPanel);
        });
        
        // Close on escape key
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                document.body.removeChild(detailPanel);
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
    }
    
    /**
     * Update memory statistics in the control panel
     */
    updateMemoryStats() {
        const statsElement = this.controlPanel.querySelector('#memoryStats');
        if (!statsElement || !this.rawData.events) return;
        
        const events = this.rawData.events;
        const peakMemory = Math.max(...events.map(e => e.cumulative_memory));
        const avgMemory = events.reduce((sum, e) => sum + e.cumulative_memory, 0) / events.length;
        const problematicCount = events.filter(e => e.exceeds_threshold).length;
        
        statsElement.innerHTML = `
            Peak: ${peakMemory.toFixed(1)}MB | 
            Avg: ${avgMemory.toFixed(1)}MB | 
            Issues: ${problematicCount}
        `;
    }
    
    /**
     * Reset zoom to show full timeline
     */
    resetZoom() {
        if (this.chart && this.chart.resetZoom) {
            this.chart.resetZoom();
        }
    }
    
    /**
     * Toggle peak highlighting
     */
    togglePeakHighlights() {
        // Implementation for highlighting memory peaks
        if (!this.rawData.peaks) return;
        
        const peakIndices = this.rawData.peaks.map(peak => {
            return this.rawData.events.findIndex(event => 
                Math.abs(event.timestamp - peak.timestamp) < 0.1
            );
        }).filter(index => index >= 0);
        
        // Add peak markers to chart
        this.highlightedPeaks.clear();
        peakIndices.forEach(index => this.highlightedPeaks.add(index));
        
        console.log(`Highlighted ${peakIndices.length} memory peaks`);
        this.updateChart();
    }
    
    /**
     * Export timeline data to CSV
     */
    exportToCSV() {
        if (!this.rawData.events) return;
        
        const headers = [
            'timestamp_ms', 'operation', 'module_path', 'memory_delta_mb', 
            'cumulative_memory_mb', 'task_head', 'event_type', 'exceeds_threshold',
            'is_bev_operation', 'compute_time_ms'
        ];
        
        const rows = this.rawData.events.map(event => [
            event.timestamp,
            event.operation,
            event.module_path,
            event.memory_delta,
            event.cumulative_memory,
            event.task_head || '',
            event.event_type,
            event.exceeds_threshold,
            event.is_bev_operation,
            event.compute_time || ''
        ]);
        
        const csvContent = [headers, ...rows]
            .map(row => row.map(field => `"${field}"`).join(','))
            .join('\n');
        
        this.downloadFile(csvContent, 'memory_timeline.csv', 'text/csv');
    }
    
    /**
     * Export chart as PNG image
     */
    exportToPNG() {
        if (!this.chart) return;
        
        const url = this.chart.toBase64Image('image/png', 1.0);
        const link = document.createElement('a');
        link.href = url;
        link.download = `memory_timeline_stage${this.stage}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    /**
     * Helper method to download files
     */
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
    
    /**
     * Get empty chart data structure
     */
    getEmptyChartData() {
        return {
            labels: [],
            datasets: [{
                label: 'No Data Available',
                data: [],
                borderColor: '#cccccc',
                backgroundColor: '#f8f9fa'
            }]
        };
    }
    
    /**
     * Update the chart with current data and filters
     */
    updateChart() {
        if (!this.chart) return;
        
        this.chart.data = this.processData();
        this.chart.update('none'); // No animation for updates
        this.updateMemoryStats();
    }
    
    /**
     * Render the memory timeline chart
     */
    render() {
        const chartData = this.processData();
        const chartOptions = this.getChartOptions();
        
        // Destroy existing chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        // Create new Chart.js instance
        this.chart = new Chart(this.ctx, {
            type: 'line',
            data: chartData,
            options: chartOptions
        });
        
        this.updateMemoryStats();
        console.log('Memory timeline chart rendered successfully');
    }
    
    /**
     * Update chart data (for real-time updates)
     */
    updateData(newData) {
        this.rawData = newData;
        this.render();
    }
    
    /**
     * Destroy the chart and clean up
     */
    destroy() {
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
        
        if (this.tooltip && this.tooltip.parentNode) {
            this.tooltip.parentNode.removeChild(this.tooltip);
        }
        
        if (this.controlPanel && this.controlPanel.parentNode) {
            this.controlPanel.parentNode.removeChild(this.controlPanel);
        }
        
        console.log('MemoryTimelineChart destroyed');
    }
}

// Global functions for template integration
window.MemoryTimelineChart = MemoryTimelineChart;

/**
 * Initialize memory timeline chart from canvas element
 * @param {HTMLCanvasElement} canvas - Canvas element for the chart
 * @param {Object} data - Memory timeline data from MemoryTimelineVisualizer
 * @param {Object} options - Chart configuration options
 * @returns {MemoryTimelineChart} Initialized chart instance
 */
function initializeMemoryTimeline(canvas, data, options = {}) {
    const timeline = new MemoryTimelineChart(canvas, data, options);
    timeline.render();
    return timeline;
}

/**
 * Create memory timeline from JSON data
 * @param {string} canvasId - Canvas element ID
 * @param {Object|string} jsonData - Data object or JSON string
 * @param {Object} options - Chart options
 * @returns {MemoryTimelineChart} Initialized chart instance
 */
function createMemoryTimelineFromJSON(canvasId, jsonData, options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas element with ID '${canvasId}' not found`);
        return null;
    }
    
    const data = typeof jsonData === 'string' ? JSON.parse(jsonData) : jsonData;
    return initializeMemoryTimeline(canvas, data, options);
}

// Export for CommonJS environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        MemoryTimelineChart,
        initializeMemoryTimeline,
        createMemoryTimelineFromJSON
    };
}