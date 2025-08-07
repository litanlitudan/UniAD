/**
 * Interactive Connection Highlighting and Zoom/Pan for UniAD Dataflow Visualization
 * 
 * This module provides connection highlighting functionality and zoom/pan navigation
 * for the dataflow visualization, allowing users to explore large graphs interactively.
 * 
 * Features:
 * - Connection click event handling
 * - Data path highlighting with animation
 * - Transformation detail display
 * - Multiple path highlighting support
 * - Path traversal animation
 * - Shape/memory/dtype transformation visualization
 * - Clear highlight functionality (ESC key or second click)
 * - Zoom and pan controls for graph navigation
 * - Smooth zoom transitions with performance optimization
 * - Zoom boundaries and reset functionality
 * - Integration with existing D3.js force-directed graphs
 */

/**
 * ZoomPanController - Handles zoom and pan functionality for graph visualization
 * 
 * Features:
 * - D3.js zoom behavior integration
 * - Smooth zoom transitions with easing
 * - Zoom boundaries and constraints
 * - Pan with mouse drag and keyboard
 * - Zoom controls (buttons, wheel, programmatic)
 * - Reset zoom functionality
 * - Performance optimization for large graphs
 * - Integration with ConnectionHighlighter
 */
class ZoomPanController {
    constructor(svg, graphGroup, config) {
        this.svg = svg;
        this.graphGroup = graphGroup; // The group element containing nodes and edges
        this.config = config || {};
        
        // Zoom configuration
        this.minZoom = this.config.minZoom || 0.1;
        this.maxZoom = this.config.maxZoom || 8.0;
        this.zoomSpeed = this.config.zoomSpeed || 1.5;
        this.transitionDuration = this.config.transitionDuration || 500;
        
        // State tracking
        this.currentTransform = d3.zoomIdentity;
        this.isZooming = false;
        this.isPanning = false;
        
        // Performance settings
        this.throttleDelay = this.config.throttleDelay || 16; // ~60 FPS
        this.lastThrottleTime = 0;
        
        // Zoom behavior instance
        this.zoomBehavior = null;
        
        this.init();
    }
    
    init() {
        this.setupZoomBehavior();
        this.setupKeyboardControls();
        this.createZoomControls();
        this.setupPerformanceOptimization();
        
        console.log('ZoomPanController initialized with bounds:', this.minZoom, 'to', this.maxZoom);
    }
    
    /**
     * Set up D3.js zoom behavior
     */
    setupZoomBehavior() {
        this.zoomBehavior = d3.zoom()
            .scaleExtent([this.minZoom, this.maxZoom])
            .filter((event) => {
                // Allow zoom with wheel, disable on right click
                if (event.type === 'wheel') {
                    return !event.ctrlKey; // Prevent browser zoom when Ctrl is held
                }
                // Allow pan with left mouse button
                if (event.type === 'mousedown') {
                    return event.button === 0; // Left button only
                }
                return true;
            })
            .on('start', (event) => this.handleZoomStart(event))
            .on('zoom', (event) => this.handleZoom(event))
            .on('end', (event) => this.handleZoomEnd(event));
        
        // Apply zoom behavior to SVG
        this.svg.call(this.zoomBehavior);
        
        // Prevent browser context menu on right click
        this.svg.on('contextmenu', (event) => {
            if (!event.defaultPrevented) {
                event.preventDefault();
            }
        });
    }
    
    /**
     * Handle zoom start event
     */
    handleZoomStart(event) {
        this.isZooming = event.sourceEvent && event.sourceEvent.type === 'wheel';
        this.isPanning = event.sourceEvent && event.sourceEvent.type === 'mousedown';
        
        // Add visual feedback
        if (this.isPanning) {
            this.svg.style('cursor', 'grabbing');
        }
        
        // Disable node interactions during zoom/pan for performance
        this.setInteractionsEnabled(false);
    }
    
    /**
     * Handle zoom event with throttling for performance
     */
    handleZoom(event) {
        const now = Date.now();
        if (now - this.lastThrottleTime < this.throttleDelay) {
            return; // Throttle for performance
        }
        this.lastThrottleTime = now;
        
        this.currentTransform = event.transform;
        
        // Apply transform to graph group
        this.graphGroup.attr('transform', this.currentTransform);
        
        // Update zoom level display
        this.updateZoomLevelDisplay(this.currentTransform.k);
        
        // Adjust element visibility based on zoom level for performance
        this.adjustElementVisibility(this.currentTransform.k);
    }
    
    /**
     * Handle zoom end event
     */
    handleZoomEnd(event) {
        this.isZooming = false;
        this.isPanning = false;
        
        // Restore cursor
        this.svg.style('cursor', 'default');
        
        // Re-enable interactions
        this.setInteractionsEnabled(true);
        
        // Final optimization pass
        this.optimizeForZoomLevel(this.currentTransform.k);
    }
    
    /**
     * Set up keyboard controls for zoom and pan
     */
    setupKeyboardControls() {
        document.addEventListener('keydown', (event) => {
            // Only handle if no input is focused
            if (document.activeElement.tagName === 'INPUT' || 
                document.activeElement.tagName === 'TEXTAREA') {
                return;
            }
            
            const step = 50; // Pan step size
            const zoomStep = 1.2; // Zoom step multiplier
            
            switch (event.key) {
                case 'ArrowUp':
                    event.preventDefault();
                    this.panBy(0, -step);
                    break;
                case 'ArrowDown':
                    event.preventDefault();
                    this.panBy(0, step);
                    break;
                case 'ArrowLeft':
                    event.preventDefault();
                    this.panBy(-step, 0);
                    break;
                case 'ArrowRight':
                    event.preventDefault();
                    this.panBy(step, 0);
                    break;
                case '+':
                case '=':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        this.zoomBy(zoomStep);
                    }
                    break;
                case '-':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        this.zoomBy(1 / zoomStep);
                    }
                    break;
                case '0':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        this.resetZoom();
                    }
                    break;
                case 'f':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        this.fitToContent();
                    }
                    break;
            }
        });
    }
    
    /**
     * Create zoom control UI elements
     */
    createZoomControls() {
        // Remove existing controls
        d3.select('.zoom-controls').remove();
        
        const controls = d3.select('body')
            .append('div')
            .attr('class', 'zoom-controls')
            .style('position', 'fixed')
            .style('top', '20px')
            .style('left', '20px')
            .style('z-index', '1000')
            .style('background', 'rgba(255, 255, 255, 0.9)')
            .style('border-radius', '8px')
            .style('box-shadow', '0 2px 8px rgba(0,0,0,0.15)')
            .style('padding', '10px')
            .style('display', 'flex')
            .style('flex-direction', 'column')
            .style('gap', '5px')
            .style('min-width', '120px');
        
        // Zoom level display
        const zoomDisplay = controls.append('div')
            .attr('class', 'zoom-level-display')
            .style('text-align', 'center')
            .style('font-size', '12px')
            .style('color', '#666')
            .style('margin-bottom', '5px')
            .text('100%');
        
        // Zoom in button
        controls.append('button')
            .attr('class', 'zoom-control zoom-in')
            .style('background', '#007bff')
            .style('color', 'white')
            .style('border', 'none')
            .style('padding', '8px 12px')
            .style('border-radius', '4px')
            .style('cursor', 'pointer')
            .style('font-size', '14px')
            .text('Zoom In (+)')
            .on('click', () => this.zoomBy(this.zoomSpeed))
            .on('mouseover', function() {
                d3.select(this).style('background', '#0056b3');
            })
            .on('mouseout', function() {
                d3.select(this).style('background', '#007bff');
            });
        
        // Zoom out button
        controls.append('button')
            .attr('class', 'zoom-control zoom-out')
            .style('background', '#6c757d')
            .style('color', 'white')
            .style('border', 'none')
            .style('padding', '8px 12px')
            .style('border-radius', '4px')
            .style('cursor', 'pointer')
            .style('font-size', '14px')
            .text('Zoom Out (-)')
            .on('click', () => this.zoomBy(1 / this.zoomSpeed))
            .on('mouseover', function() {
                d3.select(this).style('background', '#545b62');
            })
            .on('mouseout', function() {
                d3.select(this).style('background', '#6c757d');
            });
        
        // Reset zoom button
        controls.append('button')
            .attr('class', 'zoom-control zoom-reset')
            .style('background', '#28a745')
            .style('color', 'white')
            .style('border', 'none')
            .style('padding', '8px 12px')
            .style('border-radius', '4px')
            .style('cursor', 'pointer')
            .style('font-size', '14px')
            .text('Reset (0)')
            .on('click', () => this.resetZoom())
            .on('mouseover', function() {
                d3.select(this).style('background', '#1e7e34');
            })
            .on('mouseout', function() {
                d3.select(this).style('background', '#28a745');
            });
        
        // Fit to content button
        controls.append('button')
            .attr('class', 'zoom-control zoom-fit')
            .style('background', '#ffc107')
            .style('color', '#212529')
            .style('border', 'none')
            .style('padding', '8px 12px')
            .style('border-radius', '4px')
            .style('cursor', 'pointer')
            .style('font-size', '14px')
            .text('Fit (F)')
            .on('click', () => this.fitToContent())
            .on('mouseover', function() {
                d3.select(this).style('background', '#e0a800');
            })
            .on('mouseout', function() {
                d3.select(this).style('background', '#ffc107');
            });
        
        // Store reference to zoom display
        this.zoomDisplay = zoomDisplay;
    }
    
    /**
     * Set up performance optimizations
     */
    setupPerformanceOptimization() {
        // Use CSS transforms for better performance
        this.graphGroup.style('transform-origin', '0 0');
        
        // Enable hardware acceleration
        this.svg.style('transform', 'translateZ(0)');
    }
    
    /**
     * Programmatically zoom by a factor
     */
    zoomBy(factor, center) {
        if (!center) {
            // Use center of viewport if no center specified
            const rect = this.svg.node().getBoundingClientRect();
            center = [rect.width / 2, rect.height / 2];
        }
        
        this.svg.transition()
            .duration(this.transitionDuration)
            .ease(d3.easeQuadOut)
            .call(this.zoomBehavior.scaleBy, factor);
    }
    
    /**
     * Programmatically zoom to a specific level
     */
    zoomTo(scale, center) {
        if (!center) {
            const rect = this.svg.node().getBoundingClientRect();
            center = [rect.width / 2, rect.height / 2];
        }
        
        this.svg.transition()
            .duration(this.transitionDuration)
            .ease(d3.easeQuadOut)
            .call(this.zoomBehavior.scaleTo, scale);
    }
    
    /**
     * Pan by specific amounts
     */
    panBy(dx, dy) {
        this.svg.transition()
            .duration(this.transitionDuration / 2)
            .ease(d3.easeQuadOut)
            .call(this.zoomBehavior.translateBy, dx, dy);
    }
    
    /**
     * Pan to specific coordinates
     */
    panTo(x, y) {
        const transform = d3.zoomIdentity
            .translate(-x * this.currentTransform.k, -y * this.currentTransform.k)
            .scale(this.currentTransform.k);
        
        this.svg.transition()
            .duration(this.transitionDuration)
            .ease(d3.easeQuadOut)
            .call(this.zoomBehavior.transform, transform);
    }
    
    /**
     * Reset zoom to initial state
     */
    resetZoom() {
        this.svg.transition()
            .duration(this.transitionDuration)
            .ease(d3.easeQuadOut)
            .call(this.zoomBehavior.transform, d3.zoomIdentity);
    }
    
    /**
     * Fit content to viewport
     */
    fitToContent(padding = 50) {
        const graphBounds = this.getGraphBounds();
        if (!graphBounds) return;
        
        const svgNode = this.svg.node();
        const svgRect = svgNode.getBoundingClientRect();
        const viewportWidth = svgRect.width;
        const viewportHeight = svgRect.height;
        
        const graphWidth = graphBounds.maxX - graphBounds.minX;
        const graphHeight = graphBounds.maxY - graphBounds.minY;
        
        if (graphWidth === 0 || graphHeight === 0) return;
        
        // Calculate scale to fit content with padding
        const scaleX = (viewportWidth - 2 * padding) / graphWidth;
        const scaleY = (viewportHeight - 2 * padding) / graphHeight;
        const scale = Math.min(scaleX, scaleY);
        
        // Constrain scale to zoom limits
        const constrainedScale = Math.min(Math.max(scale, this.minZoom), this.maxZoom);
        
        // Calculate center position
        const centerX = graphBounds.minX + graphWidth / 2;
        const centerY = graphBounds.minY + graphHeight / 2;
        
        // Calculate transform
        const transform = d3.zoomIdentity
            .translate(viewportWidth / 2, viewportHeight / 2)
            .scale(constrainedScale)
            .translate(-centerX, -centerY);
        
        this.svg.transition()
            .duration(this.transitionDuration * 1.5)
            .ease(d3.easeQuadOut)
            .call(this.zoomBehavior.transform, transform);
    }
    
    /**
     * Get bounds of the graph content
     */
    getGraphBounds() {
        try {
            const bbox = this.graphGroup.node().getBBox();
            return {
                minX: bbox.x,
                minY: bbox.y,
                maxX: bbox.x + bbox.width,
                maxY: bbox.y + bbox.height
            };
        } catch (e) {
            // Fallback if getBBox() fails
            console.warn('Could not get graph bounds, using fallback');
            return {
                minX: -500,
                minY: -500,
                maxX: 500,
                maxY: 500
            };
        }
    }
    
    /**
     * Update zoom level display
     */
    updateZoomLevelDisplay(scale) {
        if (this.zoomDisplay) {
            const percentage = Math.round(scale * 100);
            this.zoomDisplay.text(`${percentage}%`);
        }
    }
    
    /**
     * Adjust element visibility based on zoom level for performance
     */
    adjustElementVisibility(zoomLevel) {
        // Hide labels at very low zoom levels for performance
        const showLabels = zoomLevel > 0.3;
        const showDetails = zoomLevel > 0.6;
        
        // Adjust text visibility
        this.graphGroup.selectAll('text')
            .style('display', showLabels ? 'block' : 'none');
        
        // Adjust edge details
        this.graphGroup.selectAll('.edge-label')
            .style('display', showDetails ? 'block' : 'none');
        
        // Simplify node rendering at low zoom
        if (zoomLevel < 0.5) {
            this.graphGroup.selectAll('.node circle')
                .style('stroke-width', '1px');
        } else {
            this.graphGroup.selectAll('.node circle')
                .style('stroke-width', null);
        }
    }
    
    /**
     * Optimize rendering for specific zoom level
     */
    optimizeForZoomLevel(zoomLevel) {
        // Enable/disable CSS transitions based on zoom level
        const enableTransitions = zoomLevel > 0.2;
        
        this.graphGroup.style('transition', enableTransitions ? 'all 0.2s ease' : 'none');
        
        // Update rendering quality based on zoom
        if (zoomLevel < 0.3) {
            // Low quality for performance
            this.svg.style('shape-rendering', 'optimizeSpeed');
        } else {
            // High quality for detail
            this.svg.style('shape-rendering', 'geometricPrecision');
        }
    }
    
    /**
     * Enable/disable interactions during zoom/pan
     */
    setInteractionsEnabled(enabled) {
        const pointerEvents = enabled ? 'all' : 'none';
        
        // Temporarily disable node interactions for performance
        this.graphGroup.selectAll('.node')
            .style('pointer-events', pointerEvents);
        
        this.graphGroup.selectAll('.edge')
            .style('pointer-events', pointerEvents);
            
        // Hide tooltips during zoom/pan for better performance
        if (!enabled && this.tooltipManager) {
            this.tooltipManager.hideTooltip();
        }
    }
    
    /**
     * Focus on a specific node or area
     */
    focusOnNode(nodeId, zoomLevel = 2.0) {
        const node = this.graphGroup.selectAll('.node')
            .filter(d => d.id === nodeId);
        
        if (node.empty()) return;
        
        const nodeData = node.datum();
        if (nodeData && nodeData.x !== undefined && nodeData.y !== undefined) {
            this.zoomToPoint(nodeData.x, nodeData.y, zoomLevel);
        }
    }
    
    /**
     * Zoom to a specific point
     */
    zoomToPoint(x, y, zoomLevel) {
        const svgNode = this.svg.node();
        const svgRect = svgNode.getBoundingClientRect();
        
        const transform = d3.zoomIdentity
            .translate(svgRect.width / 2, svgRect.height / 2)
            .scale(zoomLevel)
            .translate(-x, -y);
        
        this.svg.transition()
            .duration(this.transitionDuration)
            .ease(d3.easeQuadOut)
            .call(this.zoomBehavior.transform, transform);
    }
    
    /**
     * Get current zoom level
     */
    getCurrentZoom() {
        return this.currentTransform.k;
    }
    
    /**
     * Get current pan position
     */
    getCurrentPan() {
        return {
            x: this.currentTransform.x,
            y: this.currentTransform.y
        };
    }
    
    /**
     * Check if currently zooming or panning
     */
    isActive() {
        return this.isZooming || this.isPanning;
    }
    
    /**
     * Cleanup method
     */
    destroy() {
        // Remove zoom behavior
        if (this.zoomBehavior) {
            this.svg.on('.zoom', null);
        }
        
        // Remove controls
        d3.select('.zoom-controls').remove();
        
        console.log('ZoomPanController destroyed');
    }
}

/**
 * TooltipManager - Handles tooltip display for graph nodes and edges
 * 
 * Features:
 * - Hover event handlers for nodes and edges
 * - Intelligent tooltip positioning (avoid viewport edges)
 * - Display tensor shape, dtype, memory usage information
 * - Integration with VisualizationMetadata.tooltip_data
 * - Styled tooltips matching visualization theme
 * - Performance optimized with debouncing
 */
class TooltipManager {
    constructor(svg, nodeElements, edgeElements, data, config) {
        this.svg = svg;
        this.nodeElements = nodeElements;
        this.edgeElements = edgeElements;
        this.data = data;
        this.config = config || {};
        
        // Configuration
        this.showDelay = this.config.showDelay || 300; // ms delay before showing
        this.hideDelay = this.config.hideDelay || 100; // ms delay before hiding
        this.maxWidth = this.config.maxWidth || 350;
        this.offset = this.config.offset || { x: 10, y: -10 };
        
        // State management
        this.currentTooltip = null;
        this.showTimeout = null;
        this.hideTimeout = null;
        this.isVisible = false;
        this.currentTarget = null;
        
        // Performance tracking
        this.lastUpdateTime = 0;
        this.throttleDelay = 16; // ~60 FPS
        
        this.init();
    }
    
    init() {
        this.createTooltipElement();
        this.setupEventHandlers();
        this.addTooltipStyles();
        
        console.log('TooltipManager initialized');
    }
    
    /**
     * Create the main tooltip element
     */
    createTooltipElement() {
        // Remove existing tooltip
        d3.select('.uniad-tooltip').remove();
        
        this.tooltip = d3.select('body')
            .append('div')
            .attr('class', 'uniad-tooltip')
            .style('position', 'absolute')
            .style('background', 'rgba(0, 0, 0, 0.9)')
            .style('color', 'white')
            .style('padding', '12px 16px')
            .style('border-radius', '8px')
            .style('font-size', '12px')
            .style('font-family', 'system-ui, -apple-system, monospace')
            .style('line-height', '1.4')
            .style('pointer-events', 'none')
            .style('opacity', 0)
            .style('z-index', '2000')
            .style('box-shadow', '0 4px 12px rgba(0, 0, 0, 0.3)')
            .style('border', '1px solid rgba(255, 255, 255, 0.2)')
            .style('backdrop-filter', 'blur(8px)')
            .style('max-width', this.maxWidth + 'px')
            .style('word-wrap', 'break-word')
            .style('transition', 'opacity 0.2s ease-in-out, transform 0.1s ease-out')
            .style('transform', 'translateY(5px)');
    }
    
    /**
     * Set up event handlers for tooltips
     */
    setupEventHandlers() {
        // Node tooltip handlers
        if (this.nodeElements) {
            this.nodeElements
                .on('mouseenter.tooltip', (event, d) => {
                    this.scheduleShowTooltip(event, d, 'node');
                })
                .on('mouseleave.tooltip', (event, d) => {
                    this.scheduleHideTooltip();
                })
                .on('mousemove.tooltip', (event, d) => {
                    if (this.isVisible && this.currentTarget === d) {
                        this.updateTooltipPosition(event);
                    }
                });
        }
        
        // Edge tooltip handlers
        if (this.edgeElements) {
            this.edgeElements
                .on('mouseenter.tooltip', (event, d) => {
                    this.scheduleShowTooltip(event, d, 'edge');
                })
                .on('mouseleave.tooltip', (event, d) => {
                    this.scheduleHideTooltip();
                })
                .on('mousemove.tooltip', (event, d) => {
                    if (this.isVisible && this.currentTarget === d) {
                        this.updateTooltipPosition(event);
                    }
                });
        }
        
        // Hide tooltip when scrolling or window events
        window.addEventListener('scroll', () => this.hideTooltip(), { passive: true });
        window.addEventListener('resize', () => this.hideTooltip());
        
        // Hide tooltip when zoom/pan is active
        if (this.svg) {
            this.svg.on('wheel.tooltip', () => this.hideTooltip());
            this.svg.on('mousedown.tooltip', () => this.hideTooltip());
        }
    }
    
    /**
     * Schedule tooltip show with debouncing
     */
    scheduleShowTooltip(event, data, type) {
        // Clear any pending hide
        if (this.hideTimeout) {
            clearTimeout(this.hideTimeout);
            this.hideTimeout = null;
        }
        
        // If already showing for the same target, just update position
        if (this.isVisible && this.currentTarget === data) {
            this.updateTooltipPosition(event);
            return;
        }
        
        // Clear any pending show
        if (this.showTimeout) {
            clearTimeout(this.showTimeout);
        }
        
        // Schedule show
        this.showTimeout = setTimeout(() => {
            this.showTooltip(event, data, type);
            this.showTimeout = null;
        }, this.showDelay);
        
        // Store current target for comparison
        this.currentTarget = data;
    }
    
    /**
     * Schedule tooltip hide with debouncing
     */
    scheduleHideTooltip() {
        // Clear any pending show
        if (this.showTimeout) {
            clearTimeout(this.showTimeout);
            this.showTimeout = null;
        }
        
        // Schedule hide
        if (this.hideTimeout) {
            clearTimeout(this.hideTimeout);
        }
        
        this.hideTimeout = setTimeout(() => {
            this.hideTooltip();
            this.hideTimeout = null;
        }, this.hideDelay);
    }
    
    /**
     * Show tooltip with content
     */
    showTooltip(event, data, type) {
        if (!this.tooltip) return;
        
        // Generate tooltip content
        const content = this.generateTooltipContent(data, type);
        if (!content) return;
        
        // Update tooltip content
        this.tooltip.html(content);
        
        // Position and show tooltip
        this.positionTooltip(event);
        
        this.tooltip
            .style('opacity', 1)
            .style('transform', 'translateY(0px)');
        
        this.isVisible = true;
        this.currentTarget = data;
    }
    
    /**
     * Hide tooltip
     */
    hideTooltip() {
        if (!this.tooltip || !this.isVisible) return;
        
        this.tooltip
            .style('opacity', 0)
            .style('transform', 'translateY(5px)');
        
        this.isVisible = false;
        this.currentTarget = null;
        
        // Clear timeouts
        if (this.showTimeout) {
            clearTimeout(this.showTimeout);
            this.showTimeout = null;
        }
        if (this.hideTimeout) {
            clearTimeout(this.hideTimeout);
            this.hideTimeout = null;
        }
    }
    
    /**
     * Generate tooltip content based on data type
     */
    generateTooltipContent(data, type) {
        if (type === 'node') {
            return this.generateNodeTooltipContent(data);
        } else if (type === 'edge') {
            return this.generateEdgeTooltipContent(data);
        }
        return null;
    }
    
    /**
     * Generate node tooltip content
     */
    generateNodeTooltipContent(nodeData) {
        // Check if there's tooltip_data from VisualizationMetadata
        if (nodeData.tooltip_data) {
            return this.formatTooltipFromMetadata(nodeData.tooltip_data);
        }
        
        // Fallback to generating from node data
        let content = `<div class="tooltip-title">${this.escapeHtml(nodeData.name || nodeData.id)}</div>`;
        
        // Module path
        if (nodeData.module_path) {
            content += `<div class="tooltip-section">
                <span class="tooltip-label">Module:</span>
                <span class="tooltip-value">${this.escapeHtml(nodeData.module_path)}</span>
            </div>`;
        }
        
        // Operation name
        if (nodeData.op_name) {
            content += `<div class="tooltip-section">
                <span class="tooltip-label">Operation:</span>
                <span class="tooltip-value">${this.escapeHtml(nodeData.op_name)}</span>
            </div>`;
        }
        
        // Tensor shape
        if (nodeData.shape_info) {
            const shapes = this.extractShapeInfo(nodeData.shape_info);
            if (shapes.input || shapes.output) {
                content += `<div class="tooltip-section">
                    <span class="tooltip-label">Shape:</span>
                    <div class="tooltip-shape-info">`;
                
                if (shapes.input) {
                    content += `<div class="shape-item">
                        <span class="shape-label">In:</span>
                        <span class="shape-value">${this.formatShape(shapes.input)}</span>
                    </div>`;
                }
                
                if (shapes.output) {
                    content += `<div class="shape-item">
                        <span class="shape-label">Out:</span>
                        <span class="shape-value">${this.formatShape(shapes.output)}</span>
                    </div>`;
                }
                
                content += `</div></div>`;
            }
        }
        
        // Data type
        if (nodeData.dtype_info) {
            const dtypes = this.extractDtypeInfo(nodeData.dtype_info);
            if (dtypes.input || dtypes.output) {
                content += `<div class="tooltip-section">
                    <span class="tooltip-label">Data Type:</span>
                    <div class="tooltip-dtype-info">`;
                
                if (dtypes.input) {
                    content += `<div class="dtype-item">
                        <span class="dtype-label">In:</span>
                        <span class="dtype-value">${this.formatDtype(dtypes.input)}</span>
                    </div>`;
                }
                
                if (dtypes.output) {
                    content += `<div class="dtype-item">
                        <span class="dtype-label">Out:</span>
                        <span class="dtype-value">${this.formatDtype(dtypes.output)}</span>
                    </div>`;
                }
                
                content += `</div></div>`;
            }
        }
        
        // Memory usage
        if (nodeData.memory_mb !== undefined && nodeData.memory_mb > 0) {
            content += `<div class="tooltip-section">
                <span class="tooltip-label">Memory:</span>
                <span class="tooltip-value tooltip-memory">${this.formatMemory(nodeData.memory_mb)}</span>
            </div>`;
        }
        
        // Additional metadata
        if (nodeData.task_head) {
            content += `<div class="tooltip-section">
                <span class="tooltip-label">Task Head:</span>
                <span class="tooltip-value">${this.escapeHtml(nodeData.task_head)}</span>
            </div>`;
        }
        
        return content;
    }
    
    /**
     * Generate edge tooltip content
     */
    generateEdgeTooltipContent(edgeData) {
        const sourceNode = this.getNodeData(edgeData.source);
        const targetNode = this.getNodeData(edgeData.target);
        
        let content = `<div class="tooltip-title">Data Connection</div>`;
        
        // Connection info
        content += `<div class="tooltip-section">
            <span class="tooltip-label">From:</span>
            <span class="tooltip-value">${this.escapeHtml(this.getNodeName(sourceNode))}</span>
        </div>`;
        
        content += `<div class="tooltip-section">
            <span class="tooltip-label">To:</span>
            <span class="tooltip-value">${this.escapeHtml(this.getNodeName(targetNode))}</span>
        </div>`;
        
        // Shape transformation
        if (sourceNode && targetNode) {
            const shapeChange = this.getShapeTransformation(sourceNode, targetNode);
            if (shapeChange) {
                content += `<div class="tooltip-section">
                    <span class="tooltip-label">Shape Change:</span>
                    <div class="tooltip-transformation">
                        <span class="transform-from">${this.formatShape(shapeChange.from)}</span>
                        <span class="transform-arrow">→</span>
                        <span class="transform-to">${this.formatShape(shapeChange.to)}</span>
                    </div>
                </div>`;
            }
            
            // Memory change
            const memoryChange = this.getMemoryChange(sourceNode, targetNode);
            if (memoryChange !== null) {
                const changeClass = memoryChange > 0 ? 'memory-increase' : 'memory-decrease';
                const changeSymbol = memoryChange > 0 ? '+' : '';
                
                content += `<div class="tooltip-section">
                    <span class="tooltip-label">Memory Δ:</span>
                    <span class="tooltip-value ${changeClass}">
                        ${changeSymbol}${this.formatMemory(Math.abs(memoryChange))}
                    </span>
                </div>`;
            }
        }
        
        content += `<div class="tooltip-hint">Click to highlight data path</div>`;
        
        return content;
    }
    
    /**
     * Format tooltip content from VisualizationMetadata
     */
    formatTooltipFromMetadata(tooltipData) {
        let content = '';
        
        // Title
        if (tooltipData.title) {
            content += `<div class="tooltip-title">${this.escapeHtml(tooltipData.title)}</div>`;
        }
        
        // Sections
        if (tooltipData.sections) {
            tooltipData.sections.forEach(section => {
                content += `<div class="tooltip-section">`;
                
                if (section.label) {
                    content += `<span class="tooltip-label">${this.escapeHtml(section.label)}:</span>`;
                }
                
                if (section.value) {
                    content += `<span class="tooltip-value">${this.escapeHtml(section.value)}</span>`;
                }
                
                content += `</div>`;
            });
        }
        
        // Raw content
        if (tooltipData.content && !tooltipData.sections) {
            content += this.escapeHtml(tooltipData.content);
        }
        
        return content;
    }
    
    /**
     * Position tooltip intelligently to avoid viewport edges
     */
    positionTooltip(event) {
        if (!this.tooltip) return;
        
        const tooltipNode = this.tooltip.node();
        const rect = tooltipNode.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        let left = event.pageX + this.offset.x;
        let top = event.pageY + this.offset.y;
        
        // Adjust horizontal position if tooltip would overflow
        if (left + rect.width > viewportWidth - 20) {
            left = event.pageX - rect.width - Math.abs(this.offset.x);
        }
        
        // Adjust vertical position if tooltip would overflow
        if (top + rect.height > viewportHeight - 20) {
            top = event.pageY - rect.height - Math.abs(this.offset.y);
        }
        
        // Ensure tooltip doesn't go off the left edge
        if (left < 20) {
            left = 20;
        }
        
        // Ensure tooltip doesn't go off the top edge
        if (top < 20) {
            top = event.pageY + Math.abs(this.offset.y) + 10;
        }
        
        this.tooltip
            .style('left', left + 'px')
            .style('top', top + 'px');
    }
    
    /**
     * Update tooltip position during mouse move
     */
    updateTooltipPosition(event) {
        const now = Date.now();
        if (now - this.lastUpdateTime < this.throttleDelay) {
            return; // Throttle for performance
        }
        this.lastUpdateTime = now;
        
        this.positionTooltip(event);
    }
    
    /**
     * Add CSS styles for tooltips
     */
    addTooltipStyles() {
        const existingStyle = document.getElementById('uniad-tooltip-styles');
        if (existingStyle) return; // Already added
        
        const style = document.createElement('style');
        style.id = 'uniad-tooltip-styles';
        style.textContent = `
            .uniad-tooltip {
                font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
                font-size: 12px;
                line-height: 1.4;
            }
            
            .tooltip-title {
                font-weight: 600;
                font-size: 13px;
                color: #ffd700;
                margin-bottom: 8px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                padding-bottom: 4px;
            }
            
            .tooltip-section {
                display: flex;
                align-items: flex-start;
                margin-bottom: 6px;
                gap: 8px;
            }
            
            .tooltip-section:last-child {
                margin-bottom: 0;
            }
            
            .tooltip-label {
                color: #adb5bd;
                font-weight: 500;
                min-width: 60px;
                flex-shrink: 0;
            }
            
            .tooltip-value {
                color: #ffffff;
                word-break: break-word;
                flex: 1;
            }
            
            .tooltip-memory {
                color: #17a2b8;
                font-weight: 500;
            }
            
            .tooltip-shape-info,
            .tooltip-dtype-info {
                display: flex;
                flex-direction: column;
                gap: 2px;
                flex: 1;
            }
            
            .shape-item,
            .dtype-item {
                display: flex;
                gap: 6px;
                font-size: 11px;
            }
            
            .shape-label,
            .dtype-label {
                color: #6c757d;
                min-width: 25px;
                font-weight: 500;
            }
            
            .shape-value,
            .dtype-value {
                color: #28a745;
                font-family: monospace;
            }
            
            .tooltip-transformation {
                display: flex;
                align-items: center;
                gap: 6px;
                flex: 1;
                font-family: monospace;
                font-size: 11px;
            }
            
            .transform-from {
                color: #ffc107;
            }
            
            .transform-arrow {
                color: #6c757d;
                font-weight: bold;
            }
            
            .transform-to {
                color: #28a745;
            }
            
            .memory-increase {
                color: #dc3545 !important;
            }
            
            .memory-decrease {
                color: #28a745 !important;
            }
            
            .tooltip-hint {
                margin-top: 8px;
                padding-top: 6px;
                border-top: 1px solid rgba(255, 255, 255, 0.2);
                font-style: italic;
                color: #6c757d;
                font-size: 10px;
                text-align: center;
            }
            
            /* Animation for tooltip appearance */
            .uniad-tooltip {
                animation-duration: 0.2s;
                animation-timing-function: ease-out;
            }
            
            @keyframes tooltipFadeIn {
                from {
                    opacity: 0;
                    transform: translateY(5px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        `;
        
        document.head.appendChild(style);
    }
    
    // Utility methods
    
    /**
     * Get node data by ID or object
     */
    getNodeData(nodeIdOrObj) {
        const nodeId = nodeIdOrObj?.id || nodeIdOrObj;
        return this.data.nodes?.find(n => n.id === nodeId) || nodeIdOrObj;
    }
    
    /**
     * Get node name for display
     */
    getNodeName(node) {
        if (!node) return 'Unknown';
        return node.name || node.id || 'Unnamed';
    }
    
    /**
     * Extract shape information from shape_info
     */
    extractShapeInfo(shapeInfo) {
        const result = {};
        
        if (shapeInfo.input_shape) {
            result.input = shapeInfo.input_shape;
        }
        
        if (shapeInfo.output_shape) {
            result.output = shapeInfo.output_shape;
        }
        
        return result;
    }
    
    /**
     * Extract dtype information from dtype_info
     */
    extractDtypeInfo(dtypeInfo) {
        const result = {};
        
        if (dtypeInfo.input_dtype) {
            result.input = dtypeInfo.input_dtype;
        }
        
        if (dtypeInfo.output_dtype) {
            result.output = dtypeInfo.output_dtype;
        }
        
        return result;
    }
    
    /**
     * Format shape array for display
     */
    formatShape(shape) {
        if (!shape || !Array.isArray(shape)) return 'N/A';
        return `[${shape.join(', ')}]`;
    }
    
    /**
     * Format dtype for display
     */
    formatDtype(dtype) {
        if (!dtype) return 'N/A';
        return String(dtype);
    }
    
    /**
     * Format memory value for display
     */
    formatMemory(memoryMb) {
        if (memoryMb === undefined || memoryMb === null) return 'N/A';
        
        if (memoryMb >= 1024) {
            return `${(memoryMb / 1024).toFixed(1)} GB`;
        } else if (memoryMb >= 1) {
            return `${memoryMb.toFixed(1)} MB`;
        } else {
            return `${(memoryMb * 1024).toFixed(0)} KB`;
        }
    }
    
    /**
     * Get shape transformation between two nodes
     */
    getShapeTransformation(sourceNode, targetNode) {
        if (!sourceNode || !targetNode) return null;
        
        const sourceShape = sourceNode.shape_info?.output_shape;
        const targetShape = targetNode.shape_info?.input_shape;
        
        if (!sourceShape || !targetShape) return null;
        
        // Check if shapes are different
        if (JSON.stringify(sourceShape) !== JSON.stringify(targetShape)) {
            return {
                from: sourceShape,
                to: targetShape
            };
        }
        
        return null;
    }
    
    /**
     * Get memory change between two nodes
     */
    getMemoryChange(sourceNode, targetNode) {
        if (!sourceNode || !targetNode) return null;
        
        const sourceMem = sourceNode.memory_mb;
        const targetMem = targetNode.memory_mb;
        
        if (sourceMem === undefined || targetMem === undefined) return null;
        
        return targetMem - sourceMem;
    }
    
    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        if (typeof text !== 'string') return String(text);
        
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    /**
     * Update tooltip manager when data changes
     */
    updateData(newData) {
        this.data = newData;
        this.hideTooltip(); // Hide any existing tooltip
    }
    
    /**
     * Cleanup method
     */
    destroy() {
        this.hideTooltip();
        
        // Remove event listeners
        if (this.nodeElements) {
            this.nodeElements.on('.tooltip', null);
        }
        if (this.edgeElements) {
            this.edgeElements.on('.tooltip', null);
        }
        
        // Remove tooltip element
        if (this.tooltip) {
            this.tooltip.remove();
        }
        
        // Clear timeouts
        if (this.showTimeout) {
            clearTimeout(this.showTimeout);
        }
        if (this.hideTimeout) {
            clearTimeout(this.hideTimeout);
        }
        
        console.log('TooltipManager destroyed');
    }
}

class ConnectionHighlighter {
    constructor(svg, nodeElements, edgeElements, data, config) {
        this.svg = svg;
        this.nodeElements = nodeElements;
        this.edgeElements = edgeElements;
        this.data = data;
        this.config = config || {};
        
        // State management
        this.highlightedPaths = new Set();
        this.activeConnections = new Set();
        this.animationSpeed = this.config.animationSpeed || 750;
        this.highlightColors = {
            primary: '#ff6b35',
            secondary: '#4ecdc4',
            tertiary: '#45b7d1',
            path: '#feca57',
            node: '#ff9f43'
        };
        
        // Detail panel management
        this.detailPanel = null;
        this.tooltipElement = null;
        
        this.init();
    }
    
    init() {
        this.setupEventHandlers();
        this.createDetailPanel();
        this.setupKeyboardHandlers();
        console.log('ConnectionHighlighter initialized');
    }
    
    /**
     * Set up event handlers for connections and nodes
     */
    setupEventHandlers() {
        // Add click handlers to edges/connections
        if (this.edgeElements) {
            this.edgeElements
                .style('cursor', 'pointer')
                .on('click', (event, d) => {
                    event.stopPropagation();
                    this.handleConnectionClick(event, d);
                })
                .on('mouseover', (event, d) => {
                    this.showConnectionPreview(event, d);
                })
                .on('mouseout', () => {
                    this.hideConnectionPreview();
                });
        }
        
        // Add click handlers to nodes for path exploration
        if (this.nodeElements) {
            this.nodeElements
                .on('click', (event, d) => {
                    if (event.ctrlKey || event.metaKey) {
                        event.stopPropagation();
                        this.handleNodePathClick(event, d);
                    }
                })
                .on('contextmenu', (event, d) => {
                    event.preventDefault();
                    this.showNodeConnectionMenu(event, d);
                });
        }
        
        // Clear highlights on background click
        this.svg.on('click', () => {
            this.clearAllHighlights();
        });
    }
    
    /**
     * Set up keyboard event handlers
     */
    setupKeyboardHandlers() {
        document.addEventListener('keydown', (event) => {
            // Only handle if no input is focused and zoom/pan is not active
            if (document.activeElement.tagName === 'INPUT' || 
                document.activeElement.tagName === 'TEXTAREA') {
                return;
            }
            
            // Avoid conflicts with zoom/pan controller
            if (this.zoomPanController && this.zoomPanController.isActive()) {
                return;
            }
            
            switch (event.key) {
                case 'Escape':
                    this.clearAllHighlights();
                    break;
                case 'h':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        this.toggleHighlightMode();
                    }
                    break;
                case 'a':
                    if (event.ctrlKey || event.metaKey && this.highlightedPaths.size > 0) {
                        event.preventDefault();
                        this.animateActivePaths();
                    }
                    break;
            }
        });
    }
    
    /**
     * Handle connection click events
     */
    handleConnectionClick(event, edgeData) {
        const connectionId = this.getConnectionId(edgeData);
        
        // Hide tooltips during connection interactions
        if (this.tooltipManager) {
            this.tooltipManager.hideTooltip();
        }
        
        if (this.activeConnections.has(connectionId)) {
            // Second click - clear highlight
            this.clearConnectionHighlight(connectionId);
        } else {
            // First click - highlight connection and path
            this.highlightConnection(connectionId, edgeData);
        }
    }
    
    /**
     * Handle node click for path exploration (Ctrl/Cmd + Click)
     */
    handleNodePathClick(event, nodeData) {
        // Hide tooltips during path exploration
        if (this.tooltipManager) {
            this.tooltipManager.hideTooltip();
        }
        
        // Find all paths from/to this node
        const pathsFromNode = this.findPathsFromNode(nodeData.id);
        const pathsToNode = this.findPathsToNode(nodeData.id);
        
        // Highlight all connected paths
        [...pathsFromNode, ...pathsToNode].forEach(path => {
            this.highlightDataPath(path, 'secondary');
        });
        
        // Show node connection summary
        this.showNodeConnectionSummary(event, nodeData, pathsFromNode, pathsToNode);
    }
    
    /**
     * Highlight a connection and its data path
     */
    highlightConnection(connectionId, edgeData) {
        this.activeConnections.add(connectionId);
        
        // Highlight the edge itself
        this.highlightEdge(edgeData, 'primary');
        
        // Find and highlight the complete data path
        const dataPath = this.findDataPath(edgeData.source, edgeData.target);
        if (dataPath) {
            this.highlightDataPath(dataPath, 'primary');
            this.showTransformationDetails(edgeData, dataPath);
        }
        
        // Animate the highlight
        this.animateConnectionHighlight(edgeData);
    }
    
    /**
     * Clear connection highlight
     */
    clearConnectionHighlight(connectionId) {
        this.activeConnections.delete(connectionId);
        
        // Remove edge highlighting
        this.edgeElements
            .filter(d => this.getConnectionId(d) === connectionId)
            .classed('highlighted', false)
            .style('stroke', null)
            .style('stroke-width', null)
            .style('opacity', null);
        
        // Clear associated path highlighting
        this.clearPathHighlights();
        this.hideTransformationDetails();
    }
    
    /**
     * Highlight an individual edge
     */
    highlightEdge(edgeData, colorType = 'primary') {
        const color = this.highlightColors[colorType];
        
        this.edgeElements
            .filter(d => d === edgeData)
            .classed('highlighted', true)
            .style('stroke', color)
            .style('stroke-width', '4px')
            .style('opacity', 1.0)
            .style('filter', 'drop-shadow(0 0 6px rgba(255, 107, 53, 0.8))')
            .style('animation', 'pulse 2s infinite');
    }
    
    /**
     * Highlight a complete data path
     */
    highlightDataPath(pathInfo, colorType = 'primary') {
        if (!pathInfo || !pathInfo.path_nodes) return;
        
        const pathId = pathInfo.path_nodes.join('-');
        this.highlightedPaths.add(pathId);
        
        // Highlight nodes in the path
        this.highlightPathNodes(pathInfo.path_nodes, colorType);
        
        // Highlight edges in the path
        this.highlightPathEdges(pathInfo.path_nodes, colorType);
        
        // Store path info for later use
        pathInfo.colorType = colorType;
        this.storePathInfo(pathId, pathInfo);
    }
    
    /**
     * Highlight nodes in a path
     */
    highlightPathNodes(nodeIds, colorType = 'primary') {
        const color = this.highlightColors[colorType];
        
        this.nodeElements
            .filter(d => nodeIds.includes(d.id))
            .selectAll('circle')
            .classed('path-highlighted', true)
            .style('stroke', color)
            .style('stroke-width', '3px')
            .style('filter', 'drop-shadow(0 0 8px rgba(255, 107, 53, 0.6))');
        
        // Add path index numbers to nodes
        this.addPathIndexLabels(nodeIds, colorType);
    }
    
    /**
     * Highlight edges in a path
     */
    highlightPathEdges(nodeIds, colorType = 'primary') {
        const color = this.highlightColors[colorType];
        
        // Find edges that connect consecutive nodes in the path
        for (let i = 0; i < nodeIds.length - 1; i++) {
            const sourceId = nodeIds[i];
            const targetId = nodeIds[i + 1];
            
            this.edgeElements
                .filter(d => (d.source.id === sourceId && d.target.id === targetId) ||
                            (d.source === sourceId && d.target === targetId))
                .classed('path-highlighted', true)
                .style('stroke', color)
                .style('stroke-width', '3px')
                .style('opacity', 1.0)
                .style('marker-end', `url(#arrowhead-${colorType})`);
        }
    }
    
    /**
     * Add path index labels to nodes
     */
    addPathIndexLabels(nodeIds, colorType) {
        const color = this.highlightColors[colorType];
        
        nodeIds.forEach((nodeId, index) => {
            const nodeElement = this.nodeElements.filter(d => d.id === nodeId);
            
            // Remove existing path labels
            nodeElement.selectAll('.path-index').remove();
            
            // Add new path index label
            nodeElement
                .append('text')
                .attr('class', 'path-index')
                .attr('x', 0)
                .attr('y', -25)
                .attr('text-anchor', 'middle')
                .style('font-size', '10px')
                .style('font-weight', 'bold')
                .style('fill', color)
                .style('background', 'white')
                .style('padding', '2px')
                .style('border-radius', '3px')
                .text(index + 1);
        });
    }
    
    /**
     * Animate connection highlight with traversal effect
     */
    animateConnectionHighlight(edgeData) {
        // Create animated dot that travels along the edge
        const pathElement = this.edgeElements.filter(d => d === edgeData);
        
        // Add dot to graph group if it exists, otherwise to SVG
        const graphGroup = this.svg.select('.graph-group');
        const container = graphGroup.empty() ? this.svg : graphGroup;
        
        // Create traveling dot
        const dot = container.append('circle')
            .attr('class', 'connection-tracer')
            .attr('r', 4)
            .attr('fill', this.highlightColors.primary)
            .style('filter', 'drop-shadow(0 0 6px rgba(255, 107, 53, 0.8))');
        
        // Animate dot along the edge
        dot.transition()
            .duration(this.animationSpeed)
            .ease(d3.easeLinear)
            .attrTween('transform', () => {
                return (t) => {
                    const sourceX = edgeData.source.x;
                    const sourceY = edgeData.source.y;
                    const targetX = edgeData.target.x;
                    const targetY = edgeData.target.y;
                    
                    const x = sourceX + (targetX - sourceX) * t;
                    const y = sourceY + (targetY - sourceY) * t;
                    
                    return `translate(${x}, ${y})`;
                };
            })
            .on('end', () => {
                dot.remove();
            });
    }
    
    /**
     * Animate all active paths
     */
    animateActivePaths() {
        this.highlightedPaths.forEach(pathId => {
            const pathInfo = this.getStoredPathInfo(pathId);
            if (pathInfo && pathInfo.path_nodes) {
                this.animatePathTraversal(pathInfo.path_nodes);
            }
        });
    }
    
    /**
     * Animate path traversal
     */
    animatePathTraversal(nodeIds) {
        const duration = this.animationSpeed * 2;
        const delayBetweenNodes = duration / nodeIds.length;
        
        nodeIds.forEach((nodeId, index) => {
            setTimeout(() => {
                // Pulse animation for each node
                const nodeElement = this.nodeElements.filter(d => d.id === nodeId);
                
                nodeElement.selectAll('circle')
                    .transition()
                    .duration(300)
                    .attr('r', d => d.size * 1.3)
                    .style('opacity', 0.8)
                    .transition()
                    .duration(300)
                    .attr('r', d => d.size)
                    .style('opacity', 1.0);
            }, delayBetweenNodes * index);
        });
    }
    
    /**
     * Show connection preview on hover
     */
    showConnectionPreview(event, edgeData) {
        if (this.tooltipElement) {
            const transformInfo = this.getTransformationInfo(edgeData);
            const content = this.formatConnectionPreview(edgeData, transformInfo);
            
            this.tooltipElement
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px')
                .style('opacity', 1)
                .html(content);
        }
    }
    
    /**
     * Hide connection preview
     */
    hideConnectionPreview() {
        if (this.tooltipElement) {
            this.tooltipElement.style('opacity', 0);
        }
    }
    
    /**
     * Format connection preview content
     */
    formatConnectionPreview(edgeData, transformInfo) {
        return `
            <div class="connection-preview">
                <div class="preview-title">Data Connection</div>
                <div class="preview-section">
                    <span class="preview-label">From:</span> ${edgeData.source.id || edgeData.source}
                </div>
                <div class="preview-section">
                    <span class="preview-label">To:</span> ${edgeData.target.id || edgeData.target}
                </div>
                ${transformInfo.shape_change ? `
                <div class="preview-section">
                    <span class="preview-label">Shape:</span> 
                    ${transformInfo.shape_change.from} → ${transformInfo.shape_change.to}
                </div>` : ''}
                ${transformInfo.memory_change ? `
                <div class="preview-section">
                    <span class="preview-label">Memory Δ:</span> 
                    ${transformInfo.memory_change > 0 ? '+' : ''}${transformInfo.memory_change.toFixed(1)} MB
                </div>` : ''}
                <div class="preview-hint">Click to highlight path</div>
            </div>
        `;
    }
    
    /**
     * Show detailed transformation information
     */
    showTransformationDetails(edgeData, pathInfo) {
        if (!this.detailPanel) return;
        
        const content = this.formatTransformationDetails(edgeData, pathInfo);
        
        this.detailPanel
            .style('display', 'block')
            .html(content);
        
        // Position the detail panel
        this.positionDetailPanel();
    }
    
    /**
     * Hide transformation details
     */
    hideTransformationDetails() {
        if (this.detailPanel) {
            this.detailPanel.style('display', 'none');
        }
    }
    
    /**
     * Format detailed transformation information
     */
    formatTransformationDetails(edgeData, pathInfo) {
        const transformations = pathInfo.transformations || [];
        const sourceNode = this.getNodeData(edgeData.source);
        const targetNode = this.getNodeData(edgeData.target);
        
        let content = `
            <div class="transformation-details">
                <h3>Data Path Analysis</h3>
                <div class="path-summary">
                    <div class="summary-item">
                        <span class="label">Path Length:</span>
                        <span class="value">${pathInfo.path_nodes.length} nodes</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Total Memory:</span>
                        <span class="value">${pathInfo.total_memory.toFixed(1)} MB</span>
                    </div>
                    ${pathInfo.bottleneck_node ? `
                    <div class="summary-item">
                        <span class="label">Bottleneck:</span>
                        <span class="value">${pathInfo.bottleneck_node}</span>
                    </div>` : ''}
                </div>
                
                <div class="transformations-list">
                    <h4>Transformations</h4>
        `;
        
        transformations.forEach((transform, index) => {
            content += `
                <div class="transformation-item">
                    <div class="transform-step">${index + 1}</div>
                    <div class="transform-details">
                        <div class="transform-nodes">
                            ${transform.from_node} → ${transform.to_node}
                        </div>
                        ${transform.shape_change ? `
                        <div class="shape-change">
                            Shape: [${transform.shape_change.from.join(',')}] → [${transform.shape_change.to.join(',')}]
                        </div>` : ''}
                        ${transform.dtype_change ? `
                        <div class="dtype-change">
                            Type: ${transform.dtype_change.from} → ${transform.dtype_change.to}
                        </div>` : ''}
                        <div class="memory-change ${transform.memory_change >= 0 ? 'positive' : 'negative'}">
                            Memory: ${transform.memory_change >= 0 ? '+' : ''}${transform.memory_change.toFixed(1)} MB
                        </div>
                    </div>
                </div>
            `;
        });
        
        content += `
                </div>
                <div class="detail-actions">
                    <button onclick="connectionHighlighter.animateActivePaths()">Animate Path</button>
                    <button onclick="connectionHighlighter.exportPathData()">Export Data</button>
                    <button onclick="connectionHighlighter.clearAllHighlights()">Clear Highlights</button>
                </div>
            </div>
        `;
        
        return content;
    }
    
    /**
     * Show node connection menu
     */
    showNodeConnectionMenu(event, nodeData) {
        const menu = this.createContextMenu(event, [
            {
                label: 'Highlight All Incoming Paths',
                action: () => this.highlightNodePaths(nodeData.id, 'incoming')
            },
            {
                label: 'Highlight All Outgoing Paths', 
                action: () => this.highlightNodePaths(nodeData.id, 'outgoing')
            },
            {
                label: 'Show Node Analytics',
                action: () => this.showNodeAnalytics(nodeData)
            },
            { separator: true },
            {
                label: 'Clear All Highlights',
                action: () => this.clearAllHighlights()
            }
        ]);
    }
    
    /**
     * Create detail panel for transformation information
     */
    createDetailPanel() {
        // Remove existing panel
        d3.select('.transformation-detail-panel').remove();
        
        this.detailPanel = d3.select('body')
            .append('div')
            .attr('class', 'transformation-detail-panel')
            .style('display', 'none')
            .style('position', 'fixed')
            .style('top', '20px')
            .style('right', '20px')
            .style('width', '400px')
            .style('max-height', '80vh')
            .style('background', 'white')
            .style('border', '1px solid #ddd')
            .style('border-radius', '8px')
            .style('box-shadow', '0 4px 12px rgba(0,0,0,0.15)')
            .style('padding', '20px')
            .style('overflow-y', 'auto')
            .style('z-index', '1001')
            .style('font-family', 'system-ui, -apple-system, sans-serif');
        
        // Create tooltip element if it doesn't exist
        if (!this.tooltipElement) {
            this.tooltipElement = d3.select('body')
                .append('div')
                .attr('class', 'connection-tooltip')
                .style('position', 'absolute')
                .style('background', 'rgba(0,0,0,0.9)')
                .style('color', 'white')
                .style('padding', '10px')
                .style('border-radius', '4px')
                .style('font-size', '12px')
                .style('pointer-events', 'none')
                .style('opacity', 0)
                .style('z-index', '1000')
                .style('transition', 'opacity 0.2s');
        }
        
        this.addDetailPanelStyles();
    }
    
    /**
     * Add CSS styles for the detail panel
     */
    addDetailPanelStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .transformation-detail-panel {
                font-size: 14px;
                line-height: 1.4;
            }
            
            .transformation-detail-panel h3 {
                margin: 0 0 15px 0;
                color: #333;
                border-bottom: 2px solid #ff6b35;
                padding-bottom: 5px;
            }
            
            .transformation-detail-panel h4 {
                margin: 20px 0 10px 0;
                color: #555;
                font-size: 16px;
            }
            
            .path-summary {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 20px;
            }
            
            .summary-item {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
            }
            
            .summary-item:last-child {
                margin-bottom: 0;
            }
            
            .summary-item .label {
                font-weight: 500;
                color: #666;
            }
            
            .summary-item .value {
                font-weight: 600;
                color: #333;
            }
            
            .transformation-item {
                display: flex;
                margin-bottom: 15px;
                padding: 12px;
                background: #fff;
                border: 1px solid #e9ecef;
                border-radius: 6px;
            }
            
            .transform-step {
                background: #ff6b35;
                color: white;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: bold;
                margin-right: 12px;
                flex-shrink: 0;
            }
            
            .transform-details {
                flex: 1;
            }
            
            .transform-nodes {
                font-weight: 600;
                color: #333;
                margin-bottom: 4px;
            }
            
            .shape-change, .dtype-change {
                font-size: 12px;
                color: #666;
                margin-bottom: 2px;
            }
            
            .memory-change {
                font-size: 12px;
                font-weight: 500;
            }
            
            .memory-change.positive {
                color: #e74c3c;
            }
            
            .memory-change.negative {
                color: #27ae60;
            }
            
            .detail-actions {
                margin-top: 20px;
                padding-top: 15px;
                border-top: 1px solid #e9ecef;
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }
            
            .detail-actions button {
                background: #007bff;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: background-color 0.2s;
            }
            
            .detail-actions button:hover {
                background: #0056b3;
            }
            
            .connection-preview {
                max-width: 250px;
            }
            
            .preview-title {
                font-weight: bold;
                color: #ffc107;
                margin-bottom: 8px;
            }
            
            .preview-section {
                margin-bottom: 4px;
                font-size: 11px;
            }
            
            .preview-label {
                color: #adb5bd;
                font-weight: 500;
            }
            
            .preview-hint {
                margin-top: 8px;
                font-style: italic;
                color: #6c757d;
                font-size: 10px;
            }
            
            /* Highlight animation styles */
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.6; }
                100% { opacity: 1; }
            }
            
            .highlighted {
                animation: pulse 2s infinite;
            }
            
            .path-highlighted {
                filter: drop-shadow(0 0 8px rgba(255, 107, 53, 0.6));
            }
            
            .path-index {
                background: white;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 2px 4px;
            }
        `;
        
        document.head.appendChild(style);
    }
    
    /**
     * Position the detail panel optimally
     */
    positionDetailPanel() {
        if (!this.detailPanel) return;
        
        const panelNode = this.detailPanel.node();
        const rect = panelNode.getBoundingClientRect();
        const windowWidth = window.innerWidth;
        const windowHeight = window.innerHeight;
        
        let left = 20;
        let top = 20;
        
        // Adjust position if panel would be off-screen
        if (rect.right > windowWidth - 20) {
            left = windowWidth - rect.width - 40;
        }
        
        if (rect.bottom > windowHeight - 20) {
            top = windowHeight - rect.height - 40;
        }
        
        this.detailPanel
            .style('left', left + 'px')
            .style('top', top + 'px');
    }
    
    /**
     * Create context menu
     */
    createContextMenu(event, items) {
        // Remove existing menu
        d3.selectAll('.connection-context-menu').remove();
        
        const menu = d3.select('body')
            .append('div')
            .attr('class', 'connection-context-menu')
            .style('position', 'absolute')
            .style('left', event.pageX + 'px')
            .style('top', event.pageY + 'px')
            .style('background', 'white')
            .style('border', '1px solid #ccc')
            .style('border-radius', '4px')
            .style('box-shadow', '0 2px 8px rgba(0,0,0,0.15)')
            .style('z-index', '1002')
            .style('min-width', '180px');
        
        items.forEach(item => {
            if (item.separator) {
                menu.append('div')
                    .style('height', '1px')
                    .style('background', '#e9ecef')
                    .style('margin', '4px 0');
            } else {
                menu.append('div')
                    .style('padding', '8px 12px')
                    .style('cursor', 'pointer')
                    .style('font-size', '14px')
                    .text(item.label)
                    .on('click', () => {
                        item.action();
                        menu.remove();
                    })
                    .on('mouseover', function() {
                        d3.select(this).style('background', '#f8f9fa');
                    })
                    .on('mouseout', function() {
                        d3.select(this).style('background', null);
                    });
            }
        });
        
        // Remove menu when clicking elsewhere
        d3.select('body').on('click.context-menu', function() {
            menu.remove();
            d3.select('body').on('click.context-menu', null);
        });
        
        return menu;
    }
    
    /**
     * Clear all highlights
     */
    clearAllHighlights() {
        // Clear connection highlights
        this.activeConnections.clear();
        this.highlightedPaths.clear();
        
        // Reset edge styles
        if (this.edgeElements) {
            this.edgeElements
                .classed('highlighted path-highlighted', false)
                .style('stroke', null)
                .style('stroke-width', null)
                .style('opacity', null)
                .style('filter', null)
                .style('animation', null)
                .style('marker-end', null);
        }
        
        // Reset node styles
        if (this.nodeElements) {
            this.nodeElements
                .selectAll('circle')
                .classed('path-highlighted', false)
                .style('stroke', null)
                .style('stroke-width', null)
                .style('filter', null);
            
            // Remove path index labels
            this.nodeElements.selectAll('.path-index').remove();
        }
        
        // Remove tracer dots from graph group if it exists
        const graphGroup = this.svg.select('.graph-group');
        if (!graphGroup.empty()) {
            graphGroup.selectAll('.connection-tracer').remove();
        } else {
            this.svg.selectAll('.connection-tracer').remove();
        }
        
        // Hide detail panel
        this.hideTransformationDetails();
        
        console.log('All highlights cleared');
    }
    
    /**
     * Toggle highlight mode
     */
    toggleHighlightMode() {
        // Implementation for toggling between different highlight modes
        console.log('Toggle highlight mode');
    }
    
    // Utility methods
    
    /**
     * Get connection ID from edge data
     */
    getConnectionId(edgeData) {
        const sourceId = edgeData.source.id || edgeData.source;
        const targetId = edgeData.target.id || edgeData.target;
        return `${sourceId}->${targetId}`;
    }
    
    /**
     * Get node data by ID
     */
    getNodeData(nodeIdOrObj) {
        const nodeId = nodeIdOrObj.id || nodeIdOrObj;
        return this.data.nodes.find(n => n.id === nodeId);
    }
    
    /**
     * Find data path between two nodes using the filter engine
     */
    findDataPath(sourceId, targetId) {
        // Use the FilterEngine's path finding capability if available
        if (window.filterEngine && window.filterEngine.find_data_path) {
            return window.filterEngine.find_data_path(this.data.nodes, sourceId, targetId);
        }
        
        // Fallback to simple path finding
        return this.findSimpleDataPath(sourceId, targetId);
    }
    
    /**
     * Simple data path finding fallback
     */
    findSimpleDataPath(sourceId, targetId) {
        // Build adjacency map from edges
        const adjacencyMap = new Map();
        this.data.edges.forEach(edge => {
            const source = edge.source.id || edge.source;
            const target = edge.target.id || edge.target;
            
            if (!adjacencyMap.has(source)) {
                adjacencyMap.set(source, []);
            }
            adjacencyMap.get(source).push(target);
        });
        
        // BFS to find path
        const queue = [[sourceId]];
        const visited = new Set();
        
        while (queue.length > 0) {
            const path = queue.shift();
            const currentNode = path[path.length - 1];
            
            if (currentNode === targetId) {
                // Found path
                const pathNodes = path;
                const transformations = this.calculatePathTransformations(pathNodes);
                const totalMemory = this.calculatePathMemory(pathNodes);
                
                return {
                    path_nodes: pathNodes,
                    transformations: transformations,
                    total_memory: totalMemory,
                    bottleneck_node: this.findBottleneckNode(pathNodes)
                };
            }
            
            if (!visited.has(currentNode)) {
                visited.add(currentNode);
                
                const neighbors = adjacencyMap.get(currentNode) || [];
                neighbors.forEach(neighbor => {
                    if (!visited.has(neighbor)) {
                        queue.push([...path, neighbor]);
                    }
                });
            }
        }
        
        return null; // No path found
    }
    
    /**
     * Calculate transformations along a path
     */
    calculatePathTransformations(pathNodes) {
        const transformations = [];
        
        for (let i = 0; i < pathNodes.length - 1; i++) {
            const fromNode = this.getNodeData(pathNodes[i]);
            const toNode = this.getNodeData(pathNodes[i + 1]);
            
            if (fromNode && toNode) {
                const transform = {
                    from_node: fromNode.id,
                    to_node: toNode.id,
                    memory_change: toNode.memory_mb - fromNode.memory_mb
                };
                
                // Add shape change if available
                if (fromNode.shape_info && toNode.shape_info) {
                    transform.shape_change = {
                        from: fromNode.shape_info.output_shape || [],
                        to: toNode.shape_info.input_shape || []
                    };
                }
                
                transformations.push(transform);
            }
        }
        
        return transformations;
    }
    
    /**
     * Calculate total memory along a path
     */
    calculatePathMemory(pathNodes) {
        let totalMemory = 0;
        pathNodes.forEach(nodeId => {
            const nodeData = this.getNodeData(nodeId);
            if (nodeData && nodeData.memory_mb) {
                totalMemory += nodeData.memory_mb;
            }
        });
        return totalMemory;
    }
    
    /**
     * Find bottleneck node in a path
     */
    findBottleneckNode(pathNodes) {
        let maxMemory = 0;
        let bottleneckNode = null;
        
        pathNodes.forEach(nodeId => {
            const nodeData = this.getNodeData(nodeId);
            if (nodeData && nodeData.memory_mb > maxMemory) {
                maxMemory = nodeData.memory_mb;
                bottleneckNode = nodeId;
            }
        });
        
        return bottleneckNode;
    }
    
    /**
     * Get transformation information for an edge
     */
    getTransformationInfo(edgeData) {
        const sourceNode = this.getNodeData(edgeData.source);
        const targetNode = this.getNodeData(edgeData.target);
        
        const info = {};
        
        if (sourceNode && targetNode) {
            // Memory change
            if (sourceNode.memory_mb !== undefined && targetNode.memory_mb !== undefined) {
                info.memory_change = targetNode.memory_mb - sourceNode.memory_mb;
            }
            
            // Shape change
            if (sourceNode.shape_info && targetNode.shape_info) {
                const sourceShape = sourceNode.shape_info.output_shape;
                const targetShape = targetNode.shape_info.input_shape;
                
                if (sourceShape && targetShape && 
                    JSON.stringify(sourceShape) !== JSON.stringify(targetShape)) {
                    info.shape_change = {
                        from: sourceShape,
                        to: targetShape
                    };
                }
            }
        }
        
        return info;
    }
    
    /**
     * Find paths from a node
     */
    findPathsFromNode(nodeId) {
        const paths = [];
        // Implementation for finding outgoing paths
        return paths;
    }
    
    /**
     * Find paths to a node
     */
    findPathsToNode(nodeId) {
        const paths = [];
        // Implementation for finding incoming paths
        return paths;
    }
    
    /**
     * Store path information
     */
    storePathInfo(pathId, pathInfo) {
        if (!this.storedPaths) {
            this.storedPaths = new Map();
        }
        this.storedPaths.set(pathId, pathInfo);
    }
    
    /**
     * Get stored path information
     */
    getStoredPathInfo(pathId) {
        return this.storedPaths ? this.storedPaths.get(pathId) : null;
    }
    
    /**
     * Export path data
     */
    exportPathData() {
        const pathData = {
            highlighted_paths: Array.from(this.highlightedPaths),
            active_connections: Array.from(this.activeConnections),
            timestamp: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(pathData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'uniad-connection-highlights.json';
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Global instances for easy access
let connectionHighlighter = null;
let zoomPanController = null;
let tooltipManager = null;

// Integration function for existing visualizations with zoom/pan support
function initializeInteractiveVisualization(svg, nodeElements, edgeElements, data, config) {
    // Get or create the graph group (container for nodes and edges)
    let graphGroup = svg.select('.graph-group');
    if (graphGroup.empty()) {
        graphGroup = svg.append('g').attr('class', 'graph-group');
        // Move existing nodes and edges to the graph group if they exist
        if (nodeElements && !nodeElements.empty()) {
            nodeElements.each(function() {
                graphGroup.node().appendChild(this);
            });
        }
        if (edgeElements && !edgeElements.empty()) {
            edgeElements.each(function() {
                graphGroup.node().appendChild(this);
            });
        }
    }
    
    // Initialize zoom/pan controller first
    zoomPanController = new ZoomPanController(svg, graphGroup, config);
    
    // Initialize tooltip manager
    tooltipManager = new TooltipManager(svg, nodeElements, edgeElements, data, config);
    
    // Initialize connection highlighter with zoom integration
    connectionHighlighter = new ConnectionHighlighter(svg, nodeElements, edgeElements, data, config);
    
    // Integrate all systems
    connectionHighlighter.zoomPanController = zoomPanController;
    connectionHighlighter.tooltipManager = tooltipManager;
    zoomPanController.connectionHighlighter = connectionHighlighter;
    zoomPanController.tooltipManager = tooltipManager;
    tooltipManager.connectionHighlighter = connectionHighlighter;
    tooltipManager.zoomPanController = zoomPanController;
    
    // Make globally available for template integration
    window.connectionHighlighter = connectionHighlighter;
    window.zoomPanController = zoomPanController;
    window.tooltipManager = tooltipManager;
    
    console.log('Interactive visualization initialized with zoom/pan, connection highlighting, and tooltips');
    
    return {
        connectionHighlighter,
        zoomPanController,
        tooltipManager,
        graphGroup
    };
}

// Legacy function for backward compatibility
function initializeConnectionHighlighter(svg, nodeElements, edgeElements, data, config) {
    console.warn('initializeConnectionHighlighter is deprecated. Use initializeInteractiveVisualization for full functionality.');
    return initializeInteractiveVisualization(svg, nodeElements, edgeElements, data, config).connectionHighlighter;
}

// Enhanced initialization for advanced users
function initializeZoomPanOnly(svg, graphGroup, config) {
    zoomPanController = new ZoomPanController(svg, graphGroup, config);
    window.zoomPanController = zoomPanController;
    
    console.log('Zoom/pan controller initialized independently');
    return zoomPanController;
}

// Initialize tooltip manager independently
function initializeTooltipOnly(svg, nodeElements, edgeElements, data, config) {
    tooltipManager = new TooltipManager(svg, nodeElements, edgeElements, data, config);
    window.tooltipManager = tooltipManager;
    
    console.log('Tooltip manager initialized independently');
    return tooltipManager;
}

// Utility function to focus on a connection using both systems
function focusOnConnection(connectionId, zoomLevel = 2.0) {
    if (!connectionHighlighter || !zoomPanController) {
        console.warn('Interactive visualization not initialized');
        return;
    }
    
    // Find the connection
    const edgeData = connectionHighlighter.edgeElements
        .data()
        .find(d => connectionHighlighter.getConnectionId(d) === connectionId);
    
    if (!edgeData) {
        console.warn('Connection not found:', connectionId);
        return;
    }
    
    // Calculate center point of the connection
    const sourceX = edgeData.source.x || 0;
    const sourceY = edgeData.source.y || 0;
    const targetX = edgeData.target.x || 0;
    const targetY = edgeData.target.y || 0;
    
    const centerX = (sourceX + targetX) / 2;
    const centerY = (sourceY + targetY) / 2;
    
    // Zoom to the connection
    zoomPanController.zoomToPoint(centerX, centerY, zoomLevel);
    
    // Highlight the connection after zoom completes
    setTimeout(() => {
        connectionHighlighter.highlightConnection(connectionId, edgeData);
    }, zoomPanController.transitionDuration);
}

// Utility function to focus on a data path
function focusOnDataPath(sourceId, targetId, zoomLevel = 1.5) {
    if (!connectionHighlighter || !zoomPanController) {
        console.warn('Interactive visualization not initialized');
        return;
    }
    
    // Find the data path
    const pathInfo = connectionHighlighter.findDataPath(sourceId, targetId);
    if (!pathInfo || !pathInfo.path_nodes) {
        console.warn('Data path not found between', sourceId, 'and', targetId);
        return;
    }
    
    // Calculate bounding box of the path
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    pathInfo.path_nodes.forEach(nodeId => {
        const nodeData = connectionHighlighter.getNodeData(nodeId);
        if (nodeData && nodeData.x !== undefined && nodeData.y !== undefined) {
            minX = Math.min(minX, nodeData.x);
            minY = Math.min(minY, nodeData.y);
            maxX = Math.max(maxX, nodeData.x);
            maxY = Math.max(maxY, nodeData.y);
        }
    });
    
    if (minX !== Infinity && minY !== Infinity) {
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        
        // Zoom to encompass the path
        zoomPanController.zoomToPoint(centerX, centerY, zoomLevel);
        
        // Highlight the path after zoom completes
        setTimeout(() => {
            connectionHighlighter.highlightDataPath(pathInfo, 'primary');
        }, zoomPanController.transitionDuration);
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { 
        ConnectionHighlighter, 
        ZoomPanController,
        TooltipManager,
        initializeInteractiveVisualization,
        initializeConnectionHighlighter, // legacy
        initializeZoomPanOnly,
        initializeTooltipOnly,
        focusOnConnection,
        focusOnDataPath
    };
}