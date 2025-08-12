/**
 * Temporal Animation Controller for UniAD Dataflow Visualization
 * 
 * This module provides comprehensive temporal flow animation functionality for UniAD's
 * multi-frame processing, supporting both Stage 1 (5 frames) and Stage 2 (3 frames)
 * queue systems with interactive controls and smooth animations.
 * 
 * Features:
 * - Frame-by-frame stepping and animation
 * - Variable playback speed control
 * - Frame navigation with slider/selector
 * - Temporal dependency visualization
 * - Queue visualization for different stages
 * - Smooth transitions and animations
 * - Integration with D3.js visualizations
 * 
 * Requirements: D3.js v7+, compatible with existing interactive.js module
 */

/**
 * Main temporal animation controller class
 * Handles all aspects of temporal flow animation for UniAD dataflow visualization
 */
class TemporalAnimator {
    constructor(svg, graphGroup, animationData, config = {}) {
        this.svg = svg;
        this.graphGroup = graphGroup;
        this.animationData = animationData;
        this.config = {
            defaultSpeed: 1.0,
            minSpeed: 0.1,
            maxSpeed: 5.0,
            frameDuration: 1000, // ms per frame at 1x speed
            transitionDuration: 300,
            enableSmoothTransitions: true,
            showTemporalDependencies: true,
            showQueueVisualization: true,
            enableFrameLabels: true,
            colorScheme: {
                frame0: '#ff6b35',     // Current frame
                frame1: '#4ecdc4',     // Previous frame  
                frame2: '#45b7d1',     // Older frame
                dependency: '#95a5a6', // Temporal dependencies
                active: '#e74c3c',     // Active operations
                queue: '#f39c12'       // Queue operations
            },
            ...config
        };

        // Animation state
        this.currentFrame = 0;
        this.isPlaying = false;
        this.playbackSpeed = this.config.defaultSpeed;
        this.animationTimer = null;
        this.transitionTimer = null;

        // Data processing
        this.processAnimationData();
        
        // Initialize UI
        this.initializeControls();
        this.initializeVisualization();
    }

    /**
     * Process and validate animation data from TemporalTracer
     */
    processAnimationData() {
        const { metadata, frames, temporal_dependencies, flow_sequences } = this.animationData;
        
        this.queueLength = metadata?.queue_length || 3;
        this.stage = metadata?.stage || 2;
        this.totalFrames = metadata?.total_frames || this.queueLength;
        
        // Process frame data
        this.frames = frames || [];
        this.temporalDependencies = temporal_dependencies || [];
        this.flowSequences = flow_sequences || [];
        
        // Validate data
        if (this.frames.length === 0) {
            console.warn('TemporalAnimator: No frame data available');
            this.frames = this.generateDummyFrames();
        }
        
        console.log(`TemporalAnimator: Initialized with ${this.frames.length} frames for Stage ${this.stage}`);
    }

    /**
     * Generate dummy frame data for testing when no real data is available
     */
    generateDummyFrames() {
        const dummyFrames = [];
        for (let i = 0; i < this.queueLength; i++) {
            dummyFrames.push({
                frame_index: i,
                timestamp: i * 100, // 100ms intervals
                active_nodes: [`frame_${i}_node_1`, `frame_${i}_node_2`],
                operations: [`operation_${i}_1`, `operation_${i}_2`],
                memory_usage: 50 + i * 10,
                compute_time: 100 + i * 20
            });
        }
        return dummyFrames;
    }

    /**
     * Initialize animation control UI
     */
    initializeControls() {
        // Remove existing controls
        d3.select('#temporal-controls').remove();
        
        // Create control container
        const controlsContainer = d3.select('body').append('div')
            .attr('id', 'temporal-controls')
            .style('position', 'fixed')
            .style('top', '20px')
            .style('right', '20px')
            .style('background', 'rgba(255, 255, 255, 0.95)')
            .style('border', '1px solid #ddd')
            .style('border-radius', '8px')
            .style('padding', '15px')
            .style('box-shadow', '0 4px 12px rgba(0,0,0,0.15)')
            .style('z-index', '1000')
            .style('font-family', 'Arial, sans-serif')
            .style('font-size', '14px')
            .style('min-width', '280px');

        // Title
        controlsContainer.append('h4')
            .text(`Temporal Animation (Stage ${this.stage})`)
            .style('margin', '0 0 10px 0')
            .style('color', '#333');

        // Play/Pause button
        const playButton = controlsContainer.append('button')
            .attr('id', 'play-pause-btn')
            .style('background', '#e74c3c')
            .style('color', 'white')
            .style('border', 'none')
            .style('padding', '8px 16px')
            .style('border-radius', '4px')
            .style('cursor', 'pointer')
            .style('margin-right', '10px')
            .text('Play')
            .on('click', () => this.togglePlayPause());

        // Step buttons
        controlsContainer.append('button')
            .style('background', '#95a5a6')
            .style('color', 'white')
            .style('border', 'none')
            .style('padding', '8px 12px')
            .style('border-radius', '4px')
            .style('cursor', 'pointer')
            .style('margin-right', '5px')
            .text('◀')
            .on('click', () => this.stepBackward());

        controlsContainer.append('button')
            .style('background', '#95a5a6')
            .style('color', 'white')
            .style('border', 'none')
            .style('padding', '8px 12px')
            .style('border-radius', '4px')
            .style('cursor', 'pointer')
            .style('margin-right', '10px')
            .text('▶')
            .on('click', () => this.stepForward());

        // Frame slider
        const sliderContainer = controlsContainer.append('div')
            .style('margin', '15px 0');
        
        sliderContainer.append('label')
            .text('Frame: ')
            .style('display', 'block')
            .style('margin-bottom', '5px');
        
        const frameSlider = sliderContainer.append('input')
            .attr('type', 'range')
            .attr('id', 'frame-slider')
            .attr('min', 0)
            .attr('max', Math.max(0, this.totalFrames - 1))
            .attr('value', 0)
            .style('width', '100%')
            .on('input', (event) => this.goToFrame(parseInt(event.target.value)));

        // Frame display
        const frameDisplay = sliderContainer.append('div')
            .attr('id', 'frame-display')
            .style('text-align', 'center')
            .style('margin-top', '5px')
            .text(`Frame 1 of ${this.totalFrames}`);

        // Speed control
        const speedContainer = controlsContainer.append('div')
            .style('margin', '15px 0');
        
        speedContainer.append('label')
            .text('Speed: ')
            .style('display', 'block')
            .style('margin-bottom', '5px');
        
        const speedSlider = speedContainer.append('input')
            .attr('type', 'range')
            .attr('id', 'speed-slider')
            .attr('min', this.config.minSpeed * 10)
            .attr('max', this.config.maxSpeed * 10)
            .attr('value', this.config.defaultSpeed * 10)
            .attr('step', 1)
            .style('width', '100%')
            .on('input', (event) => this.setPlaybackSpeed(parseInt(event.target.value) / 10));

        const speedDisplay = speedContainer.append('div')
            .attr('id', 'speed-display')
            .style('text-align', 'center')
            .style('margin-top', '5px')
            .text(`${this.playbackSpeed.toFixed(1)}x`);

        // Options
        const optionsContainer = controlsContainer.append('div')
            .style('margin', '15px 0');

        const dependenciesCheckbox = optionsContainer.append('label')
            .style('display', 'block')
            .style('margin-bottom', '5px');
        
        dependenciesCheckbox.append('input')
            .attr('type', 'checkbox')
            .attr('id', 'show-dependencies')
            .attr('checked', this.config.showTemporalDependencies)
            .on('change', (event) => this.toggleTemporalDependencies(event.target.checked));
        
        dependenciesCheckbox.append('span')
            .text(' Show temporal dependencies')
            .style('margin-left', '5px');

        const queueCheckbox = optionsContainer.append('label')
            .style('display', 'block');
        
        queueCheckbox.append('input')
            .attr('type', 'checkbox')
            .attr('id', 'show-queue')
            .attr('checked', this.config.showQueueVisualization)
            .on('change', (event) => this.toggleQueueVisualization(event.target.checked));
        
        queueCheckbox.append('span')
            .text(' Show queue visualization')
            .style('margin-left', '5px');

        // Queue info
        const queueInfo = controlsContainer.append('div')
            .style('margin-top', '15px')
            .style('padding', '10px')
            .style('background', '#f8f9fa')
            .style('border-radius', '4px')
            .style('font-size', '12px');
        
        queueInfo.append('div')
            .text(`Queue Length: ${this.queueLength} frames`);
        
        queueInfo.append('div')
            .text(`Stage: ${this.stage} (${this.stage === 1 ? 'Perception' : 'End-to-End'})`);
    }

    /**
     * Initialize visualization elements for temporal flow
     */
    initializeVisualization() {
        // Create temporal overlay group
        this.temporalGroup = this.graphGroup.append('g')
            .attr('class', 'temporal-overlay')
            .style('pointer-events', 'none');

        // Initialize frame 0
        this.updateFrame(0);
    }

    /**
     * Toggle play/pause state
     */
    togglePlayPause() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    /**
     * Start animation playback
     */
    play() {
        if (this.isPlaying) return;
        
        this.isPlaying = true;
        d3.select('#play-pause-btn')
            .text('Pause')
            .style('background', '#f39c12');
        
        this.scheduleNextFrame();
    }

    /**
     * Pause animation playback
     */
    pause() {
        this.isPlaying = false;
        d3.select('#play-pause-btn')
            .text('Play')
            .style('background', '#e74c3c');
        
        if (this.animationTimer) {
            clearTimeout(this.animationTimer);
            this.animationTimer = null;
        }
    }

    /**
     * Schedule next frame in animation sequence
     */
    scheduleNextFrame() {
        if (!this.isPlaying) return;
        
        const frameDuration = this.config.frameDuration / this.playbackSpeed;
        
        this.animationTimer = setTimeout(() => {
            this.stepForward();
            this.scheduleNextFrame();
        }, frameDuration);
    }

    /**
     * Step to next frame
     */
    stepForward() {
        const nextFrame = (this.currentFrame + 1) % this.totalFrames;
        this.goToFrame(nextFrame);
    }

    /**
     * Step to previous frame  
     */
    stepBackward() {
        const prevFrame = (this.currentFrame - 1 + this.totalFrames) % this.totalFrames;
        this.goToFrame(prevFrame);
    }

    /**
     * Go to specific frame
     */
    goToFrame(frameIndex) {
        const clampedFrame = Math.max(0, Math.min(frameIndex, this.totalFrames - 1));
        
        if (clampedFrame === this.currentFrame) return;
        
        this.currentFrame = clampedFrame;
        this.updateFrame(this.currentFrame);
        this.updateControls();
    }

    /**
     * Update visualization for current frame
     */
    updateFrame(frameIndex) {
        if (!this.frames[frameIndex]) {
            console.warn(`TemporalAnimator: No data for frame ${frameIndex}`);
            return;
        }

        const frameData = this.frames[frameIndex];
        
        // Clear existing temporal visualizations
        this.temporalGroup.selectAll('.temporal-highlight').remove();
        this.temporalGroup.selectAll('.temporal-dependency').remove();
        this.temporalGroup.selectAll('.queue-indicator').remove();
        
        // Highlight active nodes for this frame
        this.highlightActiveNodes(frameData);
        
        // Show temporal dependencies if enabled
        if (this.config.showTemporalDependencies) {
            this.visualizeTemporalDependencies(frameIndex);
        }
        
        // Show queue visualization if enabled
        if (this.config.showQueueVisualization) {
            this.visualizeQueue(frameIndex);
        }
        
        // Update frame labels
        if (this.config.enableFrameLabels) {
            this.updateFrameLabels(frameIndex);
        }
    }

    /**
     * Highlight nodes active in current frame
     */
    highlightActiveNodes(frameData) {
        const activeNodes = frameData.active_nodes || [];
        
        // Get color for this frame
        const frameColor = this.getFrameColor(frameData.frame_index || 0);
        
        // Highlight nodes in the graph
        this.graphGroup.selectAll('.node')
            .filter(d => activeNodes.includes(d.id))
            .each(function(d) {
                const node = d3.select(this);
                
                // Add temporal highlight
                const highlight = node.select('circle') || node.select('rect');
                if (!highlight.empty()) {
                    highlight
                        .classed('temporal-highlight', true)
                        .style('stroke', frameColor)
                        .style('stroke-width', '3px')
                        .style('filter', `drop-shadow(0 0 8px ${frameColor})`);
                }
            });
    }

    /**
     * Visualize temporal dependencies between frames
     */
    visualizeTemporalDependencies(frameIndex) {
        const dependencies = this.temporalDependencies.filter(dep => 
            dep.target_frame === frameIndex || dep.source_frame === frameIndex
        );
        
        dependencies.forEach(dep => {
            this.drawTemporalDependency(dep);
        });
    }

    /**
     * Draw temporal dependency arrow
     */
    drawTemporalDependency(dependency) {
        const { source_node, target_node, dependency_type } = dependency;
        
        // Find source and target node elements
        const sourceElement = this.graphGroup.select(`[data-node-id="${source_node}"]`).node();
        const targetElement = this.graphGroup.select(`[data-node-id="${target_node}"]`).node();
        
        if (!sourceElement || !targetElement) return;
        
        // Get positions
        const sourcePos = this.getNodePosition(sourceElement);
        const targetPos = this.getNodePosition(targetElement);
        
        // Create curved dependency arrow
        const dependencyPath = this.temporalGroup.append('path')
            .attr('class', 'temporal-dependency')
            .attr('d', this.createDependencyPath(sourcePos, targetPos))
            .style('stroke', this.config.colorScheme.dependency)
            .style('stroke-width', '2px')
            .style('stroke-dasharray', '5,5')
            .style('fill', 'none')
            .style('marker-end', 'url(#temporal-arrow)');
        
        // Animate dependency
        this.animateDependency(dependencyPath);
    }

    /**
     * Visualize queue state for current frame
     */
    visualizeQueue(frameIndex) {
        // Create queue visualization in corner
        const queueViz = this.temporalGroup.append('g')
            .attr('class', 'queue-indicator')
            .attr('transform', 'translate(50, 50)');
        
        // Queue background
        queueViz.append('rect')
            .attr('width', this.queueLength * 30 + 20)
            .attr('height', 60)
            .attr('rx', 5)
            .style('fill', 'rgba(255, 255, 255, 0.9)')
            .style('stroke', '#ddd')
            .style('stroke-width', 1);
        
        // Queue title
        queueViz.append('text')
            .attr('x', 10)
            .attr('y', 15)
            .text('Temporal Queue')
            .style('font-size', '12px')
            .style('font-weight', 'bold');
        
        // Queue frames
        for (let i = 0; i < this.queueLength; i++) {
            const frameRect = queueViz.append('rect')
                .attr('x', 10 + i * 30)
                .attr('y', 25)
                .attr('width', 25)
                .attr('height', 25)
                .attr('rx', 3)
                .style('fill', i === frameIndex ? this.config.colorScheme.frame0 : '#f0f0f0')
                .style('stroke', i === frameIndex ? this.config.colorScheme.active : '#ccc')
                .style('stroke-width', i === frameIndex ? 2 : 1);
            
            // Frame number
            queueViz.append('text')
                .attr('x', 22.5 + i * 30)
                .attr('y', 42)
                .text(i)
                .style('font-size', '10px')
                .style('text-anchor', 'middle')
                .style('fill', i === frameIndex ? 'white' : '#666');
        }
    }

    /**
     * Update frame labels on nodes
     */
    updateFrameLabels(frameIndex) {
        // Remove existing labels
        this.temporalGroup.selectAll('.frame-label').remove();
        
        const frameData = this.frames[frameIndex];
        const activeNodes = frameData.active_nodes || [];
        
        activeNodes.forEach(nodeId => {
            const nodeElement = this.graphGroup.select(`[data-node-id="${nodeId}"]`).node();
            if (!nodeElement) return;
            
            const nodePos = this.getNodePosition(nodeElement);
            
            // Add frame label
            this.temporalGroup.append('text')
                .attr('class', 'frame-label')
                .attr('x', nodePos.x)
                .attr('y', nodePos.y - 20)
                .text(`F${frameIndex}`)
                .style('text-anchor', 'middle')
                .style('font-size', '10px')
                .style('font-weight', 'bold')
                .style('fill', this.getFrameColor(frameIndex))
                .style('background', 'white')
                .style('padding', '2px');
        });
    }

    /**
     * Get color for frame index
     */
    getFrameColor(frameIndex) {
        const colors = [
            this.config.colorScheme.frame0,
            this.config.colorScheme.frame1,
            this.config.colorScheme.frame2
        ];
        return colors[frameIndex % colors.length] || this.config.colorScheme.frame0;
    }

    /**
     * Get position of node element
     */
    getNodePosition(nodeElement) {
        const bbox = nodeElement.getBBox();
        return {
            x: bbox.x + bbox.width / 2,
            y: bbox.y + bbox.height / 2
        };
    }

    /**
     * Create curved path for temporal dependency
     */
    createDependencyPath(source, target) {
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const dr = Math.sqrt(dx * dx + dy * dy) * 0.3;
        
        return `M${source.x},${source.y} A${dr},${dr} 0 0,1 ${target.x},${target.y}`;
    }

    /**
     * Animate temporal dependency
     */
    animateDependency(pathElement) {
        const totalLength = pathElement.node().getTotalLength();
        
        pathElement
            .attr('stroke-dasharray', `0 ${totalLength}`)
            .transition()
            .duration(this.config.transitionDuration)
            .attr('stroke-dasharray', `${totalLength} 0`);
    }

    /**
     * Update control UI elements
     */
    updateControls() {
        // Update frame slider
        d3.select('#frame-slider').property('value', this.currentFrame);
        
        // Update frame display
        d3.select('#frame-display').text(`Frame ${this.currentFrame + 1} of ${this.totalFrames}`);
    }

    /**
     * Set playback speed
     */
    setPlaybackSpeed(speed) {
        this.playbackSpeed = Math.max(this.config.minSpeed, Math.min(speed, this.config.maxSpeed));
        d3.select('#speed-display').text(`${this.playbackSpeed.toFixed(1)}x`);
    }

    /**
     * Toggle temporal dependency visualization
     */
    toggleTemporalDependencies(enabled) {
        this.config.showTemporalDependencies = enabled;
        this.updateFrame(this.currentFrame);
    }

    /**
     * Toggle queue visualization
     */
    toggleQueueVisualization(enabled) {
        this.config.showQueueVisualization = enabled;
        this.updateFrame(this.currentFrame);
    }

    /**
     * Clean up resources
     */
    destroy() {
        this.pause();
        d3.select('#temporal-controls').remove();
        if (this.temporalGroup) {
            this.temporalGroup.remove();
        }
    }

    /**
     * Reset animation to first frame
     */
    reset() {
        this.pause();
        this.goToFrame(0);
    }

    /**
     * Get current animation state
     */
    getState() {
        return {
            currentFrame: this.currentFrame,
            isPlaying: this.isPlaying,
            playbackSpeed: this.playbackSpeed,
            totalFrames: this.totalFrames,
            queueLength: this.queueLength,
            stage: this.stage
        };
    }
}

/**
 * Factory function to initialize temporal animator
 * Compatible with existing codebase patterns
 */
function initializeTemporalAnimator(svg, graphGroup, animationData, config = {}) {
    return new TemporalAnimator(svg, graphGroup, animationData, config);
}

/**
 * Global functions for template integration
 */
function playAnimation() {
    if (window.temporalAnimator) {
        window.temporalAnimator.play();
    }
}

function pauseAnimation() {
    if (window.temporalAnimator) {
        window.temporalAnimator.pause();
    }
}

function resetAnimation() {
    if (window.temporalAnimator) {
        window.temporalAnimator.reset();
    }
}

function goToFrame(frameIndex) {
    if (window.temporalAnimator) {
        window.temporalAnimator.goToFrame(frameIndex);
    }
}

function setAnimationSpeed(speed) {
    if (window.temporalAnimator) {
        window.temporalAnimator.setPlaybackSpeed(speed);
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TemporalAnimator,
        initializeTemporalAnimator,
        playAnimation,
        pauseAnimation,
        resetAnimation,
        goToFrame,
        setAnimationSpeed
    };
}