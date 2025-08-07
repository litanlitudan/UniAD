/**
 * Browser Compatibility Detection for UniAD Dataflow Visualization
 * 
 * This module detects browser capabilities and provides fallback mechanisms
 * for older or incompatible browsers, ensuring broad compatibility while
 * maintaining functionality where possible.
 */

(function(window) {
    'use strict';

    /**
     * Main compatibility checker class
     */
    class CompatibilityChecker {
        constructor() {
            this.features = {};
            this.warnings = [];
            this.errors = [];
            this.browserInfo = this.detectBrowser();
            this.runAllChecks();
        }

        /**
         * Detect browser type and version
         */
        detectBrowser() {
            const ua = navigator.userAgent;
            let browserName = 'Unknown';
            let browserVersion = '0';
            let isModern = false;

            // Chrome
            if (/Chrome\/(\d+)/.test(ua) && !/Edge|Edg/.test(ua)) {
                browserName = 'Chrome';
                browserVersion = RegExp.$1;
                isModern = parseInt(browserVersion) >= 80;
            }
            // Edge (Chromium-based)
            else if (/Edg\/(\d+)/.test(ua)) {
                browserName = 'Edge';
                browserVersion = RegExp.$1;
                isModern = parseInt(browserVersion) >= 80;
            }
            // Firefox
            else if (/Firefox\/(\d+)/.test(ua)) {
                browserName = 'Firefox';
                browserVersion = RegExp.$1;
                isModern = parseInt(browserVersion) >= 75;
            }
            // Safari
            else if (/Version\/(\d+).*Safari/.test(ua)) {
                browserName = 'Safari';
                browserVersion = RegExp.$1;
                isModern = parseInt(browserVersion) >= 13;
            }
            // Legacy Edge
            else if (/Edge\/(\d+)/.test(ua)) {
                browserName = 'Edge Legacy';
                browserVersion = RegExp.$1;
                isModern = false;
            }
            // IE
            else if (/Trident/.test(ua)) {
                browserName = 'Internet Explorer';
                browserVersion = '11';
                isModern = false;
            }

            return {
                name: browserName,
                version: browserVersion,
                versionNumber: parseInt(browserVersion),
                userAgent: ua,
                isModern: isModern,
                isMobile: /Mobile|Android|iPhone|iPad/.test(ua),
                isTablet: /iPad|Android.*Tablet/.test(ua),
                platform: navigator.platform || 'Unknown'
            };
        }

        /**
         * Run all compatibility checks
         */
        runAllChecks() {
            // Core JavaScript features
            this.checkES6Support();
            this.checkPromiseSupport();
            this.checkFetchSupport();
            this.checkWebWorkersSupport();
            
            // D3.js requirements
            this.checkSVGSupport();
            this.checkCanvasSupport();
            this.checkTransformSupport();
            this.checkTransitionSupport();
            
            // CSS features
            this.checkFlexboxSupport();
            this.checkGridSupport();
            this.checkCustomPropertiesSupport();
            
            // Performance features
            this.checkRequestAnimationFrame();
            this.checkIntersectionObserver();
            this.checkResizeObserver();
            
            // Storage features
            this.checkLocalStorage();
            this.checkSessionStorage();
            
            // UniAD-specific requirements
            this.checkWebGLSupport();
            this.checkMemoryAvailability();
        }

        /**
         * Check ES6/ES2015+ support
         */
        checkES6Support() {
            try {
                // Test arrow functions
                eval('(() => {})');
                // Test classes
                eval('class Test {}');
                // Test template literals
                eval('`test`');
                // Test destructuring
                eval('const {a} = {a: 1}');
                // Test spread operator
                eval('[...[1,2,3]]');
                
                this.features.es6 = true;
            } catch (e) {
                this.features.es6 = false;
                this.errors.push('ES6 features not supported. Modern JavaScript required.');
            }

            // Check for specific ES6 features
            this.features.map = typeof Map !== 'undefined';
            this.features.set = typeof Set !== 'undefined';
            this.features.symbol = typeof Symbol !== 'undefined';
            this.features.proxy = typeof Proxy !== 'undefined';
        }

        /**
         * Check Promise support
         */
        checkPromiseSupport() {
            this.features.promise = typeof Promise !== 'undefined';
            if (!this.features.promise) {
                this.warnings.push('Promise not supported. Async operations may fail.');
            }
        }

        /**
         * Check Fetch API support
         */
        checkFetchSupport() {
            this.features.fetch = typeof fetch !== 'undefined';
            if (!this.features.fetch) {
                this.warnings.push('Fetch API not supported. Using XMLHttpRequest fallback.');
            }
        }

        /**
         * Check Web Workers support
         */
        checkWebWorkersSupport() {
            this.features.webWorkers = typeof Worker !== 'undefined';
            if (!this.features.webWorkers) {
                this.warnings.push('Web Workers not supported. Heavy computations may freeze UI.');
            }
        }

        /**
         * Check SVG support (required for D3.js)
         */
        checkSVGSupport() {
            this.features.svg = !!(
                document.createElementNS &&
                document.createElementNS('http://www.w3.org/2000/svg', 'svg').createSVGRect
            );
            
            if (!this.features.svg) {
                this.errors.push('SVG not supported. D3.js visualizations will not work.');
            }

            // Check for specific SVG features
            if (this.features.svg) {
                const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                this.features.svgForeignObject = 'foreignObject' in svg;
                this.features.svgFilters = 'filter' in svg;
            }
        }

        /**
         * Check Canvas support
         */
        checkCanvasSupport() {
            const canvas = document.createElement('canvas');
            this.features.canvas = !!(canvas.getContext && canvas.getContext('2d'));
            
            if (this.features.canvas) {
                this.features.canvasText = typeof canvas.getContext('2d').fillText === 'function';
            }
        }

        /**
         * Check CSS Transform support
         */
        checkTransformSupport() {
            const el = document.createElement('div');
            const transforms = ['transform', 'WebkitTransform', 'MozTransform', 'OTransform', 'msTransform'];
            
            this.features.cssTransform = transforms.some(t => el.style[t] !== undefined);
            
            if (!this.features.cssTransform) {
                this.warnings.push('CSS transforms not supported. Animations may be limited.');
            }
        }

        /**
         * Check CSS Transition support
         */
        checkTransitionSupport() {
            const el = document.createElement('div');
            const transitions = ['transition', 'WebkitTransition', 'MozTransition', 'OTransition'];
            
            this.features.cssTransition = transitions.some(t => el.style[t] !== undefined);
            
            if (!this.features.cssTransition) {
                this.warnings.push('CSS transitions not supported. Animations will be disabled.');
            }
        }

        /**
         * Check Flexbox support
         */
        checkFlexboxSupport() {
            const el = document.createElement('div');
            const flexProps = ['flex', 'WebkitFlex', 'msFlex'];
            
            this.features.flexbox = flexProps.some(p => el.style[p] !== undefined);
        }

        /**
         * Check CSS Grid support
         */
        checkGridSupport() {
            const el = document.createElement('div');
            this.features.cssGrid = el.style.grid !== undefined;
        }

        /**
         * Check CSS Custom Properties support
         */
        checkCustomPropertiesSupport() {
            this.features.cssVariables = window.CSS && window.CSS.supports && 
                                         window.CSS.supports('--test', 'test');
        }

        /**
         * Check requestAnimationFrame support
         */
        checkRequestAnimationFrame() {
            this.features.raf = typeof requestAnimationFrame !== 'undefined';
            
            // Polyfill if needed
            if (!this.features.raf) {
                window.requestAnimationFrame = (function() {
                    return window.webkitRequestAnimationFrame ||
                           window.mozRequestAnimationFrame ||
                           function(callback) {
                               window.setTimeout(callback, 1000 / 60);
                           };
                })();
            }
        }

        /**
         * Check IntersectionObserver support
         */
        checkIntersectionObserver() {
            this.features.intersectionObserver = typeof IntersectionObserver !== 'undefined';
        }

        /**
         * Check ResizeObserver support
         */
        checkResizeObserver() {
            this.features.resizeObserver = typeof ResizeObserver !== 'undefined';
        }

        /**
         * Check localStorage support
         */
        checkLocalStorage() {
            try {
                const test = '__localStorage_test__';
                localStorage.setItem(test, test);
                localStorage.removeItem(test);
                this.features.localStorage = true;
            } catch(e) {
                this.features.localStorage = false;
                this.warnings.push('localStorage not available. Settings will not persist.');
            }
        }

        /**
         * Check sessionStorage support
         */
        checkSessionStorage() {
            try {
                const test = '__sessionStorage_test__';
                sessionStorage.setItem(test, test);
                sessionStorage.removeItem(test);
                this.features.sessionStorage = true;
            } catch(e) {
                this.features.sessionStorage = false;
            }
        }

        /**
         * Check WebGL support (for advanced visualizations)
         */
        checkWebGLSupport() {
            try {
                const canvas = document.createElement('canvas');
                const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                this.features.webgl = !!gl;
            } catch(e) {
                this.features.webgl = false;
            }
            
            if (!this.features.webgl) {
                this.warnings.push('WebGL not supported. 3D visualizations will be disabled.');
            }
        }

        /**
         * Check available memory
         */
        checkMemoryAvailability() {
            if (performance && performance.memory) {
                const limit = performance.memory.jsHeapSizeLimit;
                const used = performance.memory.usedJSHeapSize;
                const available = limit - used;
                
                this.features.memoryInfo = {
                    limit: Math.round(limit / 1048576), // Convert to MB
                    used: Math.round(used / 1048576),
                    available: Math.round(available / 1048576)
                };
                
                // Warn if less than 100MB available
                if (available < 104857600) {
                    this.warnings.push('Low memory available. Large graphs may cause issues.');
                }
            } else {
                this.features.memoryInfo = null;
            }
        }

        /**
         * Check if D3.js v7 specific features are available
         */
        checkD3Requirements() {
            const requirements = {
                svg: this.features.svg,
                es6: this.features.es6,
                promise: this.features.promise,
                cssTransform: this.features.cssTransform,
                cssTransition: this.features.cssTransition
            };
            
            const allMet = Object.values(requirements).every(v => v === true);
            
            if (!allMet) {
                const missing = Object.entries(requirements)
                    .filter(([k, v]) => !v)
                    .map(([k]) => k);
                
                this.errors.push(`D3.js v7 requirements not met: ${missing.join(', ')}`);
            }
            
            return allMet;
        }

        /**
         * Determine if browser can run interactive visualization
         */
        canRunInteractive() {
            return this.features.svg && 
                   this.features.es6 && 
                   this.features.promise &&
                   this.browserInfo.isModern;
        }

        /**
         * Get fallback mode based on capabilities
         */
        getFallbackMode() {
            if (!this.features.svg || !this.features.es6) {
                return 'static'; // Use static Mermaid diagrams
            }
            
            if (!this.features.cssTransition || !this.features.cssTransform) {
                return 'no-animation'; // Disable animations
            }
            
            if (!this.features.webWorkers) {
                return 'limited'; // Limit to smaller graphs
            }
            
            return 'full'; // All features available
        }

        /**
         * Generate compatibility report
         */
        generateReport() {
            return {
                browser: this.browserInfo,
                features: this.features,
                canRunInteractive: this.canRunInteractive(),
                fallbackMode: this.getFallbackMode(),
                warnings: this.warnings,
                errors: this.errors,
                recommendations: this.getRecommendations()
            };
        }

        /**
         * Get recommendations based on detected issues
         */
        getRecommendations() {
            const recommendations = [];
            
            if (!this.browserInfo.isModern) {
                recommendations.push(`Upgrade to ${this.browserInfo.name} version 80+ for best experience`);
            }
            
            if (this.browserInfo.name === 'Internet Explorer') {
                recommendations.push('Internet Explorer is not supported. Please use Chrome, Firefox, Edge, or Safari.');
            }
            
            if (this.features.memoryInfo && this.features.memoryInfo.available < 500) {
                recommendations.push('Close unnecessary tabs to free up memory for large visualizations.');
            }
            
            if (!this.features.webgl) {
                recommendations.push('Enable hardware acceleration for better performance.');
            }
            
            return recommendations;
        }
    }

    /**
     * Fallback renderer for incompatible browsers
     */
    class FallbackRenderer {
        constructor(compatibilityReport) {
            this.report = compatibilityReport;
            this.mode = compatibilityReport.fallbackMode;
        }

        /**
         * Render static Mermaid diagram as fallback
         */
        renderStaticMermaid(container, mermaidCode) {
            const pre = document.createElement('pre');
            pre.className = 'mermaid-fallback';
            pre.textContent = mermaidCode;
            
            const message = document.createElement('div');
            message.className = 'compatibility-message';
            message.innerHTML = `
                <h3>Interactive Visualization Not Available</h3>
                <p>Your browser doesn't support interactive visualizations.</p>
                <p>Below is a static diagram representation:</p>
                <pre>${this.escapeHtml(mermaidCode)}</pre>
                <h4>Recommendations:</h4>
                <ul>
                    ${this.report.recommendations.map(r => `<li>${r}</li>`).join('')}
                </ul>
            `;
            
            container.innerHTML = '';
            container.appendChild(message);
        }

        /**
         * Escape HTML for safe display
         */
        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        /**
         * Show compatibility warning
         */
        showWarning(container) {
            const warning = document.createElement('div');
            warning.className = 'compatibility-warning';
            warning.innerHTML = `
                <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin: 10px; border-radius: 5px;">
                    <h4 style="margin: 0 0 10px 0; color: #856404;">⚠️ Limited Compatibility Mode</h4>
                    <p style="margin: 5px 0;">Some features may not work correctly in your browser.</p>
                    <details>
                        <summary>View Details</summary>
                        <ul>
                            ${this.report.warnings.map(w => `<li>${w}</li>`).join('')}
                        </ul>
                    </details>
                </div>
            `;
            container.insertBefore(warning, container.firstChild);
        }

        /**
         * Apply compatibility fixes
         */
        applyFixes() {
            // Disable animations if not supported
            if (this.mode === 'no-animation') {
                document.documentElement.style.setProperty('--animation-duration', '0ms');
                
                // Override D3 transitions
                if (window.d3) {
                    const originalTransition = window.d3.selection.prototype.transition;
                    window.d3.selection.prototype.transition = function() {
                        return this; // Return selection without transition
                    };
                }
            }
            
            // Limit graph size for limited mode
            if (this.mode === 'limited' && window.graphConfig) {
                window.graphConfig.maxNodes = Math.min(window.graphConfig.maxNodes || 100, 50);
            }
        }
    }

    /**
     * Initialize compatibility checking
     */
    function initCompatibility() {
        const checker = new CompatibilityChecker();
        const report = checker.generateReport();
        
        // Store globally for other modules
        window.compatibilityReport = report;
        
        // Log report to console
        console.log('Browser Compatibility Report:', report);
        
        // Show warnings if any
        if (report.warnings.length > 0) {
            console.warn('Compatibility warnings:', report.warnings);
        }
        
        // Show errors if any
        if (report.errors.length > 0) {
            console.error('Compatibility errors:', report.errors);
        }
        
        // Initialize fallback renderer if needed
        if (report.fallbackMode !== 'full') {
            const fallback = new FallbackRenderer(report);
            window.fallbackRenderer = fallback;
            
            // Apply fixes
            fallback.applyFixes();
            
            // Show warning if in limited mode
            if (report.fallbackMode === 'limited' || report.fallbackMode === 'no-animation') {
                document.addEventListener('DOMContentLoaded', function() {
                    const container = document.getElementById('visualization-container');
                    if (container) {
                        fallback.showWarning(container);
                    }
                });
            }
        }
        
        return report;
    }

    // Export for use in other modules
    window.CompatibilityChecker = CompatibilityChecker;
    window.FallbackRenderer = FallbackRenderer;
    window.initCompatibility = initCompatibility;
    
    // Auto-initialize on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCompatibility);
    } else {
        initCompatibility();
    }

})(window);