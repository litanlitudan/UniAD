#!/usr/bin/env python3
"""Example usage of MemoryTimelineVisualizer for UniAD models

This example demonstrates how to use the MemoryTimelineVisualizer to analyze
memory usage patterns in UniAD's multi-task architecture.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_structures import MemoryTimelineEvent, MemoryEventType
from visualizers.memory_timeline import MemoryTimelineVisualizer, create_memory_timeline_from_trace_nodes

def create_sample_uniad_memory_timeline():
    """Create sample memory timeline events representing UniAD Stage 2 execution"""
    
    events = [
        # Backbone feature extraction
        MemoryTimelineEvent(
            timestamp=0.0,
            operation='ResNet50.conv1',
            module_path='uniad.backbone.resnet50.conv1',
            memory_delta=500.0,
            cumulative_memory=500.0,
            tensor_info={'input_shape': [6, 3, 928, 1600], 'output_shape': [6, 64, 464, 800]}
        ),
        MemoryTimelineEvent(
            timestamp=5.2,
            operation='ResNet50.layer4',
            module_path='uniad.backbone.resnet50.layer4',
            memory_delta=1500.0,
            cumulative_memory=2000.0,
            tensor_info={'input_shape': [6, 1024, 58, 100], 'output_shape': [6, 2048, 29, 50]}
        ),
        
        # BEV Encoder - major memory allocation
        MemoryTimelineEvent(
            timestamp=12.1,
            operation='BEVFormerEncoder.forward',
            module_path='uniad.bev_encoder.encoder',
            memory_delta=8000.0,
            cumulative_memory=10000.0,
            is_bev_operation=True,
            tensor_info={'bev_shape': [1, 256, 200, 200], 'num_cameras': 6}
        ),
        
        # Temporal fusion
        MemoryTimelineEvent(
            timestamp=18.7,
            operation='TemporalSelfAttention.forward',
            module_path='uniad.bev_encoder.temporal_attention',
            memory_delta=2000.0,
            cumulative_memory=12000.0,
            is_bev_operation=True,
            temporal_index=2,  # Current frame in queue
            tensor_info={'queue_length': 3, 'bev_shape': [1, 256, 200, 200]}
        ),
        
        # Task heads - tracking
        MemoryTimelineEvent(
            timestamp=25.3,
            operation='TrackHead.forward',
            module_path='uniad.dense_heads.track_head',
            memory_delta=1200.0,
            cumulative_memory=13200.0,
            task_head='track',
            tensor_info={'num_queries': 900, 'feature_dim': 256}
        ),
        
        # Task heads - segmentation
        MemoryTimelineEvent(
            timestamp=28.9,
            operation='PansegformerHead.forward',
            module_path='uniad.dense_heads.panseg_head',
            memory_delta=800.0,
            cumulative_memory=14000.0,
            task_head='seg',
            tensor_info={'seg_channels': 4, 'bev_shape': [200, 200]}
        ),
        
        # Task heads - motion prediction
        MemoryTimelineEvent(
            timestamp=34.1,
            operation='MotionHead.forward',
            module_path='uniad.dense_heads.motion_head',
            memory_delta=2500.0,
            cumulative_memory=16500.0,
            task_head='motion',
            tensor_info={'future_frames': 12, 'num_modes': 6, 'num_agents': 20}
        ),
        
        # Task heads - occupancy
        MemoryTimelineEvent(
            timestamp=38.4,
            operation='OccHead.forward',
            module_path='uniad.dense_heads.occ_head',
            memory_delta=1500.0,
            cumulative_memory=18000.0,
            task_head='occ',
            tensor_info={'occ_shape': [200, 200, 16], 'flow_frames': 4}
        ),
        
        # Task heads - planning (final task)
        MemoryTimelineEvent(
            timestamp=42.6,
            operation='PlanningHeadSingleMode.forward',
            module_path='uniad.dense_heads.planning_head',
            memory_delta=1000.0,
            cumulative_memory=19000.0,
            task_head='planning',
            tensor_info={'planning_horizon': 6, 'trajectory_points': 6}
        ),
        
        # Memory cleanup/deallocation
        MemoryTimelineEvent(
            timestamp=45.0,
            operation='backward_cleanup',
            module_path='torch.autograd',
            memory_delta=-2000.0,
            cumulative_memory=17000.0,
            event_type=MemoryEventType.DEALLOCATION
        )
    ]
    
    return events

def demonstrate_memory_timeline_analysis():
    """Demonstrate comprehensive memory timeline analysis"""
    
    print("🚀 UniAD Memory Timeline Analysis Example")
    print("=" * 60)
    
    # Create sample timeline
    events = create_sample_uniad_memory_timeline()
    print(f"📊 Created sample timeline with {len(events)} memory events")
    
    # Initialize visualizer for Stage 2 
    visualizer = MemoryTimelineVisualizer(
        stage=2,
        warning_threshold_mb=15000,    # 15GB warning
        critical_threshold_mb=20000,   # 20GB critical  
        enable_task_breakdown=True
    )
    
    # Generate timeline analysis
    print("\n🔍 Analyzing memory timeline...")
    timeline_data = visualizer.generate_timeline(events)
    
    # Display results
    print(f"✅ Timeline Analysis Complete:")
    print(f"   Events processed: {len(timeline_data.events)}")
    print(f"   Memory peaks detected: {len(timeline_data.peaks)}")
    print(f"   Task heads analyzed: {len(timeline_data.task_breakdown)}")
    print(f"   Memory pressure score: {timeline_data.memory_pressure:.3f}")
    print(f"   Problematic operations: {len(timeline_data.problematic_operations)}")
    
    # Show memory peaks
    if timeline_data.peaks:
        print(f"\n🏔️  Top Memory Peaks:")
        for i, peak in enumerate(timeline_data.peaks[:3], 1):
            status = "⚠️ PROBLEMATIC" if peak.is_problematic else "✅ Normal"
            task_info = f" ({peak.task_head})" if peak.task_head else ""
            print(f"   {i}. {peak.operation}{task_info}")
            print(f"      Memory: {peak.peak_memory_mb:.1f}MB at {peak.timestamp:.1f}ms {status}")
    
    # Show top memory consumers
    print(f"\n💾 Top Memory Consumers:")
    top_ops = visualizer.rank_operations_by_memory(events, top_k=5)
    for i, (operation, memory, task_head) in enumerate(top_ops, 1):
        task_info = f" ({task_head})" if task_head != "unknown" else ""
        print(f"   {i}. {operation}{task_info}: {memory:.1f}MB")
    
    # Show task breakdown
    if timeline_data.task_breakdown:
        print(f"\n🎯 Task Head Memory Breakdown:")
        for task_head, points in timeline_data.task_breakdown.items():
            if points:
                max_memory = max(point[1] for point in points)
                print(f"   {task_head}: {max_memory:.1f}MB peak")
    
    # Show recommendations
    if timeline_data.recommendations:
        print(f"\n💡 Optimization Recommendations:")
        for i, rec in enumerate(timeline_data.recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Generate reports
    print(f"\n📄 Generating reports...")
    
    # Markdown report
    markdown_report = visualizer.generate_markdown_report(timeline_data)
    with open('uniad_memory_analysis.md', 'w') as f:
        f.write(markdown_report)
    print(f"   📝 Markdown report: uniad_memory_analysis.md ({len(markdown_report)} chars)")
    
    # HTML visualization
    html_content = visualizer.generate_interactive_html(
        timeline_data, 
        output_path='uniad_memory_timeline.html'
    )
    print(f"   🌐 Interactive HTML: uniad_memory_timeline.html ({len(html_content)} chars)")
    
    print(f"\n🎉 Memory timeline analysis complete!")
    return timeline_data

if __name__ == "__main__":
    # Run the demonstration
    timeline_data = demonstrate_memory_timeline_analysis()
    
    print(f"\n📈 Summary Statistics:")
    if timeline_data.timeline_points:
        max_memory = max(point[1] for point in timeline_data.timeline_points)
        avg_memory = sum(point[1] for point in timeline_data.timeline_points) / len(timeline_data.timeline_points)
        print(f"   Peak Memory: {max_memory:.1f}MB")
        print(f"   Average Memory: {avg_memory:.1f}MB")
        print(f"   Memory Efficiency: {1.0 - timeline_data.memory_pressure:.2%}")
    
    print(f"\n🔧 Usage Instructions:")
    print(f"   1. Import: from visualizers.memory_timeline import MemoryTimelineVisualizer")
    print(f"   2. Create: visualizer = MemoryTimelineVisualizer(stage=2)")
    print(f"   3. Analyze: timeline_data = visualizer.generate_timeline(memory_events)")
    print(f"   4. Visualize: visualizer.generate_interactive_html(timeline_data)")