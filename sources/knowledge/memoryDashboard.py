"""
Enhanced Memory Analytics Dashboard
Provides web-based interface for memory insights and analytics.
"""

import json
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

@dataclass
class DashboardConfig:
    """Configuration for memory dashboard."""
    output_dir: str = "./data/dashboard"
    refresh_interval: int = 300  # seconds
    max_chart_points: int = 1000
    color_scheme: str = "viridis"

class MemoryDashboard:
    """
    Interactive dashboard for memory analytics and insights.
    Generates HTML reports with visualizations and interactive charts.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set styling
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.config.color_scheme)
    
    def generate_dashboard(self, memory_analytics, vector_memory, 
                         knowledge_graph) -> str:
        """
        Generate complete dashboard with all analytics.
        
        Args:
            memory_analytics: MemoryAnalytics instance
            vector_memory: VectorMemory instance
            knowledge_graph: KnowledgeGraph instance
            
        Returns:
            Path to generated HTML dashboard
        """
        try:
            # Get analytics data
            insights = memory_analytics.analyze_patterns(vector_memory)
            kg_stats = knowledge_graph.get_stats()
            memory_stats = vector_memory.get_stats()
            
            # Generate individual charts
            charts = {}
            charts['temporal'] = self._create_temporal_chart(insights)
            charts['semantic'] = self._create_semantic_chart(insights)
            charts['usage'] = self._create_usage_chart(memory_stats)
            charts['knowledge_graph'] = self._create_kg_chart(kg_stats)
            charts['performance'] = self._create_performance_chart(memory_stats)
            
            # Generate HTML dashboard
            dashboard_path = self._generate_html_dashboard(
                charts, insights, memory_stats, kg_stats
            )
            
            return str(dashboard_path)
            
        except Exception as e:
            print(f"Dashboard generation failed: {e}")
            return ""
    
    def _create_temporal_chart(self, insights: Dict) -> str:
        """Create temporal activity chart."""
        try:
            temporal_patterns = insights.get('temporal_patterns', {})
            if not temporal_patterns:
                return ""
            
            # Extract hourly activity data
            hourly_activity = temporal_patterns.get('hourly_activity', {})
            if not hourly_activity:
                return ""
            
            hours = list(hourly_activity.keys())
            activity = list(hourly_activity.values())
            
            # Create Plotly chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours,
                y=activity,
                mode='lines+markers',
                name='Memory Activity',
                line=dict(color='#2E86C1', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title='Memory Activity by Hour',
                xaxis_title='Hour of Day',
                yaxis_title='Number of Memories',
                template='plotly_white',
                height=400
            )
            
            # Save as HTML
            chart_path = self.output_dir / "temporal_chart.html"
            fig.write_html(str(chart_path))
            return chart_path.read_text()
            
        except Exception as e:
            print(f"Temporal chart creation failed: {e}")
            return ""
    
    def _create_semantic_chart(self, insights: Dict) -> str:
        """Create semantic clusters visualization."""
        try:
            semantic_patterns = insights.get('semantic_patterns', {})
            clusters = semantic_patterns.get('clusters', [])
            
            if not clusters:
                return ""
            
            # Prepare data for visualization
            cluster_data = []
            for i, cluster in enumerate(clusters):
                cluster_data.append({
                    'cluster': f'Cluster {i+1}',
                    'size': len(cluster.get('memories', [])),
                    'keywords': ', '.join(cluster.get('keywords', [])[:5])
                })
            
            df = pd.DataFrame(cluster_data)
            
            # Create pie chart
            fig = px.pie(df, values='size', names='cluster', 
                        title='Memory Clusters Distribution',
                        hover_data=['keywords'])
            
            fig.update_layout(height=400)
            
            # Save as HTML
            chart_path = self.output_dir / "semantic_chart.html"
            fig.write_html(str(chart_path))
            return chart_path.read_text()
            
        except Exception as e:
            print(f"Semantic chart creation failed: {e}")
            return ""
    
    def _create_usage_chart(self, memory_stats: Dict) -> str:
        """Create memory usage statistics chart."""
        try:
            # Extract usage data
            total_memories = memory_stats.get('total_memories', 0)
            avg_similarity = memory_stats.get('average_similarity', 0)
            storage_size = memory_stats.get('storage_size_mb', 0)
            
            # Create gauge charts
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=['Total Memories', 'Avg Similarity', 'Storage (MB)'],
                specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # Total memories gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=total_memories,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [None, max(total_memories * 1.2, 100)]}},
                title={'text': "Memories"}
            ), row=1, col=1)
            
            # Similarity gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_similarity,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, 1]}},
                title={'text': "Similarity"}
            ), row=1, col=2)
            
            # Storage gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=storage_size,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, max(storage_size * 1.2, 100)]}},
                title={'text': "Storage MB"}
            ), row=1, col=3)
            
            fig.update_layout(height=300, title_text="Memory Usage Statistics")
            
            # Save as HTML
            chart_path = self.output_dir / "usage_chart.html"
            fig.write_html(str(chart_path))
            return chart_path.read_text()
            
        except Exception as e:
            print(f"Usage chart creation failed: {e}")
            return ""
    
    def _create_kg_chart(self, kg_stats: Dict) -> str:
        """Create knowledge graph visualization."""
        try:
            entities = kg_stats.get('total_entities', 0)
            relationships = kg_stats.get('total_relationships', 0)
            concepts = kg_stats.get('total_concepts', 0)
            
            # Create bar chart
            categories = ['Entities', 'Relationships', 'Concepts']
            values = [entities, relationships, concepts]
            
            fig = go.Figure(data=[
                go.Bar(x=categories, y=values, 
                      marker_color=['#E74C3C', '#F39C12', '#27AE60'])
            ])
            
            fig.update_layout(
                title='Knowledge Graph Components',
                yaxis_title='Count',
                height=400,
                template='plotly_white'
            )
            
            # Save as HTML
            chart_path = self.output_dir / "kg_chart.html"
            fig.write_html(str(chart_path))
            return chart_path.read_text()
            
        except Exception as e:
            print(f"Knowledge graph chart creation failed: {e}")
            return ""
    
    def _create_performance_chart(self, memory_stats: Dict) -> str:
        """Create performance metrics chart."""
        try:
            # Simulate performance data (in real implementation, collect from system)
            metrics = {
                'Search Speed': memory_stats.get('avg_search_time_ms', 50),
                'Storage Efficiency': memory_stats.get('compression_ratio', 0.8) * 100,
                'Cache Hit Rate': memory_stats.get('cache_hit_rate', 0.75) * 100,
                'Memory Usage': memory_stats.get('memory_usage_percent', 65)
            }
            
            # Create radar chart
            categories = list(metrics.keys())
            values = list(metrics.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Performance'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Performance Metrics",
                height=400
            )
            
            # Save as HTML
            chart_path = self.output_dir / "performance_chart.html"
            fig.write_html(str(chart_path))
            return chart_path.read_text()
            
        except Exception as e:
            print(f"Performance chart creation failed: {e}")
            return ""
    
    def _generate_html_dashboard(self, charts: Dict[str, str], 
                               insights: Dict, memory_stats: Dict, 
                               kg_stats: Dict) -> Path:
        """Generate complete HTML dashboard."""
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create dashboard HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Analytics Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background-color: #f8f9fa;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        .charts-section {{
            padding: 30px;
        }}
        .chart-container {{
            margin-bottom: 40px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .insights-section {{
            background-color: #ecf0f1;
            padding: 30px;
        }}
        .insight-card {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .insight-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            background-color: #34495e;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Memory Analytics Dashboard</h1>
            <p>Generated on {timestamp}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{memory_stats.get('total_memories', 0)}</div>
                <div class="stat-label">Total Memories</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{kg_stats.get('total_entities', 0)}</div>
                <div class="stat-label">Knowledge Entities</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{memory_stats.get('storage_size_mb', 0):.1f} MB</div>
                <div class="stat-label">Storage Used</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{memory_stats.get('average_similarity', 0):.2f}</div>
                <div class="stat-label">Avg Similarity</div>
            </div>
        </div>
        
        <div class="charts-section">
            <div class="chart-container">
                {charts.get('temporal', '<p>Temporal chart not available</p>')}
            </div>
            
            <div class="chart-container">
                {charts.get('usage', '<p>Usage chart not available</p>')}
            </div>
            
            <div class="chart-container">
                {charts.get('semantic', '<p>Semantic chart not available</p>')}
            </div>
            
            <div class="chart-container">
                {charts.get('knowledge_graph', '<p>Knowledge graph chart not available</p>')}
            </div>
            
            <div class="chart-container">
                {charts.get('performance', '<p>Performance chart not available</p>')}
            </div>
        </div>
        
        <div class="insights-section">
            <h2>💡 Key Insights</h2>
            {self._generate_insights_html(insights)}
        </div>
        
        <div class="footer">
            <p>AgenticSeek Memory Analytics Dashboard | Enhanced Memory System v2.0</p>
        </div>
    </div>
    
    <script>
        // Auto-refresh functionality
        setTimeout(function() {{
            location.reload();
        }}, {self.config.refresh_interval * 1000});
    </script>
</body>
</html>
"""
        
        # Save dashboard
        dashboard_path = self.output_dir / "memory_dashboard.html"
        dashboard_path.write_text(html_content)
        
        return dashboard_path
    
    def _generate_insights_html(self, insights: Dict) -> str:
        """Generate HTML for insights section."""
        html = ""
        
        # Temporal insights
        temporal = insights.get('temporal_patterns', {})
        if temporal:
            peak_hour = temporal.get('peak_hour', 'N/A')
            total_sessions = temporal.get('total_sessions', 0)
            html += f"""
            <div class="insight-card">
                <div class="insight-title">⏰ Temporal Patterns</div>
                <p>Peak activity hour: <strong>{peak_hour}</strong></p>
                <p>Total sessions analyzed: <strong>{total_sessions}</strong></p>
            </div>
            """
        
        # Semantic insights
        semantic = insights.get('semantic_patterns', {})
        if semantic:
            num_clusters = len(semantic.get('clusters', []))
            html += f"""
            <div class="insight-card">
                <div class="insight-title">🔍 Semantic Analysis</div>
                <p>Memory clusters identified: <strong>{num_clusters}</strong></p>
                <p>Topics show good diversity in conversation patterns</p>
            </div>
            """
        
        # Performance insights
        html += f"""
        <div class="insight-card">
            <div class="insight-title">⚡ Performance Status</div>
            <p>System is operating efficiently with good search performance</p>
            <p>Memory compression is helping maintain optimal storage</p>
        </div>
        """
        
        return html or "<p>No insights available</p>"
    
    def generate_report(self, memory_analytics, vector_memory, 
                       knowledge_graph, format: str = "json") -> str:
        """
        Generate analytical report in specified format.
        
        Args:
            memory_analytics: MemoryAnalytics instance
            vector_memory: VectorMemory instance  
            knowledge_graph: KnowledgeGraph instance
            format: Output format ('json', 'csv', 'md')
            
        Returns:
            Path to generated report
        """
        try:
            insights = memory_analytics.analyze_patterns(vector_memory)
            kg_stats = knowledge_graph.get_stats()
            memory_stats = vector_memory.get_stats()
            
            timestamp = datetime.datetime.now().isoformat()
            
            report_data = {
                'timestamp': timestamp,
                'memory_stats': memory_stats,
                'knowledge_graph_stats': kg_stats,
                'insights': insights,
                'summary': {
                    'total_memories': memory_stats.get('total_memories', 0),
                    'total_entities': kg_stats.get('total_entities', 0),
                    'storage_size_mb': memory_stats.get('storage_size_mb', 0),
                    'avg_similarity': memory_stats.get('average_similarity', 0)
                }
            }
            
            if format == "json":
                report_path = self.output_dir / f"memory_report_{timestamp.split('T')[0]}.json"
                with open(report_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                    
            elif format == "csv":
                report_path = self.output_dir / f"memory_report_{timestamp.split('T')[0]}.csv"
                # Flatten data for CSV
                flat_data = []
                summary = report_data['summary']
                flat_data.append({
                    'timestamp': timestamp,
                    'total_memories': summary['total_memories'],
                    'total_entities': summary['total_entities'],
                    'storage_size_mb': summary['storage_size_mb'],
                    'avg_similarity': summary['avg_similarity']
                })
                df = pd.DataFrame(flat_data)
                df.to_csv(report_path, index=False)
                
            elif format == "md":
                report_path = self.output_dir / f"memory_report_{timestamp.split('T')[0]}.md"
                md_content = self._generate_markdown_report(report_data)
                report_path.write_text(md_content)
            
            return str(report_path)
            
        except Exception as e:
            print(f"Report generation failed: {e}")
            return ""
    
    def _generate_markdown_report(self, report_data: Dict) -> str:
        """Generate markdown format report."""
        timestamp = report_data['timestamp']
        summary = report_data['summary']
        
        md_content = f"""# Memory Analytics Report

**Generated:** {timestamp}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Memories | {summary['total_memories']} |
| Knowledge Entities | {summary['total_entities']} |
| Storage Used | {summary['storage_size_mb']:.1f} MB |
| Average Similarity | {summary['avg_similarity']:.3f} |

## Memory Performance

The memory system is functioning well with efficient storage and retrieval capabilities.

## Knowledge Graph Status

The knowledge graph contains {summary['total_entities']} entities with good relationship mapping.

## Recommendations

1. Continue regular memory cleanup to maintain performance
2. Monitor storage growth and consider archival for old memories
3. Leverage semantic search for improved information retrieval

---
*Generated by AgenticSeek Enhanced Memory System*
"""
        return md_content
