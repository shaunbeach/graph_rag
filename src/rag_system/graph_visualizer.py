"""
Graph Visualization - Create visual representations of code graphs

Supports multiple output formats:
- DOT (Graphviz) - For high-quality static visualizations
- HTML/D3.js - Interactive web-based visualizations
- Mermaid - Simple text-based diagrams
- ASCII - Terminal-based tree views
"""

import json
from pathlib import Path
from typing import Optional, List, Set, Dict, Any
from .knowledge_graph import CodeKnowledgeGraph, GraphNode


class GraphVisualizer:
    """
    Creates visualizations of code knowledge graphs.
    """

    def __init__(self, graph: CodeKnowledgeGraph):
        """
        Initialize visualizer.

        Args:
            graph: CodeKnowledgeGraph instance
        """
        self.graph = graph

    def to_dot(self, output_path: str,
               node_types: Optional[List[str]] = None,
               max_nodes: int = 100,
               layout: str = "dot") -> str:
        """
        Export graph to DOT format for Graphviz.

        Args:
            output_path: Path to save DOT file
            node_types: Filter by node types (e.g., ['class', 'function'])
            max_nodes: Maximum nodes to include (prevents huge graphs)
            layout: Graphviz layout engine (dot, neato, fdp, circo, twopi)

        Returns:
            Path to generated DOT file
        """
        output_path = Path(output_path)

        # Color scheme for different node types
        node_colors = {
            'module': '#E8F4F8',      # Light blue
            'class': '#FFF4E6',       # Light orange
            'function': '#F0F4F8',    # Light gray
            'method': '#E8F5E9',      # Light green
            'import': '#F3E5F5'       # Light purple
        }

        edge_colors = {
            'imports': '#90A4AE',     # Blue gray
            'defines': '#78909C',     # Dark gray
            'has_method': '#66BB6A',  # Green
            'calls': '#42A5F5',       # Blue
            'inherits_from': '#EF5350'  # Red
        }

        # Start DOT file
        lines = [
            f'digraph CodeGraph {{',
            f'    rankdir=LR;',
            f'    node [shape=box, style=filled, fontname="Arial"];',
            f'    edge [fontname="Arial", fontsize=10];',
            ''
        ]

        # Filter nodes
        nodes_to_include = []
        for node_id, data in self.graph.graph.nodes(data=True):
            if node_types and data.get('type') not in node_types:
                continue
            if len(nodes_to_include) >= max_nodes:
                break
            nodes_to_include.append((node_id, data))

        # Add nodes
        for node_id, data in nodes_to_include:
            node_type = data.get('type', 'unknown')
            name = data.get('name', node_id)
            color = node_colors.get(node_type, '#FFFFFF')

            # Sanitize label
            label = name.replace('"', '\\"')

            # Add type prefix for clarity
            if node_type == 'class':
                label = f'C: {label}'
            elif node_type == 'function':
                label = f'F: {label}'
            elif node_type == 'method':
                label = f'M: {label}'

            lines.append(
                f'    "{node_id}" [label="{label}", fillcolor="{color}"];'
            )

        lines.append('')

        # Add edges
        node_ids = {node_id for node_id, _ in nodes_to_include}
        for source, target, data in self.graph.graph.edges(data=True):
            if source not in node_ids or target not in node_ids:
                continue

            relationship = data.get('relationship', '')
            color = edge_colors.get(relationship, '#000000')

            # Sanitize label
            label = relationship.replace('_', ' ').title()

            lines.append(
                f'    "{source}" -> "{target}" '
                f'[label="{label}", color="{color}"];'
            )

        lines.append('}')

        # Write file
        dot_content = '\n'.join(lines)
        output_path.write_text(dot_content)

        return str(output_path)

    def to_html(self, output_path: str,
                title: str = "Code Knowledge Graph",
                max_nodes: int = 200) -> str:
        """
        Export graph to interactive HTML with D3.js.

        Args:
            output_path: Path to save HTML file
            title: Graph title
            max_nodes: Maximum nodes to include

        Returns:
            Path to generated HTML file
        """
        output_path = Path(output_path)

        # Collect nodes and edges
        nodes = []
        edges = []

        node_ids = set()
        for node_id, data in list(self.graph.graph.nodes(data=True))[:max_nodes]:
            nodes.append({
                'id': node_id,
                'name': data.get('name', node_id),
                'type': data.get('type', 'unknown'),
                'source_file': data.get('source_file', '')
            })
            node_ids.add(node_id)

        for source, target, data in self.graph.graph.edges(data=True):
            if source in node_ids and target in node_ids:
                edges.append({
                    'source': source,
                    'target': target,
                    'relationship': data.get('relationship', '')
                })

        # Create HTML with embedded D3.js visualization
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        #graph {{
            width: 100%;
            height: 800px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .node {{
            cursor: pointer;
        }}
        .node circle {{
            stroke: #fff;
            stroke-width: 2px;
        }}
        .node text {{
            font-size: 12px;
            pointer-events: none;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
            border-radius: 50%;
        }}
        #info {{
            position: fixed;
            top: 80px;
            right: 20px;
            width: 300px;
            padding: 15px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: none;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <svg id="graph"></svg>

    <div class="legend">
        <strong>Node Types:</strong><br>
        <div class="legend-item">
            <span class="legend-color" style="background: #4A90E2;"></span>
            <span>Module</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #F5A623;"></span>
            <span>Class</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #7ED321;"></span>
            <span>Function</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #50E3C2;"></span>
            <span>Method</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #BD10E0;"></span>
            <span>Import</span>
        </div>
    </div>

    <div id="info">
        <h3 id="info-name"></h3>
        <p><strong>Type:</strong> <span id="info-type"></span></p>
        <p><strong>File:</strong> <span id="info-file"></span></p>
    </div>

    <script>
        const data = {{
            nodes: {json.dumps(nodes)},
            links: {json.dumps(edges)}
        }};

        const width = document.getElementById('graph').clientWidth;
        const height = 800;

        const colorScale = {{
            'module': '#4A90E2',
            'class': '#F5A623',
            'function': '#7ED321',
            'method': '#50E3C2',
            'import': '#BD10E0'
        }};

        const svg = d3.select('#graph')
            .attr('width', width)
            .attr('height', height);

        const simulation = d3.forceSimulation(data.nodes)
            .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));

        const link = svg.append('g')
            .selectAll('line')
            .data(data.links)
            .join('line')
            .attr('class', 'link')
            .attr('stroke-width', 2);

        const node = svg.append('g')
            .selectAll('g')
            .data(data.nodes)
            .join('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended))
            .on('click', showInfo);

        node.append('circle')
            .attr('r', 10)
            .attr('fill', d => colorScale[d.type] || '#999');

        node.append('text')
            .text(d => d.name)
            .attr('x', 15)
            .attr('y', 5);

        simulation.on('tick', () => {{
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
        }});

        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}

        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}

        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}

        function showInfo(event, d) {{
            document.getElementById('info-name').textContent = d.name;
            document.getElementById('info-type').textContent = d.type;
            document.getElementById('info-file').textContent = d.source_file;
            document.getElementById('info').style.display = 'block';
        }}
    </script>
</body>
</html>'''

        output_path.write_text(html_content)
        return str(output_path)

    def to_mermaid(self, output_path: str,
                   node_types: Optional[List[str]] = None,
                   max_nodes: int = 50) -> str:
        """
        Export graph to Mermaid diagram format.

        Args:
            output_path: Path to save Mermaid file
            node_types: Filter by node types
            max_nodes: Maximum nodes to include

        Returns:
            Path to generated Mermaid file
        """
        output_path = Path(output_path)

        lines = ['graph LR']

        # Collect nodes
        nodes_to_include = []
        for node_id, data in self.graph.graph.nodes(data=True):
            if node_types and data.get('type') not in node_types:
                continue
            if len(nodes_to_include) >= max_nodes:
                break
            nodes_to_include.append((node_id, data))

        # Node ID mapping (Mermaid needs simple IDs)
        node_map = {node_id: f"N{i}" for i, (node_id, _) in enumerate(nodes_to_include)}

        # Add nodes with labels
        for node_id, data in nodes_to_include:
            mermaid_id = node_map[node_id]
            name = data.get('name', node_id)
            node_type = data.get('type', 'unknown')

            # Different shapes for different types
            if node_type == 'class':
                lines.append(f'    {mermaid_id}["{name}"]')
            elif node_type == 'function':
                lines.append(f'    {mermaid_id}("{name}")')
            elif node_type == 'method':
                lines.append(f'    {mermaid_id}("{name}")')
            else:
                lines.append(f'    {mermaid_id}["{name}"]')

        # Add edges
        node_ids = {node_id for node_id, _ in nodes_to_include}
        for source, target, data in self.graph.graph.edges(data=True):
            if source not in node_ids or target not in node_ids:
                continue

            source_id = node_map[source]
            target_id = node_map[target]
            relationship = data.get('relationship', '')

            lines.append(f'    {source_id} -->|{relationship}| {target_id}')

        mermaid_content = '\n'.join(lines)
        output_path.write_text(mermaid_content)

        return str(output_path)

    def dependency_tree_ascii(self, module_name: str,
                              max_depth: int = 3) -> str:
        """
        Create ASCII tree visualization of module dependencies.

        Args:
            module_name: Name of module to start from
            max_depth: Maximum depth to traverse

        Returns:
            ASCII tree string
        """
        # Find module node
        module_nodes = self.graph.find_nodes(node_type='module', name=module_name)
        if not module_nodes:
            return f"Module '{module_name}' not found"

        module_node = module_nodes[0]

        lines = [f"📦 {module_name}"]
        visited = set()

        def build_tree(node_id: str, depth: int, prefix: str):
            if depth >= max_depth or node_id in visited:
                return

            visited.add(node_id)

            # Get imports
            relationships = self.graph.get_relationships(
                node_id,
                relationship_type='imports',
                direction='outgoing'
            )

            for i, (_, target_id, _) in enumerate(relationships):
                target_node = self.graph.get_node(target_id)
                if not target_node:
                    continue

                is_last = i == len(relationships) - 1
                connector = "└── " if is_last else "├── "
                extension = "    " if is_last else "│   "

                lines.append(f"{prefix}{connector}📄 {target_node.name}")

                # Recursively add dependencies
                build_tree(target_id, depth + 1, prefix + extension)

        build_tree(module_node.id, 0, "")

        return '\n'.join(lines)

    def class_hierarchy_ascii(self, class_name: str) -> str:
        """
        Create ASCII tree visualization of class hierarchy.

        Args:
            class_name: Name of class

        Returns:
            ASCII tree string
        """
        inheritance = self.graph.get_inheritance_tree(class_name)

        if not inheritance:
            return f"Class '{class_name}' not found"

        lines = [f"🏛️  {class_name}"]

        # Show parents
        if inheritance.get('parents'):
            lines.append("  ⬆️  Inherits from:")
            for parent in inheritance['parents']:
                lines.append(f"    └── {parent}")

        # Show children
        if inheritance.get('children'):
            lines.append("  ⬇️  Subclasses:")
            for child in inheritance['children']:
                lines.append(f"    └── {child}")

        # Show methods
        methods = self.graph.get_methods(class_name)
        if methods:
            lines.append("  🔧 Methods:")
            for method in methods:
                args = method.metadata.get('args', [])
                args_str = ', '.join(args) if args else ''
                lines.append(f"    • {method.name}({args_str})")

        return '\n'.join(lines)
