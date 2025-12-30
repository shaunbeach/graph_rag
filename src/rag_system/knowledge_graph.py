"""
Knowledge Graph for Code Relationships

Stores and queries relationships between code entities:
- Classes, functions, methods
- Imports and dependencies
- Call graphs
- Inheritance hierarchies
"""

import ast
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx
from dataclasses import dataclass, asdict


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str  # Unique identifier
    type: str  # module, class, function, method, variable, import
    name: str  # Entity name
    source_file: str  # Source file path
    metadata: Dict[str, Any]  # Additional metadata


@dataclass
class GraphEdge:
    """Represents an edge (relationship) in the knowledge graph."""
    source_id: str
    target_id: str
    relationship: str  # imports, defines, has_method, calls, inherits_from, uses
    metadata: Dict[str, Any]


class CodeKnowledgeGraph:
    """
    Manages a knowledge graph of code relationships.

    Uses NetworkX for in-memory graph operations and SQLite for persistence.
    """

    def __init__(self, db_path: str):
        """
        Initialize the knowledge graph.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.graph = nx.DiGraph()  # Directed graph
        self.conn = None
        self._initialize_db()
        self._load_from_db()

    def _initialize_db(self):
        """Create database tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()

        # Nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                source_file TEXT NOT NULL,
                metadata TEXT
            )
        """)

        # Edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            )
        """)

        # Indexes for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source_file)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_relationship ON edges(relationship)
        """)

        self.conn.commit()

    def _load_from_db(self):
        """Load graph from database into NetworkX."""
        cursor = self.conn.cursor()

        # Load nodes
        cursor.execute("SELECT id, type, name, source_file, metadata FROM nodes")
        for row in cursor.fetchall():
            node_id, node_type, name, source_file, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            self.graph.add_node(
                node_id,
                type=node_type,
                name=name,
                source_file=source_file,
                **metadata
            )

        # Load edges
        cursor.execute("SELECT source_id, target_id, relationship, metadata FROM edges")
        for row in cursor.fetchall():
            source_id, target_id, relationship, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            self.graph.add_edge(
                source_id,
                target_id,
                relationship=relationship,
                **metadata
            )

    def add_node(self, node: GraphNode):
        """Add a node to the graph."""
        # Add to NetworkX graph
        self.graph.add_node(
            node.id,
            type=node.type,
            name=node.name,
            source_file=node.source_file,
            **node.metadata
        )

        # Persist to database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO nodes (id, type, name, source_file, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            node.id,
            node.type,
            node.name,
            node.source_file,
            json.dumps(node.metadata)
        ))
        self.conn.commit()

    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph."""
        # Add to NetworkX graph
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            relationship=edge.relationship,
            **edge.metadata
        )

        # Persist to database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO edges (source_id, target_id, relationship, metadata)
            VALUES (?, ?, ?, ?)
        """, (
            edge.source_id,
            edge.target_id,
            edge.relationship,
            json.dumps(edge.metadata)
        ))
        self.conn.commit()

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        if node_id not in self.graph:
            return None

        data = self.graph.nodes[node_id]
        return GraphNode(
            id=node_id,
            type=data['type'],
            name=data['name'],
            source_file=data['source_file'],
            metadata={k: v for k, v in data.items()
                     if k not in ['type', 'name', 'source_file']}
        )

    def find_nodes(self, node_type: Optional[str] = None,
                   name: Optional[str] = None,
                   source_file: Optional[str] = None) -> List[GraphNode]:
        """Find nodes matching criteria."""
        results = []

        for node_id, data in self.graph.nodes(data=True):
            if node_type and data.get('type') != node_type:
                continue
            if name and data.get('name') != name:
                continue
            if source_file and data.get('source_file') != source_file:
                continue

            results.append(GraphNode(
                id=node_id,
                type=data['type'],
                name=data['name'],
                source_file=data['source_file'],
                metadata={k: v for k, v in data.items()
                         if k not in ['type', 'name', 'source_file']}
            ))

        return results

    def get_relationships(self, node_id: str,
                         relationship_type: Optional[str] = None,
                         direction: str = 'outgoing') -> List[Tuple[str, str, Dict]]:
        """
        Get relationships for a node.

        Args:
            node_id: Node ID
            relationship_type: Filter by relationship type
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of (source_id, target_id, edge_data) tuples
        """
        results = []

        if direction in ['outgoing', 'both']:
            for target in self.graph.successors(node_id):
                edge_data = self.graph[node_id][target]
                if relationship_type and edge_data.get('relationship') != relationship_type:
                    continue
                results.append((node_id, target, edge_data))

        if direction in ['incoming', 'both']:
            for source in self.graph.predecessors(node_id):
                edge_data = self.graph[source][node_id]
                if relationship_type and edge_data.get('relationship') != relationship_type:
                    continue
                results.append((source, node_id, edge_data))

        return results

    def get_methods(self, class_name: str) -> List[GraphNode]:
        """Get all methods of a class."""
        # Find class node
        class_nodes = self.find_nodes(node_type='class', name=class_name)
        if not class_nodes:
            return []

        class_node = class_nodes[0]

        # Find methods via HAS_METHOD relationship
        methods = []
        for _, target_id, edge_data in self.get_relationships(
            class_node.id,
            relationship_type='has_method',
            direction='outgoing'
        ):
            method = self.get_node(target_id)
            if method:
                methods.append(method)

        return methods

    def get_callers(self, function_name: str) -> List[GraphNode]:
        """Get all functions that call a given function."""
        # Find function node
        func_nodes = self.find_nodes(node_type='function', name=function_name)
        if not func_nodes:
            # Try method
            func_nodes = self.find_nodes(node_type='method', name=function_name)

        if not func_nodes:
            return []

        func_node = func_nodes[0]

        # Find callers via CALLS relationship (incoming)
        callers = []
        for source_id, _, edge_data in self.get_relationships(
            func_node.id,
            relationship_type='calls',
            direction='incoming'
        ):
            caller = self.get_node(source_id)
            if caller:
                callers.append(caller)

        return callers

    def get_callees(self, function_name: str) -> List[GraphNode]:
        """Get all functions called by a given function."""
        # Find function node
        func_nodes = self.find_nodes(node_type='function', name=function_name)
        if not func_nodes:
            func_nodes = self.find_nodes(node_type='method', name=function_name)

        if not func_nodes:
            return []

        func_node = func_nodes[0]

        # Find callees via CALLS relationship (outgoing)
        callees = []
        for _, target_id, edge_data in self.get_relationships(
            func_node.id,
            relationship_type='calls',
            direction='outgoing'
        ):
            callee = self.get_node(target_id)
            if callee:
                callees.append(callee)

        return callees

    def get_dependencies(self, module_path: str) -> List[GraphNode]:
        """Get all modules imported by a given module."""
        # Find module node
        module_nodes = self.find_nodes(node_type='module', source_file=module_path)
        if not module_nodes:
            return []

        module_node = module_nodes[0]

        # Find imports via IMPORTS relationship
        imports = []
        for _, target_id, edge_data in self.get_relationships(
            module_node.id,
            relationship_type='imports',
            direction='outgoing'
        ):
            imported = self.get_node(target_id)
            if imported:
                imports.append(imported)

        return imports

    def get_inheritance_tree(self, class_name: str) -> Dict[str, List[str]]:
        """Get inheritance hierarchy for a class."""
        class_nodes = self.find_nodes(node_type='class', name=class_name)
        if not class_nodes:
            return {}

        class_node = class_nodes[0]

        # Get parent classes
        parents = []
        for _, target_id, _ in self.get_relationships(
            class_node.id,
            relationship_type='inherits_from',
            direction='outgoing'
        ):
            parent = self.get_node(target_id)
            if parent:
                parents.append(parent.name)

        # Get child classes
        children = []
        for source_id, _, _ in self.get_relationships(
            class_node.id,
            relationship_type='inherits_from',
            direction='incoming'
        ):
            child = self.get_node(source_id)
            if child:
                children.append(child.name)

        return {
            'class': class_name,
            'parents': parents,
            'children': children
        }

    def query(self, cypher_like_query: str) -> List[Dict]:
        """
        Simple graph query interface (inspired by Cypher).

        Examples:
            "MATCH (c:class)-[HAS_METHOD]->(m:method) WHERE c.name='MyClass' RETURN m"
            "MATCH (f:function)-[CALLS]->(g:function) WHERE f.name='main' RETURN g"
        """
        # This is a simplified query parser - full Cypher would be complex
        # For now, we'll implement a few common patterns

        # Pattern: Find methods of a class
        if 'class' in cypher_like_query and 'HAS_METHOD' in cypher_like_query:
            import re
            match = re.search(r"c\.name='(\w+)'", cypher_like_query)
            if match:
                class_name = match.group(1)
                methods = self.get_methods(class_name)
                return [asdict(m) for m in methods]

        # Pattern: Find callers of a function
        if 'CALLS' in cypher_like_query and 'direction' in cypher_like_query.lower():
            import re
            match = re.search(r"f\.name='(\w+)'", cypher_like_query)
            if match:
                func_name = match.group(1)
                callers = self.get_callers(func_name)
                return [asdict(c) for c in callers]

        return []

    def clear(self):
        """Clear all nodes and edges."""
        self.graph.clear()
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM edges")
        cursor.execute("DELETE FROM nodes")
        self.conn.commit()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
