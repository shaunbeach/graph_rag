"""
Graph Builder - Extracts code relationships and builds knowledge graph

Analyzes Python source code to extract:
- Module structure
- Class definitions and inheritance
- Function/method definitions
- Function calls
- Import dependencies
"""

import ast
import os
from pathlib import Path
from typing import List, Set, Optional, Dict, Any
from .knowledge_graph import CodeKnowledgeGraph, GraphNode, GraphEdge


class PythonGraphBuilder:
    """
    Builds a knowledge graph from Python source code.

    Uses AST parsing to extract entities and relationships.
    """

    def __init__(self, graph: CodeKnowledgeGraph):
        """
        Initialize the graph builder.

        Args:
            graph: CodeKnowledgeGraph instance to populate
        """
        self.graph = graph

    def build_from_file(self, filepath: str) -> Dict[str, int]:
        """
        Extract code structure from a Python file and add to graph.

        Args:
            filepath: Path to Python file

        Returns:
            Dictionary with counts of entities extracted
        """
        stats = {
            'modules': 0,
            'classes': 0,
            'functions': 0,
            'methods': 0,
            'imports': 0,
            'calls': 0,
            'inheritance': 0
        }

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source_code = f.read()

            tree = ast.parse(source_code, filename=filepath)

            # Get relative filepath for cleaner IDs
            rel_path = os.path.relpath(filepath)
            module_id = f"module:{rel_path}"

            # Add module node
            module_node = GraphNode(
                id=module_id,
                type='module',
                name=os.path.basename(filepath),
                source_file=rel_path,
                metadata={'full_path': filepath}
            )
            self.graph.add_node(module_node)
            stats['modules'] += 1

            # Extract entities and relationships
            visitor = CodeVisitor(self.graph, module_id, rel_path, stats)
            visitor.visit(tree)

        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}")

        return stats

    def build_from_directory(self, directory: str,
                            pattern: str = "*.py",
                            recursive: bool = True) -> Dict[str, int]:
        """
        Extract code structure from all Python files in a directory.

        Args:
            directory: Root directory to scan
            pattern: File pattern to match (default: "*.py")
            recursive: Whether to scan subdirectories

        Returns:
            Dictionary with total counts of entities extracted
        """
        total_stats = {
            'files': 0,
            'modules': 0,
            'classes': 0,
            'functions': 0,
            'methods': 0,
            'imports': 0,
            'calls': 0,
            'inheritance': 0
        }

        dir_path = Path(directory)

        if recursive:
            files = dir_path.rglob(pattern)
        else:
            files = dir_path.glob(pattern)

        for filepath in files:
            if filepath.is_file():
                file_stats = self.build_from_file(str(filepath))
                total_stats['files'] += 1
                for key, value in file_stats.items():
                    if key in total_stats:
                        total_stats[key] += value

        return total_stats


class CodeVisitor(ast.NodeVisitor):
    """
    AST visitor that extracts code entities and relationships.
    """

    def __init__(self, graph: CodeKnowledgeGraph,
                 module_id: str, source_file: str,
                 stats: Dict[str, int]):
        """
        Initialize visitor.

        Args:
            graph: Knowledge graph to populate
            module_id: ID of the current module
            source_file: Source file path
            stats: Statistics dictionary to update
        """
        self.graph = graph
        self.module_id = module_id
        self.source_file = source_file
        self.stats = stats
        self.current_class = None  # Track current class context
        self.current_function = None  # Track current function context

    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class definition."""
        class_id = f"class:{self.source_file}:{node.name}"

        # Add class node
        class_node = GraphNode(
            id=class_id,
            type='class',
            name=node.name,
            source_file=self.source_file,
            metadata={
                'line': node.lineno,
                'docstring': ast.get_docstring(node)
            }
        )
        self.graph.add_node(class_node)
        self.stats['classes'] += 1

        # Link module -> class
        edge = GraphEdge(
            source_id=self.module_id,
            target_id=class_id,
            relationship='defines',
            metadata={'entity_type': 'class'}
        )
        self.graph.add_edge(edge)

        # Extract base classes (inheritance)
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_class_name = base.id
                # Create a reference to base class (may not exist yet)
                base_class_id = f"class:{base_class_name}"

                # Add base class node (as reference)
                base_node = GraphNode(
                    id=base_class_id,
                    type='class',
                    name=base_class_name,
                    source_file='unknown',  # May be in another module
                    metadata={'is_reference': True}
                )
                self.graph.add_node(base_node)

                # Add inheritance relationship
                edge = GraphEdge(
                    source_id=class_id,
                    target_id=base_class_id,
                    relationship='inherits_from',
                    metadata={}
                )
                self.graph.add_edge(edge)
                self.stats['inheritance'] += 1

        # Visit class body with class context
        old_class = self.current_class
        self.current_class = class_id
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract function/method definition."""
        # Determine if this is a method or function
        if self.current_class:
            # It's a method
            func_id = f"method:{self.source_file}:{self.current_class.split(':')[-1]}.{node.name}"
            func_type = 'method'
            parent_id = self.current_class
            self.stats['methods'] += 1
        else:
            # It's a function
            func_id = f"function:{self.source_file}:{node.name}"
            func_type = 'function'
            parent_id = self.module_id
            self.stats['functions'] += 1

        # Add function/method node
        func_node = GraphNode(
            id=func_id,
            type=func_type,
            name=node.name,
            source_file=self.source_file,
            metadata={
                'line': node.lineno,
                'docstring': ast.get_docstring(node),
                'args': [arg.arg for arg in node.args.args]
            }
        )
        self.graph.add_node(func_node)

        # Link parent -> function/method
        relationship = 'has_method' if func_type == 'method' else 'defines'
        edge = GraphEdge(
            source_id=parent_id,
            target_id=func_id,
            relationship=relationship,
            metadata={'entity_type': func_type}
        )
        self.graph.add_edge(edge)

        # Visit function body with function context
        old_function = self.current_function
        self.current_function = func_id
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handle async functions like regular functions."""
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call):
        """Extract function calls."""
        if not self.current_function:
            # Only track calls within functions
            self.generic_visit(node)
            return

        # Try to extract the called function name
        called_func_name = None

        if isinstance(node.func, ast.Name):
            # Simple function call: foo()
            called_func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method()
            called_func_name = node.func.attr

        if called_func_name:
            # Create reference to called function (may not exist in graph yet)
            called_func_id = f"function:{called_func_name}"

            # Add called function node (as reference)
            called_node = GraphNode(
                id=called_func_id,
                type='function',
                name=called_func_name,
                source_file='unknown',
                metadata={'is_reference': True}
            )
            self.graph.add_node(called_node)

            # Add call relationship
            edge = GraphEdge(
                source_id=self.current_function,
                target_id=called_func_id,
                relationship='calls',
                metadata={'line': node.lineno}
            )
            self.graph.add_edge(edge)
            self.stats['calls'] += 1

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        """Extract import statements."""
        for alias in node.names:
            import_name = alias.name
            import_id = f"import:{import_name}"

            # Add import node
            import_node = GraphNode(
                id=import_id,
                type='import',
                name=import_name,
                source_file=self.source_file,
                metadata={
                    'alias': alias.asname,
                    'line': node.lineno
                }
            )
            self.graph.add_node(import_node)

            # Link module -> import
            edge = GraphEdge(
                source_id=self.module_id,
                target_id=import_id,
                relationship='imports',
                metadata={}
            )
            self.graph.add_edge(edge)
            self.stats['imports'] += 1

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Extract 'from X import Y' statements."""
        if node.module:
            for alias in node.names:
                import_name = f"{node.module}.{alias.name}" if alias.name != '*' else node.module
                import_id = f"import:{import_name}"

                # Add import node
                import_node = GraphNode(
                    id=import_id,
                    type='import',
                    name=import_name,
                    source_file=self.source_file,
                    metadata={
                        'alias': alias.asname,
                        'line': node.lineno,
                        'from_module': node.module
                    }
                )
                self.graph.add_node(import_node)

                # Link module -> import
                edge = GraphEdge(
                    source_id=self.module_id,
                    target_id=import_id,
                    relationship='imports',
                    metadata={'from': node.module}
                )
                self.graph.add_edge(edge)
                self.stats['imports'] += 1

        self.generic_visit(node)
