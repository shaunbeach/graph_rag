"""
Graph RAG - Knowledge Graph enhanced Retrieval-Augmented Generation

This module implements Graph RAG by combining vector similarity search
with knowledge graph traversal for improved context retrieval.

Architecture:
1. Extract entities from documents during ingestion (AST + pattern matching)
2. Build knowledge graph of entities and relationships
3. During query, use vector search + graph traversal for context
4. Provide enriched context to LLM for better answers

Extraction Methods:
- AST Parsing: 100% accurate for Python code (classes, functions, imports, calls)
- Pattern Matching: For concepts, metrics, formulas in any content

Entity Types:
- CODE_FUNCTION: Python functions (AST or pattern)
- CODE_METHOD: Python class methods (AST)
- CODE_CLASS: Python classes (AST or pattern)
- IMPORT: Python imports (AST)
- CONCEPT: ML/Statistical concepts (pattern)
- METRIC: Performance metrics (pattern)
- FORMULA: Mathematical formulas (pattern)
- TERM: Domain-specific terminology (pattern)

Relationship Types:
- CALLS: function calls another function (AST)
- INHERITS_FROM: class inheritance (AST)
- IMPLEMENTS: function implements concept (pattern)
- CALCULATES: function calculates metric (pattern)
- USES: concept uses formula/metric (pattern)
- RELATED_TO: general relationship (pattern)
- EXAMPLE_OF: code demonstrates concept (pattern)
"""

import re
import ast
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity."""
    id: str
    type: str  # CODE_FUNCTION, CONCEPT, METRIC, etc.
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    chunk_ids: Optional[Set[str]] = None

    def __post_init__(self):
        if self.chunk_ids is None:
            self.chunk_ids = set()


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_id: str
    target_id: str
    type: str  # IMPLEMENTS, CALCULATES, USES, etc.
    metadata: Optional[Dict[str, Any]] = None


class EntityExtractor:
    """Extract entities from document chunks using both pattern matching and AST parsing."""

    def __init__(self):
        # Patterns for entity extraction (fallback for non-Python code)
        self.code_function_pattern = re.compile(r'def\s+(\w+)\s*\(')
        self.code_class_pattern = re.compile(r'class\s+(\w+)\s*[\(:]')
        self.formula_pattern = re.compile(r'([A-Z][a-z]*)\s*=\s*\([^)]+\)\s*/\s*\([^)]+\)')

        # Statistical/ML concepts (expandable)
        self.ml_concepts = {
            'logistic regression', 'linear regression', 'random forest',
            'neural network', 'decision tree', 'svm', 'knn',
            'gradient descent', 'cross-validation', 'regularization',
            'overfitting', 'underfitting', 'bias-variance tradeoff',
            'feature engineering', 'dimensionality reduction', 'pca',
            'clustering', 'classification', 'regression'
        }

        # Performance metrics
        self.metrics = {
            'accuracy', 'precision', 'recall', 'sensitivity', 'specificity',
            'f1 score', 'f1-score', 'roc auc', 'roc-auc', 'auc', 'roc',
            'confusion matrix', 'true positive', 'false positive',
            'true negative', 'false negative', 'tpr', 'fpr', 'tnr', 'fnr',
            'mean squared error', 'mse', 'rmse', 'mae', 'r-squared', 'r2'
        }

        # Statistical terms
        self.statistical_terms = {
            'p-value', 'confidence interval', 'hypothesis test',
            't-test', 'anova', 'chi-square', 'correlation',
            'mean', 'median', 'mode', 'standard deviation', 'variance',
            'distribution', 'normal distribution', 'probability'
        }

    def _is_python_code(self, content: str) -> bool:
        """Check if content appears to be Python code."""
        # Try to parse as Python AST
        try:
            ast.parse(content)
            return True
        except SyntaxError:
            # Check for Python-like patterns
            python_indicators = [
                r'\bdef\s+\w+\s*\(',  # function definitions
                r'\bclass\s+\w+\s*[\(:]',  # class definitions
                r'\bimport\s+\w+',  # import statements
                r'\bfrom\s+\w+\s+import',  # from imports
            ]
            return any(re.search(pattern, content) for pattern in python_indicators)

    def _extract_ast_entities(self, content: str, chunk_id: str, source_file: str) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities from Python code using AST parsing (100% accurate).

        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Not valid Python, return empty
            return entities, relationships

        # Track current class for method context
        current_class = None

        for node in ast.walk(tree):
            # Extract classes
            if isinstance(node, ast.ClassDef):
                class_name = node.name

                # Get base classes for inheritance
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)

                # Get docstring
                docstring = ast.get_docstring(node)

                # Get methods
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

                class_entity = Entity(
                    id=f"class:{class_name}",
                    type="CODE_CLASS",
                    name=class_name,
                    description=docstring or f"Class: {class_name}",
                    metadata={
                        "source_file": source_file,
                        "bases": bases,
                        "methods": methods,
                        "line": node.lineno,
                        "extraction_method": "ast"
                    },
                    chunk_ids={chunk_id}
                )
                entities.append(class_entity)

                # Create inheritance relationships
                for base in bases:
                    relationships.append(Relationship(
                        source_id=f"class:{class_name}",
                        target_id=f"class:{base}",
                        type="INHERITS_FROM",
                        metadata={"context": "class inheritance"}
                    ))

            # Extract functions and methods
            elif isinstance(node, ast.FunctionDef):
                func_name = node.name

                # Get arguments
                args = [arg.arg for arg in node.args.args]

                # Get docstring
                docstring = ast.get_docstring(node)

                # Determine if method or standalone function
                # Check if this function is inside a class
                is_method = False
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef) and hasattr(parent, 'body'):
                        if node in parent.body:
                            is_method = True
                            break

                entity_type = "CODE_METHOD" if is_method else "CODE_FUNCTION"

                func_entity = Entity(
                    id=f"func:{func_name}",
                    type=entity_type,
                    name=func_name,
                    description=docstring or f"{'Method' if is_method else 'Function'}: {func_name}",
                    metadata={
                        "source_file": source_file,
                        "args": args,
                        "line": node.lineno,
                        "extraction_method": "ast"
                    },
                    chunk_ids={chunk_id}
                )
                entities.append(func_entity)

                # Extract function calls within this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            called_func = child.func.id
                            relationships.append(Relationship(
                                source_id=f"func:{func_name}",
                                target_id=f"func:{called_func}",
                                type="CALLS",
                                metadata={"context": "function call"}
                            ))
                        elif isinstance(child.func, ast.Attribute):
                            called_func = child.func.attr
                            relationships.append(Relationship(
                                source_id=f"func:{func_name}",
                                target_id=f"func:{called_func}",
                                type="CALLS",
                                metadata={"context": "method call"}
                            ))

            # Extract imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    import_entity = Entity(
                        id=f"import:{alias.name}",
                        type="IMPORT",
                        name=alias.name,
                        description=f"Import: {alias.name}",
                        metadata={
                            "source_file": source_file,
                            "line": node.lineno,
                            "extraction_method": "ast"
                        },
                        chunk_ids={chunk_id}
                    )
                    entities.append(import_entity)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    import_entity = Entity(
                        id=f"import:{module}.{alias.name}" if module else f"import:{alias.name}",
                        type="IMPORT",
                        name=f"{module}.{alias.name}" if module else alias.name,
                        description=f"Import: {alias.name} from {module}",
                        metadata={
                            "source_file": source_file,
                            "module": module,
                            "line": node.lineno,
                            "extraction_method": "ast"
                        },
                        chunk_ids={chunk_id}
                    )
                    entities.append(import_entity)

        return entities, relationships

    def extract_entities(self, chunk: Document, chunk_id: str) -> List[Entity]:
        """
        Extract entities from a document chunk.

        Uses AST parsing for Python code (100% accurate), falls back to pattern matching
        for other content types and concepts/metrics.

        Args:
            chunk: LangChain Document object
            chunk_id: Unique identifier for the chunk

        Returns:
            List of extracted Entity objects
        """
        entities = []
        content = chunk.page_content
        content_lower = content.lower()
        source_file = chunk.metadata.get("source_file", "unknown")

        # Try AST-based extraction first for Python code
        if self._is_python_code(content):
            ast_entities, ast_relationships = self._extract_ast_entities(
                content, chunk_id, source_file
            )
            entities.extend(ast_entities)

            # Store relationships for later (we'll return them separately)
            if not hasattr(self, '_pending_relationships'):
                self._pending_relationships = []
            self._pending_relationships.extend(ast_relationships)

            logger.info(f"AST extraction: {len(ast_entities)} entities, "
                       f"{len(ast_relationships)} relationships")
        else:
            # Fall back to pattern matching for non-Python code
            # Extract code functions
            for match in self.code_function_pattern.finditer(content):
                func_name = match.group(1)
                entities.append(Entity(
                    id=f"func:{func_name}",
                    type="CODE_FUNCTION",
                    name=func_name,
                    description=f"Function: {func_name}",
                    metadata={
                        "source_file": source_file,
                        "extraction_method": "pattern"
                    },
                    chunk_ids={chunk_id}
                ))

            # Extract code classes
            for match in self.code_class_pattern.finditer(content):
                class_name = match.group(1)
                entities.append(Entity(
                    id=f"class:{class_name}",
                    type="CODE_CLASS",
                    name=class_name,
                    description=f"Class: {class_name}",
                    metadata={
                        "source_file": source_file,
                        "extraction_method": "pattern"
                    },
                    chunk_ids={chunk_id}
                ))

        # Extract ML concepts
        for concept in self.ml_concepts:
            if concept in content_lower:
                entities.append(Entity(
                    id=f"concept:{concept.replace(' ', '_')}",
                    type="CONCEPT",
                    name=concept,
                    description=f"ML/Statistical concept: {concept}",
                    chunk_ids={chunk_id}
                ))

        # Extract metrics
        for metric in self.metrics:
            if metric in content_lower:
                entities.append(Entity(
                    id=f"metric:{metric.replace(' ', '_').replace('-', '_')}",
                    type="METRIC",
                    name=metric,
                    description=f"Performance metric: {metric}",
                    chunk_ids={chunk_id}
                ))

        # Extract statistical terms
        for term in self.statistical_terms:
            if term in content_lower:
                entities.append(Entity(
                    id=f"term:{term.replace(' ', '_').replace('-', '_')}",
                    type="TERM",
                    name=term,
                    description=f"Statistical term: {term}",
                    chunk_ids={chunk_id}
                ))

        # Extract formulas (simplified - looks for assignment patterns)
        for match in self.formula_pattern.finditer(content):
            formula_name = match.group(1)
            entities.append(Entity(
                id=f"formula:{formula_name.lower()}",
                type="FORMULA",
                name=formula_name,
                description=f"Mathematical formula: {formula_name}",
                chunk_ids={chunk_id}
            ))

        return entities

    def get_pending_relationships(self) -> List[Relationship]:
        """
        Get and clear pending relationships from AST extraction.

        Returns:
            List of Relationship objects from AST parsing
        """
        relationships = getattr(self, '_pending_relationships', [])
        self._pending_relationships = []
        return relationships

    def extract_relationships(
        self,
        chunk: Document,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships between entities in a chunk.

        Combines AST-extracted relationships (from get_pending_relationships)
        with pattern-based relationship detection.

        Args:
            chunk: Document chunk
            entities: Entities found in this chunk

        Returns:
            List of Relationship objects
        """
        relationships = []
        content_lower = chunk.page_content.lower()

        # Include AST-extracted relationships
        relationships.extend(self.get_pending_relationships())

        # Map entity types for quick lookup
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.type].append(entity)

        # Rule: Functions that calculate metrics
        for func in entities_by_type.get("CODE_FUNCTION", []):
            for metric in entities_by_type.get("METRIC", []):
                # Check if metric name appears near function definition
                if metric.name in content_lower:
                    relationships.append(Relationship(
                        source_id=func.id,
                        target_id=metric.id,
                        type="CALCULATES",
                        metadata={"context": "function calculates metric"}
                    ))

        # Rule: Concepts use metrics
        for concept in entities_by_type.get("CONCEPT", []):
            for metric in entities_by_type.get("METRIC", []):
                # If both appear in same chunk, they're likely related
                relationships.append(Relationship(
                    source_id=concept.id,
                    target_id=metric.id,
                    type="USES",
                    metadata={"context": "concept uses metric"}
                ))

        # Rule: Functions implement concepts
        for func in entities_by_type.get("CODE_FUNCTION", []):
            for concept in entities_by_type.get("CONCEPT", []):
                # Check if concept appears in function context
                if concept.name in func.name.lower().replace('_', ' '):
                    relationships.append(Relationship(
                        source_id=func.id,
                        target_id=concept.id,
                        type="IMPLEMENTS",
                        metadata={"context": "function implements concept"}
                    ))

        # Rule: Formulas calculate metrics
        for formula in entities_by_type.get("FORMULA", []):
            for metric in entities_by_type.get("METRIC", []):
                if formula.name.lower() == metric.name.lower():
                    relationships.append(Relationship(
                        source_id=formula.id,
                        target_id=metric.id,
                        type="CALCULATES",
                        metadata={"context": "formula calculates metric"}
                    ))

        return relationships


class DocumentKnowledgeGraph:
    """Knowledge graph for document entities and relationships."""

    def __init__(self, workspace_path: Path):
        """
        Initialize knowledge graph.

        Args:
            workspace_path: Path to workspace directory
        """
        self.workspace_path = Path(workspace_path)
        self.graph_path = self.workspace_path / "document_graph.json"

        # NetworkX graph for relationships
        self.graph = nx.DiGraph()

        # Entity storage
        self.entities: Dict[str, Entity] = {}

        # Chunk to entity mapping
        self.chunk_entities: Dict[str, Set[str]] = defaultdict(set)

        # Load existing graph if available
        self.load()

    def add_entity(self, entity: Entity):
        """Add or update an entity in the graph."""
        if entity.id in self.entities:
            # Merge chunk_ids if entity exists
            self.entities[entity.id].chunk_ids.update(entity.chunk_ids)
        else:
            self.entities[entity.id] = entity
            self.graph.add_node(
                entity.id,
                type=entity.type,
                name=entity.name,
                description=entity.description
            )

        # Update chunk mappings
        for chunk_id in entity.chunk_ids:
            self.chunk_entities[chunk_id].add(entity.id)

    def add_relationship(self, relationship: Relationship):
        """Add a relationship to the graph."""
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            type=relationship.type,
            metadata=relationship.metadata or {}
        )

    def get_related_entities(
        self,
        entity_id: str,
        max_depth: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> Set[str]:
        """
        Get entities related to a given entity via graph traversal.

        Args:
            entity_id: Starting entity ID
            max_depth: Maximum traversal depth
            relationship_types: Filter by relationship types (optional)

        Returns:
            Set of related entity IDs
        """
        if entity_id not in self.graph:
            return set()

        related = set()

        # BFS traversal
        visited = {entity_id}
        queue = [(entity_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            # Get neighbors (both incoming and outgoing)
            neighbors = set(self.graph.successors(current_id)) | \
                       set(self.graph.predecessors(current_id))

            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue

                # Check relationship type filter
                if relationship_types:
                    edge_data = self.graph.get_edge_data(current_id, neighbor_id) or \
                               self.graph.get_edge_data(neighbor_id, current_id)
                    if edge_data and edge_data.get('type') not in relationship_types:
                        continue

                related.add(neighbor_id)
                visited.add(neighbor_id)
                queue.append((neighbor_id, depth + 1))

        return related

    def get_chunks_for_entities(self, entity_ids: Set[str]) -> Set[str]:
        """Get all chunk IDs that contain any of the given entities."""
        chunk_ids = set()
        for entity_id in entity_ids:
            if entity_id in self.entities:
                chunk_ids.update(self.entities[entity_id].chunk_ids)
        return chunk_ids

    def save(self):
        """Save graph to disk."""
        data = {
            "entities": {
                entity_id: {
                    "id": entity.id,
                    "type": entity.type,
                    "name": entity.name,
                    "description": entity.description,
                    "metadata": entity.metadata,
                    "chunk_ids": list(entity.chunk_ids)
                }
                for entity_id, entity in self.entities.items()
            },
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "type": data.get("type"),
                    "metadata": data.get("metadata", {})
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }

        with open(self.graph_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved document knowledge graph: {len(self.entities)} entities, "
                   f"{len(self.graph.edges)} relationships")

    def load(self):
        """Load graph from disk."""
        if not self.graph_path.exists():
            logger.info("No existing document knowledge graph found")
            return

        try:
            with open(self.graph_path, 'r') as f:
                data = json.load(f)

            # Load entities
            for entity_data in data.get("entities", {}).values():
                entity = Entity(
                    id=entity_data["id"],
                    type=entity_data["type"],
                    name=entity_data["name"],
                    description=entity_data.get("description"),
                    metadata=entity_data.get("metadata"),
                    chunk_ids=set(entity_data.get("chunk_ids", []))
                )
                self.add_entity(entity)

            # Load edges
            for edge_data in data.get("edges", []):
                relationship = Relationship(
                    source_id=edge_data["source"],
                    target_id=edge_data["target"],
                    type=edge_data.get("type"),
                    metadata=edge_data.get("metadata")
                )
                self.add_relationship(relationship)

            logger.info(f"Loaded document knowledge graph: {len(self.entities)} entities, "
                       f"{len(self.graph.edges)} relationships")

        except Exception as e:
            logger.error(f"Error loading document knowledge graph: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics with extraction method breakdown."""
        entity_counts = defaultdict(int)
        extraction_methods = defaultdict(int)

        for entity in self.entities.values():
            entity_counts[entity.type] += 1
            # Count extraction methods
            method = entity.metadata.get("extraction_method", "unknown") if entity.metadata else "unknown"
            extraction_methods[method] += 1

        relationship_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            relationship_counts[data.get("type", "unknown")] += 1

        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.graph.edges),
            "entity_types": dict(entity_counts),
            "relationship_types": dict(relationship_counts),
            "extraction_methods": dict(extraction_methods),
            "total_chunks_with_entities": len(self.chunk_entities)
        }


class GraphRAGRetriever:
    """Enhanced retrieval using knowledge graph."""

    def __init__(
        self,
        knowledge_graph: DocumentKnowledgeGraph,
        entity_extractor: EntityExtractor
    ):
        """
        Initialize Graph RAG retriever.

        Args:
            knowledge_graph: Document knowledge graph
            entity_extractor: Entity extractor instance
        """
        self.kg = knowledge_graph
        self.extractor = entity_extractor

    def enhance_retrieval(
        self,
        query: str,
        initial_docs: List[Document],
        max_additional_chunks: int = 5,
        graph_depth: int = 2
    ) -> List[str]:
        """
        Enhance retrieval by using knowledge graph.

        Process:
        1. Extract entities from initial retrieved documents
        2. Find related entities via graph traversal
        3. Get chunks containing related entities
        4. Return chunk IDs for enhanced context

        Args:
            query: User query
            initial_docs: Initial documents from vector search
            max_additional_chunks: Max additional chunks to retrieve
            graph_depth: Max depth for graph traversal

        Returns:
            List of chunk IDs (original + graph-enhanced)
        """
        # Get chunk IDs from initial docs
        initial_chunk_ids = {
            doc.metadata.get("chunk_id", str(i))
            for i, doc in enumerate(initial_docs)
        }

        # Extract entities from retrieved documents
        entity_ids = set()
        for doc in initial_docs:
            chunk_id = doc.metadata.get("chunk_id", "")
            if chunk_id in self.kg.chunk_entities:
                entity_ids.update(self.kg.chunk_entities[chunk_id])

        logger.info(f"Found {len(entity_ids)} entities in retrieved chunks")

        # Expand via graph traversal
        related_entity_ids = set()
        for entity_id in entity_ids:
            related = self.kg.get_related_entities(
                entity_id,
                max_depth=graph_depth
            )
            related_entity_ids.update(related)

        logger.info(f"Found {len(related_entity_ids)} related entities via graph")

        # Get chunks for related entities
        related_chunk_ids = self.kg.get_chunks_for_entities(related_entity_ids)

        # Exclude already retrieved chunks
        additional_chunk_ids = related_chunk_ids - initial_chunk_ids

        # Limit additional chunks
        additional_chunk_ids = list(additional_chunk_ids)[:max_additional_chunks]

        logger.info(f"Adding {len(additional_chunk_ids)} graph-enhanced chunks")

        return list(initial_chunk_ids) + additional_chunk_ids
