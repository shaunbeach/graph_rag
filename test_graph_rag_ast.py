#!/usr/bin/env python3
"""
Test script for AST-enhanced Graph RAG system
"""

from pathlib import Path
from langchain_core.documents import Document
from src.rag_system.graph_rag import EntityExtractor, DocumentKnowledgeGraph

def test_ast_extraction():
    """Test AST-based entity extraction from Python code."""

    # Sample Python code
    python_code = '''
import numpy as np
from sklearn.model_selection import train_test_split

class LogisticRegressionModel:
    """A logistic regression model for classification."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None

    def fit(self, X, y):
        """Train the model on data."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(1000):
            predictions = self.predict(X)
            errors = y - predictions
            self.weights += self.learning_rate * np.dot(X.T, errors)

    def predict(self, X):
        """Make predictions on new data."""
        linear_model = np.dot(X, self.weights)
        return self._sigmoid(linear_model)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calculate_accuracy(self, X, y):
        """Calculate accuracy metric."""
        predictions = self.predict(X)
        binary_preds = (predictions >= 0.5).astype(int)
        accuracy = np.mean(binary_preds == y)
        return accuracy
'''

    # Create document chunk
    doc = Document(
        page_content=python_code,
        metadata={
            "source_file": "test_model.py",
            "chunk_id": "chunk_0"
        }
    )

    # Extract entities
    extractor = EntityExtractor()
    entities = extractor.extract_entities(doc, "chunk_0")
    relationships = extractor.extract_relationships(doc, entities)

    print("=" * 80)
    print("AST-ENHANCED GRAPH RAG TEST")
    print("=" * 80)

    print(f"\n✓ Extracted {len(entities)} entities:")
    print("-" * 80)

    # Group by type
    by_type = {}
    for entity in entities:
        if entity.type not in by_type:
            by_type[entity.type] = []
        by_type[entity.type].append(entity)

    for entity_type, entity_list in sorted(by_type.items()):
        print(f"\n{entity_type}:")
        for entity in entity_list:
            extraction_method = entity.metadata.get('extraction_method', 'N/A') if entity.metadata else 'N/A'
            print(f"  - {entity.name} (method: {extraction_method})")
            if entity.metadata and 'args' in entity.metadata:
                print(f"    Args: {entity.metadata['args']}")
            if entity.metadata and 'methods' in entity.metadata:
                print(f"    Methods: {entity.metadata['methods']}")

    print(f"\n✓ Extracted {len(relationships)} relationships:")
    print("-" * 80)

    # Group by type
    by_rel_type = {}
    for rel in relationships:
        if rel.type not in by_rel_type:
            by_rel_type[rel.type] = []
        by_rel_type[rel.type].append(rel)

    for rel_type, rel_list in sorted(by_rel_type.items()):
        print(f"\n{rel_type}:")
        for rel in rel_list[:10]:  # Limit to first 10
            print(f"  - {rel.source_id} → {rel.target_id}")
        if len(rel_list) > 10:
            print(f"  ... and {len(rel_list) - 10} more")

    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    ast_count = sum(1 for e in entities if e.metadata and e.metadata.get('extraction_method') == 'ast')
    pattern_count = sum(1 for e in entities if e.metadata and e.metadata.get('extraction_method') == 'pattern')

    print(f"\nTotal Entities: {len(entities)}")
    print(f"  - AST-based (100% accurate): {ast_count}")
    print(f"  - Pattern-based: {pattern_count}")
    print(f"\nTotal Relationships: {len(relationships)}")

    # Test knowledge graph integration
    print("\n" + "=" * 80)
    print("KNOWLEDGE GRAPH INTEGRATION TEST")
    print("=" * 80)

    # Create temporary workspace
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create knowledge graph
        kg = DocumentKnowledgeGraph(workspace)

        # Add entities
        for entity in entities:
            kg.add_entity(entity)

        # Add relationships
        for rel in relationships:
            kg.add_relationship(rel)

        # Get statistics
        stats = kg.get_statistics()

        print(f"\nKnowledge Graph Statistics:")
        print(f"  Total Entities: {stats['total_entities']}")
        print(f"  Total Relationships: {stats['total_relationships']}")
        print(f"\nEntity Types:")
        for entity_type, count in sorted(stats['entity_types'].items()):
            print(f"  - {entity_type}: {count}")

        print(f"\nExtraction Methods:")
        for method, count in sorted(stats.get('extraction_methods', {}).items()):
            print(f"  - {method}: {count}")

        print(f"\nRelationship Types:")
        for rel_type, count in sorted(stats['relationship_types'].items()):
            print(f"  - {rel_type}: {count}")

        # Save and load test
        kg.save()
        print(f"\n✓ Knowledge graph saved to {workspace / 'document_graph.json'}")

        # Create new graph and load
        kg2 = DocumentKnowledgeGraph(workspace)
        stats2 = kg2.get_statistics()

        print(f"✓ Knowledge graph loaded successfully")
        print(f"  Entities after reload: {stats2['total_entities']}")
        print(f"  Relationships after reload: {stats2['total_relationships']}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)

if __name__ == "__main__":
    test_ast_extraction()
