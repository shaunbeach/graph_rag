"""
RAG System - Standalone Retrieval-Augmented Generation

A flexible, production-ready RAG system for document ingestion and querying.
Perfect for class notes, research papers, documentation, and any searchable knowledge base.
"""

__version__ = "1.0.0"
__author__ = "AiForge Team"
__license__ = "MIT"

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent

# Project root (two levels up from this file: src/rag_system/__init__.py -> /)
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Default workspace directory - use chroma_db in project root
DEFAULT_WORKSPACE = PROJECT_ROOT / "chroma_db"
