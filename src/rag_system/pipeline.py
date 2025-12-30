import os
import json
import yaml
import nbformat
import re
import uuid
import io
import argparse
import base64
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import requests

# Import metadata extraction
try:
    from .metadata_config import MetadataExtractor
    HAS_METADATA_EXTRACTOR = True
except ImportError:
    HAS_METADATA_EXTRACTOR = False
    print("Warning: metadata_config not available. Custom metadata extraction disabled.")

# Import Graph RAG
try:
    from .graph_rag import EntityExtractor, DocumentKnowledgeGraph, GraphRAGRetriever
    HAS_GRAPH_RAG = True
except ImportError:
    HAS_GRAPH_RAG = False
    print("Warning: graph_rag not available. Graph RAG features disabled.")

# Image processing imports
try:
    import fitz  # PyMuPDF
    from PIL import Image
    HAS_IMAGE_SUPPORT = True
except ImportError:
    HAS_IMAGE_SUPPORT = False
    print("Warning: PyMuPDF or PIL not installed. Image extraction will be disabled.")

# PDF text extraction with pdfplumber (for better table/layout handling)
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Warning: pdfplumber not installed. PDF text extraction will use basic method (no table support).")

# CLIP embeddings for images
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("Warning: transformers or torch not installed. CLIP embeddings will be disabled.")

# Load environment variables from .env file
load_dotenv()

# Try to import optional dependencies for HTML parsing
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("Warning: BeautifulSoup4 not installed. HTML parsing will use basic text extraction.")

# --- CONFIGURATION ---
# Standalone configuration (no AiForge dependencies)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up to project root (src/rag_system -> src -> project_root)
SOURCE_FOLDER_PATH = os.getenv("SOURCE_FOLDER_PATH", str(PROJECT_ROOT / "documents"))
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", str(PROJECT_ROOT / "chroma_db"))
IMAGE_STORE_PATH = os.getenv("IMAGE_STORE_PATH", str(PROJECT_ROOT / "image_store"))
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", str(PROJECT_ROOT / "docstore"))

# Embedding models
# Using BGE-small for better semantic understanding while maintaining speed
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
IMAGE_EMBEDDING_MODEL_NAME = os.getenv("IMAGE_EMBEDDING_MODEL_NAME", "openai/clip-vit-base-patch32")

# Chunk sizes optimized for different content types
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # For code: smaller chunks for precise retrieval
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Larger chunks for prose-heavy content (articles, documentation)
PROSE_CHUNK_SIZE = int(os.getenv("PROSE_CHUNK_SIZE", "1200"))  # Larger to preserve context
PROSE_CHUNK_OVERLAP = int(os.getenv("PROSE_CHUNK_OVERLAP", "200"))

# Image extraction settings
MIN_IMAGE_SIZE = int(os.getenv("MIN_IMAGE_SIZE", "100"))  # Minimum width/height in pixels
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "95"))  # JPEG quality for saved images

# Vision model for image captioning (optional)
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "llava:7b")  # Ollama vision model
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

# Initialize metadata extractor (global instance)
METADATA_EXTRACTOR = MetadataExtractor() if HAS_METADATA_EXTRACTOR else None

# File tracking for incremental ingestion
FILE_TRACKING_DB = os.getenv("FILE_TRACKING_DB", str(PROJECT_ROOT / "chroma_db" / "file_tracking.json"))
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", str(PROJECT_ROOT / "chroma_db" / "bm25_index.pkl"))

EMBEDDING_KWARGS = {"prompt_name": "Retrieval-document"}

# File extensions to process
SUPPORTED_EXTENSIONS = {
    '.pdf', '.ipynb', '.html', '.htm', '.txt', '.md',
    '.py', '.json', '.yaml', '.yml', '.rst', '.jsonl',
    '.jpeg', '.jpg', '.png', '.bmp', '.gif', '.tiff', '.webp'
}

# Image file extensions
IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

# --- HELPER FUNCTIONS ---

def enrich_metadata(base_metadata: Dict[str, Any], filepath: str) -> Dict[str, Any]:
    """
    Enrich base metadata with custom metadata from MetadataExtractor.

    Args:
        base_metadata: Base metadata dict with source, file_type, etc.
        filepath: Path to the file being processed

    Returns:
        Enriched metadata dict
    """
    if METADATA_EXTRACTOR:
        try:
            custom_metadata = METADATA_EXTRACTOR.extract_metadata(Path(filepath))
            # Merge custom metadata, base metadata takes precedence for conflicts
            enriched = {**custom_metadata, **base_metadata}
            return enriched
        except Exception as e:
            print(f"Warning: Failed to extract custom metadata from {filepath}: {e}")
            return base_metadata
    return base_metadata

def extract_text_from_html(filepath: str) -> str:
    """Extract text from HTML file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if HAS_BS4:
        soup = BeautifulSoup(content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n')
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        return text
    else:
        # Basic fallback: just return raw content
        return content


def extract_html_structures(filepath: str) -> List[Document]:
    """
    Extract HTML with STRUCTURE-AWARE CHUNKING.
    Chunks by semantic sections (article, section, div with headings, etc.)
    """
    filename = os.path.basename(filepath)
    documents = []

    if not HAS_BS4:
        # Fallback to simple text extraction if BeautifulSoup not available
        text = extract_text_from_html(filepath)
        context_header = f"[Source: {filename}]\n\n"
        metadata = {"source": filename, "file_type": "html"}
        return [Document(page_content=context_header + text, metadata=metadata)]

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Strategy: Find semantic sections
    # Priority: article > section > div with heading > main > body
    semantic_sections = []

    # Try to find article tags (common in modern HTML)
    articles = soup.find_all('article')
    if articles:
        for idx, article in enumerate(articles):
            semantic_sections.append(('article', idx, article))

    # Try to find section tags
    sections = soup.find_all('section')
    if sections:
        for idx, section in enumerate(sections):
            semantic_sections.append(('section', idx, section))

    # If no semantic tags, try div with headings
    if not semantic_sections:
        # Find divs that contain headings (h1-h6)
        for idx, div in enumerate(soup.find_all('div')):
            heading = div.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if heading:
                semantic_sections.append(('div_with_heading', idx, div))

    # If still nothing, just chunk by main or body
    if not semantic_sections:
        main = soup.find('main')
        if main:
            semantic_sections.append(('main', 0, main))
        else:
            body = soup.find('body')
            if body:
                semantic_sections.append(('body', 0, body))

    # Extract text from each section
    for tag_type, idx, element in semantic_sections:
        # Get the heading if available
        heading = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        heading_text = heading.get_text(strip=True) if heading else f"{tag_type}_{idx}"

        # Extract text from the section
        text = element.get_text(separator='\n', strip=True)

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        clean_text = '\n'.join(line for line in lines if line)

        if clean_text:
            context_header = f"[Source: {filename}, Section: {heading_text}]\n\n"
            content_with_context = context_header + clean_text

            metadata = {
                "source": filename,
                "file_type": "html",
                "structure_type": tag_type,
                "section_index": idx,
                "heading": heading_text
            }

            documents.append(Document(page_content=content_with_context, metadata=metadata))

    # If no documents were created, fall back to whole-page extraction
    if not documents:
        text = soup.get_text(separator='\n', strip=True)
        lines = (line.strip() for line in text.splitlines())
        clean_text = '\n'.join(line for line in lines if line)

        context_header = f"[Source: {filename}]\n\n"
        metadata = {"source": filename, "file_type": "html", "structure_type": "full_page"}
        documents.append(Document(page_content=context_header + clean_text, metadata=metadata))

    return documents


def extract_text_from_notebook(filepath: str) -> Tuple[List[Document], int]:
    """
    Extract text from Jupyter notebook with SECTION-AWARE CHUNKING.

    CRITICAL: Keeps problem sections together (question + code + output).
    Instead of splitting by cells, this groups cells into logical sections based on
    markdown headers. This ensures that a problem, its solution code, and outputs
    all stay in the same chunk - perfect for study-focused RAG.

    Strategy:
    - Detect markdown cells with headers (## Problem 1, ### Exercise 2.1, etc.)
    - These headers mark section boundaries
    - Group all cells (markdown, code, outputs) between headers into one Document
    - Each Document = one complete problem/section with question + code + output

    Returns:
        Tuple of (list of Document objects, total_cell_count)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    documents = []
    filename = os.path.basename(filepath)
    total_cells = len(nb.cells)

    # Section tracking
    current_section_content = []  # Accumulates cells for current section
    current_header = None  # Title of current section (e.g., "Problem 1")
    current_header_level = 0  # Header level (1-6)
    section_start_cell = 0  # First cell index in current section

    def save_current_section():
        """Save accumulated section as a Document."""
        if not current_section_content:
            return

        # Combine all cells in section
        section_text = "\n\n---\n\n".join(current_section_content)

        if section_text.strip():
            # Create context header with section information
            context_header = f"[Source: {filename}"
            if current_header:
                context_header += f", Section: {current_header}"
            else:
                context_header += f", Cell {section_start_cell + 1}"
            context_header += "]\n\n"

            # Base metadata
            metadata = {
                "source": filename,
                "file_type": "jupyter_notebook",
                "structure_type": "section",
                "section_header": current_header or f"Cell {section_start_cell + 1}",
                "header_level": current_header_level,
                "start_cell": section_start_cell,
                "num_cells": len(current_section_content),
                "total_cells": total_cells
            }

            # Enrich with custom metadata
            metadata = enrich_metadata(metadata, filepath)

            documents.append(Document(
                page_content=context_header + section_text,
                metadata=metadata
            ))

    # Process cells
    for cell_idx, cell in enumerate(nb.cells):
        cell_text = ""

        # --- MARKDOWN CELLS: Check for headers (section boundaries) ---
        if cell.cell_type == 'markdown':
            # Check if this markdown cell contains a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', cell.source.strip(), re.MULTILINE)

            if header_match:
                # Found a new section header!
                # First, save the previous section
                save_current_section()

                # Start new section
                current_header_level = len(header_match.group(1))
                current_header = header_match.group(2).strip()
                current_section_content = []
                section_start_cell = cell_idx

            # Add markdown content to current section
            cell_text = cell.source

        # --- CODE CELLS: Add code + output ---
        elif cell.cell_type == 'code':
            cell_text = f"```python\n{cell.source}\n```"

            # --- CAPTURE CELL OUTPUTS (CRITICAL FOR ANALYSIS) ---
            # For data science notebooks, outputs contain key information:
            # - DataFrame displays (df.head(), df.describe())
            # - Model metrics (accuracy scores, loss values)
            # - Print statements and debugging output
            output_text = ""
            if hasattr(cell, 'outputs') and cell.outputs:
                for output in cell.outputs:
                    if output.output_type == 'stream':
                        # Capture print() statements and stdout
                        if hasattr(output, 'text'):
                            output_text += output.text
                    elif output.output_type == 'execute_result':
                        # Capture the result of the last expression
                        if hasattr(output, 'data') and 'text/plain' in output.data:
                            output_text += output.data['text/plain'] + "\n"
                    elif output.output_type == 'display_data':
                        # Capture display() outputs
                        if hasattr(output, 'data') and 'text/plain' in output.data:
                            output_text += output.data['text/plain'] + "\n"

            if output_text.strip():
                # Format output clearly
                cell_text += f"\n\n[CELL OUTPUT]:\n{output_text.strip()}"
            # --- END OF OUTPUT CAPTURE ---

        else:
            # Skip other cell types
            continue

        # Add cell to current section
        if cell_text.strip():
            current_section_content.append(cell_text)

    # Save final section
    save_current_section()

    # If no sections were created (no headers found), fall back to cell-by-cell
    # This handles notebooks without clear section structure
    if not documents:
        print(f"    Note: No section headers found in {filename}, using cell-by-cell chunking")
        for cell_idx, cell in enumerate(nb.cells):
            cell_text = ""

            if cell.cell_type == 'markdown':
                cell_text = cell.source
            elif cell.cell_type == 'code':
                cell_text = f"```python\n{cell.source}\n```"

                output_text = ""
                if hasattr(cell, 'outputs') and cell.outputs:
                    for output in cell.outputs:
                        if output.output_type == 'stream':
                            if hasattr(output, 'text'):
                                output_text += output.text
                        elif output.output_type == 'execute_result':
                            if hasattr(output, 'data') and 'text/plain' in output.data:
                                output_text += output.data['text/plain'] + "\n"
                        elif output.output_type == 'display_data':
                            if hasattr(output, 'data') and 'text/plain' in output.data:
                                output_text += output.data['text/plain'] + "\n"

                if output_text.strip():
                    cell_text += f"\n\n[CELL OUTPUT]:\n{output_text.strip()}"
            else:
                continue

            if cell_text.strip():
                context_header = f"[Source: {filename}, Cell {cell_idx + 1}]\n\n"
                metadata = {
                    "source": filename,
                    "file_type": "jupyter_notebook",
                    "structure_type": "cell",
                    "cell_index": cell_idx,
                    "total_cells": total_cells
                }
                documents.append(Document(
                    page_content=context_header + cell_text,
                    metadata=metadata
                ))

    return documents, total_cells


def extract_text_from_text_file(filepath: str) -> str:
    """Extract text from plain text files (.txt, .md, .py, .rst)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def extract_python_structures(filepath: str) -> List[Document]:
    """
    Extract Python code with STRUCTURE-AWARE CHUNKING.
    Chunks by functions, classes, and top-level code blocks.
    """
    import ast

    filename = os.path.basename(filepath)
    documents = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Parse the AST
        tree = ast.parse(source_code)
        lines = source_code.split('\n')

        # Track which lines have been processed
        processed_lines = set()

        for node in ast.walk(tree):
            # Extract functions
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno - 1
                end_line = node.end_lineno

                if start_line not in processed_lines:
                    # Get function code including decorators
                    function_lines = lines[start_line:end_line]
                    function_code = '\n'.join(function_lines)

                    # Add context header
                    context_header = f"[Source: {filename}, Function: {node.name}]\n\n"
                    code_with_context = context_header + function_code

                    metadata = {
                        "source": filename,
                        "file_type": "python_code",
                        "structure_type": "function",
                        "function_name": node.name,
                        "start_line": start_line + 1,
                        "end_line": end_line
                    }

                    documents.append(Document(page_content=code_with_context, metadata=metadata))
                    processed_lines.update(range(start_line, end_line))

            # Extract classes (with their methods)
            elif isinstance(node, ast.ClassDef) and node.col_offset == 0:  # Top-level classes only
                start_line = node.lineno - 1
                end_line = node.end_lineno

                if start_line not in processed_lines:
                    class_lines = lines[start_line:end_line]
                    class_code = '\n'.join(class_lines)

                    context_header = f"[Source: {filename}, Class: {node.name}]\n\n"
                    code_with_context = context_header + class_code

                    # Extract method names
                    method_names = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]

                    metadata = {
                        "source": filename,
                        "file_type": "python_code",
                        "structure_type": "class",
                        "class_name": node.name,
                        "methods": ','.join(method_names),
                        "start_line": start_line + 1,
                        "end_line": end_line
                    }

                    documents.append(Document(page_content=code_with_context, metadata=metadata))
                    processed_lines.update(range(start_line, end_line))

        # Collect remaining top-level code (imports, constants, etc.)
        remaining_lines = []
        current_block_start = None

        for i, line in enumerate(lines):
            if i not in processed_lines and line.strip():
                if current_block_start is None:
                    current_block_start = i
                remaining_lines.append((i, line))
            elif remaining_lines and (i in processed_lines or not line.strip()):
                # End of a block of unprocessed code
                if len(remaining_lines) > 0:
                    block_code = '\n'.join([l[1] for l in remaining_lines])

                    context_header = f"[Source: {filename}, Top-level code, Line {remaining_lines[0][0] + 1}]\n\n"
                    code_with_context = context_header + block_code

                    metadata = {
                        "source": filename,
                        "file_type": "python_code",
                        "structure_type": "top_level",
                        "start_line": remaining_lines[0][0] + 1,
                        "end_line": remaining_lines[-1][0] + 1
                    }

                    documents.append(Document(page_content=code_with_context, metadata=metadata))

                remaining_lines = []
                current_block_start = None

        # Add any remaining code at the end
        if remaining_lines:
            block_code = '\n'.join([l[1] for l in remaining_lines])
            context_header = f"[Source: {filename}, Top-level code, Line {remaining_lines[0][0] + 1}]\n\n"
            code_with_context = context_header + block_code

            metadata = {
                "source": filename,
                "file_type": "python_code",
                "structure_type": "top_level",
                "start_line": remaining_lines[0][0] + 1,
                "end_line": remaining_lines[-1][0] + 1
            }

            documents.append(Document(page_content=code_with_context, metadata=metadata))

        return documents

    except SyntaxError as e:
        # If AST parsing fails, fall back to simple text extraction
        print(f"  Warning: Could not parse Python AST for {filename}: {e}")
        print(f"  Falling back to simple text extraction")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        context_header = f"[Source: {filename}]\n\n"
        metadata = {
            "source": filename,
            "file_type": "python_code",
            "structure_type": "unparsed"
        }
        return [Document(page_content=context_header + content, metadata=metadata)]


def extract_markdown_sections(filepath: str) -> List[Document]:
    """
    Extract Markdown with STRUCTURE-AWARE CHUNKING.
    Chunks by sections (headers).
    """
    filename = os.path.basename(filepath)
    documents = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    current_section = []
    current_header = None
    current_level = 0

    for line in lines:
        # Check if this is a header
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

        if header_match:
            # Save previous section if it exists
            if current_section:
                section_text = '\n'.join(current_section)
                if section_text.strip():
                    context_header = f"[Source: {filename}"
                    if current_header:
                        context_header += f", Section: {current_header}"
                    context_header += "]\n\n"

                    metadata = {
                        "source": filename,
                        "file_type": "markdown",
                        "structure_type": "section",
                        "header": current_header or "Introduction",
                        "header_level": current_level
                    }

                    documents.append(Document(
                        page_content=context_header + section_text,
                        metadata=metadata
                    ))

            # Start new section
            current_level = len(header_match.group(1))
            current_header = header_match.group(2).strip()
            current_section = [line]
        else:
            current_section.append(line)

    # Add final section
    if current_section:
        section_text = '\n'.join(current_section)
        if section_text.strip():
            context_header = f"[Source: {filename}"
            if current_header:
                context_header += f", Section: {current_header}"
            context_header += "]\n\n"

            metadata = {
                "source": filename,
                "file_type": "markdown",
                "structure_type": "section",
                "header": current_header or "Introduction",
                "header_level": current_level
            }

            documents.append(Document(
                page_content=context_header + section_text,
                metadata=metadata
            ))

    return documents if documents else [Document(
        page_content=f"[Source: {filename}]\n\n{content}",
        metadata={"source": filename, "file_type": "markdown"}
    )]


def extract_text_from_json(filepath: str) -> str:
    """Extract and format JSON content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Format JSON nicely for better readability in RAG
    formatted = json.dumps(data, indent=2)
    return f"JSON Content:\n\n{formatted}"


def extract_json_structures(filepath: str) -> List[Document]:
    """
    Extract JSON with STRUCTURE-AWARE CHUNKING.
    Chunks by top-level keys for objects or batches for arrays.
    """
    filename = os.path.basename(filepath)
    documents = []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle JSON objects (dictionaries)
    if isinstance(data, dict):
        for key, value in data.items():
            # Create a chunk for each top-level key
            chunk_data = {key: value}
            formatted = json.dumps(chunk_data, indent=2)

            context_header = f"[Source: {filename}, Key: {key}]\n\n"
            content = context_header + formatted

            metadata = {
                "source": filename,
                "file_type": "json",
                "structure_type": "object_key",
                "key": key,
                "value_type": type(value).__name__
            }

            documents.append(Document(page_content=content, metadata=metadata))

    # Handle JSON arrays
    elif isinstance(data, list):
        batch_size = 10  # Chunk arrays into batches of 10 items
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            formatted = json.dumps(batch, indent=2)

            context_header = f"[Source: {filename}, Items {i}-{min(i + batch_size - 1, len(data) - 1)}]\n\n"
            content = context_header + formatted

            metadata = {
                "source": filename,
                "file_type": "json",
                "structure_type": "array_batch",
                "start_index": i,
                "end_index": min(i + batch_size - 1, len(data) - 1),
                "batch_size": len(batch)
            }

            documents.append(Document(page_content=content, metadata=metadata))

    # Handle other JSON primitives (rare but possible)
    else:
        formatted = json.dumps(data, indent=2)
        context_header = f"[Source: {filename}]\n\n"
        content = context_header + formatted

        metadata = {
            "source": filename,
            "file_type": "json",
            "structure_type": "primitive"
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def extract_text_from_yaml(filepath: str) -> str:
    """Extract and format YAML content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Convert to JSON-like format for readability
    formatted = yaml.dump(data, default_flow_style=False, sort_keys=False)
    return f"YAML Content:\n\n{formatted}"


def extract_yaml_structures(filepath: str) -> List[Document]:
    """
    Extract YAML with STRUCTURE-AWARE CHUNKING.
    Chunks by top-level keys for mappings or batches for sequences.
    """
    filename = os.path.basename(filepath)
    documents = []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Handle YAML mappings (dictionaries)
    if isinstance(data, dict):
        for key, value in data.items():
            # Create a chunk for each top-level key
            chunk_data = {key: value}
            formatted = yaml.dump(chunk_data, default_flow_style=False, sort_keys=False)

            context_header = f"[Source: {filename}, Key: {key}]\n\n"
            content = context_header + formatted

            metadata = {
                "source": filename,
                "file_type": "yaml",
                "structure_type": "mapping_key",
                "key": key,
                "value_type": type(value).__name__
            }

            documents.append(Document(page_content=content, metadata=metadata))

    # Handle YAML sequences (lists)
    elif isinstance(data, list):
        batch_size = 10
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            formatted = yaml.dump(batch, default_flow_style=False, sort_keys=False)

            context_header = f"[Source: {filename}, Items {i}-{min(i + batch_size - 1, len(data) - 1)}]\n\n"
            content = context_header + formatted

            metadata = {
                "source": filename,
                "file_type": "yaml",
                "structure_type": "sequence_batch",
                "start_index": i,
                "end_index": min(i + batch_size - 1, len(data) - 1),
                "batch_size": len(batch)
            }

            documents.append(Document(page_content=content, metadata=metadata))

    # Handle other YAML primitives
    else:
        formatted = yaml.dump(data, default_flow_style=False, sort_keys=False)
        context_header = f"[Source: {filename}]\n\n"
        content = context_header + formatted

        metadata = {
            "source": filename,
            "file_type": "yaml",
            "structure_type": "primitive"
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def detect_code_language(text: str) -> List[str]:
    """Detect programming languages in code blocks."""
    languages = set()

    # Extract code block language tags
    code_block_langs = re.findall(r'```(\w+)', text)
    languages.update(code_block_langs)

    # Detect by keywords/patterns
    if 'import pandas' in text or 'import numpy' in text or 'def ' in text:
        languages.add('python')
    if 'SELECT' in text.upper() and 'FROM' in text.upper():
        languages.add('sql')
    if ('const ' in text or 'let ' in text or 'function' in text) and 'javascript' not in languages:
        languages.add('javascript')
    if 'public class' in text or 'public static void' in text:
        languages.add('java')

    return list(languages) if languages else ['general']


def generate_image_caption(image_path: str, vision_model: str = VISION_MODEL_NAME) -> Optional[str]:
    """
    Generate a descriptive caption for an image using Ollama's vision model.

    Args:
        image_path: Path to the image file
        vision_model: Name of the Ollama vision model to use (e.g., 'llava:7b', 'bakllava')

    Returns:
        Generated caption string, or None if captioning fails
    """
    try:
        # Read and encode image as base64
        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')

        # Prepare request to Ollama API
        prompt = (
            "Describe this image in detail. Focus on:\n"
            "- Main subjects and objects\n"
            "- Colors and visual characteristics\n"
            "- Actions or activities depicted\n"
            "- Setting or background\n"
            "- Any text or symbols visible\n\n"
            "Provide a clear, concise description (2-3 sentences)."
        )

        payload = {
            "model": vision_model,
            "prompt": prompt,
            "images": [image_data],
            "stream": False
        }

        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            caption = result.get('response', '').strip()
            return caption if caption else None
        else:
            print(f"  Warning: Vision model returned status {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"  Warning: Could not connect to Ollama at {OLLAMA_API_URL}")
        print(f"  Tip: Start Ollama with 'ollama serve' and pull vision model: 'ollama pull {vision_model}'")
        return None
    except Exception as e:
        print(f"  Warning: Failed to generate caption: {e}")
        return None


class CLIPEmbeddings:
    """
    Custom CLIP embeddings class for image and text embeddings.
    Uses HuggingFace transformers CLIP model.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        if not HAS_CLIP:
            raise ImportError("CLIP support requires transformers and torch. Install with: pip install transformers torch")

        self.model_name = model_name

        # Check for MPS (Apple Silicon GPU) first, then CUDA, then fall back to CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("🚀 Apple Silicon GPU detected - using Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 NVIDIA GPU detected - using CUDA")
        else:
            self.device = "cpu"
            print("⚠️  Using CPU (slow) - GPU not available")

        print(f"Loading CLIP model '{model_name}' on {self.device}...")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

        print(f"✓ CLIP model loaded successfully on {self.device.upper()}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed text documents using CLIP text encoder."""
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_documents([text])[0]

    def embed_images(self, image_paths: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed images from file paths using CLIP vision encoder in batches.

        Args:
            image_paths: List of image file paths to embed
            batch_size: Number of images to process simultaneously (default: 32)
                       Higher values use more GPU memory but are faster

        Returns:
            List of normalized 512-dimensional embedding vectors
        """
        all_embeddings = []
        num_images = len(image_paths)

        if num_images == 0:
            return all_embeddings

        print(f"  Embedding {num_images} images in batches of {batch_size}...")

        with torch.no_grad():
            for i in range(0, num_images, batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []

                # Load images for the current batch
                for image_path in batch_paths:
                    try:
                        image = Image.open(image_path).convert("RGB")
                        batch_images.append(image)
                    except Exception as e:
                        print(f"  Warning: Failed to load image {image_path}: {e}")
                        # Create blank placeholder image
                        batch_images.append(Image.new("RGB", (224, 224), "white"))

                if not batch_images:
                    continue

                try:
                    # Process the entire batch at once - this is where GPU shines!
                    inputs = self.processor(
                        images=batch_images,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)

                    # Get features for the whole batch in one forward pass
                    image_features = self.model.get_image_features(**inputs)

                    # Normalize embeddings (L2 normalization for cosine similarity)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    # Convert to list and add to results
                    all_embeddings.extend(image_features.cpu().numpy().tolist())

                except Exception as e:
                    print(f"  Error processing batch {i // batch_size + 1}: {e}")
                    # Fallback: add zero vectors for failed batch
                    all_embeddings.extend([[0.0] * 512] * len(batch_paths))

                # Progress indicator every 10 batches (e.g., every 320 images with batch_size=32)
                if (i // batch_size) % 10 == 0 and i > 0:
                    print(f"    ... processed {min(i + batch_size, num_images)} / {num_images} images")

        print(f"  ✓ Finished embedding {len(all_embeddings)} images")
        return all_embeddings


def create_standalone_image_document(
    filepath: str,
    filename: str,
    image_store_path: str,
    use_captioning: bool = False
) -> Document:
    """
    Create a Document for a standalone image file.

    Args:
        filepath: Path to the image file
        filename: Name of the file
        image_store_path: Directory to save/copy the image
        use_captioning: If True, generate AI caption using vision model

    Returns:
        Document object for the image, or None if processing fails
    """
    if not HAS_IMAGE_SUPPORT:
        print(f"  Skipping {filename} (image support not available)")
        return None

    try:
        # Open image to validate and get metadata
        image = Image.open(filepath)
        width, height = image.size
        image_format = image.format or 'unknown'

        # Filter out small images (likely thumbnails or icons)
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            print(f"  Skipping small image: {filename} ({width}x{height}px)")
            return None

        # Generate unique ID and target path
        image_uuid = str(uuid.uuid4())
        file_ext = Path(filepath).suffix.lower()
        new_filename = f"{image_uuid}{file_ext}"
        saved_image_path = os.path.join(image_store_path, new_filename)

        # Convert to RGB if needed for consistency
        if image.mode in ("RGBA", "LA", "P"):
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            if image.mode in ("RGBA", "LA"):
                rgb_image.paste(image, mask=image.split()[-1])
            else:
                rgb_image.paste(image)
            image = rgb_image

        # Save with appropriate quality
        if file_ext in ['.jpg', '.jpeg']:
            image.save(saved_image_path, "JPEG", quality=IMAGE_QUALITY)
        elif file_ext == '.png':
            image.save(saved_image_path, "PNG", optimize=True)
        elif file_ext == '.webp':
            image.save(saved_image_path, "WEBP", quality=IMAGE_QUALITY)
        else:
            # For other formats, save as JPEG
            image.save(saved_image_path, "JPEG", quality=IMAGE_QUALITY)

        # Create metadata
        image_metadata = {
            "source": filename,
            "file_type": "image",
            "image_path": saved_image_path,
            "image_id": image_uuid,
            "width": width,
            "height": height,
            "format": image_format,
            "standalone": True  # Flag to indicate this is a standalone file
        }

        # Create descriptive content for text-based context
        if use_captioning:
            # Generate AI caption using vision model
            caption = generate_image_caption(saved_image_path)
            if caption:
                page_content = f"Image description: {caption}"
                print(f"  ✓ Generated caption: {caption[:80]}...")
            else:
                # Fallback to filename if captioning fails
                name_without_ext = Path(filename).stem
                page_content = f"Standalone image file: {filename}\n"
                page_content += f"Filename context: {name_without_ext.replace('_', ' ').replace('-', ' ')}"
        else:
            # Use filename-based context (fast mode)
            name_without_ext = Path(filename).stem
            page_content = f"Standalone image file: {filename}\n"
            page_content += f"Filename context: {name_without_ext.replace('_', ' ').replace('-', ' ')}"

        return Document(page_content=page_content, metadata=image_metadata)

    except Exception as e:
        print(f"  Warning: Failed to process standalone image {filename}: {e}")
        return None


def process_pdf_and_extract_images(
    filepath: str,
    filename: str,
    image_store_path: str,
    use_captioning: bool = False
) -> Tuple[List[Document], List[Document]]:
    """
    Extract text and images from a PDF file.
    - Uses pdfplumber (if available) for high-quality text and table extraction
    - Uses PyMuPDF/fitz for fast image extraction
    - Gracefully falls back to basic text extraction if pdfplumber unavailable

    Args:
        filepath: Path to the PDF file
        filename: Name of the file
        image_store_path: Directory to save extracted images
        use_captioning: If True, generate AI captions for images

    Returns:
        Tuple of (text_documents, image_documents)
    """
    text_docs = []
    image_docs = []

    # --- PART 1: Extract Text (with pdfplumber for tables/layout) ---
    if HAS_PDFPLUMBER:
        # Use pdfplumber for smart text/table extraction
        try:
            with pdfplumber.open(filepath) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text with layout awareness (x_tolerance helps with columns)
                    page_text = page.extract_text(x_tolerance=2) or ""

                    # Extract tables and format them clearly
                    tables = page.extract_tables()
                    if tables:
                        table_text = ""
                        for table_index, table in enumerate(tables):
                            # Convert table to markdown-like format for better readability
                            # Filter out None values and convert to strings
                            table_text += f"\n\n[TABLE {table_index + 1}]:\n"
                            for row in table:
                                cleaned_row = [str(cell) if cell is not None else "" for cell in row]
                                table_text += " | ".join(cleaned_row) + "\n"
                            table_text += "\n"

                        page_text += table_text

                    # Create document for this page
                    if page_text and page_text.strip():
                        # Add filename and page context
                        context_header = f"[Source: {filename}, Page {page_num}]\n\n"
                        page_text_with_context = context_header + page_text

                        text_metadata = {
                            "source": filename,
                            "file_type": "pdf_text",
                            "page_number": page_num,
                            "total_pages": total_pages,
                            "has_tables": bool(tables)
                        }
                        text_docs.append(Document(page_content=page_text_with_context, metadata=text_metadata))

        except Exception as e:
            print(f"  Warning: pdfplumber failed for {filename}, falling back to basic extraction: {e}")
            HAS_PDFPLUMBER_TEMP = False
        else:
            HAS_PDFPLUMBER_TEMP = True
    else:
        HAS_PDFPLUMBER_TEMP = False

    # Fallback to basic text extraction if pdfplumber not available or failed
    if not HAS_PDFPLUMBER_TEMP:
        if not HAS_IMAGE_SUPPORT:
            print(f"  Skipping {filename} (no PDF libraries available)")
            return text_docs, image_docs

        try:
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text and text.strip():
                    # Add filename and page context (consistent with pdfplumber path)
                    context_header = f"[Source: {filename}, Page {page_num + 1}]\n\n"
                    text_with_context = context_header + text

                    text_metadata = {
                        "source": filename,
                        "file_type": "pdf_text",
                        "page_number": page_num + 1,
                        "total_pages": len(doc)
                    }
                    text_docs.append(Document(page_content=text_with_context, metadata=text_metadata))
            doc.close()
        except Exception as e:
            print(f"  Error extracting text from {filename}: {e}")

    # --- PART 2: Extract Images (always use fitz/PyMuPDF - it's excellent for images) ---
    if not HAS_IMAGE_SUPPORT:
        return text_docs, image_docs

    try:
        doc = fitz.open(filepath)

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Open image to check dimensions
                    image = Image.open(io.BytesIO(image_bytes))
                    width, height = image.size

                    # Filter out small images (likely icons, logos, etc.)
                    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                        continue

                    # Generate unique filename
                    image_uuid = str(uuid.uuid4())
                    image_filename = f"{image_uuid}.{image_ext}"
                    saved_image_path = os.path.join(image_store_path, image_filename)

                    # Convert to RGB if needed and save
                    if image.mode in ("RGBA", "LA", "P"):
                        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                        if image.mode == "P":
                            image = image.convert("RGBA")
                        rgb_image.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
                        image = rgb_image

                    # Save with high quality
                    if image_ext.lower() in ['jpg', 'jpeg']:
                        image.save(saved_image_path, "JPEG", quality=IMAGE_QUALITY)
                    else:
                        image.save(saved_image_path)

                    # Create document for the image with rich metadata
                    image_metadata = {
                        "source": filename,
                        "file_type": "image",
                        "page_number": page_num + 1,
                        "image_index": img_index,
                        "image_path": saved_image_path,
                        "image_id": image_uuid,
                        "width": width,
                        "height": height,
                        "format": image_ext
                    }

                    # Page content includes context for text-based search
                    if use_captioning:
                        caption = generate_image_caption(saved_image_path)
                        if caption:
                            page_content = f"Image description: {caption}"
                            page_content += f"\nSource: {filename}, page {page_num + 1}"
                        else:
                            page_content = f"Image from {filename}, page {page_num + 1}"
                    else:
                        page_content = f"Image from {filename}, page {page_num + 1}"

                    image_docs.append(Document(page_content=page_content, metadata=image_metadata))

                except Exception as img_error:
                    print(f"  Warning: Failed to extract image {img_index} from page {page_num + 1}: {img_error}")
                    continue

        doc.close()

    except Exception as e:
        print(f"  Error processing PDF images {filename}: {e}")

    return text_docs, image_docs


def process_jsonl_conversations(filepath: str) -> List[Document]:
    """
    Process JSONL file containing conversational Q&A data.
    Each line is a complete conversation that gets split intelligently.
    """
    documents = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                # Extract messages
                messages = data.get('messages', [])
                if not messages:
                    continue

                # Build conversation text
                user_query = ""
                assistant_response = ""

                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')

                    if role == 'user':
                        user_query = content
                    elif role == 'assistant':
                        assistant_response = content

                if not user_query or not assistant_response:
                    continue

                # Detect programming languages in the response
                languages = detect_code_language(assistant_response)

                # Create a structured format that preserves context
                conversation_text = f"# Question:\n{user_query}\n\n# Answer:\n{assistant_response}"

                # Create document with rich metadata
                doc = Document(
                    page_content=conversation_text,
                    metadata={
                        "source": os.path.basename(filepath),
                        "file_type": "code_conversation",
                        "line_number": line_num,
                        "languages": ",".join(languages),
                        "query": user_query[:200],  # First 200 chars for reference
                    }
                )

                documents.append(doc)

            except json.JSONDecodeError as e:
                print(f"  Warning: Invalid JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"  Error processing line {line_num}: {e}")
                continue

    return documents


def process_file(filepath: str, filename: str) -> List[Document]:
    """
    Process a single file and return LangChain Document(s) with STRUCTURE-AWARE CHUNKING.

    This function now intelligently chunks files based on their natural structure:
    - Jupyter notebooks: chunked by cells
    - Python files: chunked by functions, classes, and top-level blocks
    - Markdown: chunked by sections (headers)
    - JSON/YAML: chunked by top-level keys or array batches
    - HTML: chunked by semantic sections (article, section, etc.)
    - Plain text: falls back to traditional text extraction
    """
    ext = Path(filepath).suffix.lower()

    try:
        # PDFs are handled separately with image extraction in process_all_files
        if ext == '.pdf':
            return []

        # JSONL files: one document per conversation (already structure-aware)
        elif ext == '.jsonl':
            print(f"  Processing JSONL (Structure-Aware): {filename}")
            return process_jsonl_conversations(filepath)

        # Jupyter notebooks: chunk by sections (STRUCTURE-AWARE)
        elif ext == '.ipynb':
            print(f"  Processing Jupyter Notebook (Section-based Chunking): {filename}")
            docs, cell_count = extract_text_from_notebook(filepath)
            print(f"    ✓ Extracted {len(docs)} sections from {cell_count} total cells")
            return docs

        # HTML: chunk by semantic sections (STRUCTURE-AWARE)
        elif ext in ['.html', '.htm']:
            print(f"  Processing HTML (Section-based Chunking): {filename}")
            docs = extract_html_structures(filepath)
            print(f"    ✓ Extracted {len(docs)} semantic sections")
            return docs

        # Python code: chunk by functions and classes (STRUCTURE-AWARE)
        elif ext == '.py':
            print(f"  Processing Python (Function/Class-based Chunking): {filename}")
            docs = extract_python_structures(filepath)
            print(f"    ✓ Extracted {len(docs)} code structures")
            return docs

        # Markdown: chunk by sections/headers (STRUCTURE-AWARE)
        elif ext == '.md':
            print(f"  Processing Markdown (Section-based Chunking): {filename}")
            docs = extract_markdown_sections(filepath)
            print(f"    ✓ Extracted {len(docs)} sections")
            return docs

        # Plain text and RST: traditional extraction (no structure)
        elif ext in ['.txt', '.rst']:
            print(f"  Processing Text: {filename}")
            text = extract_text_from_text_file(filepath)
            file_type_map = {'.txt': 'text', '.rst': 'restructuredtext'}

            if text and text.strip():
                context_header = f"[Source: {filename}]\n\n"
                text_with_context = context_header + text
                metadata = {
                    "source": filename,
                    "file_type": file_type_map.get(ext, 'text')
                }
                # Enrich with custom metadata
                metadata = enrich_metadata(metadata, filepath)
                return [Document(page_content=text_with_context, metadata=metadata)]
            return []

        # JSON: chunk by top-level keys (STRUCTURE-AWARE)
        elif ext == '.json':
            print(f"  Processing JSON (Key-based Chunking): {filename}")
            docs = extract_json_structures(filepath)
            print(f"    ✓ Extracted {len(docs)} JSON structures")
            return docs

        # YAML: chunk by top-level keys (STRUCTURE-AWARE)
        elif ext in ['.yaml', '.yml']:
            print(f"  Processing YAML (Key-based Chunking): {filename}")
            docs = extract_yaml_structures(filepath)
            print(f"    ✓ Extracted {len(docs)} YAML structures")
            return docs

        else:
            return []

    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return []


# --- FILE TRACKING FOR INCREMENTAL INGESTION ---

def load_file_tracking() -> Dict[str, Dict[str, Any]]:
    """Load the file tracking database.

    Returns:
        Dictionary mapping filepath -> {mtime, size, hash}
    """
    if os.path.exists(FILE_TRACKING_DB):
        try:
            with open(FILE_TRACKING_DB, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load file tracking database: {e}")
            return {}
    return {}


def save_file_tracking(tracking_data: Dict[str, Dict[str, Any]]):
    """Save the file tracking database."""
    try:
        with open(FILE_TRACKING_DB, 'w') as f:
            json.dump(tracking_data, f, indent=2)
    except Exception as e:
        print(f"  Warning: Could not save file tracking database: {e}")


def get_file_info(filepath: str) -> Dict[str, Any]:
    """Get file metadata for tracking.

    Returns:
        Dictionary with mtime, size, text_ids, image_ids
    """
    stat = os.stat(filepath)
    return {
        'mtime': stat.st_mtime,
        'size': stat.st_size,
        'text_ids': [],    # Will be populated during ingestion
        'image_ids': []    # Will be populated during ingestion
    }


def has_file_changed(filepath: str, tracking_data: Dict[str, Dict[str, Any]]) -> bool:
    """Check if a file has been modified since last ingestion.

    Returns:
        True if file is new or has been modified
    """
    if filepath not in tracking_data:
        return True  # New file

    current_info = get_file_info(filepath)
    stored_info = tracking_data[filepath]

    # Check if file has been modified (compare mtime and size)
    if (current_info['mtime'] != stored_info.get('mtime') or
        current_info['size'] != stored_info.get('size')):
        return True

    return False


def get_deleted_files(folder_path: str, tracking_data: Dict[str, Dict[str, Any]]) -> List[str]:
    """Find files that were tracked but no longer exist.

    Returns:
        List of filepaths that were deleted
    """
    current_files = set()
    for filename in os.listdir(folder_path):
        if filename.startswith('._'):
            continue
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            current_files.add(filepath)

    tracked_files = set(tracking_data.keys())
    deleted_files = tracked_files - current_files

    return list(deleted_files)


def cleanup_deleted_files(
    deleted_files: List[str],
    tracking_data: Dict[str, Dict[str, Any]],
    vectorstore_path: str
) -> Tuple[int, int]:
    """Remove embeddings for deleted files from vector database.

    Args:
        deleted_files: List of deleted file paths
        tracking_data: File tracking data containing document IDs
        vectorstore_path: Path to ChromaDB storage

    Returns:
        Tuple of (text_chunks_deleted, images_deleted)
    """
    if not deleted_files:
        return 0, 0

    import chromadb

    text_deleted = 0
    images_deleted = 0

    try:
        client = chromadb.PersistentClient(path=vectorstore_path)

        # Collect all IDs to delete
        text_ids_to_delete = []
        image_ids_to_delete = []

        for filepath in deleted_files:
            file_data = tracking_data.get(filepath, {})
            text_ids_to_delete.extend(file_data.get('text_ids', []))
            image_ids_to_delete.extend(file_data.get('image_ids', []))

        # Delete from text collection
        if text_ids_to_delete:
            try:
                text_collection = client.get_collection("text_collection")
                text_collection.delete(ids=text_ids_to_delete)
                text_deleted = len(text_ids_to_delete)
                print(f"  ✓ Deleted {text_deleted} text chunks from deleted files")
            except Exception as e:
                print(f"  Warning: Could not delete text chunks: {e}")

        # Delete from image collection
        if image_ids_to_delete:
            try:
                image_collection = client.get_collection("image_collection")
                image_collection.delete(ids=image_ids_to_delete)
                images_deleted = len(image_ids_to_delete)
                print(f"  ✓ Deleted {images_deleted} images from deleted files")
            except Exception as e:
                print(f"  Warning: Could not delete images: {e}")

    except Exception as e:
        print(f"  Warning: Could not access vector database for cleanup: {e}")

    return text_deleted, images_deleted


def process_all_files(
    folder_path: str,
    image_store_path: str,
    use_captioning: bool = False,
    tracking_data: Dict[str, Dict[str, Any]] = None
) -> Tuple[List[Document], List[Document], Dict[str, int], List[str]]:
    """
    Process all supported files in the folder.

    Args:
        folder_path: Path to folder containing documents
        image_store_path: Path to store extracted images
        use_captioning: If True, generate AI captions for images using vision model
        tracking_data: File tracking data to detect changes (optional)

    Returns:
        Tuple of (text_documents, image_documents, file_counts, processed_files)
    """
    all_text_docs = []
    all_image_docs = []
    file_counts = {}
    processed_files = []

    if tracking_data is None:
        tracking_data = {}

    print(f"\nScanning folder: {folder_path}")

    # Create image store directory if it doesn't exist
    os.makedirs(image_store_path, exist_ok=True)

    print("\n--- Processing Files (Incremental - Only New/Modified) ---")

    total_files = 0
    skipped_files = 0

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        # Skip macOS metadata files
        if filename.startswith('._'):
            continue

        if not os.path.isfile(filepath):
            continue

        total_files += 1

        # Check if file has changed since last ingestion
        if not has_file_changed(filepath, tracking_data):
            skipped_files += 1
            continue  # Skip unchanged files

        ext = Path(filepath).suffix.lower()

        # Handle PDFs with image extraction
        if ext == '.pdf':
            print(f"Processing PDF (Text+Images): {filename}")
            text_docs, image_docs = process_pdf_and_extract_images(filepath, filename, image_store_path, use_captioning)
            all_text_docs.extend(text_docs)
            all_image_docs.extend(image_docs)
            file_counts['pdf_text'] = file_counts.get('pdf_text', 0) + len(text_docs)
            file_counts['image'] = file_counts.get('image', 0) + len(image_docs)
            processed_files.append(filepath)
            continue

        # Handle standalone image files
        elif ext in IMAGE_EXTENSIONS:
            print(f"Processing (Standalone Image): {filename}")
            try:
                img_doc = create_standalone_image_document(filepath, filename, image_store_path, use_captioning)
                if img_doc:
                    all_image_docs.append(img_doc)
                    file_counts['standalone_image'] = file_counts.get('standalone_image', 0) + 1
                    processed_files.append(filepath)
            except Exception as e:
                print(f"  Error processing standalone image {filename}: {e}")
            continue

        # Handle other file types
        if ext in SUPPORTED_EXTENSIONS:
            print(f"Processing (Text): {filename}")
            docs = process_file(filepath, filename)
            if docs:
                all_text_docs.extend(docs)
                # Count documents by file type
                for doc in docs:
                    file_type = doc.metadata.get('file_type', 'unknown')
                    file_counts[file_type] = file_counts.get(file_type, 0) + 1
                processed_files.append(filepath)

    # Print summary of skipped files
    if skipped_files > 0:
        print(f"\n✓ Skipped {skipped_files}/{total_files} unchanged files")

    return all_text_docs, all_image_docs, file_counts, processed_files


# --- MAIN SCRIPT ---
def ingest_documents_pipeline(
    source_files: List[str],
    vectorstore_path: str,
    force_rebuild: bool = False,
    caption_images: bool = False
) -> Dict[str, int]:
    """
    Ingest documents using the advanced pipeline (CLI-compatible wrapper).

    Args:
        source_files: List of file paths to ingest
        vectorstore_path: Path to ChromaDB storage
        force_rebuild: If True, rebuild everything from scratch
        caption_images: If True, generate AI captions for images

    Returns:
        Dictionary with statistics:
        - files_processed: Number of files processed
        - chunks_created: Number of text chunks
        - images_extracted: Number of images extracted
    """
    import shutil
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    # Override global paths for this run
    global VECTORSTORE_PATH, IMAGE_STORE_PATH, FILE_TRACKING_DB, BM25_INDEX_PATH
    VECTORSTORE_PATH = vectorstore_path
    IMAGE_STORE_PATH = str(Path(vectorstore_path).parent / "image_store")
    FILE_TRACKING_DB = str(Path(vectorstore_path).parent / "file_tracking.json")
    BM25_INDEX_PATH = str(Path(vectorstore_path) / "bm25_index.pkl")

    # Initialize storage
    if force_rebuild:
        for path in [VECTORSTORE_PATH, IMAGE_STORE_PATH]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        if os.path.exists(FILE_TRACKING_DB):
            os.remove(FILE_TRACKING_DB)
    else:
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        os.makedirs(IMAGE_STORE_PATH, exist_ok=True)

    # Load file tracking
    tracking_data = load_file_tracking()

    # Process each file
    all_text_docs = []
    all_image_docs = []
    file_counts = {}
    processed_files = []

    print(f"Processing {len(source_files)} of {len(source_files)} files...")

    for filepath in source_files:
        if not os.path.exists(filepath):
            continue

        # Skip if unchanged
        if not force_rebuild and not has_file_changed(str(filepath), tracking_data):
            continue

        filename = os.path.basename(filepath)
        ext = Path(filepath).suffix.lower()

        print(f"Processing: {filename}")

        try:
            # Handle PDFs with images
            if ext == '.pdf':
                text_docs, image_docs = process_pdf_and_extract_images(
                    str(filepath), filename, IMAGE_STORE_PATH, caption_images
                )
                all_text_docs.extend(text_docs)
                all_image_docs.extend(image_docs)
                file_counts['pdf'] = file_counts.get('pdf', 0) + 1

            # Handle standalone images
            elif ext in IMAGE_EXTENSIONS:
                img_doc = create_standalone_image_document(
                    str(filepath), filename, IMAGE_STORE_PATH, caption_images
                )
                if img_doc:
                    all_image_docs.append(img_doc)
                    file_counts['image'] = file_counts.get('image', 0) + 1

            # Handle other text files
            elif ext in SUPPORTED_EXTENSIONS:
                docs = process_file(str(filepath), filename)
                if docs:
                    all_text_docs.extend(docs)
                    file_type = docs[0].metadata.get('file_type', 'text')
                    file_counts[file_type] = file_counts.get(file_type, 0) + 1

            processed_files.append(str(filepath))

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Create embeddings and store
    chunks_created = 0
    images_extracted = 0

    if all_text_docs or all_image_docs:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        text_chunks = []
        for doc in all_text_docs:
            chunks = text_splitter.split_documents([doc])
            text_chunks.extend(chunks)

        # Add unique chunk IDs for Graph RAG
        for idx, chunk in enumerate(text_chunks):
            chunk.metadata['chunk_id'] = f"chunk_{idx}"

        chunks_created = len(text_chunks)

        # Build knowledge graph from chunks (if enabled)
        if text_chunks and HAS_GRAPH_RAG:
            print(f"Building knowledge graph from {len(text_chunks)} chunks...")
            try:
                workspace_path = Path(VECTORSTORE_PATH).parent
                knowledge_graph = DocumentKnowledgeGraph(workspace_path)
                entity_extractor = EntityExtractor()

                # Extract entities and relationships
                for chunk in text_chunks:
                    chunk_id = chunk.metadata.get('chunk_id', '')

                    # Extract entities from this chunk
                    entities = entity_extractor.extract_entities(chunk, chunk_id)

                    # Add entities to graph
                    for entity in entities:
                        knowledge_graph.add_entity(entity)

                    # Extract relationships
                    relationships = entity_extractor.extract_relationships(chunk, entities)

                    # Add relationships to graph
                    for relationship in relationships:
                        knowledge_graph.add_relationship(relationship)

                # Save knowledge graph
                knowledge_graph.save()

                # Print statistics
                stats = knowledge_graph.get_statistics()
                print(f"Knowledge Graph: {stats['total_entities']} entities, "
                      f"{stats['total_relationships']} relationships")

            except Exception as e:
                print(f"Warning: Graph RAG processing failed: {e}")

        # Create embeddings
        if text_chunks:
            print(f"Creating embeddings for {len(text_chunks)} chunks...")
            text_embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

            vectorstore_text = Chroma.from_documents(
                documents=text_chunks,
                embedding=text_embedder,
                persist_directory=VECTORSTORE_PATH,
                collection_name="text_collection"
            )

        # Handle images
        if all_image_docs and HAS_CLIP:
            print(f"Creating CLIP embeddings for {len(all_image_docs)} images...")
            image_embedder = CLIPEmbeddings()

            vectorstore_images = Chroma.from_documents(
                documents=all_image_docs,
                embedding=image_embedder,
                persist_directory=VECTORSTORE_PATH,
                collection_name="image_collection"
            )
            images_extracted = len(all_image_docs)

        # Update tracking
        for filepath in processed_files:
            file_info = get_file_info(filepath)
            tracking_data[filepath] = file_info

        save_file_tracking(tracking_data)

        # Build BM25 index
        try:
            from rank_bm25 import BM25Okapi

            text_embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            vectorstore_text = Chroma(
                persist_directory=VECTORSTORE_PATH,
                embedding_function=text_embedder,
                collection_name="text_collection"
            )

            all_docs_data = vectorstore_text.get(include=["documents"])

            if all_docs_data['documents']:
                tokenized_docs = [doc.lower().split() for doc in all_docs_data['documents']]
                bm25 = BM25Okapi(tokenized_docs)

                os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
                with open(BM25_INDEX_PATH, 'wb') as f:
                    pickle.dump(bm25, f)
        except Exception as e:
            print(f"Warning: Could not build BM25 index: {e}")

    return {
        'files_processed': len(processed_files),
        'chunks_created': chunks_created,
        'images_extracted': images_extracted
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Multimodal RAG Document Ingestion with optional AI image captioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update database with new/modified documents (RECOMMENDED - smart & fast)
  python ingest.py
  python ingest.py --update         # Same as above (explicit)

  # Update with AI-generated image captions
  python ingest.py --update --caption-images

  # Force complete rebuild from scratch (deletes existing database)
  python ingest.py --force-rebuild

  # Rebuild with AI captions (best quality, slowest)
  python ingest.py --force-rebuild --caption-images

  # Specify custom vision model
  VISION_MODEL_NAME=llava:13b python ingest.py --caption-images

Requirements for AI captioning:
  - Ollama must be running: ollama serve
  - Vision model installed: ollama pull llava:7b

How it works:
  DEFAULT (no flags):     Updates database with new/modified files only (incremental)
  --update:               Same as default, but more explicit
  --force-rebuild:        Deletes everything and rebuilds from scratch
  --caption-images:       Generates AI descriptions for images (works with both modes)

Note: You do NOT need to delete the database manually. The system automatically
      tracks changes and only processes what's new or modified!
        """
    )

    parser.add_argument(
        '--update',
        action='store_true',
        help='Update database with new/modified files (DEFAULT behavior - use for clarity)'
    )

    parser.add_argument(
        '--caption-images',
        action='store_true',
        help='Generate AI captions for images using Ollama vision model (slower but more accurate)'
    )

    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force complete database rebuild (deletes existing data and reprocesses all files)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Starting Multimodal Document Ingestion")
    if args.caption_images:
        print(f"📸 AI Captioning: ENABLED (using {VISION_MODEL_NAME})")
    else:
        print("📸 AI Captioning: DISABLED (using filename context)")

    if args.force_rebuild:
        print("🔄 Mode: FORCE REBUILD (all data will be deleted)")
    else:
        print("⚡ Mode: UPDATE (only new/modified files)")
        if args.update:
            print("   (--update flag specified for clarity)")
    print("=" * 60)

    # --- 1. Initialize Storage ---
    if args.force_rebuild:
        # Force rebuild: delete everything and start fresh
        print("\n--- Force Rebuild: Cleaning Previous Data ---")
        import shutil
        for path in [VECTORSTORE_PATH, IMAGE_STORE_PATH]:
            if os.path.exists(path):
                print(f"  Removing existing directory: {path}")
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

        # Also remove file tracking database
        if os.path.exists(FILE_TRACKING_DB):
            print(f"  Removing file tracking database: {FILE_TRACKING_DB}")
            os.remove(FILE_TRACKING_DB)

        print("✓ All previous data removed. Starting fresh ingestion...")
    else:
        # Incremental mode: preserve existing data
        print("\n--- Initializing Storage (Incremental Mode) ---")
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        os.makedirs(IMAGE_STORE_PATH, exist_ok=True)
        print(f"✓ Vector store: {VECTORSTORE_PATH}")
        print(f"✓ Image store: {IMAGE_STORE_PATH}")
        print("  (Existing data preserved - only processing new/modified files)")

    # Check if source folder exists
    if not os.path.exists(SOURCE_FOLDER_PATH):
        print(f"\nError: Source folder not found: {SOURCE_FOLDER_PATH}")
        return

    # --- 2. Load File Tracking Database ---
    print("\n--- Loading File Tracking Database ---")
    tracking_data = load_file_tracking()
    if tracking_data:
        print(f"✓ Loaded tracking data for {len(tracking_data)} previously processed files")
    else:
        print("  No previous tracking data found (first run)")

    # Check for deleted files and clean up their data
    deleted_files = get_deleted_files(SOURCE_FOLDER_PATH, tracking_data)
    if deleted_files:
        print(f"\n⚠️  Found {len(deleted_files)} deleted files")
        print("  Cleaning up stale data from vector database...")

        # Remove embeddings from collections
        text_deleted, images_deleted = cleanup_deleted_files(
            deleted_files,
            tracking_data,
            VECTORSTORE_PATH
        )

        # Remove from tracking database
        for filepath in deleted_files:
            del tracking_data[filepath]

        print(f"  ✓ Cleanup complete: {text_deleted} text chunks, {images_deleted} images removed")

    # --- 3. Process All Files (Incremental) ---
    all_text_docs, all_image_docs, file_counts, processed_files = process_all_files(
        SOURCE_FOLDER_PATH,
        IMAGE_STORE_PATH,
        use_captioning=args.caption_images,
        tracking_data=tracking_data
    )

    # Check if there are documents to process
    has_new_documents = all_text_docs or all_image_docs

    if not has_new_documents:
        if processed_files:
            print(f"\nWarning: Processed files but no documents extracted.")
        else:
            print(f"\nNo new or modified files found. Database is up to date!")
        # Don't return early - we'll still build/update BM25 index at the end
    else:
        # Print summary only if we have new documents
        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)
        print(f"Total text documents/pages: {len(all_text_docs)}")
        print(f"Total images extracted: {len(all_image_docs)}")
        print("\nBreakdown by file type:")
        for file_type, count in sorted(file_counts.items()):
            print(f"  - {file_type}: {count}")

    # Only process documents if we have new ones
    if has_new_documents:
        # --- 3. Chunk Text Documents (STRUCTURE-AWARE + Content-Aware) ---
        print("\n--- Processing Text Documents (Structure-Aware + Content-Aware Chunking) ---")

        # Create specialized splitters for files that need traditional chunking
        # (only used for plain text, RST, and PDF pages)

        # Create specialized splitter for PROSE (larger chunks to preserve context)
        # Used for: plain text, RST files, and PDF text pages
        prose_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PROSE_CHUNK_SIZE,
            chunk_overlap=PROSE_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]  # Prefer breaking at paragraphs/sentences
        )

        # Splitter for code conversations (JSONL files)
        markdown_code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        # Generic splitter for other content
        generic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        # Separate structure-aware documents from traditional documents
        text_chunks = []
        structure_aware_types = {
            'jupyter_notebook',  # Pre-chunked by cells
            'python_code',       # Pre-chunked by functions/classes
            'markdown',          # Pre-chunked by sections
            'json',              # Pre-chunked by keys
            'yaml',              # Pre-chunked by keys
            'html'               # Pre-chunked by semantic sections
        }

        structure_aware_count = 0
        traditionally_chunked_count = 0

        for doc in all_text_docs:
            file_type = doc.metadata.get('file_type', '')
            structure_type = doc.metadata.get('structure_type', None)

            # If document is already structure-aware chunked, use as-is
            if file_type in structure_aware_types or structure_type is not None:
                # Document is already optimally chunked - add directly
                text_chunks.append(doc)
                structure_aware_count += 1

            # For code conversations (JSONL), apply markdown splitter
            elif file_type == 'code_conversation':
                doc_chunks = markdown_code_splitter.split_documents([doc])
                text_chunks.extend(doc_chunks)
                traditionally_chunked_count += len(doc_chunks)

            # For PDF text pages and plain text files, use prose splitter
            elif file_type in {'pdf_text', 'text', 'restructuredtext'}:
                doc_chunks = prose_splitter.split_documents([doc])
                text_chunks.extend(doc_chunks)
                traditionally_chunked_count += len(doc_chunks)

            # Fallback to generic splitter for unknown types
            else:
                doc_chunks = generic_splitter.split_documents([doc])
                text_chunks.extend(doc_chunks)
                traditionally_chunked_count += len(doc_chunks)

        print(f"✓ Processed {len(text_chunks)} total chunks:")
        print(f"  - {structure_aware_count} structure-aware chunks (no re-chunking needed)")
        print(f"  - {traditionally_chunked_count} traditionally chunked documents")

    # --- 4. Initialize Embeddings ---
    print("\n--- Initializing Embedding Models ---")

    # Text embedder
    text_embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"prompts": EMBEDDING_KWARGS}
    )
    print(f"✓ Text embedder loaded: {EMBEDDING_MODEL_NAME}")

    # Image embedder (CLIP) - only if we have images and CLIP is available
    image_embedder = None
    if all_image_docs and HAS_CLIP:
        try:
            image_embedder = CLIPEmbeddings(model_name=IMAGE_EMBEDDING_MODEL_NAME)
        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            print("Images will be skipped.")
            all_image_docs = []

    # --- 5. Add Text Chunks to Vector Store (Incremental) ---
    print("\n--- Building Vector Database (Text) ---")

    text_ids_by_file = {}  # Track text chunk IDs per source file

    if text_chunks:
        # Connect to or create text collection
        print("  Connecting to text collection...")
        vectorstore_text = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=text_embedder,
            collection_name="text_collection"
        )

        # Generate explicit IDs for text chunks (for tracking and deletion)
        text_ids = []
        for idx, chunk in enumerate(text_chunks):
            source_file = chunk.metadata.get('source', 'unknown')
            # Create unique ID: source_file + chunk index + timestamp
            chunk_id = f"text_{hash(source_file)}_{idx}_{int(chunk.metadata.get('page_number', 0))}"
            text_ids.append(chunk_id)

            # Track IDs by source file for cleanup
            if source_file not in text_ids_by_file:
                text_ids_by_file[source_file] = []
            text_ids_by_file[source_file].append(chunk_id)

        print(f"  Adding {len(text_chunks)} new text chunks to collection...")
        vectorstore_text.add_documents(documents=text_chunks, ids=text_ids)
        print(f"✓ Successfully added {len(text_chunks)} text chunks")
    else:
        print("  No new text chunks to add")

    # --- 6. Add Image Embeddings to Vector Store ---
    image_ids_by_file = {}  # Track image IDs per source file

    if all_image_docs and image_embedder:
        print("\n--- Building Vector Database (Images) ---")
        print(f"Embedding {len(all_image_docs)} images with CLIP...")

        # Extract image paths
        image_paths = [doc.metadata['image_path'] for doc in all_image_docs]

        # Generate image embeddings
        image_embeddings_list = image_embedder.embed_images(image_paths)

        # Create a separate collection for images or use unified approach
        # Here we'll create documents with the image embeddings
        print("Adding image embeddings to vector store...")

        # For compatibility with older Chroma, we need to access the underlying collection
        # and add embeddings directly
        import chromadb

        # Initialize Chroma client
        client = chromadb.PersistentClient(path=VECTORSTORE_PATH)

        # Get or create the image collection (preserves existing data)
        print("  Connecting to image collection...")
        collection = client.get_or_create_collection(
            name="image_collection",
            metadata={"hnsw:space": "cosine"}  # CLIP uses normalized vectors
        )
        print("  ✓ Image collection ready")

        # Extract metadatas, documents (page_content), and generate IDs
        metadatas = [doc.metadata for doc in all_image_docs]
        documents = [doc.page_content for doc in all_image_docs]  # Include page_content
        ids = [doc.metadata.get("image_id", str(uuid.uuid4())) for doc in all_image_docs]

        # Track image IDs by source file for cleanup
        for doc, img_id in zip(all_image_docs, ids):
            source_file = doc.metadata.get('source', 'unknown')
            if source_file not in image_ids_by_file:
                image_ids_by_file[source_file] = []
            image_ids_by_file[source_file].append(img_id)

        # Add embeddings directly to the collection
        # This ensures CLIP image vectors are stored, not text re-embeddings
        collection.add(
            embeddings=image_embeddings_list,
            metadatas=metadatas,
            documents=documents,  # Add page_content for LangChain compatibility
            ids=ids
        )

        print(f"✓ Added {len(all_image_docs)} image embeddings to vector store (CLIP vectors)")

    elif all_image_docs and not HAS_CLIP:
        print("\n⚠️  Images extracted but CLIP not available - images will not be searchable")
        print("   Install with: pip install transformers torch")

        # --- 7. Update File Tracking Database with IDs ---
        print("\n--- Updating File Tracking Database ---")
        for filepath in processed_files:
            # Get basic file info (mtime, size)
            file_info = get_file_info(filepath)

            # Add document IDs for this file
            # Note: Use just the filename for matching (source metadata contains filename, not full path)
            filename = os.path.basename(filepath)

            file_info['text_ids'] = text_ids_by_file.get(filename, [])
            file_info['image_ids'] = image_ids_by_file.get(filename, [])

            tracking_data[filepath] = file_info

        save_file_tracking(tracking_data)
        print(f"✓ Updated tracking for {len(processed_files)} processed files")
        print(f"  (Tracking {sum(len(ids) for ids in text_ids_by_file.values())} text chunk IDs, "
              f"{sum(len(ids) for ids in image_ids_by_file.values())} image IDs)")

    # --- 8. Build and Save BM25 Index for Hybrid Search ---
    print("\n--- Building and Saving BM25 Index for Hybrid Search ---")
    try:
        from rank_bm25 import BM25Okapi

        # Connect to Chroma to get all documents (necessary for BM25)
        # Note: We only need this to read documents, not to embed anything new
        text_embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
        vectorstore_text = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=text_embedder,
            collection_name="text_collection"
        )

        all_docs_data = vectorstore_text.get(include=["documents"])

        if not all_docs_data['documents']:
            print("  Warning: No documents found in text_collection. Skipping BM25 index creation.")
        else:
            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in all_docs_data['documents']]

            # Build BM25 index
            bm25 = BM25Okapi(tokenized_docs)

            # Save the built index to disk
            with open(BM25_INDEX_PATH, 'wb') as f:
                pickle.dump(bm25, f)

            print(f"✓ BM25 index built and saved to {BM25_INDEX_PATH}")
            print(f"  Index contains {len(tokenized_docs)} documents")

    except ImportError:
        print("  Warning: rank-bm25 not installed. Run: pip install rank-bm25")
        print("  Hybrid search will not be available in GUI.")
    except Exception as e:
        print(f"  Warning: Could not build and save BM25 index: {e}")
        print("  GUI will fall back to building BM25 index on startup.")

    # --- 9. Finalize ---
    print("\n" + "=" * 60)
    print("✓ Multimodal Ingestion Complete!")
    print("=" * 60)
    print(f"Vector database saved to: {VECTORSTORE_PATH}")
    print(f"Extracted images saved to: {IMAGE_STORE_PATH}")
    print(f"\nCollections created:")
    if text_chunks:
        print(f"  - text_collection: {len(text_chunks)} text chunks")
    if all_image_docs and image_embedder:
        print(f"  - image_collection: {len(all_image_docs)} images")
    print("\nYour multimodal RAG knowledge base is ready to use!")
    print("=" * 60)


if __name__ == "__main__":
    main()
