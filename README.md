# RAG System - Standalone Retrieval-Augmented Generation

A flexible, production-ready RAG (Retrieval-Augmented Generation) system for document ingestion and semantic search. Perfect for class notes, research papers, code documentation, or any searchable knowledge base.

## Key Features

### Core Capabilities
- **Two Interfaces** - Web dashboard (Streamlit) or command-line (CLI)
- **Flexible Document Ingestion** - Process documents from any folder
- **Multiple File Formats** - Markdown, PDF, HTML, text, Jupyter notebooks
- **Smart Filtering** - Pattern-based file filtering (`*.py`, `*.md`, `*.ipynb`, etc.)
- **Incremental Updates** - Only re-processes modified files
- **Deduplication** - Remove duplicate documents by content or source
- **Semantic Search** - Query documents using natural language

### Advanced Features
- **Graph RAG** - Knowledge graph with AST parsing for code understanding
- **Multi-Provider LLM** - Supports Ollama, OpenAI, Anthropic, Google Gemini, OpenRouter
- **Dynamic Model Selection** - Automatically detects available models
- **Modern Web Dashboard** - Streamlit-based UI with dark theme
- **Analytics** - Complete ingestion history and database statistics
- **GPU Acceleration** - Automatic GPU detection for fast embeddings

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Install the package
pip install -e .

# Verify installation
./test_installation.sh
```

### Choose Your Interface

#### Option 1: Web Dashboard (Recommended for Beginners)

```bash
# Launch Streamlit dashboard
cd dashboard
./start_dashboard.sh
```

Open **http://localhost:8501** in your browser.

**Dashboard Features:**
- Interactive chat with multiple LLM providers
- Drag-and-drop document upload
- System status and knowledge graph visualization
- Easy API key configuration
- Ingestion analytics

#### Option 2: Command Line (For Power Users)

```bash
# Ingest documents
rag ingest ~/Documents/notes/ --pattern "*.md" --recursive

# Query your documents
rag query "what is machine learning?"

# Check status
rag status
```

## CLI Documentation

### Available Commands

```bash
rag ingest [PATH]          # Ingest documents into the system
rag query "question"       # Search using natural language
rag status                 # Show database statistics
rag history                # View ingestion history
rag deduplicate            # Remove duplicate documents
rag info                   # System information
```

### `rag ingest [PATH]`

Ingest documents into the RAG system.

**Options:**
- `--pattern TEXT` - File pattern to match (default: `*`)
- `--recursive` - Process subdirectories recursively
- `--force-rebuild` - Re-ingest all files, even if unchanged
- `--workspace PATH` - Custom workspace directory

**Examples:**

```bash
# Ingest all files from a folder
rag ingest ~/Documents/notes/

# Ingest only Markdown files recursively
rag ingest ~/Documents/ --pattern "*.md" --recursive

# Ingest Python source code (enables Graph RAG)
rag ingest ~/Projects/myapp/src/ --pattern "*.py" --recursive

# Ingest Jupyter notebooks
rag ingest ~/Research/ --pattern "*.ipynb" --recursive

# Force rebuild of entire database
rag ingest ~/Documents/notes/ --force-rebuild
```

### `rag query "your question"`

Search the document database using natural language.

**Options:**
- `--top-k INTEGER` - Number of results to return (default: 5)
- `--workspace PATH` - Custom workspace directory

**Examples:**

```bash
# Basic query
rag query "what is machine learning?"

# Get more results
rag query "python best practices" --top-k 10

# Query specific workspace
rag query "API design" --workspace ~/work_rag
```

### Other Commands

```bash
# Show database status
rag status

# View ingestion history
rag history --limit 10

# Remove duplicate documents
rag deduplicate --by-content --dry-run

# System information
rag info
```

## Graph RAG (Knowledge Graph)

Graph RAG automatically builds a knowledge graph during ingestion:

### What It Does

- **Extracts Code Structure** - Classes, functions, methods, imports (via AST parsing - 100% accurate)
- **Finds Relationships** - Function calls, inheritance, implementations
- **Identifies Concepts** - ML terms, metrics, statistical concepts
- **Connects Ideas** - How concepts and code relate to each other

### How to Use

```bash
# Ingest Python code - Graph RAG extracts relationships
rag ingest ~/Projects/myapp/src/ --pattern "*.py" --recursive

# Query leverages both vector search AND graph traversal
rag query "how does the authentication system work?"
```

**Learn More:** See [docs/WHAT_IS_GRAPH_RAG.md](docs/WHAT_IS_GRAPH_RAG.md) for a beginner-friendly explanation.

## Multi-Provider LLM Support

The system supports 5 LLM providers:

### Local (No API Key)
- **Ollama** - Run models locally (llama3.1, qwen2.5, mistral, etc.)

### Cloud (Requires API Key)
- **OpenAI** - GPT-4, GPT-3.5 Turbo
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Opus
- **Google** - Gemini 1.5 Pro, Gemini Flash
- **OpenRouter** - Access to multiple models via proxy

### Setup

**For Ollama (Free, Local):**
```bash
# Install Ollama
brew install ollama  # macOS
# or visit https://ollama.ai

# Start service
ollama serve

# Pull a model
ollama pull llama3.1

# Test
./test_ollama_detection.py
```

**For Cloud Providers (API Keys):**

Create `.env` file in project root:
```bash
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
OPENROUTER_API_KEY=your-key-here
```

Or configure via the dashboard: Settings → API Keys → Save

**Dynamic Model Detection:** The system automatically detects available models for each provider—no hardcoded lists!

## Use Cases

### 1. Study Notes for Exams

```bash
# Ingest all your class notes
rag ingest ~/Documents/Classes/CS101/ --pattern "*.md" --recursive

# Query when studying
rag query "explain quicksort algorithm"
rag query "what are the key concepts in lecture 5?"
```

### 2. Research Paper Library

```bash
# Ingest research papers
rag ingest ~/Papers/ML/ --pattern "*.pdf" --recursive

# Find relevant papers
rag query "attention mechanism in transformers"
rag query "reinforcement learning applications"
```

### 3. Code Documentation & Understanding

```bash
# Ingest project code and docs
rag ingest ~/Projects/myapp/src/ --pattern "*.py" --recursive
rag ingest ~/Projects/myapp/docs/ --pattern "*.md" --recursive

# Search for implementation details
rag query "how does authentication work?"
rag query "API endpoint for user registration"

# Graph RAG shows relationships between functions
```

### 4. Jupyter Notebook Analysis

```bash
# Ingest notebooks
rag ingest ~/Research/ --pattern "*.ipynb" --recursive

# Query code and markdown cells
rag query "data preprocessing steps"
rag query "model training approach"
```

## File Structure

```
~/.rag_system/              # Default workspace (configurable)
├── vector_store/           # ChromaDB vector database
├── file_index.json         # Tracks processed files
├── ingestion_log.json      # Complete ingestion history
├── document_graph.json     # Knowledge graph (entities & relationships)
└── image_store/            # Extracted PDF images (if enabled)
```

## Configuration

### Custom Workspace

```bash
# Set via environment variable
export RAG_WORKSPACE=~/my_custom_rag

# Or use --workspace flag
rag ingest ~/Documents/ --workspace ~/my_custom_rag
rag query "test" --workspace ~/my_custom_rag
```

### Environment Variables

- `RAG_WORKSPACE` - Custom workspace directory (default: `~/.rag_system/` or `./chroma_db/`)
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GOOGLE_API_KEY` - Google Gemini API key
- `OPENROUTER_API_KEY` - OpenRouter API key

### Custom Metadata Extraction

Create `.rag_metadata.json` to extract custom fields from filenames/paths:

```json
{
  "enabled": true,
  "rules": [
    {
      "field": "project",
      "type": "path_contains",
      "patterns": [
        {"match": "ProjectA", "value": "Project A"}
      ]
    }
  ]
}
```

See [docs/METADATA.md](docs/METADATA.md) for details.

## 🔧 Advanced Usage

### Incremental Updates

The system automatically tracks file changes:

```bash
# First ingestion
rag ingest ~/Documents/notes/

# Later, only new/modified files are processed
rag ingest ~/Documents/notes/
```

### Force Rebuild

```bash
rag ingest ~/Documents/notes/ --force-rebuild
```

### Multiple Workspaces

```bash
# Work notes
rag ingest ~/Work/docs/ --workspace ~/work_rag

# Personal notes
rag ingest ~/Personal/notes/ --workspace ~/personal_rag

# Query specific workspace
rag query "project timeline" --workspace ~/work_rag
```

## Web Dashboard Features

### Chat Tab
- Select LLM provider (Ollama, OpenAI, Anthropic, Google, OpenRouter)
- Dynamic model selection (auto-detects available models)
- Graph RAG enhancement toggle
- Interactive Q&A with context sources

### Ingest Tab
- Drag-and-drop file upload
- Directory ingestion with pattern filtering
- Progress tracking
- Supports all file types

### Analytics Tab
- Ingestion history with timestamps
- File counts and chunk statistics
- Visual charts and metrics

### System Status Tab
- Database statistics
- Knowledge graph metrics (entities, relationships, extraction methods)
- Ollama status and available models
- File tracking information

## Troubleshooting

### No results found

```bash
# Check database status
rag status

# Verify documents were ingested
rag history

# Try force rebuild
rag ingest ~/Documents/notes/ --force-rebuild
```

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Or reinstall the package
pip install -e . --force-reinstall
```

### Ollama not detected

```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags

# Test detection
./test_ollama_detection.py
```

### Dashboard won't start

```bash
# Install Streamlit dependencies
cd dashboard
pip install -r streamlit_app/requirements.txt

# Try again
./start_dashboard.sh
```

## Performance

- **Initial Ingestion:** ~3 seconds per file (includes embedding generation)
- **Incremental Updates:** Only processes new/modified files (major speedup)
- **Query Speed:** <1 second after embeddings cached
- **Database Size:** ~1MB per 100 documents (approximate)
- **GPU Support:** Automatic detection and usage for embeddings

## Privacy

All data is stored locally on your machine:
- Vector embeddings: `~/.rag_system/vector_store/`
- File tracking: `~/.rag_system/file_index.json`
- History log: `~/.rag_system/ingestion_log.json`
- Knowledge graph: `~/.rag_system/document_graph.json`

**No data is sent to external servers** (except when using cloud LLM providers with your API keys).

## Complete Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 5 minutes
- **[Installation Guide](docs/INSTALLATION.md)** - Detailed installation instructions
- **[Metadata Configuration](docs/METADATA.md)** - Custom metadata extraction
- **[What is Graph RAG?](docs/WHAT_IS_GRAPH_RAG.md)** - Beginner-friendly explanation
- **[Graph RAG Enhancements](GRAPH_RAG_ENHANCEMENTS.md)** - Technical details
- **[Ollama Model Selection](OLLAMA_MODEL_SELECTION.md)** - Dynamic model detection

## 🧪 Testing

```bash
# Run comprehensive installation test
./test_installation.sh

# Test Ollama detection
./test_ollama_detection.py

# Test Graph RAG functionality
./test_graph_rag_ast.py

# Run pytest suite
pytest
```

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [HuggingFace](https://huggingface.co/) - Embedding models (BGE-small-en-v1.5)
- [Streamlit](https://streamlit.io/) - Web dashboard
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting
- [NetworkX](https://networkx.org/) - Knowledge graph

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/rag-system/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/rag-system/discussions)
- **Documentation:** See `docs/` directory


