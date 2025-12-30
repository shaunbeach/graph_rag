# Installation Guide

Complete installation instructions for RAG System.

## Requirements

- Python 3.8 or higher
- pip (Python package manager)
- 1GB free disk space (for embedding models)

## Installation Methods

### Method 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Install in development mode
pip install -e .
```

This installs the package in "editable" mode, allowing you to modify the source code.

### Method 2: Install from PyPI (When Available)

```bash
pip install rag-system
```

### Method 3: Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Install dependencies
pip install -r requirements.txt

# Add to PATH (optional)
export PATH=$PATH:$(pwd)/src
```

## Verify Installation

Run the comprehensive installation test:

```bash
# Run automated test suite
./test_installation.sh
```

Or test manually:

```bash
# Check that rag command is available
rag --version

# Should output: rag, version 1.0.0

# Show help
rag --help
```

## First Run

On first use, the system will download the embedding model (~400MB):

```bash
# This will download the model
rag ingest ~/Documents/test/
```

The model is cached locally and only downloaded once.

## Virtual Environment (Recommended)

Using a virtual environment keeps dependencies isolated:

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install RAG System
cd rag-system
pip install -e .
```

## Platform-Specific Instructions

### macOS

```bash
# Install Python if needed
brew install python3

# Clone and install
git clone https://github.com/yourusername/rag-system.git
cd rag-system
pip3 install -e .
```

### Linux (Ubuntu/Debian)

```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip

# Clone and install
git clone https://github.com/yourusername/rag-system.git
cd rag-system
pip3 install -e .
```

### Windows

```bash
# Install Python from python.org
# Then in PowerShell or CMD:

git clone https://github.com/yourusername/rag-system.git
cd rag-system
pip install -e .
```

## Dependencies

Core dependencies installed automatically:

- **click** - CLI framework
- **rich** - Terminal formatting
- **langchain** - LLM framework
- **langchain-chroma** - Vector database
- **chromadb** - Vector storage
- **sentence-transformers** - Embeddings
- **unstructured** - Document loaders
- **pypdf** - PDF processing
- **python-dotenv** - Environment variable management
- **networkx** - Knowledge graph support
- **streamlit** - Web dashboard UI

### LLM Provider Dependencies

The system supports multiple LLM providers (installed automatically):

- **langchain-ollama** - Local Ollama models
- **langchain-openai** - OpenAI GPT models
- **langchain-anthropic** - Anthropic Claude models
- **langchain-google-genai** - Google Gemini models

All provider dependencies are included in `requirements.txt`.

## Optional Dependencies

For enhanced functionality:

```bash
# Better image extraction from PDFs
pip install pillow pytesseract

# Additional document formats
pip install python-docx python-pptx
```

## Setting Up LLM Providers

### Local Ollama (No API Key Required)

```bash
# Install Ollama
brew install ollama  # macOS
# or visit https://ollama.ai for other platforms

# Start Ollama
ollama serve

# Pull a model
ollama pull llama3.1
ollama pull qwen2.5
```

### Cloud Providers (Requires API Keys)

Create a `.env` file in the project root:

```bash
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Anthropic Claude
ANTHROPIC_API_KEY=your-key-here

# Google Gemini
GOOGLE_API_KEY=your-key-here

# OpenRouter (proxy for multiple providers)
OPENROUTER_API_KEY=your-key-here
```

Or configure via the Streamlit dashboard (see Web Dashboard section below).

## Web Dashboard Installation

The Streamlit dashboard is included and auto-installs dependencies:

```bash
# Launch dashboard
cd dashboard
./start_dashboard.sh
```

The script will:
1. Create virtual environment if needed
2. Install dependencies
3. Start Streamlit on http://localhost:8501

## Troubleshooting

### Issue: Command not found

**Problem:** `rag: command not found`

**Solution:**
```bash
# Reinstall with --force
pip install -e . --force-reinstall

# Or add to PATH
export PATH=$PATH:~/.local/bin
```

### Issue: Import errors

**Problem:** `ModuleNotFoundError: No module named 'langchain'`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: Permission denied

**Problem:** `Permission denied` during installation

**Solution:**
```bash
# Install for user only
pip install -e . --user

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Issue: Slow installation

**Problem:** Installation takes very long

**Solution:**
```bash
# Use faster mirror
pip install -e . --index-url https://pypi.org/simple

# Or install core dependencies first
pip install click rich langchain langchain-chroma chromadb
```

### Issue: Model download fails

**Problem:** Embedding model download fails

**Solution:**
```bash
# Try downloading manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

### Issue: Ollama not detected

**Problem:** Dashboard shows "Ollama not running"

**Solution:**
```bash
# Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags

# Test with CLI
./test_ollama_detection.py
```

## Testing Your Installation

```bash
# Run comprehensive test suite
./test_installation.sh

# Test Ollama detection
./test_ollama_detection.py

# Test Graph RAG functionality
./test_graph_rag_ast.py
```

## Uninstallation

```bash
# Uninstall the package
pip uninstall rag-system

# Remove workspace data (optional)
rm -rf chroma_db/
rm -rf ~/.rag_system
```

## Upgrading

```bash
# Update from git
cd rag-system
git pull
pip install -e . --upgrade

# Or reinstall
pip install -e . --force-reinstall
```

## Docker Installation (Advanced)

Create a Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source
COPY . .
RUN pip install -e .

# Set workspace
ENV RAG_WORKSPACE=/data

CMD ["rag", "--help"]
```

Build and run:

```bash
# Build image
docker build -t rag-system .

# Run
docker run -v ~/my_docs:/docs -v ~/.rag_system:/data rag-system rag ingest /docs
```

## Development Installation

For contributing:

```bash
# Clone with development branch
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e .

# Run tests
pytest
./test_installation.sh

# Check code style
flake8 src/
black src/
```

## Next Steps

After installation:

1. Run `./test_installation.sh` to verify setup
2. Read [Quick Start Guide](QUICK_START.md)
3. Configure API keys in `.env` or via dashboard
4. Try the web dashboard: `cd dashboard && ./start_dashboard.sh`
5. See [Full Documentation](../README.md)

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/rag-system/issues)
- **Documentation:** [README](../README.md)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/rag-system/discussions)
