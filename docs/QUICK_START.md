# Quick Start Guide

Get up and running with RAG System in under 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Install
pip install -e .

# Verify installation
./test_installation.sh
```

## Choose Your Interface

RAG System offers two interfaces:

### Option 1: Web Dashboard (Recommended for Beginners)

```bash
# Launch the Streamlit dashboard
cd dashboard
./start_dashboard.sh
```

Open **http://localhost:8501** in your browser and enjoy the graphical interface!

Features:
- 🎨 Visual document upload and ingestion
- 💬 Interactive chat interface
- 📊 System status and analytics
- 🔧 Easy LLM provider configuration
- 📈 Knowledge graph visualization

### Option 2: Command Line (For Power Users)

Continue with the CLI quick start below.

## CLI Quick Start

### Step 1: Create Sample Documents (30 seconds)

```bash
# Create a test folder
mkdir -p ~/test_rag_docs

# Add some sample documents
cat > ~/test_rag_docs/ml_basics.md << 'EOF'
# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Common types include supervised learning, unsupervised learning, and reinforcement learning.
EOF

cat > ~/test_rag_docs/python_tips.md << 'EOF'
# Python Programming Tips

Python is a versatile language known for its readability.
Use list comprehensions for cleaner code: [x*2 for x in range(10)]
EOF
```

### Step 2: Ingest Documents (1 minute)

```bash
# Ingest the documents
rag ingest ~/test_rag_docs/
```

Expected output:
```
╭─ RAG Document Ingestion ─╮
│                           │
╰───────────────────────────╯

Processing 2 files...
Processing: ml_basics.md
Processing: python_tips.md
Creating embeddings for 4 chunks...

✓ Ingestion completed in 15.2s
  Files processed: 2
  Chunks created: 4
  Knowledge graph: 8 entities, 13 relationships
```

### Step 3: Query Your Documents (10 seconds)

```bash
# Ask questions
rag query "what is machine learning?"
```

Expected output:
```
╭─ Result 1/5 ─────────────────╮
│ ml_basics.md                 │
│                              │
│ Machine learning is a subset │
│ of artificial intelligence...│
╰──────────────────────────────╯
```

### Step 4: Check Status (10 seconds)

```bash
rag status
```

Expected output:
```
╭─ RAG System Status ─╮
│                      │
╰──────────────────────╯

Database Status
─────────────────
Vector Store    ✓    0.5 MB
Text Chunks     ✓    4 documents
Graph Entities  ✓    8 entities
Graph Relations ✓    13 relationships

Recent Ingestions
─────────────────
2025-12-30 14:30  test_rag_docs  *  2  15.2s
```

## Setting Up LLM Providers

The system supports multiple LLM providers:

### Ollama (Local, Free, No API Key)

```bash
# Install Ollama
brew install ollama  # macOS
# or visit https://ollama.ai

# Start service
ollama serve

# Pull a model
ollama pull llama3.1
ollama pull qwen2.5

# Test detection
./test_ollama_detection.py
```

### Cloud Providers (Requires API Keys)

Create a `.env` file in the project root:

```bash
# Choose one or more providers
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
OPENROUTER_API_KEY=your-key-here
```

Or use the dashboard to configure API keys visually (Settings → API Keys → Save).

## Common Commands

```bash
# Ingest with filtering
rag ingest ~/Documents/ --pattern "*.md" --recursive

# Ingest Jupyter notebooks
rag ingest ~/Projects/ --pattern "*.ipynb" --recursive

# Query with more results
rag query "your question" --top-k 10

# View history
rag history

# Remove duplicates
rag deduplicate --dry-run

# System info
rag info

# Use different workspace
rag ingest ~/docs/ --workspace ~/my_custom_rag
rag query "test" --workspace ~/my_custom_rag
```

## Advanced Features

### Graph RAG (Knowledge Graph)

Graph RAG is automatically enabled during ingestion:

```bash
# Ingest Python code - extracts classes, functions, relationships
rag ingest ~/Projects/myapp/src/ --pattern "*.py" --recursive

# Query leverages both vector search AND graph traversal
rag query "how does the authentication system work?"
```

The system extracts:
- **Code entities**: Classes, functions, methods, imports (via AST parsing - 100% accurate)
- **Concepts**: ML terms, metrics, statistical concepts (via pattern matching)
- **Relationships**: Function calls, inheritance, implementations

### Using the Web Dashboard

```bash
cd dashboard
./start_dashboard.sh
```

Dashboard features:
- **Chat Tab**: Ask questions with visual results
  - Select LLM provider (Ollama, OpenAI, Anthropic, Google, OpenRouter)
  - Dynamic model selection (automatically detects available models)
  - Graph RAG enhancement toggle
  
- **Ingest Tab**: Upload files or ingest directories
  - Drag-and-drop file upload
  - Directory path with pattern filtering
  - Progress tracking
  
- **Analytics Tab**: View ingestion history and statistics

- **System Status Tab**: Check system health
  - Database statistics
  - Knowledge graph metrics (entities, relationships, extraction methods)
  - Ollama status and available models
  - File tracking information

## Next Steps

- Read [Installation Guide](INSTALLATION.md) for advanced setup
- See [Metadata Configuration](METADATA.md) for custom metadata extraction
- Check out the feature documentation:
  - [OLLAMA_MODEL_SELECTION.md](../OLLAMA_MODEL_SELECTION.md) - Dynamic model detection
  - [GRAPH_RAG_ENHANCEMENTS.md](../GRAPH_RAG_ENHANCEMENTS.md) - Knowledge graph details
- Try with your own documents!

## Troubleshooting

**No results found?**
```bash
rag status  # Check if documents were ingested
rag history # See ingestion log
```

**Import errors?**
```bash
pip install -r requirements.txt --upgrade
```

**Slow first query?**
- First query downloads embedding model (~400MB)
- Subsequent queries are much faster

**Ollama not detected in dashboard?**
```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags

# Test
./test_ollama_detection.py
```

**Dashboard won't start?**
```bash
# Install Streamlit dependencies
cd dashboard
pip install -r streamlit_app/requirements.txt

# Try again
./start_dashboard.sh
```

## Tips

1. **Start small** - Test with a few files first
2. **Use patterns** - Filter files with `--pattern`
3. **Go recursive** - Use `--recursive` for subdirectories
4. **Check status** - Use `rag status` to verify
5. **Be specific** - More specific queries get better results
6. **Try Graph RAG** - Especially powerful for code understanding
7. **Use the dashboard** - Visual interface is easier for beginners

---

**Ready to use RAG System!** 🚀

For more details, see the full [README](../README.md)
