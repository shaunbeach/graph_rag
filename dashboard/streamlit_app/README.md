# RAG System - Streamlit Dashboard

A modern, beautiful, and user-friendly interface for the RAG (Retrieval-Augmented Generation) Knowledge Base system.

## 🌟 Features

### 🔍 **Search**
- Semantic search through your knowledge base
- Adjustable number of results
- View source documents and metadata

### 💬 **Chat**
- AI-powered Q&A with your documents
- Support for multiple LLM providers (Ollama, OpenAI, Anthropic)
- Persistent chat history
- Contextual answers with source citations

### 📤 **Document Ingestion**
- **File Upload**: Drag and drop documents directly
- **Directory Path**: Ingest entire folders
- Support for multiple formats: PDF, Markdown, HTML, Jupyter notebooks, Python, JSON, etc.
- Force rebuild option for re-processing

### 📊 **Analytics**
- View ingestion history
- Track processed files and chunks
- Monitor system performance

### ⚙️ **System Status**
- Real-time database statistics
- Configuration overview
- Workspace management
- Maintenance tools

## 🎨 Design Features

- **Modern Gradient UI**: Beautiful purple gradient theme
- **Responsive Layout**: Works on all screen sizes
- **Dark Mode**: Easy on the eyes
- **Smooth Animations**: Polished user experience
- **Card-based Design**: Clean, organized interface

## 🚀 Quick Start

### Launch the Dashboard

```bash
cd dashboard
./start_dashboard.sh
```

Then open your browser to: **http://localhost:8501**

### First Time Setup

1. The script automatically:
   - Creates a Python virtual environment
   - Installs all dependencies
   - Launches the Streamlit app

2. On first launch:
   - Go to **Ingest Documents** tab
   - Upload files or specify a directory
   - Click "Start Ingestion"

3. Start using:
   - **Search**: Find specific information
   - **Chat**: Ask questions and get AI-powered answers

## 📋 Supported File Types

- **Documents**: PDF, Markdown (.md), Text (.txt), HTML
- **Code**: Python (.py), JSON, YAML
- **Notebooks**: Jupyter (.ipynb)

## 🔧 Configuration

### Workspace Location

Change your workspace in the sidebar:
- Default: `current_directory/chroma_db/`
- Custom: Enter any path

### LLM Settings

In the **Chat** tab:
- **Provider**: Ollama (default), OpenAI, or Anthropic
- **Model**: Choose your preferred model
- **Ollama Default**: Qwen3-4B-Instruct-2507:Q4_K_M

## 💡 Tips for Non-Technical Users

### Getting Started
1. **Ingest First**: Always start by adding documents
2. **Search vs Chat**:
   - Use Search for finding specific information
   - Use Chat for questions requiring explanation
3. **Check Analytics**: View what's been processed

### Best Practices
- **File Organization**: Keep related documents together
- **File Names**: Use descriptive names
- **Regular Updates**: Re-ingest when documents change
- **Question Quality**: Be specific in chat questions

### Troubleshooting
- **No Results**: Make sure documents are ingested
- **Slow Responses**: First query loads the model (normal)
- **Chat Errors**: Check that Ollama is running (`ollama serve`)

## 🎯 Use Cases

### 📚 Study Aid
Upload class notes, textbooks, and study materials. Ask questions about concepts, get summaries, and find specific information quickly.

### 📄 Document Management
Organize company documents, manuals, and reports. Search across everything instantly and get AI-powered answers.

### 🔬 Research Assistant
Ingest research papers and articles. Ask about methodologies, findings, and connections between papers.

### 💼 Knowledge Base
Build a company wiki. Upload documentation, procedures, and guides. Employees can search and ask questions.

## 🛠️ Advanced Features

### Workspace Isolation
- Create multiple workspaces for different projects
- Switch between workspaces in the sidebar
- Each workspace has independent databases

### Force Rebuild
- Re-process all files even if unchanged
- Useful after updating embedding models
- Available in the Ingest tab

### Incremental Updates
- Only processes new and changed files
- Tracks file hashes automatically
- Significantly faster for large collections

## 🔒 Privacy

All data stays on your machine:
- Documents stored locally
- Embeddings generated locally
- No external API calls (with Ollama)
- Full control over your data

## 📊 Technical Details

- **Frontend**: Streamlit (Python)
- **Backend**: LangChain + ChromaDB
- **Embeddings**: BAAI/bge-small-en-v1.5
- **LLM**: Configurable (Ollama/OpenAI/Anthropic)
- **Vector Store**: ChromaDB with BM25 hybrid search

## 🆘 Support

For issues or questions:
1. Check the **System Status** tab
2. Review ingestion logs in **Analytics**
3. Ensure all dependencies are installed
4. Verify Ollama is running (for chat features)

---

**Built with ❤️ using Streamlit and LangChain**
