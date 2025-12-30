"""
RAG System - Modern Streamlit Interface

A beautiful, user-friendly interface for the RAG (Retrieval-Augmented Generation) system.
Designed for non-technical users with full RAG capabilities.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import json
import os
import tempfile
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_system.pipeline import ingest_documents_pipeline
from rag_system.llm_handler import (
    LLMHandler,
    get_available_ollama_models,
    get_ollama_model_info,
    get_openai_models,
    get_anthropic_models,
    get_google_models,
    get_openrouter_models
)

# Import Graph RAG (optional)
try:
    from rag_system.graph_rag import DocumentKnowledgeGraph, EntityExtractor, GraphRAGRetriever
    HAS_GRAPH_RAG = True
except ImportError:
    HAS_GRAPH_RAG = False

# Helper function to load ingestion log
def load_ingestion_log(workspace_dir: Path) -> dict:
    """Load the ingestion log from the workspace."""
    log_path = workspace_dir / "ingestion_log.json"
    if not log_path.exists():
        return {"ingestions": []}

    try:
        with open(log_path, 'r') as f:
            return json.load(f)
    except:
        return {"ingestions": []}

def format_bytes(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Force dark theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Dark background */
    .stApp {
        background-color: #0e1117;
    }

    /* Main content area */
    .main {
        background-color: #0e1117;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d29;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0 !important;
    }

    /* Regular text */
    p, div, span, label {
        color: #c9d1d9 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 8px;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #21262d;
        border-radius: 6px;
        padding: 10px 20px;
        color: #8b949e;
        border: 1px solid #30363d;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
        border-color: #6366f1;
    }

    /* Input fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stNumberInput>div>div>input {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        border-radius: 6px !important;
    }

    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }

    /* Select boxes */
    .stSelectbox>div>div>div {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
    }

    /* Radio buttons */
    .stRadio>div {
        background-color: #161b22;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }

    /* Checkboxes */
    .stCheckbox {
        color: #c9d1d9 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        color: #c9d1d9 !important;
    }

    .streamlit-expanderContent {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-top: none !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #161b22;
        border: 2px dashed #30363d;
        border-radius: 8px;
        padding: 20px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
    }

    /* Success/Warning/Error boxes */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        color: #6ee7b7 !important;
    }

    .stWarning {
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        color: #fbbf24 !important;
    }

    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        color: #fca5a5 !important;
    }

    .stInfo {
        background-color: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #6366f1;
        color: #a5b4fc !important;
    }

    /* Code blocks */
    code {
        background-color: #161b22 !important;
        color: #79c0ff !important;
        padding: 2px 6px;
        border-radius: 4px;
    }

    pre {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }

    /* Markdown container */
    [data-testid="stMarkdownContainer"] {
        color: #c9d1d9;
    }

    /* Horizontal rule */
    hr {
        border-color: #30363d !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'workspace' not in st.session_state:
    # Get the project root (two levels up from this script)
    # Use resolve() to get absolute path
    project_root = Path(__file__).resolve().parent.parent.parent
    default_workspace = os.getenv("RAG_WORKSPACE", str(project_root / "chroma_db"))
    st.session_state.workspace = Path(default_workspace).resolve()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar - Workspace Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    workspace_input = st.text_input(
        "Workspace Directory",
        value=str(st.session_state.workspace),
        help="Location where your RAG database is stored"
    )

    if workspace_input != str(st.session_state.workspace):
        st.session_state.workspace = Path(workspace_input)
        st.rerun()

    vectorstore_path = st.session_state.workspace / "vector_store"

    # Debug: Show resolved paths
    with st.expander("🔍 Debug Info", expanded=False):
        st.code(f"Workspace: {st.session_state.workspace}")
        st.code(f"Vector Store: {vectorstore_path}")
        st.code(f"Exists: {vectorstore_path.exists()}")
        st.code(f"Resolved: {vectorstore_path.resolve()}")

    st.markdown("---")

    # API Keys Configuration
    st.markdown("### 🔑 API Keys")

    # Initialize session state for API keys if not exists
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}

    with st.expander("Configure LLM API Keys", expanded=False):
        st.markdown("**Enter your API keys for cloud LLM providers**")

        # Get project root (where .env should be)
        project_root = Path(__file__).resolve().parent.parent.parent
        env_file = project_root / ".env"

        # Load existing .env file
        from dotenv import load_dotenv, set_key, find_dotenv
        load_dotenv(env_file)

        # OpenAI
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.api_keys.get('openai', os.getenv('OPENAI_API_KEY', '')),
            key="sidebar_openai_key",
            help="Enter your OpenAI API key (starts with sk-)"
        )
        if openai_key and openai_key != st.session_state.api_keys.get('openai', ''):
            st.session_state.api_keys['openai'] = openai_key

        # Anthropic
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.api_keys.get('anthropic', os.getenv('ANTHROPIC_API_KEY', '')),
            key="sidebar_anthropic_key",
            help="Enter your Anthropic API key"
        )
        if anthropic_key and anthropic_key != st.session_state.api_keys.get('anthropic', ''):
            st.session_state.api_keys['anthropic'] = anthropic_key

        # Google
        google_key = st.text_input(
            "Google API Key",
            type="password",
            value=st.session_state.api_keys.get('google', os.getenv('GOOGLE_API_KEY', '')),
            key="sidebar_google_key",
            help="Enter your Google API key for Gemini"
        )
        if google_key and google_key != st.session_state.api_keys.get('google', ''):
            st.session_state.api_keys['google'] = google_key

        # OpenRouter
        openrouter_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=st.session_state.api_keys.get('openrouter', os.getenv('OPENROUTER_API_KEY', '')),
            key="sidebar_openrouter_key",
            help="Enter your OpenRouter API key"
        )
        if openrouter_key and openrouter_key != st.session_state.api_keys.get('openrouter', ''):
            st.session_state.api_keys['openrouter'] = openrouter_key

        st.markdown("---")

        # Save button
        if st.button("💾 Save API Keys to .env", key="save_api_keys"):
            saved_count = 0
            try:
                # Ensure .env file exists
                env_file.touch(exist_ok=True)

                # Save each configured key
                if st.session_state.api_keys.get('openai'):
                    set_key(str(env_file), "OPENAI_API_KEY", st.session_state.api_keys['openai'])
                    saved_count += 1

                if st.session_state.api_keys.get('anthropic'):
                    set_key(str(env_file), "ANTHROPIC_API_KEY", st.session_state.api_keys['anthropic'])
                    saved_count += 1

                if st.session_state.api_keys.get('google'):
                    set_key(str(env_file), "GOOGLE_API_KEY", st.session_state.api_keys['google'])
                    saved_count += 1

                if st.session_state.api_keys.get('openrouter'):
                    set_key(str(env_file), "OPENROUTER_API_KEY", st.session_state.api_keys['openrouter'])
                    saved_count += 1

                if saved_count > 0:
                    st.success(f"✅ Saved {saved_count} API key(s) to {env_file}")
                    st.caption("🔒 Keys are saved securely in .env file (make sure it's in .gitignore)")
                else:
                    st.warning("⚠️ No API keys to save. Enter at least one key above.")

            except Exception as e:
                st.error(f"❌ Error saving to .env: {e}")

        st.caption(f"💡 Keys will be saved to: {env_file}")

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")

    # Check if vector store exists
    if vectorstore_path.exists():
        st.success("✅ Database Initialized")

        # Get document count
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(vectorstore_path))
            try:
                collection = client.get_collection("text_collection")
                doc_count = collection.count()
                st.metric("Total Chunks", f"{doc_count:,}")
            except:
                st.warning("No documents yet")
        except Exception as e:
            st.error(f"Error reading database")
    else:
        st.warning("⚠️ No database found")
        st.info("👉 Go to 'Ingest Documents' to get started!")

# Main header
st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0;'>
        🧠 RAG Knowledge Base
    </h1>
    <p style='text-align: center; color: #a0a0a0; margin-top: 0;'>
        Semantic Search & AI-Powered Q&A
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Search",
    "💬 Chat",
    "📤 Ingest Documents",
    "📊 Analytics",
    "⚙️ System Status"
])

# Tab 1: Search
with tab1:
    st.markdown("### 🔍 Semantic Search")
    st.markdown("Search through your knowledge base using natural language.")

    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "Enter your search query",
            placeholder="e.g., What are neural networks?",
            key="search_input"
        )

    with col2:
        num_results = st.number_input(
            "Results",
            min_value=1,
            max_value=20,
            value=5,
            key="num_results"
        )

    if st.button("🔍 Search", key="search_btn"):
        if not search_query:
            st.warning("Please enter a search query")
        elif not vectorstore_path.exists():
            st.error("❌ Vector store not initialized. Please ingest documents first.")
        else:
            with st.spinner("Searching..."):
                try:
                    # Initialize embeddings and vectorstore
                    try:
                        from langchain_chroma import Chroma
                    except ImportError:
                        from langchain_community.vectorstores import Chroma

                    from langchain_huggingface import HuggingFaceEmbeddings

                    embeddings = HuggingFaceEmbeddings(
                        model_name="BAAI/bge-small-en-v1.5"
                    )

                    vectorstore = Chroma(
                        persist_directory=str(vectorstore_path),
                        embedding_function=embeddings,
                        collection_name="text_collection"
                    )

                    # Perform search
                    results = vectorstore.similarity_search(search_query, k=num_results)

                    if results:
                        st.success(f"✅ Found {len(results)} results")

                        for idx, doc in enumerate(results, 1):
                            with st.expander(f"📄 Result {idx}: {doc.metadata.get('source_file', 'Unknown')}"):
                                st.markdown(f"**Content:**")
                                st.markdown(doc.page_content)

                                st.markdown("**Metadata:**")
                                metadata_str = json.dumps(doc.metadata, indent=2)
                                st.code(metadata_str, language="json")
                    else:
                        st.warning("No results found")

                except Exception as e:
                    st.error(f"❌ Search error: {str(e)}")

# Tab 2: Chat
with tab2:
    st.markdown("### 💬 AI-Powered Q&A")
    st.markdown("Ask questions and get AI-generated answers based on your documents.")

    # LLM Provider selection
    col1, col2 = st.columns([1, 1])

    with col1:
        llm_provider = st.selectbox(
            "LLM Provider",
            ["ollama", "openai", "anthropic", "google", "openrouter"],
            key="llm_provider"
        )

    with col2:
        # Handle model selection based on provider
        if llm_provider == "ollama":
            # Get available Ollama models
            available_models = get_available_ollama_models()

            if available_models:
                # Show dropdown with available models
                default_model = "Qwen3-4B-Instruct-2507:Q4_K_M"
                default_index = available_models.index(default_model) if default_model in available_models else 0

                llm_model = st.selectbox(
                    "Model",
                    available_models,
                    index=default_index,
                    key="llm_model_ollama",
                    help=f"{len(available_models)} models available in Ollama"
                )
            else:
                # Ollama not running or no models - show text input as fallback
                st.warning("⚠️ Could not detect Ollama models. Make sure Ollama is running.")
                llm_model = st.text_input(
                    "Model Name",
                    value="Qwen3-4B-Instruct-2507:Q4_K_M",
                    key="llm_model_ollama_manual",
                    help="Enter model name manually (e.g., llama3, mistral, qwen2.5)"
                )

        elif llm_provider == "openai":
            # Check if API key is configured
            api_key = st.session_state.api_keys.get('openai', os.getenv('OPENAI_API_KEY', ''))

            if api_key:
                # Get available models dynamically
                available_models = get_openai_models(api_key)
                if available_models:
                    llm_model = st.selectbox(
                        "Model",
                        available_models,
                        key="llm_model_openai",
                        help=f"{len(available_models)} models available"
                    )
                else:
                    st.warning("⚠️ Could not fetch models. Using fallback list.")
                    llm_model = st.selectbox(
                        "Model",
                        ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                        key="llm_model_openai_fallback"
                    )
            else:
                st.warning("⚠️ OpenAI API key not configured")
                st.info("👈 Configure your API key in the sidebar under 🔑 API Keys")
                llm_model = "gpt-4"

        elif llm_provider == "anthropic":
            # Check if API key is configured
            api_key = st.session_state.api_keys.get('anthropic', os.getenv('ANTHROPIC_API_KEY', ''))

            if api_key:
                # Get available models
                available_models = get_anthropic_models(api_key)
                llm_model = st.selectbox(
                    "Model",
                    available_models,
                    key="llm_model_anthropic",
                    help=f"{len(available_models)} Claude models"
                )
            else:
                st.warning("⚠️ Anthropic API key not configured")
                st.info("👈 Configure your API key in the sidebar under 🔑 API Keys")
                llm_model = "claude-3-5-sonnet-20241022"

        elif llm_provider == "google":
            # Check if API key is configured
            api_key = st.session_state.api_keys.get('google', os.getenv('GOOGLE_API_KEY', ''))

            if api_key:
                # Get available models dynamically
                available_models = get_google_models(api_key)
                if available_models:
                    llm_model = st.selectbox(
                        "Model",
                        available_models,
                        key="llm_model_google",
                        help=f"{len(available_models)} Gemini models available"
                    )
                else:
                    st.warning("⚠️ Could not fetch models. Using fallback list.")
                    llm_model = st.selectbox(
                        "Model",
                        ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-pro"],
                        key="llm_model_google_fallback"
                    )
            else:
                st.warning("⚠️ Google API key not configured")
                st.info("👈 Configure your API key in the sidebar under 🔑 API Keys")
                llm_model = "gemini-1.5-pro-latest"

        else:  # openrouter
            # Check if API key is configured
            api_key = st.session_state.api_keys.get('openrouter', os.getenv('OPENROUTER_API_KEY', ''))

            if api_key:
                # Get available models dynamically
                available_models = get_openrouter_models(api_key)
                if available_models:
                    llm_model = st.selectbox(
                        "Model",
                        available_models,
                        key="llm_model_openrouter",
                        help=f"{len(available_models)} models available via OpenRouter"
                    )
                else:
                    st.warning("⚠️ Could not fetch models. Using fallback list.")
                    llm_model = st.selectbox(
                        "Model",
                        ["openai/gpt-4-turbo", "anthropic/claude-3-5-sonnet", "google/gemini-pro"],
                        key="llm_model_openrouter_fallback"
                    )
            else:
                st.warning("⚠️ OpenRouter API key not configured")
                st.info("👈 Configure your API key in the sidebar under 🔑 API Keys")
                llm_model = "openai/gpt-4-turbo"

    # Chat interface
    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_area(
            "Your Question",
            placeholder="e.g., Explain how neural networks work",
            key="chat_question"
        )

    with col2:
        st.markdown("") # Spacing
        num_docs = st.number_input(
            "Context Docs",
            min_value=3,
            max_value=20,
            value=10,
            help="Number of document chunks to retrieve for context",
            key="num_context_docs"
        )

        # Graph RAG toggle (if available)
        if HAS_GRAPH_RAG:
            use_graph_rag = st.checkbox(
                "Use Graph RAG",
                value=True,
                help="Enhance retrieval using knowledge graph",
                key="use_graph_rag"
            )
        else:
            use_graph_rag = False

    if st.button("🤖 Ask", key="ask_btn"):
        if not question:
            st.warning("Please enter a question")
        elif not vectorstore_path.exists():
            st.error("❌ Vector store not initialized. Please ingest documents first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Search for relevant documents
                    try:
                        from langchain_chroma import Chroma
                    except ImportError:
                        from langchain_community.vectorstores import Chroma

                    from langchain_huggingface import HuggingFaceEmbeddings

                    embeddings = HuggingFaceEmbeddings(
                        model_name="BAAI/bge-small-en-v1.5"
                    )

                    vectorstore = Chroma(
                        persist_directory=str(vectorstore_path),
                        embedding_function=embeddings,
                        collection_name="text_collection"
                    )

                    # Initial vector search
                    initial_docs = vectorstore.similarity_search(question, k=num_docs)

                    if not initial_docs:
                        st.warning("No relevant documents found")
                    else:
                        # Enhance with Graph RAG if enabled
                        final_docs = initial_docs
                        graph_enhanced = False

                        if use_graph_rag and HAS_GRAPH_RAG:
                            try:
                                # Load knowledge graph
                                knowledge_graph = DocumentKnowledgeGraph(st.session_state.workspace)
                                entity_extractor = EntityExtractor()
                                retriever = GraphRAGRetriever(knowledge_graph, entity_extractor)

                                # Get enhanced chunk IDs
                                enhanced_chunk_ids = retriever.enhance_retrieval(
                                    query=question,
                                    initial_docs=initial_docs,
                                    max_additional_chunks=5,
                                    graph_depth=2
                                )

                                # Retrieve additional chunks
                                if len(enhanced_chunk_ids) > len(initial_docs):
                                    # Get all chunks including graph-enhanced ones
                                    collection = vectorstore._collection
                                    all_results = collection.get(
                                        where={"chunk_id": {"$in": enhanced_chunk_ids}}
                                    )

                                    # Reconstruct documents
                                    from langchain.schema import Document as LCDocument
                                    final_docs = []
                                    for i, doc_id in enumerate(all_results['ids']):
                                        final_docs.append(LCDocument(
                                            page_content=all_results['documents'][i],
                                            metadata=all_results['metadatas'][i]
                                        ))

                                    graph_enhanced = True
                                    st.info(f"🔗 Graph RAG: Enhanced with {len(final_docs) - len(initial_docs)} related chunks")

                            except Exception as e:
                                st.warning(f"Graph RAG enhancement failed: {e}")
                                final_docs = initial_docs

                        # Generate answer with LLM
                        # Set API keys in environment for non-Ollama providers
                        if llm_provider != "ollama":
                            provider_key_map = {
                                "openai": "OPENAI_API_KEY",
                                "anthropic": "ANTHROPIC_API_KEY",
                                "google": "GOOGLE_API_KEY",
                                "openrouter": "OPENROUTER_API_KEY"
                            }
                            env_var = provider_key_map.get(llm_provider)
                            if env_var and llm_provider in st.session_state.api_keys:
                                os.environ[env_var] = st.session_state.api_keys[llm_provider]

                        llm_handler = LLMHandler(provider=llm_provider, model=llm_model)
                        result = llm_handler.generate_answer(question, final_docs)

                        # Display answer
                        st.markdown("### 💡 Answer")
                        st.markdown(result['answer'])

                        # Display sources
                        st.markdown("### 📚 Sources")
                        for source in result['sources']:
                            with st.expander(f"Source {source['number']}: {source['file']}"):
                                st.markdown(f"**Section:** {source['section']}")
                                st.markdown(f"**Type:** {source['type']}")
                                st.markdown(f"**Preview:**")
                                st.text(source['preview'])

                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': result['answer'],
                            'timestamp': datetime.now().isoformat()
                        })

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Show chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📜 Recent Chats")

        for idx, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            with st.expander(f"Q: {chat['question'][:50]}..."):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                st.caption(f"Time: {chat['timestamp']}")

# Tab 3: Ingest Documents
with tab3:
    st.markdown("### 📤 Document Ingestion")
    st.markdown("Add documents to your knowledge base.")

    ingest_method = st.radio(
        "Ingestion Method",
        ["📁 Upload Files", "📂 Directory Path"],
        key="ingest_method"
    )

    if ingest_method == "📁 Upload Files":
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=None,  # Accept all file types
            key="file_uploader",
            help="Upload any file type: .ipynb, .py, .pdf, .md, .c, .tsx, .csv, .yaml, etc."
        )

        st.info("💡 The system will attempt to process any file type. Common formats include: PDF, Markdown, HTML, Jupyter notebooks, Python, JSON, YAML, CSV, and plain text files.")

        col1, col2 = st.columns([1, 1])
        with col1:
            force_rebuild = st.checkbox("Force Rebuild", value=False, key="force_rebuild_upload")

        if st.button("🚀 Start Ingestion", key="ingest_upload_btn"):
            if not uploaded_files:
                st.warning("Please upload at least one file")
            else:
                with st.spinner(f"Processing {len(uploaded_files)} files..."):
                    try:
                        # Create temporary directory for uploaded files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)

                            # Save uploaded files and collect paths
                            source_files = []
                            for uploaded_file in uploaded_files:
                                file_path = temp_path / uploaded_file.name
                                with open(file_path, 'wb') as f:
                                    f.write(uploaded_file.getvalue())
                                source_files.append(str(file_path))

                            # Get vectorstore path
                            vectorstore_path = str(st.session_state.workspace / "vector_store")

                            # Run ingestion
                            result = ingest_documents_pipeline(
                                source_files=source_files,
                                vectorstore_path=vectorstore_path,
                                force_rebuild=force_rebuild,
                                caption_images=False
                            )

                            st.success(f"✅ Successfully processed {result['files_processed']} files!")
                            st.info(f"📊 Created {result['chunks_created']} chunks")
                            if HAS_GRAPH_RAG:
                                st.info(f"🔗 Knowledge graph built with entities and relationships")

                    except Exception as e:
                        st.error(f"❌ Ingestion error: {str(e)}")

    else:  # Directory Path
        dir_path = st.text_input(
            "Directory Path",
            placeholder="/path/to/documents",
            key="dir_path_input"
        )

        # File pattern selection
        st.markdown("**File Selection**")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Direct pattern input - supporting any file type
            pattern_input = st.text_input(
                "File Pattern (glob)",
                value="*.ipynb",
                key="pattern_input",
                help="Examples: *.ipynb, *.py, *.c, *.tsx, *.csv, *.yaml - any extension works!"
            )

        with col2:
            recursive = st.checkbox("Recursive", value=True, key="recursive_checkbox",
                                   help="Search subdirectories")

        # Show common pattern examples as hints
        with st.expander("💡 Pattern Examples", expanded=False):
            st.markdown("""
            **Single file type:**
            - `*.ipynb` - Jupyter notebooks
            - `*.py` - Python files
            - `*.c` - C source files
            - `*.tsx` - TypeScript React files
            - `*.csv` - CSV data files
            - `*.yaml` - YAML config files
            - `*.xlsx` - Excel files

            **Multiple patterns (comma-separated):**
            - `*.py, *.ipynb` - Python and notebooks
            - `*.js, *.jsx, *.ts, *.tsx` - All JavaScript/TypeScript
            - `*.csv, *.xlsx, *.json` - All data files

            **Wildcards:**
            - `module_*.ipynb` - Files starting with "module_"
            - `*test*.py` - Python files containing "test"
            - `*.md` - All Markdown files

            **All files:**
            - `*` - Every file (use with caution!)
            """)

        # Parse pattern (handle comma-separated patterns)
        if ',' in pattern_input:
            # Multiple patterns separated by comma
            pattern = [p.strip() for p in pattern_input.split(',')]
        else:
            # Single pattern
            pattern = pattern_input.strip()

        col1, col2 = st.columns([1, 1])

        with col1:
            force_rebuild = st.checkbox("Force Rebuild", value=False, key="force_rebuild_dir",
                                       help="Rebuild entire database from scratch")

        if st.button("🚀 Start Ingestion", key="ingest_dir_btn"):
            if not dir_path:
                st.warning("Please enter a directory path")
            elif not Path(dir_path).exists():
                st.error("❌ Directory does not exist")
            else:
                with st.spinner("Scanning directory..."):
                    try:
                        # Collect files matching pattern
                        source_files = []
                        dir_path_obj = Path(dir_path)

                        # Handle multiple patterns
                        patterns = pattern if isinstance(pattern, list) else [pattern]

                        for pat in patterns:
                            if recursive:
                                # Recursive glob
                                matching_files = list(dir_path_obj.rglob(pat))
                            else:
                                # Non-recursive glob
                                matching_files = list(dir_path_obj.glob(pat))

                            # Filter to only files (not directories)
                            source_files.extend([str(f) for f in matching_files if f.is_file()])

                        # Remove duplicates
                        source_files = list(set(source_files))

                        if not source_files:
                            st.warning(f"No files found matching pattern: {pattern}")
                        else:
                            st.info(f"Found {len(source_files)} files to process")

                            with st.spinner(f"Processing {len(source_files)} documents..."):
                                # Get vectorstore path
                                vectorstore_path = str(st.session_state.workspace / "vector_store")

                                # Run ingestion
                                result = ingest_documents_pipeline(
                                    source_files=source_files,
                                    vectorstore_path=vectorstore_path,
                                    force_rebuild=force_rebuild,
                                    caption_images=False
                                )

                                st.success(f"✅ Successfully processed {result['files_processed']} files!")
                                st.info(f"📊 Created {result['chunks_created']} chunks")
                                if HAS_GRAPH_RAG:
                                    st.info(f"🔗 Knowledge graph built with entities and relationships")

                    except Exception as e:
                        st.error(f"❌ Ingestion error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

# Tab 4: Analytics
with tab4:
    st.markdown("### 📊 Ingestion Analytics")

    try:
        log_data = load_ingestion_log(st.session_state.workspace)

        if log_data.get("ingestions"):
            st.markdown(f"**Total Ingestions:** {len(log_data['ingestions'])}")

            # Show recent ingestions
            st.markdown("#### Recent Ingestions")

            recent = log_data["ingestions"][-10:]
            for entry in reversed(recent):
                with st.expander(f"📁 {entry['source_path']} - {entry['timestamp'][:10]}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Files Processed", entry['files_processed'])

                    with col2:
                        st.metric("Chunks Created", entry['chunks_created'])

                    with col3:
                        st.metric("Duration", f"{entry['duration_seconds']:.1f}s")

                    st.markdown(f"**Pattern:** `{entry['pattern']}`")
                    st.markdown(f"**Timestamp:** {entry['timestamp']}")
        else:
            st.info("No ingestion history available yet")

    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

# Tab 5: System Status
with tab5:
    st.markdown("### ⚙️ System Status")

    # Workspace info
    st.markdown("#### 📁 Workspace Information")
    st.code(str(st.session_state.workspace))

    # Vector store status
    st.markdown("#### 🗄️ Vector Store")

    if vectorstore_path.exists():
        # Calculate size
        total_size = sum(f.stat().st_size for f in vectorstore_path.rglob("*") if f.is_file())
        size_mb = total_size / 1024 / 1024

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Status", "✅ Initialized")

        with col2:
            st.metric("Size", f"{size_mb:.1f} MB")

        # Get collection stats
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(vectorstore_path))

            try:
                collection = client.get_collection("text_collection")
                count = collection.count()
                st.metric("Document Chunks", f"{count:,}")
            except:
                st.warning("Collection not found")

        except Exception as e:
            st.error(f"Error reading stats: {str(e)}")
    else:
        st.warning("⚠️ Vector store not initialized")

    # File tracking
    st.markdown("#### 📋 File Tracking")

    tracking_file = st.session_state.workspace / "file_tracking.json"
    if tracking_file.exists():
        try:
            with open(tracking_file, 'r') as f:
                tracking_data = json.load(f)

            st.metric("Tracked Files", len(tracking_data))

            if st.checkbox("Show tracked files"):
                st.json(tracking_data)
        except:
            st.warning("Could not load file tracking data")
    else:
        st.info("No file tracking data yet")

    # Graph RAG statistics
    if HAS_GRAPH_RAG:
        st.markdown("#### 🕸️ Knowledge Graph")

        graph_file = st.session_state.workspace / "document_graph.json"
        if graph_file.exists():
            try:
                knowledge_graph = DocumentKnowledgeGraph(st.session_state.workspace)
                stats = knowledge_graph.get_statistics()

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Entities", f"{stats['total_entities']:,}")

                with col2:
                    st.metric("Total Relationships", f"{stats['total_relationships']:,}")

                with col3:
                    st.metric("Chunks with Entities", f"{stats['total_chunks_with_entities']:,}")

                # Show entity type breakdown
                if stats['entity_types']:
                    st.markdown("**Entity Types:**")
                    entity_cols = st.columns(len(stats['entity_types']))
                    for idx, (entity_type, count) in enumerate(sorted(stats['entity_types'].items())):
                        with entity_cols[idx]:
                            st.metric(entity_type.replace('_', ' ').title(), count)

                # Show extraction method breakdown
                if stats.get('extraction_methods'):
                    st.markdown("**Extraction Methods:**")
                    method_col1, method_col2 = st.columns(2)

                    with method_col1:
                        ast_count = stats['extraction_methods'].get('ast', 0)
                        st.metric("AST-based (100% accurate)", ast_count)

                    with method_col2:
                        pattern_count = stats['extraction_methods'].get('pattern', 0)
                        st.metric("Pattern-based", pattern_count)

                # Show relationship type breakdown
                if stats['relationship_types']:
                    st.markdown("**Relationship Types:**")
                    rel_cols = st.columns(min(len(stats['relationship_types']), 4))
                    for idx, (rel_type, count) in enumerate(sorted(stats['relationship_types'].items())):
                        col_idx = idx % 4
                        with rel_cols[col_idx]:
                            st.metric(rel_type.replace('_', ' ').title(), count)

            except Exception as e:
                st.warning(f"Could not load knowledge graph: {e}")
        else:
            st.info("No knowledge graph built yet. Graph RAG will be built automatically during ingestion.")
    else:
        st.markdown("#### 🕸️ Knowledge Graph")
        st.warning("Graph RAG not available. Install required dependencies.")

    # Ollama status
    st.markdown("#### 🤖 Ollama LLM Status")

    available_models = get_available_ollama_models()

    if available_models:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Status", "✅ Running")

        with col2:
            st.metric("Models Available", len(available_models))

        with st.expander("📋 Available Models", expanded=False):
            model_info = get_ollama_model_info()

            if model_info:
                for info in model_info:
                    size_str = format_bytes(info['size']) if info['size'] else "Unknown size"
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{info['name']}**")
                    with col_b:
                        st.caption(size_str)
            else:
                for model in available_models:
                    st.code(model)
    else:
        st.warning("⚠️ Ollama not running or no models installed")
        st.markdown("**Setup Instructions:**")
        st.code("""# Install Ollama
brew install ollama  # macOS
# or visit https://ollama.ai for other platforms

# Start Ollama
ollama serve

# Pull a model
ollama pull llama3.1
ollama pull qwen2.5""")

    # Environment info
    st.markdown("#### 🔧 Environment")

    env_info = {
        "Python Version": sys.version.split()[0],
        "Workspace": str(st.session_state.workspace),
        "Vector Store Path": str(vectorstore_path),
    }

    for key, value in env_info.items():
        st.text(f"{key}: {value}")

# Footer
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #a0a0a0; font-size: 14px;'>
        Built with ❤️ using Streamlit and LangChain
    </p>
""", unsafe_allow_html=True)
