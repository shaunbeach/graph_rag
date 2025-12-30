import os
import pickle
import gradio as gr
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from typing import List

# Load environment variables from .env file
load_dotenv()

# Try to import CLIP for multimodal support
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    from .ingest import CLIPEmbeddings
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("Warning: CLIP not available. Image search will be disabled.")

# --- CONFIGURATION ---
# Use workspace detection consistent with CLI
def get_workspace_path() -> Path:
    """Get workspace path - matches CLI behavior."""
    if os.getenv("RAG_WORKSPACE"):
        return Path(os.getenv("RAG_WORKSPACE"))
    return Path.cwd() / "chroma_db"

WORKSPACE_DIR = get_workspace_path()
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", str(WORKSPACE_DIR / "vector_store"))
IMAGE_STORE_PATH = os.getenv("IMAGE_STORE_PATH", str(WORKSPACE_DIR / "image_store"))
BM25_INDEX_PATH = str(WORKSPACE_DIR / "vector_store" / "bm25_index.pkl")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")  # Better semantic understanding
IMAGE_EMBEDDING_MODEL_NAME = os.getenv("IMAGE_EMBEDDING_MODEL_NAME", "openai/clip-vit-base-patch32")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "Qwen3-4B-Instruct-2507:Q4_K_M")
NUM_RETRIEVAL_DOCS = int(os.getenv("NUM_RETRIEVAL_DOCS", "10"))  # Increased to get more text context for better coverage
NUM_IMAGE_RESULTS = int(os.getenv("NUM_IMAGE_RESULTS", "3"))  # Balanced for multimodal results
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "12000"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "12000"))

# Custom prompt template - optimized for multimodal RAG with inline citations
# CRITICAL: Forces LLM to ONLY use provided context, not its own knowledge
PROMPT_TEMPLATE = """
You are an expert assistant with access to a multimodal knowledge base containing documents, code, and images.

**CRITICAL RULES:**
1. You MUST answer questions ONLY using the context provided below
2. If the answer is NOT in the context, you MUST say "I don't have this information in the provided documents"
3. NEVER use your general knowledge or training data - ONLY use the context below
4. If you use information from a source, cite it INLINE using the [TAG] provided (e.g., [TEXT_SOURCE_1] or [IMAGE_SOURCE_2])

**Instructions:**
- ALWAYS cite sources inline when using information
- ALWAYS mention relevant images when present, describing them and citing their [IMAGE_SOURCE_...] tag
- For code questions, provide clear examples WITH CITATIONS to show where the code came from
- If the retrieved context mentions the topic but lacks specific details, acknowledge what IS present and explain what's missing
- If information is incomplete in the context, say so explicitly and suggest more specific search terms
- Format code blocks properly with appropriate syntax

**Context from knowledge base:**
{context}

**User Question:**
{question}

**Your answer (using ONLY the context above):**
"""

# --- LOAD RAG SYSTEM ---
print("Loading embeddings and vector databases...")

# Text embeddings and vector store
text_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

text_db = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=text_embeddings,
    collection_name="text_collection"
)

# Initialize BM25 for hybrid search
print("Initializing BM25 keyword search...")
from langchain_core.documents import Document as LCDocument
all_docs_data = text_db.get()
all_docs = []
for i, doc_id in enumerate(all_docs_data['ids']):
    metadata = all_docs_data['metadatas'][i]
    content = all_docs_data['documents'][i]
    all_docs.append(LCDocument(page_content=content, metadata=metadata))

# Load pre-built BM25 index from disk (faster than building on startup)
bm25 = None
if os.path.exists(BM25_INDEX_PATH):
    try:
        print("Loading pre-built BM25 index from disk...")
        with open(BM25_INDEX_PATH, 'rb') as f:
            bm25 = pickle.load(f)
        print(f"BM25 index loaded with {len(all_docs)} documents")
    except Exception as e:
        print(f"Warning: Could not load BM25 index from disk: {e}")
        print("Falling back to building BM25 index on startup...")
        bm25 = None

# Fallback: Build BM25 index if loading failed
if bm25 is None:
    print("Building BM25 index from scratch...")
    tokenized_docs = [doc.page_content.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_docs)
    print(f"BM25 index built with {len(all_docs)} documents")

# Build content-to-ID lookup map for fast RRF (O(1) instead of O(N))
print("Building content-to-ID lookup map for hybrid search...")
content_to_id_map = {
    content: all_docs_data['ids'][i]
    for i, content in enumerate(all_docs_data['documents'])
}
print(f"Lookup map built with {len(content_to_id_map)} entries")

# Image embeddings and vector store (if available)
image_db = None
image_embedder = None
if HAS_CLIP:
    try:
        print("Loading CLIP model for image search...")
        image_embedder = CLIPEmbeddings(model_name=IMAGE_EMBEDDING_MODEL_NAME)
        image_db = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=image_embedder,
            collection_name="image_collection"
        )
        print("Multimodal search enabled (text + images)")
    except Exception as e:
        print(f"Warning: Could not load image database: {e}")
        print("Image search will be disabled.")
        HAS_CLIP = False
else:
    print("Image search disabled (CLIP not available)")

# --- HELPER FUNCTIONS FOR CONVERSATIONAL MEMORY ---
def _format_chat_history(history: list) -> str:
    """
    Formats Gradio's chat history into a string for the LLM.

    Args:
        history: List of [human_message, ai_message] pairs from Gradio

    Returns:
        Formatted chat history string
    """
    if not history:
        return "No previous conversation."

    formatted = []
    for human, ai in history:
        formatted.append(f"Human: {human}")
        formatted.append(f"AI: {ai}")

    return "\n".join(formatted)


# --- HYBRID RETRIEVAL (BM25 + Semantic) ---
def hybrid_retrieve(query: str, k: int = 10) -> List[LCDocument]:
    """
    Hybrid retrieval combining BM25 (keyword) and semantic search.

    Args:
        query: The search query
        k: Number of documents to retrieve

    Returns:
        List of top-k documents using reciprocal rank fusion
    """
    # 1. BM25 keyword search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*2]

    # 2. Semantic search
    semantic_docs = text_db.similarity_search(query, k=k*2)

    # 3. Reciprocal Rank Fusion (RRF)
    # Score each document based on its rank in both lists
    rrf_scores = {}
    K = 60  # RRF constant

    # Add BM25 scores
    for rank, idx in enumerate(bm25_top_indices, 1):
        doc_id = all_docs_data['ids'][idx]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (K + rank)

    # Add semantic scores (using fast O(1) lookup instead of O(N) nested loop)
    for rank, doc in enumerate(semantic_docs, 1):
        # Use the pre-built content-to-ID lookup map for instant retrieval
        doc_id = content_to_id_map.get(doc.page_content)

        if doc_id:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (K + rank)

    # Sort by RRF score and return top-k
    sorted_doc_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    # Retrieve full documents
    result_docs = []
    for doc_id, score in sorted_doc_ids:
        idx = all_docs_data['ids'].index(doc_id)
        result_docs.append(all_docs[idx])

    return result_docs


# --- UNIFIED CONTEXT FORMATTER ---
def format_multimodal_context(retrieved_data: dict) -> str:
    """
    Formats text and image documents into a single context string for the LLM.
    Adds citation-friendly source markers for inline referencing.

    Args:
        retrieved_data: Dictionary with 'text_docs' and 'image_docs' lists

    Returns:
        Formatted context string with citation tags
    """
    context_parts = []

    # --- Format Text Documents ---
    text_docs = retrieved_data.get("text_docs", [])
    if text_docs:
        context_parts.append("--- RELEVANT TEXT ---")
        for i, doc in enumerate(text_docs, 1):
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page_number', None)

            # Enhanced: Support for structure-aware metadata
            structure_type = doc.metadata.get('structure_type', None)
            file_type = doc.metadata.get('file_type', 'unknown')

            # Create a citation tag with enhanced structure information
            cite_tag = f"[TEXT_SOURCE_{i}: {source}"

            # Add structure-specific information to citations
            if structure_type == 'function':
                func_name = doc.metadata.get('function_name', '?')
                cite_tag += f", Function: {func_name}"
            elif structure_type == 'class':
                class_name = doc.metadata.get('class_name', '?')
                cite_tag += f", Class: {class_name}"
            elif structure_type == 'section':
                # Handle both markdown sections and notebook sections
                if file_type == 'jupyter_notebook':
                    section_header = doc.metadata.get('section_header', '?')
                    cite_tag += f", Section: {section_header}"
                else:
                    header = doc.metadata.get('header', '?')
                    cite_tag += f", Section: {header}"
            elif structure_type == 'cell':
                # Fallback for notebooks without sections
                cell_idx = doc.metadata.get('cell_index', '?')
                cite_tag += f", Cell {cell_idx}"
            elif page:
                cite_tag += f", Page {page}"

            cite_tag += "]"

            context_parts.append(f"{cite_tag}\n{doc.page_content}")

    # --- Format Image Documents ---
    image_docs = retrieved_data.get("image_docs", [])
    if image_docs:
        context_parts.append("\n--- RELEVANT IMAGES ---")
        for i, doc in enumerate(image_docs, 1):
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page_number', '?')
            description = doc.page_content if doc.page_content else "No description"

            # Create a citation tag for the LLM to use
            cite_tag = f"[IMAGE_SOURCE_{i}: {source}, Page {page}]"

            context_parts.append(f"{cite_tag}\n{description}")

    if not context_parts:
        return "No relevant context found."

    return "\n\n".join(context_parts)

print("Initializing LLM...")
HAS_LLM = False
llm = None
rag_chain = None
retrieval_chain = None

try:
    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_CONTEXT_WINDOW,
        num_predict=LLM_MAX_TOKENS
    )

    # Test connection
    test_response = llm.invoke("test", stop=["\n"])

    # Create retrievers for both text and images
    print("Creating unified multimodal retrieval chain with BM25 hybrid search...")
    # Use hybrid retrieval (BM25 + semantic) instead of pure semantic
    text_retriever = RunnableLambda(lambda x: hybrid_retrieve(x, k=NUM_RETRIEVAL_DOCS))

    # Create image retriever (or empty function if not available)
    if HAS_CLIP and image_db:
        image_retriever = image_db.as_retriever(search_kwargs={"k": NUM_IMAGE_RESULTS})
        print("Hybrid text retriever (BM25 + semantic) and image retriever are active.")
    else:
        # Fallback: return empty list when images not available
        image_retriever = RunnableLambda(lambda x: [])
        print("Hybrid text retriever (BM25 + semantic) is active. Image retrieval is disabled.")

    # Create the "condense question" prompt for conversational memory
    # This rephrases follow-up questions to be standalone based on chat history
    CONDENSE_PROMPT_TEMPLATE = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the conversation history.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

    condense_prompt = PromptTemplate(
        template=CONDENSE_PROMPT_TEMPLATE,
        input_variables=["chat_history", "question"]
    )

    # Create the main RAG prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Create unified retrieval chain that searches BOTH text and images in parallel
    # This runs simultaneously, not sequentially, for maximum efficiency
    retrieval_chain = RunnableParallel(
        text_docs=text_retriever,
        image_docs=image_retriever
    )

    # Create the STATEFUL unified multimodal RAG chain with conversational memory
    # This chain handles follow-up questions by first creating a standalone question

    # Sub-chain 1: Create standalone question from chat history
    _standalone_question_chain = RunnablePassthrough.assign(
        standalone_question=(
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: _format_chat_history(x.get("chat_history", []))
            }
            | condense_prompt
            | llm
            | StrOutputParser()
        )
    )

    # Sub-chain 2: Retrieve data based on the standalone question
    _retrieval_chain = RunnablePassthrough.assign(
        retrieved_data=lambda x: retrieval_chain.invoke(x["standalone_question"])
    )

    # Full stateful RAG chain
    rag_chain = (
        # Step 1: Create standalone question (handles "it", "that", "this" references)
        _standalone_question_chain
        # Step 2: Retrieve relevant documents using the standalone question
        | _retrieval_chain
        # Step 3: Format the retrieved context with citation tags
        | RunnablePassthrough.assign(
            context=lambda x: format_multimodal_context(x["retrieved_data"])
        )
        # Step 4 & 5: Generate answer AND pass retrieved_data through for UI
        | {
            "answer": (
                # Use the standalone question for the final prompt
                {"context": lambda x: x["context"], "question": lambda x: x["standalone_question"]}
                | prompt
                | llm
                | StrOutputParser()
            ),
            "retrieved_data": lambda x: x["retrieved_data"]
        }
    )

    HAS_LLM = True
    print("✓ STATEFUL Multimodal RAG chain loaded (supports conversational memory)")

except Exception as e:
    print(f"Warning: Could not connect to Ollama: {e}")
    print("✓ Fallback mode: Image search and document retrieval only")
    print("\nTo enable LLM features:")
    print("  1. Start Ollama: ollama serve")
    print("  2. Pull a model: ollama pull qwen2.5-coder:7b")
    print("  3. Restart this GUI")

    # Create basic retrievers for fallback mode (also use hybrid search)
    text_retriever = RunnableLambda(lambda x: hybrid_retrieve(x, k=NUM_RETRIEVAL_DOCS))
    if HAS_CLIP and image_db:
        image_retriever = image_db.as_retriever(search_kwargs={"k": NUM_IMAGE_RESULTS})
    else:
        image_retriever = None

    # Create retrieval chain for fallback mode
    if image_retriever:
        retrieval_chain = RunnableParallel(
            text_docs=text_retriever,
            image_docs=image_retriever
        )
    else:
        retrieval_chain = RunnableParallel(
            text_docs=text_retriever,
            image_docs=RunnableLambda(lambda _: [])
        )

# --- CHAT FUNCTION (STATEFUL WITH CONVERSATIONAL MEMORY) ---
def chat_with_rag(message, history):
    """
    Process user message and return response with sources.
    Uses STATEFUL multimodal RAG chain with conversational memory.
    Handles follow-up questions like "what about X?" or "show me an example".
    """
    if not message.strip():
        return history, "", []

    try:
        # --- Fallback Mode (No LLM) ---
        # Fallback doesn't need history - just simple retrieval
        if not HAS_LLM or not rag_chain:
            print(f"Running fallback mode for: {message}")
            retrieved_data = retrieval_chain.invoke(message)

            answer = f"**LLM not available - showing retrieved context only**\n\n"
            answer += "**Relevant document snippets:**\n\n"

            text_docs = retrieved_data.get("text_docs", [])
            for i, doc in enumerate(text_docs[:3], 1):
                content_preview = doc.page_content[:300]
                answer += f"**{i}.** {content_preview}...\n\n"

            if not text_docs:
                answer += "_No relevant documents found._\n\n"

            answer += "**To enable full AI responses:**\n"
            answer += "1. Start Ollama: `ollama serve`\n"
            answer += "2. Restart this interface"

        # --- Full LLM Mode (WITH CONVERSATIONAL MEMORY) ---
        else:
            print(f"Running stateful RAG chain for: {message}")

            # Invoke the stateful chain with question AND chat history
            chain_input = {
                "question": message,
                "chat_history": history  # Pass the full conversation history
            }
            result = rag_chain.invoke(chain_input)

            # Extract answer and retrieved data from the result dictionary
            answer = result["answer"]
            retrieved_data = result["retrieved_data"]

        # --- Build Sources Display and Image Gallery ---
        sources_text = "\n\n---\n**Sources:**\n"
        source_entries = []
        image_results = []

        # Process text sources with enhanced structure-aware information
        text_docs = retrieved_data.get("text_docs", [])
        for i, doc in enumerate(text_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            file_type = doc.metadata.get('file_type', 'unknown')
            page_number = doc.metadata.get('page_number', None)
            structure_type = doc.metadata.get('structure_type', None)

            # Format source entry with structure-aware details
            entry = f"{i}. **[TEXT] {source}** ({file_type})"

            # Add structure-specific information
            if structure_type == 'function':
                func_name = doc.metadata.get('function_name', '?')
                entry += f" - Function: `{func_name}`"
            elif structure_type == 'class':
                class_name = doc.metadata.get('class_name', '?')
                entry += f" - Class: `{class_name}`"
            elif structure_type == 'section':
                # Handle both markdown sections and notebook sections
                if file_type == 'jupyter_notebook':
                    section_header = doc.metadata.get('section_header', '?')
                    num_cells = doc.metadata.get('num_cells', '?')
                    entry += f" - Section: *{section_header}* ({num_cells} cells)"
                else:
                    header = doc.metadata.get('header', '?')
                    entry += f" - Section: *{header}*"
            elif structure_type == 'cell':
                # Fallback for notebooks without sections
                cell_idx = doc.metadata.get('cell_index', '?')
                entry += f" - Cell {cell_idx}"
            elif structure_type in ('object_key', 'mapping_key'):
                key = doc.metadata.get('key', '?')
                entry += f" - Key: `{key}`"
            elif page_number is not None:
                entry += f" - Page {page_number}"

            source_entries.append(entry)

        # Process image sources
        image_docs = retrieved_data.get("image_docs", [])
        for i, doc in enumerate(image_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page_number', '?')
            entry = f"{len(text_docs) + i}. **[IMAGE] {source}** (Page {page})"
            source_entries.append(entry)

            # Add to image gallery
            image_path = doc.metadata.get('image_path', '')
            if image_path and os.path.exists(image_path):
                image_results.append(image_path)

        if source_entries:
            sources_text += "\n".join(source_entries)
        else:
            sources_text = ""

        # --- Return Results ---
        history = history + [[message, answer + sources_text]]
        return history, "", image_results

    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nDebug info: {type(e).__name__}"
        print(f"Error in chat_with_rag: {e}")
        import traceback
        traceback.print_exc()
        history = history + [[message, error_msg]]
        return history, "", []


def search_images_only(query):
    """Search for images only without LLM generation."""
    if not HAS_CLIP or not image_db:
        return [], "Image search not available (CLIP not loaded)"

    try:
        image_docs = image_db.similarity_search_with_score(query, k=NUM_IMAGE_RESULTS)
        image_results = []
        info_text = f"**Found {len(image_docs)} images matching: \"{query}\"**\n\n"

        for i, (doc, score) in enumerate(image_docs, 1):
            image_path = doc.metadata.get('image_path', '')
            if image_path and os.path.exists(image_path):
                image_results.append(image_path)
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page_number', '?')
                similarity = 1 - score
                info_text += f"{i}. **{source}** (Page {page}) - Similarity: {similarity:.3f}\n"

        return image_results, info_text
    except Exception as e:
        return [], f"Error searching images: {str(e)}"


def clear_chat():
    """Clear the chat history."""
    return [], "", []


# --- GRADIO INTERFACE ---
with gr.Blocks(
    title="Multimodal RAG with Hybrid Search",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    ),
    css="""
    .gradio-container {max-width: 100% !important; width: 100% !important; margin: 0 auto; padding: 20px;}
    .contain {max-width: 100% !important;}
    #gallery {border-radius: 8px;}
    """
) as demo:
    # Determine status message
    status_parts = []
    if HAS_LLM:
        status_parts.append("🤖 **AI Mode Active**")
    else:
        status_parts.append("⚠️ **Retrieval Mode** (Ollama not running)")

    if HAS_CLIP:
        status_parts.append("🖼️ **Image Search Enabled**")
    else:
        status_parts.append("📝 Text-only")

    status_parts.append("🔍 **Hybrid Search** (BM25 + Semantic)")
    status_message = " | ".join(status_parts)

    gr.Markdown(
        f"""
        # 🧠 Multimodal RAG with Hybrid Search
        ### Intelligent document Q&A powered by BM25 keyword search + BGE semantic embeddings

        {status_message}

        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                f"""
                ### 💡 Example Queries

                **Intelligent Retrieval** (no exact keywords needed):
                - "What does Section 3.2 say about data validation?"
                - "What preprocessing methods are discussed?"
                - "Explain the API design patterns in Chapter 4"

                {'**Visual Search:**' if HAS_CLIP else ''}
                {'''- "Show me system architecture diagrams"
                - "Find charts about data analysis"
                - "Images with workflow diagrams"''' if HAS_CLIP else ''}

                ### ✨ Key Features
                - **Hybrid Search**: BM25 + semantic for best results
                - **Context-Only**: Zero hallucinations
                - **Conversational**: Remembers chat history
                - **Inline Citations**: Source attribution
                - {'**Multimodal**: Text + image retrieval' if HAS_CLIP else '**Text Search**: Document-based retrieval'}

                ### ⚙️ Configuration
                - **LLM:** {OLLAMA_MODEL}
                - **Embeddings:** BGE-small-en-v1.5
                - **Text Chunks:** {NUM_RETRIEVAL_DOCS} (hybrid ranked)
                {f'- **Images:** {NUM_IMAGE_RESULTS} (CLIP similarity)' if HAS_CLIP else ''}
                - **Temperature:** {LLM_TEMPERATURE}
                """
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=450,
                show_copy_button=True,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=assistant"),
                bubble_full_width=False,
                show_share_button=False
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Ask a question",
                    placeholder="No exact keywords needed - just describe what you're looking for..." if HAS_CLIP else "Ask about your documents...",
                    scale=4,
                    lines=2,
                    max_lines=4
                )
                submit_btn = gr.Button("🚀 Send", scale=1, variant="primary", size="lg")

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
                gr.Markdown(
                    f"""
                    <small>Powered by Hybrid Search (BM25 + BGE) • {NUM_RETRIEVAL_DOCS} chunks retrieved</small>
                    """,
                    elem_classes="info-text"
                )

            # Image gallery for results
            image_gallery = gr.Gallery(
                label="🖼️ Related Images from Documents",
                show_label=True,
                elem_id="gallery",
                columns=3,
                rows=1,
                height="auto",
                object_fit="contain",
                visible=HAS_CLIP,
                show_download_button=True
            )

    # Add image search section if CLIP is available
    if HAS_CLIP:
        gr.Markdown(
            """
            ---
            ### 🔍 Direct Image Search
            Search for images by description without generating an LLM response.
            """
        )
        with gr.Row():
            image_query = gr.Textbox(
                label="Image Description",
                placeholder="e.g., 'neural network architecture diagram', 'confusion matrix heatmap', 'clustering visualization'",
                scale=4,
                lines=1
            )
            image_search_btn = gr.Button("🔎 Search Images", scale=1, variant="secondary")

        image_info = gr.Markdown("")
        image_only_gallery = gr.Gallery(
            label="Image Search Results (CLIP Similarity)",
            show_label=True,
            columns=3,
            rows=2,
            height="auto",
            object_fit="contain",
            show_download_button=True
        )

    # Event handlers
    submit_btn.click(
        chat_with_rag,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, image_gallery]
    )

    msg.submit(
        chat_with_rag,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, image_gallery]
    )

    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg, image_gallery]
    )

    if HAS_CLIP:
        image_search_btn.click(
            search_images_only,
            inputs=[image_query],
            outputs=[image_only_gallery, image_info]
        )

        image_query.submit(
            search_images_only,
            inputs=[image_query],
            outputs=[image_only_gallery, image_info]
        )

    gr.Markdown(
        f"""
        ---
        <div style="text-align: center; color: #666; font-size: 0.9em;">
        <b>Multimodal RAG with Hybrid Search</b> • Built with LangChain • Ollama • ChromaDB • {'CLIP' if HAS_CLIP else 'BGE Embeddings'}<br/>
        BM25 Keyword Search + BGE Semantic Embeddings = Intelligent Retrieval<br/>
        <a href="https://github.com/yourusername/your-repo" target="_blank">Documentation</a> •
        <a href="docs/QUICKSTART.md" target="_blank">Quick Start</a> •
        <a href="docs/ARCHITECTURE.md" target="_blank">Architecture</a>
        </div>
        """
    )

# Launch function for CLI entry point
def launch():
    """Launch the Gradio web interface."""
    demo.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        share=False,
        show_error=True
    )

# Launch the app
if __name__ == "__main__":
    launch()
