"""
RAG System CLI

Command-line interface for the standalone RAG system.
"""

import click
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

try:
    from .query_classifier import QueryClassifier
except ImportError:
    QueryClassifier = None

try:
    from .feedback_logger import FeedbackLogger
except ImportError:
    FeedbackLogger = None

console = Console()

# Get workspace directory - use current working directory by default
# This is simple and portable - no hardcoded paths!
def get_default_workspace() -> Path:
    """Get the default workspace directory.

    Priority:
    1. RAG_WORKSPACE environment variable
    2. Current working directory's chroma_db/ (default)
    """
    # Check environment variable first
    if os.getenv("RAG_WORKSPACE"):
        return Path(os.getenv("RAG_WORKSPACE"))

    # Default to chroma_db in current working directory
    return Path.cwd() / "chroma_db"

WORKSPACE_DIR = get_default_workspace()
VECTORSTORE_PATH = WORKSPACE_DIR / "vector_store"
LOG_PATH = WORKSPACE_DIR / "ingestion_log.json"


def get_ingestion_log_path() -> Path:
    """Get path to ingestion log file."""
    log_path = WORKSPACE_DIR / "ingestion_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def load_ingestion_log() -> dict:
    """Load ingestion log from disk."""
    log_path = get_ingestion_log_path()

    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                return json.load(f)
        except:
            return {"ingestions": []}
    return {"ingestions": []}


def save_ingestion_log(log_data: dict):
    """Save ingestion log to disk."""
    log_path = get_ingestion_log_path()

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def add_ingestion_entry(source_path: str, pattern: str, recursive: bool,
                       files_processed: int, chunks_created: int,
                       images_extracted: int, duration_seconds: float):
    """Add an entry to the ingestion log."""
    log_data = load_ingestion_log()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "source_path": source_path,
        "pattern": pattern,
        "recursive": recursive,
        "files_processed": files_processed,
        "chunks_created": chunks_created,
        "images_extracted": images_extracted,
        "duration_seconds": round(duration_seconds, 2)
    }

    log_data["ingestions"].append(entry)
    save_ingestion_log(log_data)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    RAG System - Standalone Retrieval-Augmented Generation

    A flexible RAG system for document ingestion and querying.
    Perfect for class notes, research papers, documentation, and more.
    """
    pass


@cli.command()
@click.argument('source_path', type=str, required=False, default=None)
@click.option('--pattern', '-p', default='*',
              help='File pattern to match (e.g., "*.py", "*.md")')
@click.option('--recursive', '-r', is_flag=True,
              help='Recursively process subdirectories')
@click.option('--force-rebuild', is_flag=True,
              help='Re-ingest all files, even if unchanged')
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def ingest(source_path: Optional[str], pattern: str, recursive: bool, force_rebuild: bool, workspace: Optional[str]):
    """
    Ingest documents into the RAG system.

    Examples:

      # Ingest from current directory
      rag ingest

      # Ingest from specific folder
      rag ingest ~/Documents/ClassNotes/

      # Ingest only markdown files
      rag ingest ~/Notes/ --pattern "*.md" --recursive
    """
    import time

    start_time = time.time()

    # Set workspace if provided
    if workspace:
        global WORKSPACE_DIR, VECTORSTORE_PATH
        WORKSPACE_DIR = Path(workspace)
        VECTORSTORE_PATH = WORKSPACE_DIR / "vector_store"

    # Determine source path
    if source_path is None:
        resolved_source = Path.cwd()
        console.print(f"[dim]Using current directory: {resolved_source}[/dim]")
    else:
        resolved_source = Path(source_path).expanduser().resolve()

    # Verify source exists
    if not resolved_source.exists():
        console.print(f"[red]Error: Source path does not exist: {resolved_source}[/red]")
        raise click.Abort()

    if not resolved_source.is_dir():
        console.print(f"[red]Error: Source path is not a directory: {resolved_source}[/red]")
        raise click.Abort()

    # Display ingestion parameters
    console.print()
    console.print(Panel.fit(
        "[bold cyan]RAG Document Ingestion[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    params_table = Table(show_header=False, box=box.SIMPLE)
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="white")

    params_table.add_row("Source Path", str(resolved_source))
    params_table.add_row("Pattern", pattern)
    params_table.add_row("Recursive", "Yes" if recursive else "No")
    params_table.add_row("Workspace", str(WORKSPACE_DIR))

    console.print(params_table)
    console.print()

    # Build file list
    if recursive:
        matched_files = list(resolved_source.glob(f"**/{pattern}"))
    else:
        matched_files = list(resolved_source.glob(pattern))

    # Filter out directories
    matched_files = [f for f in matched_files if f.is_file()]

    if not matched_files:
        console.print(f"[yellow]No files matched pattern '{pattern}' in {resolved_source}[/yellow]")
        console.print("[dim]Try using --recursive or a different pattern[/dim]")
        raise click.Abort()

    console.print(f"[green]Found {len(matched_files)} files matching pattern[/green]")
    console.print()

    # Ingest files using advanced pipeline
    try:
        from rag_system.pipeline import ingest_documents_pipeline

        console.print("[cyan]Processing documents...[/cyan]")

        result = ingest_documents_pipeline(
            source_files=[str(f) for f in matched_files],
            vectorstore_path=str(VECTORSTORE_PATH),
            force_rebuild=force_rebuild,
            caption_images=False  # Can be added as CLI option later
        )

        duration = time.time() - start_time

        console.print()
        console.print(f"[green]✓ Ingestion completed in {duration:.1f}s[/green]")
        console.print(f"[dim]Processed {result['files_processed']} files, created {result['chunks_created']} chunks[/dim]")

        # Log this ingestion
        add_ingestion_entry(
            source_path=str(resolved_source),
            pattern=pattern,
            recursive=recursive,
            files_processed=result['files_processed'],
            chunks_created=result['chunks_created'],
            images_extracted=result.get('images_extracted', 0),
            duration_seconds=duration
        )

    except ImportError as e:
        console.print(f"[red]Error: Could not import ingestion module: {e}[/red]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


@cli.command()
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def status(workspace: Optional[str]):
    """Show RAG system status and ingestion history."""
    if workspace:
        global WORKSPACE_DIR, VECTORSTORE_PATH
        WORKSPACE_DIR = Path(workspace)
        VECTORSTORE_PATH = WORKSPACE_DIR / "vector_store"

    console.print()
    console.print(Panel.fit(
        "[bold cyan]RAG System Status[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    status_table = Table(title="Database Status", box=box.ROUNDED, show_header=True)
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", justify="center")
    status_table.add_column("Details", style="white")

    if VECTORSTORE_PATH.exists():
        # Count files in vector store
        chroma_files = list(VECTORSTORE_PATH.rglob("*"))
        size_mb = sum(f.stat().st_size for f in chroma_files if f.is_file()) / 1024 / 1024
        status_table.add_row(
            "Vector Store",
            "[green]✓[/green]",
            f"{size_mb:.1f} MB"
        )

        # Try to get collection stats
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(VECTORSTORE_PATH))

            try:
                text_collection = client.get_collection("text_collection")
                text_count = text_collection.count()
                status_table.add_row(
                    "  Text Chunks",
                    "[green]✓[/green]",
                    f"{text_count:,} documents"
                )
            except:
                status_table.add_row(
                    "  Text Chunks",
                    "[yellow]⚠[/yellow]",
                    "Collection not found"
                )

        except Exception as e:
            status_table.add_row(
                "  Collections",
                "[yellow]⚠[/yellow]",
                f"Could not read: {str(e)[:30]}"
            )
    else:
        status_table.add_row(
            "Vector Store",
            "[red]✗[/red]",
            "Not initialized"
        )

    console.print(status_table)
    console.print()

    # Show ingestion history
    log_data = load_ingestion_log()

    if log_data["ingestions"]:
        history_table = Table(
            title=f"Ingestion History (Last 10)",
            box=box.ROUNDED,
            show_header=True
        )
        history_table.add_column("Date/Time", style="cyan")
        history_table.add_column("Source Path", style="white")
        history_table.add_column("Pattern", style="dim")
        history_table.add_column("Files", justify="right", style="green")
        history_table.add_column("Duration", justify="right", style="dim")

        # Show last 10 ingestions
        recent = log_data["ingestions"][-10:]
        for entry in reversed(recent):
            dt = datetime.fromisoformat(entry["timestamp"])
            history_table.add_row(
                dt.strftime("%Y-%m-%d %H:%M"),
                entry["source_path"].split("/")[-1] if "/" in entry["source_path"] else entry["source_path"],
                entry["pattern"],
                str(entry["files_processed"]),
                f"{entry['duration_seconds']:.1f}s"
            )

        console.print(history_table)
    else:
        console.print("[dim]No ingestion history found[/dim]")

    console.print()


@cli.command()
@click.argument('query_text', type=str)
@click.option('--top-k', default=5, help='Number of results to return')
@click.option('--filter-author', type=str, help='Filter by author (e.g., "Shaun Beach")')
@click.option('--filter-course', type=str, help='Filter by course (e.g., "CMPINF 2100")')
@click.option('--filter-assignment', type=str, help='Filter by assignment type (midterm, final, homework, project)')
@click.option('--filter-week', type=str, help='Filter by week number')
@click.option('--filter-type', type=str, help='Filter by content type (markdown, code)')
@click.option('--filter-category', type=str, help='Filter by category (student_work, course_material)')
@click.option('--auto-classify', is_flag=True, default=False, help='Automatically classify query and apply smart filters')
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def query(query_text: str, top_k: int, filter_author: Optional[str], filter_course: Optional[str],
          filter_assignment: Optional[str], filter_week: Optional[str], filter_type: Optional[str],
          filter_category: Optional[str], auto_classify: bool, workspace: Optional[str]):
    """
    Query the RAG database with optional metadata filters.

    Examples:
      rag query "What are neural networks?"
      rag query "clustering analysis" --filter-author "Shaun Beach"
      rag query "pandas merge" --filter-type "code"
      rag query "drop test" --filter-assignment "midterm"
      rag query "visualization" --filter-week "7" --filter-type "code"
    """
    if workspace:
        global WORKSPACE_DIR, VECTORSTORE_PATH
        WORKSPACE_DIR = Path(workspace)
        VECTORSTORE_PATH = WORKSPACE_DIR / "vector_store"

    console.print()
    console.print(f"[cyan]Searching for:[/cyan] {query_text}")
    console.print()

    if not VECTORSTORE_PATH.exists():
        console.print("[red]Error: Vector store not initialized[/red]")
        console.print("[dim]Run 'rag ingest' first[/dim]")
        raise click.Abort()

    # Start timing
    import time
    start_time = time.time()

    try:
        # Try new import first, fall back to old one
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma

        from langchain_huggingface import HuggingFaceEmbeddings

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # Load vector store
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_PATH),
            embedding_function=embeddings,
            collection_name="text_collection"
        )

        # Auto-classify query if enabled and no manual filters set
        manual_filters_set = any([filter_author, filter_course, filter_assignment,
                                  filter_week, filter_type, filter_category])

        if auto_classify and not manual_filters_set and QueryClassifier:
            classifier = QueryClassifier()
            intent = classifier.classify(query_text)

            console.print(f"[dim]🤖 Auto-classified as: {intent.intent_type} (confidence: {intent.confidence:.2f})[/dim]")

            # Use suggested filters
            filter_dict = intent.suggested_filters.copy()

            # Use suggested search params
            lambda_mult = intent.search_params.get("lambda_mult", 0.5)
            fetch_k_mult = intent.search_params.get("fetch_k_multiplier", 3)
        else:
            # Build metadata filter from manual options
            filter_dict = {}
            if filter_author:
                filter_dict["author"] = filter_author
            if filter_course:
                filter_dict["course"] = filter_course
            if filter_assignment:
                filter_dict["assignment_type"] = filter_assignment
            if filter_week:
                filter_dict["week"] = filter_week
            if filter_type:
                filter_dict["content_type"] = filter_type
            if filter_category:
                filter_dict["doc_category"] = filter_category

            # Default search params
            lambda_mult = 0.5
            fetch_k_mult = 3

        # Show active filters
        if filter_dict:
            console.print(f"[dim]Filters: {filter_dict}[/dim]")
            console.print()

        # Perform search with MMR (MaxMarginalRelevance) for diverse results
        # This prevents returning multiple similar chunks from the same document
        try:
            if filter_dict:
                results = vectorstore.max_marginal_relevance_search(
                    query_text,
                    k=top_k,
                    fetch_k=top_k * fetch_k_mult,  # Dynamic based on query type
                    lambda_mult=lambda_mult,  # Dynamic based on query type
                    filter=filter_dict  # Apply metadata filters
                )
            else:
                results = vectorstore.max_marginal_relevance_search(
                    query_text,
                    k=top_k,
                    fetch_k=top_k * fetch_k_mult,
                    lambda_mult=lambda_mult
                )
        except Exception:
            # Fallback to regular similarity search if MMR not available
            if filter_dict:
                results = vectorstore.similarity_search(query_text, k=top_k, filter=filter_dict)
            else:
                results = vectorstore.similarity_search(query_text, k=top_k)

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Log query with feedback logger
        if FeedbackLogger:
            try:
                logger = FeedbackLogger()
                query_id = logger.log_query(
                    query_text=query_text,
                    intent_type=intent.intent_type if auto_classify and not manual_filters_set else "manual",
                    intent_confidence=intent.confidence if auto_classify and not manual_filters_set else 1.0,
                    filters_applied=filter_dict,
                    search_params={"lambda_mult": lambda_mult, "fetch_k_multiplier": fetch_k_mult},
                    results=results,
                    response_time_ms=response_time_ms
                )
                console.print(f"[dim]📝 Query logged: {query_id} ({response_time_ms:.0f}ms)[/dim]")
                console.print()
            except Exception as e:
                # Don't fail the query if logging fails
                console.print(f"[dim yellow]Warning: Could not log query: {e}[/dim yellow]")

        # Display results
        for idx, doc in enumerate(results, 1):
            # Show more content (1500 chars instead of 500) for better context
            preview_length = 1500
            content_preview = doc.page_content[:preview_length]

            # Add metadata badges for context
            badges = []
            if doc.metadata.get('content_type'):
                badges.append(f"[dim cyan]{doc.metadata['content_type']}[/dim cyan]")
            if doc.metadata.get('section_heading'):
                badges.append(f"[dim yellow]{doc.metadata['section_heading']}[/dim yellow]")

            badge_line = " | ".join(badges) if badges else ""

            console.print(Panel(
                f"[bold]{doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown source'))}[/bold]\n"
                f"{badge_line}\n\n"
                f"{content_preview}{'...' if len(doc.page_content) > preview_length else ''}",
                title=f"Result {idx}/{len(results)}",
                border_style="cyan" if idx == 1 else "dim"
            ))
            console.print()

    except ImportError as e:
        console.print(f"[red]Error: Missing dependencies: {e}[/red]")
        console.print("[dim]Install with: pip install langchain-chroma langchain-huggingface chromadb sentence-transformers[/dim]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error during query: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.argument('question', type=str)
@click.option('--top-k', default=5, help='Number of source documents to retrieve')
@click.option('--get-all', is_flag=True, help='Retrieve ALL documents matching filters (ignores top-k)')
@click.option('--provider', type=click.Choice(['auto', 'openai', 'anthropic', 'ollama']),
              default='auto', help='LLM provider to use')
@click.option('--model', type=str, help='Specific model name (optional)')
@click.option('--filter-author', type=str, help='Filter by author')
@click.option('--filter-course', type=str, help='Filter by course')
@click.option('--filter-assignment', type=str, help='Filter by assignment type')
@click.option('--filter-week', type=str, help='Filter by week number')
@click.option('--filter-type', type=str, help='Filter by content type (markdown, code)')
@click.option('--filter-category', type=str, help='Filter by category')
@click.option('--auto-classify', is_flag=True, default=False, help='Auto-classify query')
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def ask(question: str, top_k: int, get_all: bool, provider: str, model: Optional[str],
        filter_author: Optional[str], filter_course: Optional[str],
        filter_assignment: Optional[str], filter_week: Optional[str],
        filter_type: Optional[str], filter_category: Optional[str],
        auto_classify: bool, workspace: Optional[str]):
    """
    Ask a question and get an AI-generated answer based on your documents.

    This is full RAG (Retrieval-Augmented Generation):
    1. Retrieves relevant documents
    2. Uses an LLM to generate an answer
    3. Cites sources

    Examples:
      rag ask "When should I use hierarchical clustering vs KMeans?"
      rag ask "How do I merge dataframes in pandas?" --filter-type code
      rag ask "What was my midterm clustering analysis?" --filter-author "Shaun Beach"

    LLM Providers:
      - auto: Auto-detect (checks for API keys, falls back to Ollama)
      - openai: Requires OPENAI_API_KEY environment variable
      - anthropic: Requires ANTHROPIC_API_KEY environment variable
      - ollama: Requires Ollama running locally (free, no API key needed)
    """
    if workspace:
        global WORKSPACE_DIR, VECTORSTORE_PATH
        WORKSPACE_DIR = Path(workspace)
        VECTORSTORE_PATH = WORKSPACE_DIR / "vector_store"

    console.print()
    console.print(f"[cyan]Question:[/cyan] {question}")
    console.print()

    if not VECTORSTORE_PATH.exists():
        console.print("[red]Error: Vector store not initialized[/red]")
        console.print("[dim]Run 'rag ingest' first[/dim]")
        raise click.Abort()

    import time
    start_time = time.time()

    try:
        # Import LLM handler
        try:
            from .llm_handler import LLMHandler
        except ImportError:
            console.print("[red]Error: LLM handler not found[/red]")
            raise click.Abort()

        # Initialize LLM
        console.print(f"[dim]Initializing LLM ({provider})...[/dim]")
        try:
            llm_handler = LLMHandler(provider=provider, model=model)
            console.print(f"[dim green]✓ Using: {llm_handler.provider}[/dim green]")
        except Exception as e:
            console.print(f"[red]Error initializing LLM: {e}[/red]")
            console.print()
            console.print("[yellow]Tip: For free local LLM:[/yellow]")
            console.print("  1. Install Ollama: brew install ollama")
            console.print("  2. Start Ollama: ollama serve")
            console.print("  3. Pull a model: ollama pull llama3")
            console.print()
            console.print("[yellow]Or use cloud LLM:[/yellow]")
            console.print("  export OPENAI_API_KEY='your-key'")
            console.print("  export ANTHROPIC_API_KEY='your-key'")
            raise click.Abort()

        # Load vector store and embeddings (same as query command)
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma

        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_PATH),
            embedding_function=embeddings,
            collection_name="text_collection"
        )

        # Auto-classify query if enabled
        manual_filters_set = any([filter_author, filter_course, filter_assignment,
                                  filter_week, filter_type, filter_category])

        if auto_classify and not manual_filters_set and QueryClassifier:
            classifier = QueryClassifier()
            intent = classifier.classify(question)

            console.print(f"[dim]🤖 Classified as: {intent.intent_type} (confidence: {intent.confidence:.2f})[/dim]")

            filter_dict = intent.suggested_filters.copy()
            lambda_mult = intent.search_params.get("lambda_mult", 0.5)
            fetch_k_mult = intent.search_params.get("fetch_k_multiplier", 3)
        else:
            filter_dict = {}
            if filter_author:
                filter_dict["author"] = filter_author
            if filter_course:
                filter_dict["course"] = filter_course
            if filter_assignment:
                filter_dict["assignment_type"] = filter_assignment
            if filter_week:
                filter_dict["week"] = filter_week
            if filter_type:
                filter_dict["content_type"] = filter_type
            if filter_category:
                filter_dict["doc_category"] = filter_category

            lambda_mult = 0.5
            fetch_k_mult = 3

        if filter_dict:
            console.print(f"[dim]Filters: {filter_dict}[/dim]")

        # Retrieve documents
        if get_all and filter_dict:
            # Get ALL documents matching the filter (no semantic search)
            console.print(f"[dim]Retrieving ALL documents matching filters...[/dim]")
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(VECTORSTORE_PATH))
                collection = client.get_collection("text_collection")

                # Build where clause for chromadb
                where_clause = {}
                for key, value in filter_dict.items():
                    where_clause[key] = value

                all_docs = collection.get(where=where_clause, include=["documents", "metadatas"])

                # Convert to Document objects
                from langchain_core.documents import Document
                results = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
                ]
                console.print(f"[dim green]✓ Retrieved {len(results)} documents[/dim green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not retrieve all documents: {e}[/yellow]")
                console.print("[dim]Falling back to top-k search...[/dim]")
                get_all = False

        if not get_all:
            # Normal semantic search with top-k
            console.print(f"[dim]Retrieving top {top_k} relevant documents...[/dim]")
            try:
                if filter_dict:
                    results = vectorstore.max_marginal_relevance_search(
                        question,
                        k=top_k,
                        fetch_k=top_k * fetch_k_mult,
                        lambda_mult=lambda_mult,
                        filter=filter_dict
                    )
                else:
                    results = vectorstore.max_marginal_relevance_search(
                        question,
                        k=top_k,
                        fetch_k=top_k * fetch_k_mult,
                        lambda_mult=lambda_mult
                    )
            except Exception:
                if filter_dict:
                    results = vectorstore.similarity_search(question, k=top_k, filter=filter_dict)
                else:
                    results = vectorstore.similarity_search(question, k=top_k)

        if not results:
            console.print("[yellow]No relevant documents found[/yellow]")
            return

        console.print(f"[dim green]✓ Retrieved {len(results)} documents[/dim green]")
        console.print(f"[dim]Generating answer...[/dim]")
        console.print()

        # Generate answer using LLM
        result = llm_handler.generate_answer(
            query=question,
            retrieved_docs=results
        )

        response_time_ms = (time.time() - start_time) * 1000

        # Display answer
        console.print(Panel(
            result["answer"],
            title="[bold cyan]Answer[/bold cyan]",
            border_style="cyan"
        ))
        console.print()

        # Display sources (limit to first 10 when using --get-all to avoid terminal spam)
        sources_to_display = result["sources"]
        max_sources_display = 10 if get_all else len(sources_to_display)

        console.print("[bold]Sources:[/bold]")
        for source in sources_to_display[:max_sources_display]:
            console.print(
                f"  [{source['number']}] {source['file']}\n"
                f"      [dim]{source['section']} | {source['type']}[/dim]"
            )

        if len(sources_to_display) > max_sources_display:
            remaining = len(sources_to_display) - max_sources_display
            console.print(f"  [dim]... and {remaining} more sources[/dim]")

        console.print()

        console.print(f"[dim]Model: {result['model']} | Time: {response_time_ms:.0f}ms | Sources: {result['num_sources']}[/dim]")
        console.print()

    except ImportError as e:
        console.print(f"[red]Error: Missing dependencies: {e}[/red]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def history(workspace: Optional[str]):
    """Show complete ingestion history."""
    if workspace:
        global WORKSPACE_DIR
        WORKSPACE_DIR = Path(workspace)

    log_data = load_ingestion_log()

    if not log_data["ingestions"]:
        console.print("[yellow]No ingestion history found[/yellow]")
        return

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Complete Ingestion History[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    history_table = Table(box=box.ROUNDED, show_header=True)
    history_table.add_column("#", justify="right", style="dim")
    history_table.add_column("Date/Time", style="cyan")
    history_table.add_column("Source Path", style="white")
    history_table.add_column("Pattern", style="dim")
    history_table.add_column("Recursive", justify="center")
    history_table.add_column("Files", justify="right", style="green")
    history_table.add_column("Duration", justify="right", style="dim")

    for idx, entry in enumerate(log_data["ingestions"], 1):
        dt = datetime.fromisoformat(entry["timestamp"])
        history_table.add_row(
            str(idx),
            dt.strftime("%Y-%m-%d %H:%M:%S"),
            entry["source_path"],
            entry["pattern"],
            "✓" if entry["recursive"] else "―",
            str(entry["files_processed"]),
            f"{entry['duration_seconds']:.1f}s"
        )

    console.print(history_table)
    console.print()

    # Summary stats
    total_files = sum(e["files_processed"] for e in log_data["ingestions"])
    total_chunks = sum(e["chunks_created"] for e in log_data["ingestions"])

    console.print(f"[bold]Total:[/bold] {len(log_data['ingestions'])} ingestions, "
                 f"{total_files} files processed, "
                 f"{total_chunks} chunks created")
    console.print()


@cli.command()
@click.option('--by-content', is_flag=True, default=True,
              help='Deduplicate by content hash (default)')
@click.option('--by-source', is_flag=True,
              help='Deduplicate by source path')
@click.option('--dry-run', is_flag=True,
              help='Preview without removing')
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def deduplicate(by_content: bool, by_source: bool, dry_run: bool, workspace: Optional[str]):
    """
    Remove duplicate documents.

    Examples:

      # Preview duplicates
      rag deduplicate --dry-run

      # Remove content duplicates
      rag deduplicate
    """
    import hashlib
    from collections import defaultdict

    if workspace:
        global WORKSPACE_DIR, VECTORSTORE_PATH
        WORKSPACE_DIR = Path(workspace)
        VECTORSTORE_PATH = WORKSPACE_DIR / "vector_store"

    console.print()
    console.print(Panel.fit(
        "[bold cyan]RAG Deduplication[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    method = "source path" if by_source else "content hash"
    console.print(f"[cyan]Method:[/cyan] {method}")
    console.print(f"[cyan]Mode:[/cyan] {'Dry run' if dry_run else 'Live'}")
    console.print()

    if not VECTORSTORE_PATH.exists():
        console.print("[red]Error: Vector store not found[/red]")
        raise click.Abort()

    try:
        import chromadb

        console.print(f"[dim]Loading database from: {VECTORSTORE_PATH}[/dim]")
        client = chromadb.PersistentClient(path=str(VECTORSTORE_PATH))

        try:
            collection = client.get_collection("text_collection")
        except:
            console.print("[red]Error: text_collection not found[/red]")
            raise click.Abort()

        # Get all documents
        console.print("[cyan]Analyzing documents...[/cyan]")
        all_docs = collection.get(include=["documents", "metadatas"])

        total_docs = len(all_docs['ids'])
        console.print(f"[dim]Found {total_docs} documents[/dim]")
        console.print()

        if total_docs == 0:
            console.print("[yellow]Database is empty[/yellow]")
            return

        # Track duplicates
        seen = {}
        duplicates = []

        for idx, doc_id in enumerate(all_docs['ids']):
            content = all_docs['documents'][idx]
            metadata = all_docs['metadatas'][idx]

            if by_source:
                key = metadata.get('source', 'unknown')
            else:
                key = hashlib.md5(content.encode('utf-8')).hexdigest()

            if key in seen:
                duplicates.append((doc_id, seen[key], key))
            else:
                seen[key] = doc_id

        num_duplicates = len(duplicates)
        num_unique = len(seen)

        console.print(f"[bold]Results:[/bold]")
        console.print(f"  Total: {total_docs}")
        console.print(f"  Unique: {num_unique}")
        console.print(f"  Duplicates: {num_duplicates}")
        console.print()

        if num_duplicates == 0:
            console.print("[green]✓ No duplicates found![/green]")
            return

        if dry_run:
            console.print(f"[yellow]Would remove {num_duplicates} duplicates[/yellow]")
        else:
            console.print(f"[cyan]Removing {num_duplicates} duplicates...[/cyan]")
            dup_ids_to_remove = [dup_id for dup_id, _, _ in duplicates]
            collection.delete(ids=dup_ids_to_remove)
            console.print(f"[green]✓ Removed {num_duplicates} duplicates[/green]")

        console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
def info():
    """Show RAG system information."""
    console.print()
    console.print(Panel(
        "[bold cyan]RAG System[/bold cyan] [white]v1.0.0[/white]\n"
        "[dim]Standalone Retrieval-Augmented Generation[/dim]",
        border_style="cyan",
        padding=(1, 2)
    ))
    console.print()

    console.print("[bold]Features:[/bold]")
    console.print("  ✓ Flexible document ingestion from any folder")
    console.print("  ✓ Pattern-based file filtering (*.md, *.pdf, etc.)")
    console.print("  ✓ Recursive directory processing")
    console.print("  ✓ Natural language querying")
    console.print("  ✓ Deduplication support")
    console.print("  ✓ Complete ingestion history tracking")
    console.print()

    console.print(f"[bold]Workspace:[/bold] {WORKSPACE_DIR}")
    console.print(f"[bold]Vector Store:[/bold] {VECTORSTORE_PATH}")
    console.print()

    console.print("[bold]Quick Start:[/bold]")
    console.print("  [cyan]rag ingest ~/Documents/Notes/[/cyan]  - Ingest documents")
    console.print("  [cyan]rag query \"your question\"[/cyan]      - Query the database")
    console.print("  [cyan]rag status[/cyan]                      - Check system status")
    console.print()


@cli.command()
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def analytics(workspace: Optional[str]):
    """Show query analytics and feedback statistics."""
    if workspace:
        global WORKSPACE_DIR
        WORKSPACE_DIR = Path(workspace)

    if not FeedbackLogger:
        console.print("[red]Error: FeedbackLogger not available[/red]")
        return

    logger = FeedbackLogger()
    stats = logger.get_analytics()

    if stats.get("total_queries", 0) == 0:
        console.print("[yellow]No queries logged yet[/yellow]")
        console.print("[dim]Run some queries to see analytics[/dim]")
        return

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Query Analytics[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    # Main stats
    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Queries", str(stats["total_queries"]))
    table.add_row("Queries with Feedback", str(stats["queries_with_feedback"]))
    table.add_row("Feedback Rate", f"{stats['feedback_rate']:.1f}%")
    if stats.get("average_response_time_ms"):
        table.add_row("Avg Response Time", f"{stats['average_response_time_ms']:.0f}ms")

    console.print(table)
    console.print()

    # Intent distribution
    console.print("[bold]Intent Distribution:[/bold]")
    intent_table = Table(box=box.SIMPLE, show_header=True)
    intent_table.add_column("Intent Type", style="cyan")
    intent_table.add_column("Count", justify="right")
    intent_table.add_column("Avg Confidence", justify="right")

    for intent, count in stats["intent_distribution"].items():
        avg_conf = stats["average_confidence_by_intent"].get(intent, 0)
        intent_table.add_row(intent, str(count), f"{avg_conf:.2f}")

    console.print(intent_table)
    console.print()

    # Feedback distribution
    console.print("[bold]Feedback Distribution:[/bold]")
    feedback_table = Table(box=box.SIMPLE, show_header=True)
    feedback_table.add_column("Feedback", style="cyan")
    feedback_table.add_column("Count", justify="right")

    for feedback, count in stats["feedback_distribution"].items():
        feedback_table.add_row(feedback, str(count))

    console.print(feedback_table)
    console.print()

    # Filter usage
    if stats.get("filter_usage"):
        console.print("[bold]Filter Usage:[/bold]")
        filter_table = Table(box=box.SIMPLE, show_header=True)
        filter_table.add_column("Filter", style="cyan")
        filter_table.add_column("Times Used", justify="right")

        for filter_name, count in stats["filter_usage"].items():
            filter_table.add_row(filter_name, str(count))

        console.print(filter_table)
        console.print()


@cli.command()
@click.argument('query_id')
@click.argument('feedback', type=click.Choice(['positive', 'negative', 'neutral']))
@click.option('--comment', type=str, help='Optional feedback comment')
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def feedback(query_id: str, feedback: str, comment: Optional[str], workspace: Optional[str]):
    """Add feedback to a logged query.

    Examples:
        rag feedback 20241110_143025_123456 positive --comment "Great results!"
        rag feedback 20241110_143025_123456 negative --comment "Results not relevant"
    """
    if workspace:
        global WORKSPACE_DIR
        WORKSPACE_DIR = Path(workspace)

    if not FeedbackLogger:
        console.print("[red]Error: FeedbackLogger not available[/red]")
        return

    logger = FeedbackLogger()
    logger.add_feedback(query_id, feedback, comment)

    console.print(f"[green]✓ Feedback recorded for query {query_id}[/green]")
    if comment:
        console.print(f"[dim]Comment: {comment}[/dim]")


@cli.command()
@click.option('--limit', type=int, default=10, help='Number of recent queries to show')
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def recent(limit: int, workspace: Optional[str]):
    """Show recent queries."""
    if workspace:
        global WORKSPACE_DIR
        WORKSPACE_DIR = Path(workspace)

    if not FeedbackLogger:
        console.print("[red]Error: FeedbackLogger not available[/red]")
        return

    logger = FeedbackLogger()
    queries = logger.get_recent_queries(limit)

    if not queries:
        console.print("[yellow]No queries logged yet[/yellow]")
        return

    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Recent {limit} Queries[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    for query in queries:
        dt = datetime.fromisoformat(query["timestamp"])
        feedback_emoji = {
            "positive": "👍",
            "negative": "👎",
            "neutral": "➖",
        }.get(query.get("user_feedback"), "⏺️")

        console.print(Panel(
            f"[bold]Query:[/bold] {query['query_text']}\n"
            f"[dim]Intent:[/dim] {query['intent_type']} ({query['intent_confidence']:.2f})\n"
            f"[dim]Results:[/dim] {query['num_results']}\n"
            f"[dim]Filters:[/dim] {query['filters_applied']}\n"
            f"[dim]Feedback:[/dim] {feedback_emoji} {query.get('user_feedback', 'none')}"
            + (f"\n[dim]Comment:[/dim] {query['feedback_comment']}" if query.get('feedback_comment') else ""),
            title=f"{dt.strftime('%Y-%m-%d %H:%M:%S')} | {query['query_id']}",
            border_style="cyan"
        ))
        console.print()


@cli.command()
@click.option('--template', type=click.Choice(['generic', 'academic']), default='generic',
              help='Config template to generate')
@click.option('--output', type=click.Path(), default='.rag_metadata.json',
              help='Output file path')
def init_metadata(template: str, output: str):
    """
    Generate a metadata configuration file.

    Use this to enable custom metadata extraction from filenames and paths.

    Examples:
        rag init-metadata                          # Generic template
        rag init-metadata --template academic      # Academic use case template
        rag init-metadata --output ~/my-config.json
    """
    from .metadata_config import generate_config_file

    output_path = Path(output)

    if output_path.exists():
        console.print(f"[yellow]Warning: {output_path} already exists[/yellow]")
        if not click.confirm("Overwrite?"):
            console.print("[dim]Cancelled[/dim]")
            return

    generate_config_file(output_path, template=template)

    console.print()
    console.print(Panel.fit(
        f"[bold green]✓ Created metadata config[/bold green]\n\n"
        f"[cyan]File:[/cyan] {output_path}\n"
        f"[cyan]Template:[/cyan] {template}\n\n"
        f"[dim]Edit this file to customize metadata extraction.\n"
        f"Metadata extraction is enabled by default in the config.[/dim]",
        border_style="green"
    ))
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Edit the config file to match your file naming conventions")
    console.print("  2. Re-ingest your documents to apply metadata")
    console.print("  3. Use filters in queries: rag query 'text' --filter-assignment midterm")
    console.print()


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--pattern', default='*.py', help='File pattern to match (default: *.py)')
@click.option('--recursive', is_flag=True, default=True, help='Process subdirectories recursively')
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def build_graph(path: str, pattern: str, recursive: bool, workspace: Optional[str]):
    """
    Build knowledge graph from code files.

    Analyzes Python files to extract code structure and relationships:
    - Classes, functions, and methods
    - Function calls
    - Import dependencies
    - Inheritance hierarchies

    Examples:
        rag build-graph ~/Projects/myapp/src/
        rag build-graph ~/Code --pattern "*.py" --recursive
    """
    if workspace:
        global WORKSPACE_DIR
        WORKSPACE_DIR = Path(workspace)

    console.print()
    console.print(f"[cyan]Building knowledge graph from:[/cyan] {path}")
    console.print(f"[dim]Pattern: {pattern}, Recursive: {recursive}[/dim]")
    console.print()

    try:
        from .knowledge_graph import CodeKnowledgeGraph
        from .graph_builder import PythonGraphBuilder

        # Initialize graph
        graph_db_path = WORKSPACE_DIR / "code_graph.db"
        graph = CodeKnowledgeGraph(str(graph_db_path))

        # Build from directory
        builder = PythonGraphBuilder(graph)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing code...", total=None)
            stats = builder.build_from_directory(path, pattern=pattern, recursive=recursive)

        # Display results
        console.print()
        console.print(Panel.fit(
            "[bold green]✓ Knowledge Graph Built[/bold green]",
            border_style="green"
        ))
        console.print()

        table = Table(box=box.ROUNDED, show_header=True)
        table.add_column("Entity Type", style="cyan")
        table.add_column("Count", justify="right", style="white")

        table.add_row("Files Processed", str(stats['files']))
        table.add_row("Modules", str(stats['modules']))
        table.add_row("Classes", str(stats['classes']))
        table.add_row("Functions", str(stats['functions']))
        table.add_row("Methods", str(stats['methods']))
        table.add_row("Imports", str(stats['imports']))
        table.add_row("Function Calls", str(stats['calls']))
        table.add_row("Inheritance Relationships", str(stats['inheritance']))

        console.print(table)
        console.print()
        console.print(f"[dim]Graph database:[/dim] {graph_db_path}")
        console.print()

    except ImportError as e:
        console.print(f"[red]Error: Missing dependency - {e}[/red]")
        console.print("[dim]Install networkx: pip install networkx[/dim]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


@cli.command()
@click.argument('entity_name')
@click.option('--type', 'entity_type', type=click.Choice(['class', 'function', 'method', 'module']),
              help='Entity type to search for')
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def code_info(entity_name: str, entity_type: Optional[str], workspace: Optional[str]):
    """
    Get information about a code entity.

    Shows methods, callers, callees, dependencies, and other relationships.

    Examples:
        rag code-info DocumentIngester --type class
        rag code-info ingest_files --type function
        rag code-info pipeline --type module
    """
    if workspace:
        global WORKSPACE_DIR
        WORKSPACE_DIR = Path(workspace)

    graph_db_path = WORKSPACE_DIR / "code_graph.db"

    if not graph_db_path.exists():
        console.print("[red]Error: Knowledge graph not built yet[/red]")
        console.print("[dim]Run 'rag build-graph' first[/dim]")
        raise click.Abort()

    try:
        from .knowledge_graph import CodeKnowledgeGraph

        graph = CodeKnowledgeGraph(str(graph_db_path))

        console.print()
        console.print(f"[cyan]Searching for:[/cyan] {entity_name}")
        if entity_type:
            console.print(f"[dim]Type: {entity_type}[/dim]")
        console.print()

        # Find the entity
        nodes = graph.find_nodes(node_type=entity_type, name=entity_name)

        if not nodes:
            console.print(f"[yellow]No entities found matching '{entity_name}'[/yellow]")

            # Suggest similar names
            all_nodes = graph.find_nodes(node_type=entity_type) if entity_type else []
            if len(all_nodes) < 50:  # Only show suggestions if not too many
                similar = [n.name for n in all_nodes if entity_name.lower() in n.name.lower()]
                if similar:
                    console.print()
                    console.print("[dim]Did you mean:[/dim]")
                    for name in similar[:10]:
                        console.print(f"  • {name}")
            return

        # Display information for each match
        for node in nodes:
            console.print(Panel.fit(
                f"[bold]{node.name}[/bold]",
                subtitle=f"{node.type} | {node.source_file}",
                border_style="cyan"
            ))
            console.print()

            # Type-specific information
            if node.type == 'class':
                # Show methods
                methods = graph.get_methods(node.name)
                if methods:
                    console.print("[bold]Methods:[/bold]")
                    for method in methods:
                        args = method.metadata.get('args', [])
                        args_str = ', '.join(args) if args else ''
                        console.print(f"  • {method.name}({args_str})")
                    console.print()

                # Show inheritance
                inheritance = graph.get_inheritance_tree(node.name)
                if inheritance.get('parents'):
                    console.print(f"[bold]Inherits from:[/bold]")
                    for parent in inheritance['parents']:
                        console.print(f"  • {parent}")
                    console.print()

                if inheritance.get('children'):
                    console.print(f"[bold]Subclasses:[/bold]")
                    for child in inheritance['children']:
                        console.print(f"  • {child}")
                    console.print()

            elif node.type in ['function', 'method']:
                # Show function signature
                args = node.metadata.get('args', [])
                if args:
                    console.print(f"[bold]Arguments:[/bold] {', '.join(args)}")
                    console.print()

                # Show docstring
                docstring = node.metadata.get('docstring')
                if docstring:
                    console.print(f"[bold]Description:[/bold]")
                    console.print(f"[dim]{docstring[:200]}{'...' if len(docstring) > 200 else ''}[/dim]")
                    console.print()

                # Show callers
                callers = graph.get_callers(node.name)
                if callers:
                    console.print(f"[bold]Called by:[/bold]")
                    for caller in callers[:10]:  # Limit to 10
                        console.print(f"  • {caller.name} ({caller.source_file})")
                    if len(callers) > 10:
                        console.print(f"  [dim]... and {len(callers) - 10} more[/dim]")
                    console.print()

                # Show callees
                callees = graph.get_callees(node.name)
                if callees:
                    console.print(f"[bold]Calls:[/bold]")
                    for callee in callees[:10]:  # Limit to 10
                        console.print(f"  • {callee.name}")
                    if len(callees) > 10:
                        console.print(f"  [dim]... and {len(callees) - 10} more[/dim]")
                    console.print()

            elif node.type == 'module':
                # Show imports
                imports = graph.get_dependencies(node.source_file)
                if imports:
                    console.print(f"[bold]Imports:[/bold]")
                    for imp in imports[:15]:  # Limit to 15
                        console.print(f"  • {imp.name}")
                    if len(imports) > 15:
                        console.print(f"  [dim]... and {len(imports) - 15} more[/dim]")
                    console.print()

            # Show source location
            line = node.metadata.get('line')
            if line:
                console.print(f"[dim]Location: {node.source_file}:{line}[/dim]")
            else:
                console.print(f"[dim]File: {node.source_file}[/dim]")
            console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


@cli.command()
@click.option('--workspace', type=click.Path(),
              help=f'Workspace directory (default: {WORKSPACE_DIR})')
def graph_stats(workspace: Optional[str]):
    """
    Show knowledge graph statistics.

    Displays counts of entities and relationships in the code graph.
    """
    if workspace:
        global WORKSPACE_DIR
        WORKSPACE_DIR = Path(workspace)

    graph_db_path = WORKSPACE_DIR / "code_graph.db"

    if not graph_db_path.exists():
        console.print("[red]Error: Knowledge graph not built yet[/red]")
        console.print("[dim]Run 'rag build-graph' first[/dim]")
        raise click.Abort()

    try:
        from .knowledge_graph import CodeKnowledgeGraph

        graph = CodeKnowledgeGraph(str(graph_db_path))

        console.print()
        console.print(Panel.fit(
            "[bold cyan]Knowledge Graph Statistics[/bold cyan]",
            border_style="cyan"
        ))
        console.print()

        # Count nodes by type
        node_counts = {}
        for node_id, data in graph.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        # Count edges by relationship
        edge_counts = {}
        for _, _, data in graph.graph.edges(data=True):
            rel_type = data.get('relationship', 'unknown')
            edge_counts[rel_type] = edge_counts.get(rel_type, 0) + 1

        # Nodes table
        console.print("[bold]Nodes (Entities):[/bold]")
        nodes_table = Table(box=box.SIMPLE, show_header=True)
        nodes_table.add_column("Type", style="cyan")
        nodes_table.add_column("Count", justify="right")

        for node_type, count in sorted(node_counts.items()):
            nodes_table.add_row(node_type.capitalize(), str(count))

        nodes_table.add_row("", "", end_section=True)
        nodes_table.add_row("[bold]Total[/bold]", f"[bold]{sum(node_counts.values())}[/bold]")

        console.print(nodes_table)
        console.print()

        # Edges table
        console.print("[bold]Edges (Relationships):[/bold]")
        edges_table = Table(box=box.SIMPLE, show_header=True)
        edges_table.add_column("Relationship", style="cyan")
        edges_table.add_column("Count", justify="right")

        for rel_type, count in sorted(edge_counts.items()):
            edges_table.add_row(rel_type.replace('_', ' ').title(), str(count))

        edges_table.add_row("", "", end_section=True)
        edges_table.add_row("[bold]Total[/bold]", f"[bold]{sum(edge_counts.values())}[/bold]")

        console.print(edges_table)
        console.print()
        console.print(f"[dim]Graph database: {graph_db_path}[/dim]")
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.option('--format', 'output_format', type=click.Choice(['dot', 'html', 'mermaid', 'ascii-deps', 'ascii-class']),
              default='html', help='Output format for visualization')
@click.option('--output', '-o', type=click.Path(), help='Output file path (auto-generated if not provided)')
@click.option('--node-types', type=str, help='Comma-separated list of node types to include (e.g., "class,function")')
@click.option('--max-nodes', type=int, default=100, help='Maximum number of nodes to include')
@click.option('--entity', type=str, help='Entity name for ASCII tree visualizations (module for deps, class for hierarchy)')
@click.option('--workspace', type=click.Path(exists=True), help='Workspace directory (overrides default)')
def graph_viz(output_format, output, node_types, max_nodes, entity, workspace):
    """
    Visualize the knowledge graph in various formats.

    Examples:
        rag graph-viz --format html
        rag graph-viz --format dot --output graph.dot --node-types class,function
        rag graph-viz --format mermaid --max-nodes 50
        rag graph-viz --format ascii-deps --entity src/rag_system/pipeline.py
        rag graph-viz --format ascii-class --entity DocumentIngester
    """
    try:
        from .knowledge_graph import CodeKnowledgeGraph
        from .graph_visualizer import GraphVisualizer
    except ImportError as e:
        console.print(f"[red]Error: Knowledge graph modules not available: {e}[/red]")
        raise click.Abort()

    try:
        # Determine workspace
        if workspace:
            workspace_path = Path(workspace)
        else:
            workspace_path = WORKSPACE_DIR

        graph_db_path = workspace_path / "code_graph.db"

        if not graph_db_path.exists():
            console.print(f"[red]Knowledge graph not found at {graph_db_path}[/red]")
            console.print("[yellow]Run 'rag build-graph <path>' first to build the graph.[/yellow]")
            raise click.Abort()

        # Load graph
        graph = CodeKnowledgeGraph(str(graph_db_path))
        visualizer = GraphVisualizer(graph)

        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Generating {output_format.upper()} Visualization[/bold cyan]",
            border_style="cyan"
        ))
        console.print()

        # Parse node types filter
        node_types_list = None
        if node_types:
            node_types_list = [t.strip() for t in node_types.split(',')]

        # ASCII tree visualizations
        if output_format == 'ascii-deps':
            if not entity:
                console.print("[red]Error: --entity required for ascii-deps format[/red]")
                console.print("[yellow]Specify a module path, e.g., --entity src/rag_system/pipeline.py[/yellow]")
                raise click.Abort()

            tree_output = visualizer.dependency_tree_ascii(entity)
            console.print(tree_output)
            console.print()

            if output:
                Path(output).write_text(tree_output)
                console.print(f"[green]✓[/green] Saved to {output}")

            return

        elif output_format == 'ascii-class':
            if not entity:
                console.print("[red]Error: --entity required for ascii-class format[/red]")
                console.print("[yellow]Specify a class name, e.g., --entity DocumentIngester[/yellow]")
                raise click.Abort()

            tree_output = visualizer.class_hierarchy_ascii(entity)
            console.print(tree_output)
            console.print()

            if output:
                Path(output).write_text(tree_output)
                console.print(f"[green]✓[/green] Saved to {output}")

            return

        # File-based visualizations
        if not output:
            # Auto-generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extensions = {'dot': 'dot', 'html': 'html', 'mermaid': 'mmd'}
            output = f"graph_viz_{timestamp}.{extensions[output_format]}"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Generating {output_format} visualization...", total=None)

            if output_format == 'dot':
                result_path = visualizer.to_dot(output, node_types=node_types_list, max_nodes=max_nodes)
            elif output_format == 'html':
                result_path = visualizer.to_html(output, max_nodes=max_nodes)
            elif output_format == 'mermaid':
                result_path = visualizer.to_mermaid(output, node_types=node_types_list, max_nodes=max_nodes)

            progress.update(task, completed=True)

        console.print()
        console.print(f"[green]✓[/green] Visualization created: [bold]{result_path}[/bold]")

        # Format-specific instructions
        if output_format == 'dot':
            console.print()
            console.print("[dim]To generate an image:[/dim]")
            console.print(f"[dim]  dot -Tpng {result_path} -o graph.png[/dim]")
            console.print(f"[dim]  dot -Tsvg {result_path} -o graph.svg[/dim]")
        elif output_format == 'html':
            console.print()
            console.print("[dim]Open in browser to explore interactively:[/dim]")
            console.print(f"[dim]  open {result_path}[/dim]")

        console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.option('--format', 'output_format', type=click.Choice(['json', 'cypher', 'graphml', 'csv', 'adjacency']),
              default='json', help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file/directory path (auto-generated if not provided)')
@click.option('--pretty/--compact', default=True, help='Pretty-print JSON output (JSON format only)')
@click.option('--workspace', type=click.Path(exists=True), help='Workspace directory (overrides default)')
def graph_export(output_format, output, pretty, workspace):
    """
    Export knowledge graph to various formats.

    Examples:
        rag graph-export --format json
        rag graph-export --format cypher --output neo4j_import.cypher
        rag graph-export --format graphml
        rag graph-export --format csv --output ./exports/
        rag graph-export --format json --compact
    """
    try:
        from .knowledge_graph import CodeKnowledgeGraph
        from .graph_exporter import GraphExporter
    except ImportError as e:
        console.print(f"[red]Error: Knowledge graph modules not available: {e}[/red]")
        raise click.Abort()

    try:
        # Determine workspace
        if workspace:
            workspace_path = Path(workspace)
        else:
            workspace_path = WORKSPACE_DIR

        graph_db_path = workspace_path / "code_graph.db"

        if not graph_db_path.exists():
            console.print(f"[red]Knowledge graph not found at {graph_db_path}[/red]")
            console.print("[yellow]Run 'rag build-graph <path>' first to build the graph.[/yellow]")
            raise click.Abort()

        # Load graph
        graph = CodeKnowledgeGraph(str(graph_db_path))
        exporter = GraphExporter(graph)

        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Exporting to {output_format.upper()} Format[/bold cyan]",
            border_style="cyan"
        ))
        console.print()

        # Auto-generate output filename if not provided
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_format == 'csv':
                output = f"graph_export_{timestamp}"  # Directory for CSV
            else:
                extensions = {'json': 'json', 'cypher': 'cypher', 'graphml': 'graphml', 'adjacency': 'txt'}
                output = f"graph_export_{timestamp}.{extensions[output_format]}"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Exporting to {output_format}...", total=None)

            if output_format == 'json':
                result_path = exporter.to_json(output, pretty=pretty)
                console.print(f"[green]✓[/green] Exported to: [bold]{result_path}[/bold]")

            elif output_format == 'cypher':
                result_path = exporter.to_cypher(output)
                console.print(f"[green]✓[/green] Exported to: [bold]{result_path}[/bold]")
                console.print()
                console.print("[dim]To import into Neo4j:[/dim]")
                console.print(f"[dim]  cat {result_path} | cypher-shell -u neo4j -p <password>[/dim]")

            elif output_format == 'graphml':
                result_path = exporter.to_graphml(output)
                console.print(f"[green]✓[/green] Exported to: [bold]{result_path}[/bold]")

            elif output_format == 'csv':
                result_paths = exporter.to_csv(output)
                console.print(f"[green]✓[/green] Exported to:")
                console.print(f"  Nodes: [bold]{result_paths['nodes']}[/bold]")
                console.print(f"  Edges: [bold]{result_paths['edges']}[/bold]")

            elif output_format == 'adjacency':
                result_path = exporter.to_adjacency_list(output)
                console.print(f"[green]✓[/green] Exported to: [bold]{result_path}[/bold]")

            progress.update(task, completed=True)

        # Show statistics
        console.print()
        console.print("[bold]Graph Statistics:[/bold]")
        stats = exporter.get_statistics()

        stats_table = Table(box=box.SIMPLE, show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right")

        stats_table.add_row("Total Nodes", str(stats['total_nodes']))
        stats_table.add_row("Total Edges", str(stats['total_edges']))
        stats_table.add_row("Avg In-Degree", f"{stats['average_in_degree']:.2f}")
        stats_table.add_row("Avg Out-Degree", f"{stats['average_out_degree']:.2f}")

        console.print(stats_table)
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


def main():
    """Entry point for the rag command."""
    cli()


if __name__ == "__main__":
    main()
