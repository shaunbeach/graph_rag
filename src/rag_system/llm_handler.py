"""
LLM Handler for RAG Generation

Supports multiple LLM backends:
- OpenAI (GPT-4, GPT-3.5-turbo)
- Anthropic Claude
- Local models via Ollama
"""

import os
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path


def get_available_ollama_models() -> List[str]:
    """
    Get list of available models from Ollama.

    Returns:
        List of model names available in Ollama
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return sorted(models)
        return []
    except:
        return []


def get_ollama_model_info() -> List[Dict[str, Any]]:
    """
    Get detailed information about available Ollama models.

    Returns:
        List of dictionaries with model information (name, size, modified date)
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get('models', []):
                models.append({
                    'name': model['name'],
                    'size': model.get('size', 0),
                    'modified_at': model.get('modified_at', ''),
                    'details': model.get('details', {})
                })
            return sorted(models, key=lambda x: x['name'])
        return []
    except:
        return []


def get_openai_models(api_key: str) -> List[str]:
    """
    Get available OpenAI models.

    Args:
        api_key: OpenAI API key

    Returns:
        List of model names
    """
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            # Filter to GPT models only
            models = [m['id'] for m in data.get('data', [])
                     if 'gpt' in m['id'].lower()]
            return sorted(models, reverse=True)  # Newest first
        return []
    except:
        # Fallback to common models
        return [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]


def get_anthropic_models(api_key: str) -> List[str]:
    """
    Get available Anthropic Claude models.

    Note: Anthropic doesn't provide a models API endpoint,
    so we return known models.

    Args:
        api_key: Anthropic API key (for future use)

    Returns:
        List of model names
    """
    # Anthropic doesn't have a public models API
    # Return current known models
    return [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]


def get_google_models(api_key: str) -> List[str]:
    """
    Get available Google Gemini models.

    Args:
        api_key: Google API key

    Returns:
        List of model names
    """
    try:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            # Extract model names and filter to Gemini models
            models = []
            for model in data.get('models', []):
                name = model.get('name', '')
                if 'gemini' in name.lower():
                    # Convert from "models/gemini-pro" to "gemini-pro"
                    model_id = name.split('/')[-1]
                    models.append(model_id)
            return sorted(models, reverse=True)
        return []
    except:
        # Fallback to known models
        return [
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "gemini-pro",
        ]


def get_openrouter_models(api_key: str) -> List[str]:
    """
    Get available OpenRouter models.

    Args:
        api_key: OpenRouter API key

    Returns:
        List of model names
    """
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            # Extract model IDs
            models = [m['id'] for m in data.get('data', [])]
            return sorted(models)
        return []
    except:
        # Fallback to popular models
        return [
            "openai/gpt-4-turbo",
            "anthropic/claude-3-5-sonnet",
            "google/gemini-pro",
            "meta-llama/llama-3.1-70b-instruct",
            "mistralai/mistral-large",
        ]


class LLMHandler:
    """Handles LLM generation for RAG answers."""

    def __init__(self, provider: str = "auto", model: Optional[str] = None):
        """
        Initialize LLM handler.

        Args:
            provider: LLM provider ('openai', 'anthropic', 'ollama', 'auto')
            model: Specific model name (optional, uses defaults)
        """
        self.provider = provider
        self.model = model
        self.llm = None

        if provider == "auto":
            self.provider = self._detect_available_provider()

        self._initialize_llm()

    def _detect_available_provider(self) -> str:
        """Auto-detect which LLM provider is available."""
        # Check for API keys in environment
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        elif os.getenv("GOOGLE_API_KEY"):
            return "google"
        elif os.getenv("OPENROUTER_API_KEY"):
            return "openrouter"
        else:
            # Default to Ollama (local, free)
            return "ollama"

    def _initialize_llm(self):
        """Initialize the LLM based on provider."""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "google":
            self._init_google()
        elif self.provider == "openrouter":
            self._init_openrouter()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _init_openai(self):
        """Initialize OpenAI LLM."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI support requires: pip install langchain-openai"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it with: export OPENAI_API_KEY='your-key-here'"
            )

        model = self.model or "gpt-4"
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,  # Low temperature for factual responses
            api_key=api_key
        )

    def _init_anthropic(self):
        """Initialize Anthropic Claude LLM."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "Anthropic support requires: pip install langchain-anthropic"
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
            )

        model = self.model or "claude-3-5-sonnet-20241022"
        self.llm = ChatAnthropic(
            model=model,
            temperature=0.1,
            api_key=api_key
        )

    def _init_google(self):
        """Initialize Google Gemini LLM."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "Google support requires: pip install langchain-google-genai"
            )

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Set it with: export GOOGLE_API_KEY='your-key-here'"
            )

        model = self.model or "gemini-1.5-pro-latest"
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,
            google_api_key=api_key
        )

    def _init_openrouter(self):
        """Initialize OpenRouter LLM."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "OpenRouter support requires: pip install langchain-openai"
            )

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Set it with: export OPENROUTER_API_KEY='your-key-here'"
            )

        model = self.model or "openai/gpt-4-turbo"
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/rag-system",
                "X-Title": "RAG System"
            }
        )

    def _init_ollama(self):
        """Initialize local Ollama LLM."""
        try:
            from langchain_ollama import OllamaLLM
        except ImportError:
            # Fallback to old import for compatibility
            try:
                from langchain_community.llms import Ollama as OllamaLLM
            except ImportError:
                raise ImportError(
                    "Ollama support requires: pip install langchain-ollama"
                )

        # Check if Ollama is running
        try:
            requests.get("http://localhost:11434", timeout=2)
        except:
            raise ConnectionError(
                "Ollama is not running. Install and start it:\n"
                "1. Install: brew install ollama (Mac) or visit https://ollama.ai\n"
                "2. Start: ollama serve\n"
                "3. Pull model: ollama pull llama3"
            )

        # Determine which model to use
        if self.model:
            # User specified a model
            model = self.model
        elif os.getenv("OLLAMA_MODEL"):
            # Use environment variable
            model = os.getenv("OLLAMA_MODEL")
        else:
            # Auto-detect available models
            available_models = get_available_ollama_models()
            if available_models:
                # Prefer certain models if available
                preferred_models = [
                    "Qwen3-4B-Instruct-2507:Q4_K_M",
                    "qwen2.5:latest",
                    "llama3.1:latest",
                    "llama3:latest",
                    "mistral:latest"
                ]
                model = None
                for preferred in preferred_models:
                    if preferred in available_models:
                        model = preferred
                        break
                if not model:
                    # Use first available model
                    model = available_models[0]
            else:
                # Fallback default
                model = "Qwen3-4B-Instruct-2507:Q4_K_M"

        # Set context window and max output tokens (matching web UI defaults)
        num_ctx = int(os.getenv("LLM_CONTEXT_WINDOW", "12000"))
        num_predict = int(os.getenv("LLM_MAX_TOKENS", "12000"))  # Max output tokens

        self.llm = OllamaLLM(
            model=model,
            temperature=0.1,
            num_ctx=num_ctx,        # Context window size
            num_predict=num_predict  # Max tokens in response
        )

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer based on retrieved documents.

        Args:
            query: User's question
            retrieved_docs: List of retrieved Document objects
            system_prompt: Optional custom system prompt

        Returns:
            Dictionary with:
            - answer: Generated answer text
            - sources: List of source documents used
            - model: Model name used
        """
        # Build context from retrieved documents
        context_parts = []
        sources = []

        for idx, doc in enumerate(retrieved_docs, 1):
            source_file = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            section = doc.metadata.get('section_heading', '')
            content_type = doc.metadata.get('content_type', '')

            context_parts.append(
                f"[Source {idx}]\n"
                f"File: {source_file}\n"
                f"Section: {section}\n"
                f"Type: {content_type}\n"
                f"Content:\n{doc.page_content}\n"
            )

            sources.append({
                "number": idx,
                "file": source_file,
                "section": section,
                "type": content_type,
                "preview": doc.page_content[:200]
            })

        context = "\n\n".join(context_parts)

        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant analyzing course materials and student work.
Your task is to answer questions based ONLY on the provided context from the documents.

Guidelines:
- Provide clear, accurate answers based on the context
- Cite your sources using [Source N] notation
- For code questions, combine code snippets from multiple sources to provide complete implementations
- If code is split across sources, reconstruct the complete function/class by merging the relevant parts
- If the context contains partial code, provide what's available and note what's missing
- For concept questions, provide clear explanations with examples from the context
- Be concise but thorough
- NEVER say information is "not provided" if it exists in ANY of the sources - combine them"""

        # Build prompt
        prompt = f"""{system_prompt}

Context from retrieved documents:

{context}

Question: {query}

Answer (cite sources using [Source N] format):"""

        # Generate answer
        if self.provider in ["openai", "anthropic", "google", "openrouter"]:
            # Use chat models
            from langchain.schema import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Context:\n\n{context}\n\nQuestion: {query}")
            ]
            response = self.llm.invoke(messages)
            answer = response.content
        else:
            # Use completion models (Ollama)
            answer = self.llm.invoke(prompt)

        return {
            "answer": answer,
            "sources": sources,
            "model": f"{self.provider}/{self.model or 'default'}",
            "num_sources": len(sources)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test LLM initialization
    print("Testing LLM Handler...")

    try:
        handler = LLMHandler(provider="auto")
        print(f"✓ Initialized: {handler.provider}")

        # Mock document for testing
        from dataclasses import dataclass

        @dataclass
        class MockDoc:
            page_content: str
            metadata: Dict[str, Any]

        mock_docs = [
            MockDoc(
                page_content="KMeans clustering requires you to specify the number of clusters upfront. It is fast and efficient.",
                metadata={
                    "source_file": "week_09/clustering.ipynb",
                    "section_heading": "KMeans Overview",
                    "content_type": "markdown"
                }
            ),
            MockDoc(
                page_content="Hierarchical clustering does not require specifying the number of clusters. It creates a dendrogram showing all possible clusterings.",
                metadata={
                    "source_file": "week_09/clustering.ipynb",
                    "section_heading": "Hierarchical Clustering",
                    "content_type": "markdown"
                }
            )
        ]

        result = handler.generate_answer(
            query="When should I use hierarchical clustering vs KMeans?",
            retrieved_docs=mock_docs
        )

        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Sources: {result['num_sources']}")
        print(f"Model: {result['model']}")

    except Exception as e:
        print(f"✗ Error: {e}")
