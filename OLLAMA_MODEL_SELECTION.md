# Ollama Model Selection - Dynamic LLM Detection

## Overview

The RAG system now features **dynamic Ollama model detection**, allowing you to select from any locally installed Ollama models instead of using a hardcoded default. The system automatically queries the Ollama API to discover available models and intelligently selects the best one.

## Features

### 1. Automatic Model Discovery
- Queries Ollama API (`http://localhost:11434/api/tags`) to get installed models
- No hardcoded model lists - always shows what's actually available
- Works with any Ollama model you have installed

### 2. Smart Model Selection Priority
When no model is explicitly specified, the system chooses models in this order:

1. User-specified model (via parameter)
2. `OLLAMA_MODEL` environment variable
3. **Auto-detection with preference order:**
   - `Qwen3-4B-Instruct-2507:Q4_K_M` (current default)
   - `qwen2.5:latest`
   - `llama3.1:latest`
   - `llama3:latest`
   - `mistral:latest`
4. First available model (if none of the preferred ones exist)
5. Fallback to `Qwen3-4B-Instruct-2507:Q4_K_M` if Ollama is offline

### 3. Streamlit Dashboard Integration

**Chat Tab - LLM Settings:**
- Dropdown showing all available Ollama models
- Displays model count (e.g., "4 models available in Ollama")
- Expandable panel showing:
  - Model names with sizes (formatted as GB/MB)
  - Installation instructions for new models
- Refresh button (🔄) to re-detect models after installing new ones
- Fallback to manual text input if Ollama is offline

**System Status Tab - Ollama Status:**
- Shows Ollama running status (✅ Running / ⚠️ Not running)
- Displays count of available models
- Lists all models with their sizes
- Setup instructions if Ollama is not detected

### 4. API Functions

Three new utility functions in `llm_handler.py`:

```python
# Get list of model names
get_available_ollama_models() -> List[str]

# Get detailed model information (name, size, modified date)
get_ollama_model_info() -> List[Dict[str, Any]]

# Format bytes to human-readable size (KB, MB, GB, TB)
format_bytes(bytes_size: int) -> str  # In streamlit app
```

## Usage

### Via Streamlit Dashboard

1. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

2. **Open the dashboard**:
   ```bash
   cd dashboard/streamlit_app
   streamlit run app.py
   ```

3. **Go to Chat tab**:
   - See dropdown populated with your installed models
   - Select any model from the list
   - Click refresh (🔄) after installing new models

### Via Python Code

```python
from rag_system.llm_handler import LLMHandler, get_available_ollama_models

# Auto-detect and use best available model
handler = LLMHandler(provider="ollama")

# Use specific model
handler = LLMHandler(provider="ollama", model="llama3.1:latest")

# Get available models
models = get_available_ollama_models()
print(f"Available: {models}")
```

### Via CLI

The CLI automatically uses the same smart detection:

```bash
# Uses auto-detected model
rag query "explain machine learning"

# Use specific model via environment variable
OLLAMA_MODEL=llama3.1:latest rag query "explain machine learning"
```

## Installing New Models

After installing new Ollama models, they'll be automatically detected:

```bash
# Install a new model
ollama pull llama3.1

# Or install multiple
ollama pull qwen2.5
ollama pull mistral

# In Streamlit, click the refresh button (🔄) to see new models
```

## Model Information Display

The system shows detailed information about each model:

**Example Display:**
```
Qwen3-4B-Instruct-2507:Q4_K_M - 2.6 GB
Qwen3-emb-0.6b:Q8_0 - 669.8 MB
llava:7b - 4.7 GB
mistral:latest - 4.1 GB
```

## Testing

Run the test script to verify Ollama detection:

```bash
python test_ollama_detection.py
```

**Expected Output:**
```
================================================================================
OLLAMA MODEL DETECTION TEST
================================================================================

1. Detecting available Ollama models...
✓ Found 4 models:
  1. Qwen3-4B-Instruct-2507:Q4_K_M
  2. Qwen3-emb-0.6b:Q8_0
  3. Seed-Augmenter:latest
  4. llava:7b

2. Testing LLMHandler auto-detection...
✓ LLMHandler initialized successfully
  Provider: ollama
  Model: Qwen3-4B-Instruct-2507:Q4_K_M

3. Testing with specific model: Qwen3-4B-Instruct-2507:Q4_K_M
✓ LLMHandler initialized with specific model
```

## Error Handling

### Ollama Not Running
- Streamlit shows warning: "⚠️ Could not detect Ollama models"
- Falls back to manual text input
- Displays setup instructions

### No Models Installed
- Detects empty model list
- Shows installation instructions:
  ```bash
  ollama pull llama3.1
  ollama pull qwen2.5
  ollama pull mistral
  ```

### Network Timeout
- API requests have 2-second timeout
- Graceful fallback to manual input or default model

## Implementation Details

### API Endpoint
```
GET http://localhost:11434/api/tags
```

**Response Format:**
```json
{
  "models": [
    {
      "name": "llama3.1:latest",
      "size": 4700000000,
      "modified_at": "2024-01-15T10:30:00Z",
      "details": {...}
    }
  ]
}
```

### File Changes

**Modified Files:**
1. `src/rag_system/llm_handler.py`:
   - Added `get_available_ollama_models()`
   - Added `get_ollama_model_info()`
   - Enhanced `_init_ollama()` with smart model selection

2. `dashboard/streamlit_app/app.py`:
   - Added `format_bytes()` helper
   - Dynamic Ollama model dropdown in Chat tab
   - Ollama status section in System Status tab
   - Model size display

3. `test_ollama_detection.py`:
   - New test script for verification

## Benefits

### 1. Flexibility
- Works with **any** Ollama model you install
- No code changes needed to support new models
- Use different models for different tasks

### 2. User Experience
- See exactly what's available on your system
- Model sizes help with selection (smaller = faster, larger = better)
- One-click refresh after installing new models

### 3. Automatic Updates
- Model list always current
- No stale hardcoded options
- Detects new models immediately

### 4. Robustness
- Graceful fallback if Ollama is offline
- Clear error messages and setup instructions
- Works in CLI, Python, and Streamlit

## Examples

### Use Case 1: Switch Between Models
```python
# Small, fast model for quick queries
handler = LLMHandler(provider="ollama", model="qwen2.5:latest")

# Large model for complex analysis
handler = LLMHandler(provider="ollama", model="llama3.1:70b")

# Vision model for image tasks
handler = LLMHandler(provider="ollama", model="llava:7b")
```

### Use Case 2: Model Comparison
In Streamlit:
1. Select `qwen2.5:latest` - ask a question
2. Select `llama3.1:latest` - ask the same question
3. Compare responses to see which model works better for your use case

### Use Case 3: Resource Management
Check model sizes before loading:
- `qwen2.5:latest` - 4.1 GB (good for most tasks)
- `llama3.1:70b` - 40+ GB (requires powerful hardware)
- `phi3:mini` - 1.9 GB (fast, resource-efficient)

## Future Enhancements

Possible additions:
1. Model performance metrics (tokens/sec)
2. Model recommendations based on query type
3. Automatic model switching based on query complexity
4. Model download/installation from UI
5. Model benchmarking results

## Conclusion

The dynamic Ollama model selection feature makes the RAG system more flexible and user-friendly. Instead of being locked to a single hardcoded model, you can now:

- Use any Ollama model you have installed
- See available models with their sizes
- Refresh the list after installing new models
- Get smart automatic selection with preference ordering
- Have graceful fallbacks when Ollama is offline

This gives you full control over which LLM powers your RAG system, making it adaptable to different hardware capabilities, use cases, and performance requirements.
