# Custom Metadata Extraction

The RAG system supports flexible, user-configurable metadata extraction from file paths and filenames. This allows you to filter queries by custom attributes like project name, document type, author, version, etc.

## Overview

Metadata extraction is **disabled by default** to keep the system simple. Enable it when you need custom file organization and filtering.

## Quick Start

### 1. Create a Config File

The system looks for `.rag_metadata.json` in your current directory or home directory.

Create a basic config:

```json
{
  "enabled": true,
  "description": "Custom metadata extraction rules",
  "rules": [
    {
      "field": "project",
      "type": "path_contains",
      "patterns": [
        {"match": "ProjectA", "value": "Project A"},
        {"match": "ProjectB", "value": "Project B"}
      ]
    },
    {
      "field": "version",
      "type": "regex",
      "regex": "v(\\d+\\.\\d+)",
      "source": "filename"
    }
  ]
}
```

### 2. Ingest Documents

Metadata is extracted during ingestion:

```bash
# Delete old database to rebuild with new metadata
rm -rf chroma_db/

# Ingest with new metadata rules
rag ingest /path/to/docs --recursive
```

### 3. Use Metadata in Dashboard

The metadata fields are automatically available in the Streamlit dashboard when querying documents. They appear in the source information for each retrieved chunk.

## Config Structure

### Rule Types

#### 1. `filename_pattern` (Case-insensitive)
Matches substrings in the filename.

```json
{
  "field": "doc_type",
  "type": "filename_pattern",
  "patterns": [
    {"match": "meeting", "value": "meeting_notes"},
    {"match": "proposal", "value": "proposal"}
  ]
}
```

**Example:**
- `weekly_meeting_notes.md` → `doc_type: "meeting_notes"`
- `project_proposal.pdf` → `doc_type: "proposal"`

#### 2. `filename_contains` (Case-sensitive)
Exact case-sensitive match in filename.

```json
{
  "field": "author",
  "type": "filename_contains",
  "patterns": [
    {"match": "JohnDoe", "value": "John Doe"},
    {"match": "JSmith", "value": "Jane Smith"}
  ]
}
```

**Example:**
- `Report_JohnDoe_v1.pdf` → `author: "John Doe"`

#### 3. `path_contains` (Case-sensitive)
Matches in the full file path.

```json
{
  "field": "department",
  "type": "path_contains",
  "patterns": [
    {"match": "/engineering/", "value": "Engineering"},
    {"match": "/marketing/", "value": "Marketing"}
  ]
}
```

**Example:**
- `/docs/engineering/spec.md` → `department: "Engineering"`

#### 4. `regex` (Powerful pattern matching)
Extract values using regular expressions.

```json
{
  "field": "sprint",
  "type": "regex",
  "regex": "sprint[_-]?(\\d+)",
  "source": "filename"
}
```

**Example:**
- `sprint-23_notes.md` → `sprint: "23"`
- `sprint_15_retrospective.pdf` → `sprint: "15"`

The `source` can be `"filename"` or `"path"`.

## Real-World Examples

### Example 1: Software Project

```json
{
  "enabled": true,
  "rules": [
    {
      "field": "component",
      "type": "path_contains",
      "patterns": [
        {"match": "/backend/", "value": "backend"},
        {"match": "/frontend/", "value": "frontend"},
        {"match": "/api/", "value": "api"}
      ]
    },
    {
      "field": "issue",
      "type": "regex",
      "regex": "JIRA-(\\d+)",
      "source": "filename"
    },
    {
      "field": "version",
      "type": "regex",
      "regex": "v(\\d+\\.\\d+\\.\\d+)",
      "source": "filename"
    }
  ]
}
```

**Files:**
- `/backend/auth/JIRA-1234_oauth_fix_v2.0.1.md` → `component: "backend"`, `issue: "1234"`, `version: "2.0.1"`

### Example 2: Academic Research

```json
{
  "enabled": true,
  "rules": [
    {
      "field": "experiment",
      "type": "regex",
      "regex": "exp[_-]?(\\d+)",
      "source": "filename"
    },
    {
      "field": "subject",
      "type": "path_contains",
      "patterns": [
        {"match": "/neuroscience/", "value": "neuroscience"},
        {"match": "/psychology/", "value": "psychology"}
      ]
    },
    {
      "field": "phase",
      "type": "filename_pattern",
      "patterns": [
        {"match": "pilot", "value": "pilot"},
        {"match": "main_study", "value": "main"},
        {"match": "followup", "value": "followup"}
      ]
    }
  ]
}
```

### Example 3: Business Documents

```json
{
  "enabled": true,
  "rules": [
    {
      "field": "quarter",
      "type": "regex",
      "regex": "(Q[1-4])[-_]?(\\d{4})",
      "source": "filename"
    },
    {
      "field": "doc_type",
      "type": "filename_pattern",
      "patterns": [
        {"match": "budget", "value": "budget"},
        {"match": "forecast", "value": "forecast"},
        {"match": "report", "value": "report"}
      ]
    },
    {
      "field": "confidential",
      "type": "filename_pattern",
      "patterns": [
        {"match": "confidential", "value": "true"},
        {"match": "internal", "value": "true"}
      ]
    }
  ]
}
```

## Config Priority

The system looks for config files in this order:

1. Explicit path (if provided to `MetadataExtractor`)
2. `RAG_METADATA_CONFIG` environment variable
3. `.rag_metadata.json` in current directory
4. `~/.rag_metadata.json` in home directory

## Best Practices

### 1. Start Simple
Begin with 2-3 fields and test before adding more.

### 2. Use Consistent Naming
Establish file naming conventions across your project:
- `project-name_doc-type_version.ext`
- `YYYY-MM-DD_subject_author.ext`

### 3. Test Your Patterns
Create a test file with various names and verify the metadata is extracted correctly:

```python
from rag_system.metadata_config import MetadataExtractor
from pathlib import Path

extractor = MetadataExtractor()
metadata = extractor.extract_metadata(Path("test_file.md"))
print(metadata)
```

### 4. Document Your Schema
Add a `description` field to each rule for future reference.

### 5. Version Control
- Commit `.rag_metadata.json.example` with example patterns
- Add `.rag_metadata.json` to `.gitignore` if user-specific

## Disabling Metadata Extraction

Set `"enabled": false` in the config:

```json
{
  "enabled": false,
  "rules": []
}
```

Or simply don't create a config file - it's disabled by default.

## Base Metadata (Always Included)

These fields are always extracted, regardless of config:

- `source` - Full file path
- `filename` - File name only
- `file_type` - Detected from extension (jupyter_notebook, python, markdown, pdf, etc.)
- `modified_date` - Last modified timestamp (ISO 8601 format)
- `section_heading` - Section heading (for Jupyter notebooks and structured docs)
- `content_type` - Type of content (markdown, code, text)
- `chunk_id` - Unique identifier for the chunk

## Jupyter Notebook Metadata

For Jupyter notebooks (`.ipynb` files), additional metadata is automatically extracted:

- `cell_type` - "markdown" or "code"
- `cell_index` - Position in notebook
- `has_output` - Whether code cell has output (for code cells)
- `code_purpose` - Detected purpose: imports, visualization, data_loading, etc.

## Troubleshooting

### Metadata Not Appearing?

1. Check config file exists and `"enabled": true`
2. Verify patterns match your actual filenames
3. Re-ingest documents (metadata is set during ingestion)
4. View metadata in dashboard when querying

### Patterns Not Matching?

Test your regex patterns:

```python
import re
pattern = r"sprint[_-]?(\d+)"
filename = "sprint_23_notes.md"
match = re.search(pattern, filename, re.IGNORECASE)
if match:
    print(f"Matched: {match.group(1)}")
```

### Config Not Loading?

Check config file location:

```bash
# Current directory
ls -la .rag_metadata.json

# Home directory
ls -la ~/.rag_metadata.json

# Or set via environment variable
export RAG_METADATA_CONFIG=/path/to/config.json
```

## Python API

Use metadata extraction programmatically:

```python
from rag_system.metadata_config import MetadataExtractor
from pathlib import Path

# Load config
extractor = MetadataExtractor()

# Extract metadata
metadata = extractor.extract_metadata(Path("my_file.pdf"))
print(metadata)

# Get a template
template = extractor.get_config_template()
print(template)

# Use with custom config path
extractor = MetadataExtractor(config_path=Path("custom_config.json"))
```

## Contributing

If you create a useful config template for a specific domain (legal, medical, finance, etc.), consider contributing it as a preset template!

## See Also

- [QUICK_START.md](QUICK_START.md) - Getting started guide
- [INSTALLATION.md](INSTALLATION.md) - Installation instructions
- [metadata_config.py](../src/rag_system/metadata_config.py) - Source code
- [notebook_preprocessor.py](../src/rag_system/notebook_preprocessor.py) - Jupyter notebook processing
