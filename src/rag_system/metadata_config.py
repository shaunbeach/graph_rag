"""
Flexible Metadata Configuration for RAG System

Allows users to define custom metadata extraction rules via config file or environment variables.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class MetadataExtractor:
    """
    Flexible metadata extractor that can be configured via JSON config file.

    Config structure:
    {
        "enabled": true,
        "rules": [
            {
                "field": "assignment_type",
                "type": "filename_pattern",
                "patterns": [
                    {"match": "midterm", "value": "midterm"},
                    {"match": "final", "value": "final"}
                ]
            },
            {
                "field": "author",
                "type": "filename_contains",
                "patterns": [
                    {"match": "Beach", "value": "Shaun Beach"},
                    {"match": "john_doe", "value": "John Doe"}
                ]
            },
            {
                "field": "project",
                "type": "path_contains",
                "patterns": [
                    {"match": "ProjectA", "value": "Project A"},
                    {"match": "ProjectB", "value": "Project B"}
                ]
            }
        ]
    }
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize metadata extractor.

        Args:
            config_path: Path to JSON config file. If None, looks for .rag_metadata.json
                        in current directory or user home directory
        """
        self.config = self._load_config(config_path)
        self.enabled = self.config.get("enabled", False)
        self.rules = self.config.get("rules", [])

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load metadata configuration from file or environment."""

        # Priority 1: Explicit config path
        if config_path and config_path.exists():
            with open(config_path) as f:
                return json.load(f)

        # Priority 2: Environment variable
        config_env = os.getenv("RAG_METADATA_CONFIG")
        if config_env:
            try:
                return json.loads(config_env)
            except json.JSONDecodeError:
                # Treat as file path
                config_file = Path(config_env)
                if config_file.exists():
                    with open(config_file) as f:
                        return json.load(f)

        # Priority 3: Current directory
        local_config = Path.cwd() / ".rag_metadata.json"
        if local_config.exists():
            with open(local_config) as f:
                return json.load(f)

        # Priority 4: Home directory
        home_config = Path.home() / ".rag_metadata.json"
        if home_config.exists():
            with open(home_config) as f:
                return json.load(f)

        # Default: disabled
        return {"enabled": False, "rules": []}

    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from file path according to configured rules.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary of extracted metadata fields
        """
        # Base metadata (always included)
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": self._detect_file_type(file_path),
            "modified_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }

        # Skip custom extraction if disabled
        if not self.enabled:
            return metadata

        # Apply custom rules
        filename_lower = file_path.name.lower()
        path_str = str(file_path).lower()

        for rule in self.rules:
            field = rule.get("field")
            rule_type = rule.get("type")
            patterns = rule.get("patterns", [])

            if rule_type == "filename_pattern":
                for pattern in patterns:
                    match_str = pattern.get("match", "").lower()
                    if match_str in filename_lower:
                        metadata[field] = pattern.get("value")
                        break

            elif rule_type == "filename_contains":
                for pattern in patterns:
                    match_str = pattern.get("match")
                    if match_str in file_path.name:  # Case-sensitive
                        metadata[field] = pattern.get("value")
                        break

            elif rule_type == "path_contains":
                for pattern in patterns:
                    match_str = pattern.get("match")
                    if match_str in str(file_path):  # Case-sensitive
                        metadata[field] = pattern.get("value")
                        break

            elif rule_type == "regex":
                regex_pattern = rule.get("regex")
                source = rule.get("source", "filename")  # filename or path

                search_string = file_path.name if source == "filename" else str(file_path)
                match = re.search(regex_pattern, search_string, re.IGNORECASE)

                if match:
                    if match.groups():
                        metadata[field] = match.group(1)
                    else:
                        metadata[field] = pattern.get("value", match.group(0))

        return metadata

    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type from extension."""
        suffix = file_path.suffix.lower()
        type_map = {
            '.ipynb': 'jupyter_notebook',
            '.py': 'python',
            '.md': 'markdown',
            '.pdf': 'pdf',
            '.txt': 'text',
            '.html': 'html',
            '.json': 'json',
        }
        return type_map.get(suffix, 'unknown')

    def get_config_template(self) -> Dict[str, Any]:
        """Return a template configuration for users to customize."""
        return {
            "enabled": True,
            "description": "Custom metadata extraction rules for RAG system",
            "rules": [
                {
                    "field": "category",
                    "type": "path_contains",
                    "description": "Extract document category from path",
                    "patterns": [
                        {"match": "/docs/", "value": "documentation"},
                        {"match": "/code/", "value": "source_code"},
                        {"match": "/notes/", "value": "notes"}
                    ]
                },
                {
                    "field": "project",
                    "type": "filename_pattern",
                    "description": "Extract project name from filename",
                    "patterns": [
                        {"match": "project_a", "value": "Project A"},
                        {"match": "project_b", "value": "Project B"}
                    ]
                },
                {
                    "field": "version",
                    "type": "regex",
                    "description": "Extract version from filename (e.g., v1.2, v2.0)",
                    "regex": r"v(\d+\.\d+)",
                    "source": "filename"
                }
            ]
        }


def create_academic_config() -> Dict[str, Any]:
    """
    Create a pre-built config for academic/course use cases.
    Users can use this as a starting point.
    """
    return {
        "enabled": True,
        "description": "Academic metadata extraction (courses, assignments, etc.)",
        "rules": [
            {
                "field": "assignment_type",
                "type": "filename_pattern",
                "patterns": [
                    {"match": "midterm", "value": "midterm"},
                    {"match": "final", "value": "final"},
                    {"match": "homework", "value": "homework"},
                    {"match": "hw", "value": "homework"},
                    {"match": "project", "value": "project"},
                    {"match": "lab", "value": "lab"}
                ]
            },
            {
                "field": "course",
                "type": "path_contains",
                "patterns": [
                    {"match": "CMPINF", "value": "Computer Science"},
                    {"match": "INFSCI", "value": "Information Science"}
                ]
            },
            {
                "field": "author",
                "type": "filename_contains",
                "patterns": [
                    {"match": "Beach", "value": "Shaun Beach"}
                ]
            },
            {
                "field": "week",
                "type": "regex",
                "regex": r"week[_\s]?(\d+)",
                "source": "filename"
            },
            {
                "field": "module",
                "type": "regex",
                "regex": r"module[_\s]?(\d+)",
                "source": "filename"
            }
        ]
    }


# CLI helper function
def generate_config_file(output_path: Path, template: str = "generic"):
    """
    Generate a config file for users to customize.

    Args:
        output_path: Where to save the config
        template: 'generic' or 'academic'
    """
    extractor = MetadataExtractor()

    if template == "academic":
        config = create_academic_config()
    else:
        config = extractor.get_config_template()

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created metadata config at: {output_path}")
    print(f"Edit this file to customize metadata extraction for your use case.")


if __name__ == "__main__":
    # Example usage
    extractor = MetadataExtractor()

    # Test with a sample file (replace with your own path)
    # test_file = Path("path/to/your/notebook.ipynb")
    # metadata = extractor.extract_metadata(test_file)
    # print("Extracted metadata:")
    # print(json.dumps(metadata, indent=2))

    print("MetadataExtractor ready. Update the example path to test.")
