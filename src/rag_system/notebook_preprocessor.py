"""
Jupyter Notebook Preprocessor for RAG

Converts .ipynb files into clean JSON chunks with rich metadata.
Each chunk is a self-contained unit with:
- Content (markdown or code)
- Full metadata (file, course, week, section, author)
- Context preservation (headings, cell order)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class NotebookPreprocessor:
    """Preprocesses Jupyter notebooks into clean JSON chunks for RAG."""

    def __init__(self):
        self.current_heading_stack = []  # Track nested headings

    def extract_metadata_from_path(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file path structure."""
        parts = file_path.parts
        metadata = {
            "source_file": str(file_path),
            "file_name": file_path.name,
            "file_type": "jupyter_notebook",
            "modified_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }

        # Extract course info (e.g., CMPINF_2100)
        for part in parts:
            if "CMPINF" in part or "INFSCI" in part:
                metadata["course"] = part.replace("_", " ")

        # Extract week/module
        for part in parts:
            if part.startswith("week_"):
                metadata["week"] = part.replace("week_", "").replace("_", " ")
            elif "module" in part.lower():
                # Extract module number from filenames like "module_05_..."
                match = re.search(r'module[_\s]?(\d+)', part, re.IGNORECASE)
                if match:
                    metadata["module"] = match.group(1)

        # Assignment type
        filename_lower = file_path.name.lower()
        if "midterm" in filename_lower:
            metadata["assignment_type"] = "midterm"
        elif "final" in filename_lower:
            metadata["assignment_type"] = "final"
        elif "project" in filename_lower:
            metadata["assignment_type"] = "project"
        elif "homework" in filename_lower or "hw" in filename_lower:
            metadata["assignment_type"] = "homework"
        else:
            metadata["assignment_type"] = "coursework"

        # Detect author (student work vs course materials)
        if "Beach" in file_path.name or "Shaun" in file_path.name:
            metadata["author"] = "Shaun Beach"
            metadata["doc_category"] = "student_work"
        elif "checkpoint" in filename_lower:
            metadata["doc_category"] = "checkpoint"
            metadata["author"] = "auto_save"
        else:
            metadata["doc_category"] = "course_material"
            metadata["author"] = "instructor"

        return metadata

    def extract_heading_from_markdown(self, markdown_text: str) -> Optional[str]:
        """Extract the first heading from markdown cell."""
        lines = markdown_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # Remove # symbols and clean
                heading = re.sub(r'^#+\s*', '', line).strip()
                return heading
        return None

    def update_heading_stack(self, markdown_text: str):
        """Update the current heading hierarchy from markdown."""
        lines = markdown_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # Count heading level
                level = len(re.match(r'^#+', line).group())
                heading = re.sub(r'^#+\s*', '', line).strip()

                # Update stack based on heading level
                # H1 = level 1, H2 = level 2, etc.
                if level <= len(self.current_heading_stack):
                    # Replace or pop to this level
                    self.current_heading_stack = self.current_heading_stack[:level-1]

                self.current_heading_stack.append({
                    "level": level,
                    "text": heading
                })

    def get_current_section_path(self) -> str:
        """Get hierarchical section path (e.g., 'Introduction > Data Loading > Step 1')."""
        return " > ".join([h["text"] for h in self.current_heading_stack])

    def clean_markdown_content(self, content: str) -> str:
        """Clean markdown content for better search."""
        # Remove excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        # Clean up spacing
        content = content.strip()
        return content

    def clean_code_content(self, content: str) -> str:
        """Clean code content."""
        # Just strip excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()
        return content

    def process_notebook(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a Jupyter notebook into clean JSON chunks.

        Returns:
            List of chunk dictionaries with content and metadata
        """
        # Load notebook
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Extract base metadata
        base_metadata = self.extract_metadata_from_path(file_path)

        # Reset heading stack for this notebook
        self.current_heading_stack = []

        chunks = []
        cell_index = 0

        for cell in notebook.get('cells', []):
            cell_type = cell.get('cell_type')
            source = cell.get('source', [])

            # Convert source to string
            if isinstance(source, list):
                content = ''.join(source)
            else:
                content = source

            # Skip empty cells
            if not content.strip():
                continue

            # Skip cells with only whitespace or comments
            if cell_type == 'code' and not content.strip().replace('#', '').strip():
                continue

            # Process based on cell type
            if cell_type == 'markdown':
                # Update heading hierarchy
                self.update_heading_stack(content)

                # Clean content
                cleaned_content = self.clean_markdown_content(content)

                if not cleaned_content:
                    continue

                # Create chunk
                chunk = {
                    "content": cleaned_content,
                    "content_type": "markdown",
                    "section_heading": self.get_current_section_path(),
                    "cell_index": cell_index,
                    "metadata": {
                        **base_metadata,
                        "cell_type": "markdown",
                    }
                }
                chunks.append(chunk)

            elif cell_type == 'code':
                # Clean code
                cleaned_code = self.clean_code_content(content)

                if not cleaned_code:
                    continue

                # Check if there's an output we should note
                outputs = cell.get('outputs', [])
                has_output = len(outputs) > 0

                # Detect code purpose from comments or patterns
                code_purpose = self._detect_code_purpose(cleaned_code)

                # Create chunk
                chunk = {
                    "content": cleaned_code,
                    "content_type": "code",
                    "section_heading": self.get_current_section_path(),
                    "cell_index": cell_index,
                    "metadata": {
                        **base_metadata,
                        "cell_type": "code",
                        "has_output": has_output,
                        "code_purpose": code_purpose,
                    }
                }
                chunks.append(chunk)

            cell_index += 1

        return chunks

    def _detect_code_purpose(self, code: str) -> str:
        """Detect the purpose of code block from patterns."""
        code_lower = code.lower()

        # Check for common patterns
        if 'import' in code_lower and code_lower.strip().startswith('import'):
            return "imports"
        elif 'def ' in code_lower:
            return "function_definition"
        elif 'class ' in code_lower:
            return "class_definition"
        elif any(word in code_lower for word in ['plot', 'plt.', 'sns.', 'figure']):
            return "visualization"
        elif any(word in code_lower for word in ['read_csv', 'load', 'open(']):
            return "data_loading"
        elif any(word in code_lower for word in ['merge', 'join', 'concat']):
            return "data_merging"
        elif any(word in code_lower for word in ['groupby', 'agg', 'pivot']):
            return "data_aggregation"
        elif any(word in code_lower for word in ['test', 'assert', 'unittest']):
            return "testing"
        else:
            return "general"

    def process_directory(self, directory: Path, pattern: str = "*.ipynb") -> List[Dict[str, Any]]:
        """Process all notebooks in a directory."""
        all_chunks = []

        for notebook_path in directory.rglob(pattern):
            # Skip checkpoint files
            if '.ipynb_checkpoints' in str(notebook_path):
                continue

            try:
                chunks = self.process_notebook(notebook_path)
                all_chunks.extend(chunks)
                print(f"✓ Processed: {notebook_path.name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"✗ Error processing {notebook_path.name}: {e}")

        return all_chunks

    def save_to_jsonl(self, chunks: List[Dict[str, Any]], output_path: Path):
        """Save chunks to JSONL format (one JSON object per line)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        print(f"\n✓ Saved {len(chunks)} chunks to {output_path}")


# Example usage
if __name__ == "__main__":
    preprocessor = NotebookPreprocessor()

    # Process a single notebook (replace with your own path)
    # notebook_path = Path("path/to/your/notebook.ipynb")
    #
    # if notebook_path.exists():
    #     chunks = preprocessor.process_notebook(notebook_path)
    #     print(f"\nProcessed {len(chunks)} chunks from {notebook_path.name}")
    #     print("\nFirst chunk example:")
    #     print(json.dumps(chunks[0], indent=2))

    print("NotebookPreprocessor ready. Update the example path to test.")
