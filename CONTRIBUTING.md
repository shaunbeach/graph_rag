# Contributing to RAG System

Thank you for considering contributing to RAG System! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up development environment** (see below)
4. **Create a branch** for your changes
5. **Make your changes**
6. **Test your changes**
7. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.8 or higher
- git
- Virtual environment tool (venv, virtualenv, or conda)

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rag-system.git
cd rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (if using)
pre-commit install
```

### Development Dependencies

The development setup includes additional tools:

```bash
# Testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Code quality
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.0.0

# Documentation
sphinx>=6.0.0
sphinx-rtd-theme>=1.2.0
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-new-loader` - For new features
- `fix/query-bug` - For bug fixes
- `docs/update-readme` - For documentation
- `refactor/ingestion-module` - For refactoring

### Commit Messages

Write clear, descriptive commit messages:

```
Add support for DOCX file ingestion

- Implement DOCX loader using python-docx
- Add tests for DOCX processing
- Update documentation with DOCX examples
```

Format:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description (wrap at 72 chars)

### Types of Contributions

#### Bug Fixes

1. Create an issue describing the bug
2. Reference the issue in your PR
3. Include tests that fail before your fix
4. Ensure tests pass after your fix

#### New Features

1. Discuss in an issue first (for large features)
2. Implement the feature
3. Add tests
4. Update documentation
5. Add entry to CHANGELOG.md

#### Documentation

1. Fix typos, improve clarity
2. Add examples
3. Update API documentation
4. Add tutorials or guides

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rag_system --cov-report=html

# Run specific test file
pytest tests/test_basic.py

# Run specific test
pytest tests/test_basic.py::test_ingest_single_file

# Skip slow tests
pytest -m "not slow"
```

### Writing Tests

All new features should include tests:

```python
def test_new_feature(temp_workspace, sample_documents):
    """Test description."""
    # Arrange
    vectorstore_path = temp_workspace / "vector_store"

    # Act
    result = ingest_documents(
        source_path=sample_documents,
        vectorstore_path=vectorstore_path,
    )

    # Assert
    assert result['files_processed'] > 0
```

### Test Coverage

- Aim for >80% code coverage
- All new features must have tests
- Bug fixes should include regression tests

## Submitting Changes

### Pull Request Process

1. **Update your fork**
   ```bash
   git remote add upstream https://github.com/original/rag-system.git
   git fetch upstream
   git merge upstream/main
   ```

2. **Push to your fork**
   ```bash
   git push origin feature/your-feature
   ```

3. **Create Pull Request** on GitHub

4. **PR Description** should include:
   - What changes you made
   - Why you made them
   - Related issue numbers
   - Testing performed
   - Breaking changes (if any)

### PR Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123

## Changes Made
- Added X
- Fixed Y
- Updated Z

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed

## Documentation
- [ ] README updated
- [ ] CHANGELOG updated
- [ ] Code comments added

## Breaking Changes
None / Description of breaking changes
```

### Review Process

1. Automated checks must pass
2. Code review by maintainers
3. Address review feedback
4. Approval required before merge

## Coding Standards

### Python Style

Follow PEP 8 style guide:

```bash
# Format code with black
black src/

# Check with flake8
flake8 src/

# Sort imports with isort
isort src/
```

### Code Organization

```python
"""
Module docstring describing purpose.
"""

import standard_library
import third_party
import local_modules


class MyClass:
    """Class docstring."""

    def __init__(self, param: str):
        """Initialize instance."""
        self.param = param

    def method(self) -> str:
        """Method docstring."""
        return self.param


def function(arg: str) -> str:
    """
    Function docstring.

    Args:
        arg: Description

    Returns:
        Description
    """
    return arg.upper()
```

### Type Hints

Use type hints for all functions:

```python
from typing import List, Dict, Any, Optional

def ingest_files(
    file_paths: List[Path],
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """Function with type hints."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def complex_function(param1: str, param2: int) -> bool:
    """
    Brief description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Examples:
        >>> complex_function("test", 5)
        True
    """
    if param2 < 0:
        raise ValueError("param2 must be positive")
    return len(param1) > param2
```

## Documentation

### Updating Documentation

When making changes:

1. Update relevant `.md` files
2. Add docstrings to new code
3. Update examples if needed
4. Add entry to CHANGELOG.md

### Documentation Structure

```
docs/
├── INSTALLATION.md    # Installation instructions
├── QUICK_START.md     # Quick start guide
└── API.md            # API documentation (future)

examples/
└── basic_usage.md    # Usage examples

README.md             # Main documentation
CHANGELOG.md          # Version history
CONTRIBUTING.md       # This file
```

### Building Documentation (Future)

```bash
# Build Sphinx docs
cd docs
make html

# View docs
open _build/html/index.html
```

## Issue Guidelines

### Reporting Bugs

Include:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- System information (OS, Python version)
- Error messages/logs

Template:

```markdown
**Description**
Brief description of the bug

**Steps to Reproduce**
1. Run command X
2. Observe error Y

**Expected Behavior**
Should do Z

**Actual Behavior**
Does W instead

**Environment**
- OS: macOS 13.0
- Python: 3.10.5
- RAG System: 1.0.0

**Error Messages**
```
error log here
```
```

### Requesting Features

Include:
- Clear description of feature
- Use case/motivation
- Example usage
- Alternative solutions considered

## Questions?

- Open an issue for questions
- Tag with `question` label
- Check existing issues first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Appreciated by the community!

---

**Thank you for contributing to RAG System!** 🎉
