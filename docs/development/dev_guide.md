# Development Guide

This guide provides instructions for setting up your development environment and working with the project.

## Prerequisites

- [pyenv](https://github.com/pyenv/pyenv) - Python version management (optional)
- [Poetry](https://python-poetry.org/) - Python dependency management

## Environment Setup

### 1. Install Python with pyenv (Recommended)
Any way to install Python 3.12.6 is fine, pyenv is suggested as it's a common choice.

```bash
# Check if the Python version is already installed
pyenv install --list

# If the version is not installed, install it
pyenv install 3.12.6

# Set local Python version for this project
cd /path/to/project
pyenv local 3.12.6
```

### 2. Create and Activate Virtual Environment

```bash
# Navigate to your project directory
cd /path/to/project

# Create a virtual environment using the pyenv Python
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Poetry

Check if Poetry is already installed:

```bash
poetry --version
```

If Poetry is not installed, follow the [official installation instructions](https://python-poetry.org/docs/#installation).

```bash
# Configure Poetry to use the active Python
poetry env use python
```

### 4. Install Dependencies

```bash
# Install project dependencies
poetry install
```

### 5. Install Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### Running Tests

Tests should be placed in the `tests/` directory and follow the `test_*.py` naming pattern. The test discovery is automatically configured to find all tests in this directory.

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_example.py

# Run tests with verbose output
pytest -v
```

### Adding Dependencies

```bash
# Add a production dependency
poetry add <package-name>

# Add a development dependency
poetry add --group dev <package-name>
```

### Code Quality Checks

Pre-commit hooks will automatically run on each commit. You can also run them manually:

```bash
pre-commit run --all-files
```

You can also run static type checking and linting manually:

```bash
# Run ruff

poetry run ruff check conversation_simulator/

# Run mypy
poetry run mypy conversation_simulator/
```

### Applying Fixes Automatically (Ruff)
You can apply fixes automatically when possible:

```bash
# Fix issues automatically when possible
poetry run ruff check --fix conversation_simulator/
```

Following automatic fixes, check the changes and make sure they are correct.
