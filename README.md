# Your Project Name

**Template repository** - Replace this content with your project's documentation.

## Setting up a new project (initial setup, one-time)

1. Clone this repository
   ```bash
   git clone https://github.com/SparkBeyond/template_package_project.git your-project-name
   cd your-project-name
   ```

2. Update remote repository (replace with your repository URL)
   ```bash
   git remote set-url origin https://github.com/SparkBeyond/your-repo.git
   ```

3. Rename the project (replace `your_package_name`)
   - Update `name` in `pyproject.toml`
   - Rename `src/template_package` to `src/your_package_name`
   - Update imports in your code
   - Update `packages` in `pyproject.toml` if needed
   - Update documentation

4. Make initial commit
   ```bash
   git add .
   git commit -m "Initial commit: Project setup"
   ```

5. Update Quick Start section with your repository URL, and proceed to [Quick Start](#quick-start)

6. Push changes

7. Remove this section from README.md

## Quick Start

1. Clone this repository
   ```bash
   git clone https://github.com/SparkBeyond/your-repo.git your-project-name
   cd your-project-name
   ```

2. Set up the development environment (see [Development Guide](./docs/development/dev_guide.md))

## Project Structure

- `src/` - Your package source code
- `tests/` - Test files
- `docs/` - Project documentation

## Development

See the [Development Guide](./docs/development/dev_guide.md) for setup instructions.

### Coding Standards

Refer to [CODING_STANDARDS.md](./CODING_STANDARDS.md) for detailed guidelines.

---

*Replace this README with your project's documentation, including installation instructions, usage examples, and any other relevant information.*
