# Contributing to DA4HiFlow

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install in development mode with dev dependencies:
   ```bash
   python -m pip install -e ".[dev]"
   ```

## Code Standards

- Type hints where reasonable
- Deterministic seeds in examples (for reproducibility)
- Keep diffs small and focused

## Before Submitting

Run the full test suite and linting:

```bash
ruff check .
ruff format .
pytest -q
```

All checks must pass before submitting a PR.

## Issues

For bugs or feature requests, please open an issue on GitHub.
