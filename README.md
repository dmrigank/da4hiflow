# DA4HiFlow: Data Assimilation for High-Speed Flow Benchmarks

A clean, contributor-friendly Python package for running data assimilation (DA) methods on high-speed-flow benchmarks.

## Quick Start (WSL/Linux)

### 1. Clone and setup

```bash
cd da4hiflow
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install in development mode

```bash
python -m pip install -e ".[dev]"
```

### 3. Run the smoke test

```bash
pytest -q
```

### 4. Run linting/formatting checks

```bash
ruff check .
ruff format .
```

### 5. Run the CLI stub

```bash
python scripts/run_experiment.py --config configs/linear1d_direct_insert.json
# optional arguments:
#   --output-root <dir>  (default "runs")
#   --run-name <name>    (uses timestamp if omitted)
```

## Testing

The package includes a minimal smoke test that runs a 3-step dummy DA loop:

```bash
pytest tests/test_smoke_runner.py -v
```

This test verifies:
- System initialization and stepping
- Assimilator blending logic
- Full DA loop execution and trajectory shape

## Project Structure

```
da4hiflow/
├── da4hiflow/              # Main package
│   ├── core/               # Base classes and runner
│   ├── systems/            # System implementations
│   ├── assimilators/       # DA method implementations
│   └── viz/                # Visualization utilities
├── tests/                  # Unit tests
├── scripts/                # CLI and utility scripts
├── configs/                # Configuration files
├── docs/                   # Documentation
├── pyproject.toml          # Package metadata and tool configs
└── README.md
```

## Development

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.
