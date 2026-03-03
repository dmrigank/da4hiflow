# Project: DA4HiFlow

## Goal
Build a clean, contributor-friendly Python package to run data assimilation (DA) methods on high-speed-flow benchmarks.

## Milestone 1 (implement now)
Create repo skeleton + tooling + CI + minimal smoke test.

### Repo structure to create
- da4hiflow/da4hiflow/{core,systems,assimilators,viz}
- configs/, scripts/, tests/, docs/
- README.md, LICENSE, CONTRIBUTING.md

## Tech choices (Milestone 1)
- Python package managed by `pyproject.toml`
- Testing: pytest
- Linting/formatting: ruff (prefer ruff-format; avoid adding black unless necessary)
- Minimal dependencies: numpy, pytest, ruff only

## Coding standards
- Type hints where reasonable
- Deterministic seeds (even in dummy examples)
- Keep diffs small; no unnecessary refactors

## Acceptance criteria (must pass before finishing)
- `python -m pip install -e .` works
- `ruff check .` passes
- `pytest -q` passes
- Provide a smoke test that runs a tiny dummy System + dummy Assimilator loop for 3 steps

## Editing rules
- Don’t add heavy dependencies
- Don’t implement real CFD solvers yet
- If you add a CLI script, keep it minimal and documented in README