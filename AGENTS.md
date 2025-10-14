# Repository Guidelines

## Project Structure & Module Organization
- `arbor/` holds the Python package: `cli.py` exposes the CLI, `client.py` wraps the OpenAI-compatible client, and `server/` contains API, orchestration, and service logic.
- `tests/` mirrors the package layout; tests are currently disabled
- `docs/` captures design notes and deployment guides; `examples/` contains runnable notebooks and scripts that double as smoke tests.
- Artifacts created by packaging (`arbor_ai.egg-info/`) are generated; do not edit by hand.

## Build, Test, and Development Commands
- Install with `uv pip install -e .[dev]` to get runtime and tooling deps.
- Run the API locally via `uv run arbor serve`; use `--config ~/.arbor/config.yaml` to point to custom GPU layouts.

## Coding Style & Naming Conventions
- Follow Black’s defaults (88-char lines); run `uv run black arbor tests` before committing.
- Keep imports sorted with `uv run isort arbor tests`; prefer explicit relative imports within `arbor.server` packages.
- Name modules and directories with lowercase underscores; use PascalCase for classes, snake_case for functions and variables.
- Register `pre-commit` hooks (`uv run pre-commit install`) so formatting and lightweight linting run automatically.

## Testing Guidelines
- Testing currently disabled

## Commit & Pull Request Guidelines
- Write imperative, concise commit subjects (`Make dict frozenset`, `Process event`); keep them under ~72 characters and avoid `WIP` once ready for review.
- Squash noisy fixup commits locally; ensure the tree is formatted and tests pass.
- PRs need a short summary, linked issues if applicable, test evidence (command output or screenshots), and notes on API or config changes.
- Highlight operational impacts (GPU requirements, new env vars) in the PR description so deployment owners can assess risk.
