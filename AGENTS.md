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
- Format Python code with Ruff (`uv run ruff format arbor tests examples`).
- Lint and sort imports via Ruff (`uv run ruff check --fix arbor tests examples`); prefer explicit relative imports within `arbor.server` packages.
- Name modules and directories with lowercase underscores; use PascalCase for classes, snake_case for functions and variables.
- Register `pre-commit` hooks (`uv run pre-commit install`) so formatting and lightweight linting run automatically.

## Error Handling Guidelines
- Use the helpers in `arbor/server/utils/error_handling.py` for capturing and reporting failures (`LoggedError`, `ArborError`, `ErrorHandler`).
- Wrap risky GRPO operations with `with error_handler.context("step name", error_type=TrainingError):` to ensure consistent logging and escalation.
- Expose new service-level errors by subclassing `ArborError` and wiring them through `ErrorHandler.raise_if_failed` instead of ad-hoc exceptions.

## Testing Guidelines
- Testing currently disabled

## Commit & Pull Request Guidelines
- Write imperative, concise commit subjects (`Make dict frozenset`, `Process event`); keep them under ~72 characters and avoid `WIP` once ready for review.
- Squash noisy fixup commits locally; ensure the tree is formatted and tests pass.
- PRs need a short summary, linked issues if applicable, test evidence (command output or screenshots), and notes on API or config changes.
- Highlight operational impacts (GPU requirements, new env vars) in the PR description so deployment owners can assess risk.
