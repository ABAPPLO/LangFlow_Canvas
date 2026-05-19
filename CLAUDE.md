# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Langflow is a visual workflow builder for AI-powered agents. It has a Python/FastAPI backend, React/TypeScript frontend, and a lightweight executor CLI (lfx). This is a customized fork with additional features including i18n (Chinese/English), media preview components, and Tencent Cloud COS integration.

## Prerequisites

- **Python:** 3.10–3.13
- **uv:** >=0.4 (Python package manager)
- **Node.js:** >=20.19.0 (v22.12 LTS recommended)
- **npm:** v10.9+
- **make:** For build coordination

## Common Commands

### Development Setup
```bash
make init              # Install all dependencies + pre-commit hooks
make run_cli           # Build and run Langflow (http://localhost:7860)
make run_clic          # Clean build and run (use when frontend issues occur)
```

### Development Mode (Hot Reload)
```bash
make backend           # FastAPI on port 7860 (terminal 1)
make frontend          # Vite dev server on port 3000 (terminal 2, proxies API to 7860)
```

For component development, enable dynamic loading:
```bash
LFX_DEV=1 make backend                    # Load all components dynamically
LFX_DEV=mistral,openai make backend       # Load only specific modules
```

### Code Quality
```bash
make format_backend    # Format Python (ruff) - run FIRST before lint
make format_frontend   # Format TypeScript/JS (biome)
make format            # Both
make lint              # mypy type checking
```

### Testing
```bash
make unit_tests                    # Backend unit tests (pytest, parallel)
make unit_tests async=false        # Sequential tests
uv run pytest path/to/test.py      # Single test file
uv run pytest path/to/test.py::test_name  # Single test

make test_frontend                 # Jest unit tests
make tests_frontend                # Playwright e2e tests
```

### Database Migrations
```bash
make alembic-revision message="Description"  # Create migration
make alembic-upgrade                         # Apply migrations
make alembic-downgrade                       # Rollback one version
make alembic-current                         # Show current revision
make alembic-check                           # Check migration status
```

### Component Index
```bash
make build_component_index        # Rebuild component index after component changes
uv run python scripts/build_component_index.py  # Or run directly
```

### Docker
```bash
make docker_build                 # Build Docker image
make docker_compose_up            # Start with docker-compose
make dcdev_up                     # Dev docker-compose
```

### Other Packages
```bash
make lfx_build / make lfx_test / make lfx_format / make lfx_lint  # LFX package
make sdk_build / make sdk_test                                     # SDK package
make docs              # Docusaurus dev server (port 3030)
make storybook         # Storybook dev server (port 6006)
```

## Architecture

### uv Workspace Structure
Three packages in a uv workspace (defined in root `pyproject.toml`):

1. **langflow** (root, v1.8.4) — Main package with all integrations. CLI entry: `langflow.langflow_launcher:main`
2. **langflow-base** (`src/backend/base/`, v0.8.4) — Core framework (API, services, graph re-exports from lfx). FastAPI app factory: `langflow.main:create_app`
3. **lfx** (`src/lfx/`, v0.3.4) — Standalone CLI for executing/serving flows (`lfx serve`, `lfx run`). Owns the graph execution engine and 100+ component integrations. Entry: `lfx.__main__:main`

Dependency chain: `langflow` → `langflow-base` → `lfx`

### Backend Structure
```
src/backend/base/langflow/
├── api/              # FastAPI routes — v1/ (mature), v2/ (emerging workflow API)
├── components/       # Built-in Langflow components
├── services/         # Service layer with dependency injection (get_service from langflow.services.deps)
│   ├── auth/         # Authentication
│   ├── database/     # SQLAlchemy/SQLModel models + Alembic migrations
│   ├── cache/        # Caching layer
│   ├── storage/      # File storage
│   ├── tracing/      # Observability (LangWatch, Opik)
│   └── ...           # chat, jobs, settings, store, task, telemetry, variable
├── graph/            # Re-exports Graph/Vertex/Edge from lfx
└── custom/           # Custom component framework (base Component class)
```

### Frontend Structure
```
src/frontend/src/
├── controllers/API/  # API client layer (api.tsx, services/, queries/)
├── stores/           # Zustand state management (~20 stores)
├── CustomNodes/      # Custom ReactFlow node types (GenericNode, NoteNode)
├── CustomEdges/      # Custom ReactFlow edges
├── icons/            # ~150+ SVG icon components for integrations
├── i18n/             # Internationalization (Chinese/English locales)
├── modals/           # ~20 modal dialogs
├── types/            # TypeScript type definitions
├── pages/            # Route pages (FlowPage, Playground, MainPage, etc.)
└── customization/    # Customization framework
```

Tech: React 19 + TypeScript + Vite 7, @xyflow/react for graph canvas, Tailwind CSS + Radix UI + shadcn, Zustand, @tanstack/react-query, react-router-dom, framer-motion, i18next

## Component Development

Components live in `src/backend/base/langflow/components/` and `src/lfx/src/lfx/components/`. To add a new component:

1. Create component class inheriting from `Component`
2. Define `display_name`, `description`, `icon`, `inputs`, `outputs`
3. Add to `__init__.py` (alphabetical order)
4. Run with `LFX_DEV=1 make backend` for hot reload
5. After finalizing, rebuild the component index: `make build_component_index`

**CRITICAL: Never rename a component's class name.** The class name serves as an identifier used to match components in saved flows and to flag them for updates in the UI. Renaming it will break existing flows that use that component.

### Component Structure
```python
from langflow.custom import Component
from langflow.io import MessageTextInput, Output

class MyComponent(Component):
    display_name = "My Component"
    description = "What it does"
    icon = "component-icon"  # Lucide icon name or custom

    inputs = [
        MessageTextInput(name="input_value", display_name="Input"),
    ]
    outputs = [
        Output(display_name="Output", name="output", method="process"),
    ]

    def process(self) -> Message:
        # Component logic
        return Message(text=self.input_value)
```

### Custom Icons
1. Create SVG component in `src/frontend/src/icons/YourIcon/`
2. Export with `forwardRef` and `isDark` prop for light/dark mode
3. Add to `lazyIconImports.ts` — key must match backend `icon` string exactly (case-sensitive)
4. Set `icon = "YourIcon"` in Python component

If no custom icon exists, use a [Lucide icon](https://lucide.dev/icons) name.

### Component Testing
Tests go in `src/backend/tests/unit/components/`. Use base classes from `src/backend/tests/base.py`:

| Base Class | Creates `client`? | Use Case |
|------------|-------------------|----------|
| `ComponentTestBaseWithClient` | Yes | Components needing API access during `run()` |
| `ComponentTestBaseWithoutClient` | No | Pure-logic components with no API calls |

Required fixtures: `component_class`, `default_kwargs`, `file_names_mapping` (list of `VersionComponentMapping` dicts for backward compat testing).

The base classes auto-provide tests: `test_latest_version`, `test_all_versions_have_a_file_name_defined`, and parametrized `test_component_versions`.

For testing without external APIs, use `MockLanguageModel` from `tests.unit.mock_language_model`.

## Testing Notes

- `@pytest.mark.api_key_required` — tests requiring external API keys
- `@pytest.mark.no_blockbuster` — skip blockbuster plugin
- `@pytest.mark.noclient` — skip client fixture creation
- Database tests may fail in batch but pass individually
- Pre-commit hooks require `uv run git commit`
- Always use `uv run` when running Python commands
- Context variables may not propagate in `asyncio.to_thread`

### Graph Testing Pattern
```python
from tests.unit.build_utils import create_flow, build_flow, get_build_events, consume_and_assert_stream

flow_id = await create_flow(client, json_flow, logged_in_headers)
build_response = await build_flow(client, flow_id, logged_in_headers)
events = await get_build_events(client, job_id, logged_in_headers)
await consume_and_assert_stream(events, job_id)
```

### API Testing Pattern
```python
# client fixture provides async httpx.AsyncClient with in-memory SQLite
# logged_in_headers fixture provides authenticated headers
async def test_endpoint(client, logged_in_headers):
    response = await client.post("api/v1/flows/", json=data, headers=logged_in_headers)
    assert response.status_code == 201
```

## Pre-commit Workflow

1. Run `make format_backend` (FIRST - saves time on lint fixes)
2. Run `make format_frontend`
3. Run `make lint`
4. Run `make unit_tests`
5. Commit changes (use `uv run git commit` if pre-commit hooks are enabled)

Pre-commit hooks include: ruff (lint+format), biome (lint+format, bans `any` type), detect-secrets, Alembic migration validation, starter project JSON validation, deprecated import checks.

## Frontend Formatting

Biome is used (not ESLint). Key enforced rules:
- No `any` types (error level)
- No unused imports (error level)
- No debugger statements (error level)
- 2-space indent, double quotes

## Version Management
```bash
make patch v=1.5.0  # Update version across all packages
```

This updates: `pyproject.toml`, `src/backend/base/pyproject.toml`, `src/frontend/package.json`

## Pull Request Guidelines

- Follow [semantic commit conventions](https://www.conventionalcommits.org/)
- Reference any issues fixed (e.g., `Fixes #1234`)
- Ensure all tests pass before submitting
