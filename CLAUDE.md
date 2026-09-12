# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Strix is

Strix (`strix-agent` on PyPI) is an open-source autonomous AI pentesting tool. It spawns a graph of LLM-driven agents that run a target's code dynamically inside a Docker sandbox, find vulnerabilities, and validate them with real proofs-of-concept. It ships two ways to run the *same* engine: the open-source local CLI (this repo, BYO LLM key, needs Docker) and the managed cloud (`strix cloud ...`, driven over REST). `AGENTS.md` is the canonical reference for **using** Strix (run modes, exit codes, cloud CLI, artifacts) — read it before touching run/CLI behavior.

## Commands

Python 3.12+, managed with `uv`. All dev commands run through `uv`/`make`:

- `make setup-dev` — install dev deps + pre-commit hooks (`uv sync` + `uv run pre-commit install`)
- `make check-all` — format, lint, type-check, security in one (ruff → ruff → mypy+pyright → bandit); must pass before a PR
- `make format` / `make lint` — `uv run ruff format .` / `uv run ruff check . --fix`
- `make type-check` — `uv run mypy strix/` **and** `uv run pyright strix/` (mypy is `strict = true`)
- `make security` — `uv run bandit -r strix/ -c pyproject.toml`
- `uv run pytest` — run the test suite (`asyncio_mode = auto`, so `async def test_*` needs no decorator)
- `uv run pytest tests/test_execution.py::test_name` — run a single test
- `uv run strix --target <target>` — run Strix from source (dev mode)

Runtime env (see `strix/config/settings.py` for the full set, all `STRIX_*` / `LLM_*` aliases):

```bash
export STRIX_LLM="openrouter/z-ai/glm-5.3"   # any LiteLLM model id
export LLM_API_KEY="<key>"
```

Requires a running Docker daemon — the agent workload executes in a sandbox container, not on the host.

### TUI (Go) and viewer (React) — only if you touch them

- `make tui-build` / `make tui-test` / `make tui-lint` — the Bubble Tea TUI under `strix/interface/tui/` (Go 1.24.x). Editable installs run it from source via `go run`; wheels bundle a compiled sidecar.
- `make viewer` — rebuild the `strix view` SPA (`strix/interface/viewer/frontend/`, Vite+React). The **built output** in `strix/interface/viewer/static/` is committed and shipped; end users never run a JS build. Commit both the source change and the regenerated `static/`.

## Architecture

The engine is built on `openai-agents` (with the SDK's `SandboxAgent` + `Filesystem`/`Shell` capabilities) and LiteLLM for provider routing. Flow of a scan:

- **`strix/interface/`** — entrypoint (`main.py` → `strix.interface.main:main`), CLI arg parsing (`cli_args.py`, `cli.py`), scan setup/preflight (`scan_setup.py`), interactive TUI launch (`interactive.py`), the `strix cloud` REST client (`cloud/`), and the `strix view` local dashboard (`viewer/`).
- **`strix/core/`** — the run engine. `runner.py` is the top-level scan runner; `execution.py` is the per-agent async loop (spawn, respawn, compaction, transient-retry); `agents.py` holds `AgentCoordinator` — the SDK-native **addressable agent graph** where agents have mailboxes, statuses, and wait-kinds and can message each other. `hooks.py` enforces the token/cost **budget**; `sessions.py` manages per-agent SDK sessions.
- **`strix/agents/`** — `factory.py` (`build_strix_agent`, `make_child_factory`) assembles a `SandboxAgent`: registers every host-side tool and renders the system prompt from `prompts/system_prompt.jinja` (`prompt.py`).
- **`strix/tools/`** — the pentest toolkit. Each family is a package with a `tool.py`/`tools.py` of host-side SDK function tools, imported directly by `factory.py`. Notable: `proxy/` (Caido HTTP intercept), `agent_browser/`, `shell/`, `agents_graph/` (create/message/stop/wait sub-agents), `reporting/` (vulnerability + dependency reports), `threat_model/`, `notes/`, `todo/`, `coverage/`, `web_search/`, `mcp/`, `load_skill/`, `finish/`. Sandbox shell + filesystem tools are *not* here — the SDK emits them per-run and binds them to the live container.
- **`strix/runtime/`** — sandbox lifecycle. `backends.py` selects a backend via `STRIX_RUNTIME_BACKEND` (default `docker`); `docker_client.py` injects `NET_ADMIN`/`NET_RAW` + `host.docker.internal`; `session_manager.py`, `caido_*` bootstrap the HTTP proxy.
- **`strix/config/`** — `settings.py` (pydantic-settings, env-driven), `models.py` (`StrixProvider` extends LiteLLM's `MultiProvider`; Codex/streaming/tool-schema quirks live here), `codex.py`, `loader.py`.
- **`strix/llm/`** — `compaction.py` (context auto-compaction on overflow), `context_budget.py`, `warmup.py` (import warmup to hide startup latency).
- **`strix/report/`** — findings state, vulnerability markdown, SARIF 2.1.0, PDF/DOCX export.
- **`strix/skills/`** — internal knowledge packs (vulnerability/framework/technology/etc. `.md`) that *pentest agents* dynamically load into their prompt (up to 5 per agent via `create_agent(..., skills=...)`). **Distinct** from `skills/` at the repo root, which are the consumer SKILL.md packs for coding agents that *use* Strix.
- **`containers/`** — the sandbox Docker image (`Dockerfile`, `docker-entrypoint.sh`).

Key mental model: one root agent can spawn a **graph of specialized child agents** (recon, exploitation, etc.) that run in parallel, share discoveries via messages, and each operate inside the shared Docker sandbox. Budget (cost/turns) and context compaction are enforced centrally in `core/`.

## Conventions

- ruff line-length 100, `py312` target, a broad rule set (see `pyproject.toml` `[tool.ruff.lint]`); mypy is fully `strict`. Type hints on all functions.
- Tests mirror source by feature name under `tests/` (flat, `test_<area>.py`); `tests/conftest.py` holds shared fixtures. `tests.*` relax the untyped-decorator rule.
- `strix/tools/__init__.py` intentionally does **not** import submodules eagerly — import tool families deeply so `import strix.tools` doesn't pull in every dependency.
- Only scan targets you are authorized to test.
