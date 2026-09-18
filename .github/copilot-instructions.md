# Copilot Developer Instructions

## 1. Core Philosophy & Architecture
- **Functional Programming:** Strictly use pure functions for business logic. No OOP or logic-bearing classes.
- **Data Structures Only:** Classes (e.g., `dataclass`, `TypedDict`, `pydantic.BaseModel`) are strictly reserved for data modeling and must contain zero methods.
- **YAGNI (You Aren't Gonna Need It):** Do not Overengineer! Solve the immediate problem. Strictly avoid premature abstractions, layers, factories, wrappers, or "just in case" patterns. Code should be as simple and concise and direct as possible.
- **Feature Folders:** Organize code by domain/feature, not by technical artifact.
- **File naming:** Use descriptive names for files that reflect their feature or domain, avoiding technical artifact-based names.
- **DRY:** Strictly avoid code duplication. Extract reusable pure functions.
- **Tagged Unions:** Model domain states, events, and distinct data shapes using Tagged Unions (e.g., `TypeA | TypeB` with a literal `type` field discriminator) to enforce precise type-safety across pipelines.
- **No nesting:** Avoid deeply nested code structures. Prefer early returns, guard clauses, and flat code structures to improve readability and maintainability. No nesting of if, loops, match-case, try-except and so on beyond one level, except for simple comprehensions or context managers. Always strive for flat and readable code.

## 2. Tech Stack & Tooling
- **Language:** Python 3.14
- **Toolchain:** `uv` (exclusively for running, testing, and dependency management)
- **Core Frameworks:** FastAPI (endpoints), LangGraph (agent orchestration)
- **Data & AI:** Qdrant Cloud (vector DB), Azure AI (LLMs)
- **Environment:** Render (serverless/cron-ready structure)

## 3. Naming Conventions
- **Pattern Matching:** Prefer idiomatic Python `match-case` structural pattern matching over nested `if-else` chains, especially when unpacking Tagged Unions, handling agent states, or routing pipeline conditions.
- **Eliminate Noise:** Drop redundant prefixes (`get_`, `is_`, `calculate_`).
    - *Bad:* `get_sentiment()`, `is_duplicate()`, `calculate_price_delta()`
    - *Good:* `sentiment()`, `duplicated()`, `price_delta()`
- **Actionable Context:** Use `verb_noun` only when representing a side effect or distinct pipeline action (e.g., `fetch_news()`, `update_prices()`).
- **No underscore prefix for function and variables names:** Avoid leading underscores for functions/variables even if they are intended to be private within a module.

## 4. Optimization & Performance
- **Batching is Mandatory:** Always optimize for batched database I/O. Use Qdrant's batch operations for point insertions, semantic vector searches, and crucially, **batch payload/metadata updates** when querying via filters (e.g., matching `published_at` to backfill realized price deltas).
- **Agent Efficiency:** Ensure LangGraph state transitions and LLM calls are optimized for concurrent execution and latency.

## 5. Testing & Comments
- **Test Core Logic:** Write unit and integration tests focusing on domain logic and edge cases. Execute via `uv`.
- **Signal-to-Noise in Comments:** Comment only to explain *why* a complex decision was made. Do not comment simple or obvious code.

## 6. Documentation
- **Keep README in sync:** When making code changes that affect architecture, configuration, pipeline flows, or local dev workflow, update `README.md` to reflect those changes in the same commit.
