# Copilot Developer Instructions

## 1. Core Philosophy & Architecture
- **Functional Programming:** Strictly use pure functions for business logic. No OOP or logic-bearing classes.
- **Data Structures Only:** Classes (e.g., `dataclass`, `TypedDict`, `pydantic.BaseModel`) are strictly reserved for data modeling and must contain zero methods.
- **YAGNI (You Aren't Gonna Need It):** Solve the immediate problem. Strictly avoid premature abstractions, layers, factories, wrappers, or "just in case" patterns. 
- **Feature Folders:** Organize code by domain/feature (e.g., `news_pipeline/`, `price_updater/`), not by technical artifact (e.g., `routers/`, `services/`).
- **DRY:** Strictly avoid code duplication. Extract reusable pure functions.
- **Tagged Unions:** Model domain states, events, and distinct data shapes using Tagged Unions (e.g., `TypeA | TypeB` with a literal `type` field discriminator) to enforce precise type-safety across pipelines.

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

## 4. Optimization & Performance
- **Batching is Mandatory:** Always optimize for batched database I/O. Use Qdrant's batch operations for point insertions, semantic vector searches, and crucially, **batch payload/metadata updates** when querying via filters (e.g., matching `published_at` to backfill realized price deltas).
- **Agent Efficiency:** Ensure LangGraph state transitions and LLM calls (e.g., GPT-5 Nano vs. Mini routing) are optimized for concurrent execution and latency.

## 5. Testing & Comments
- **Test Core Logic:** Write unit and integration tests focusing on domain logic (e.g., deduplication vector thresholds, LangGraph routing fallbacks, delta math) and edge cases. Execute via `uv`.
- **Signal-to-Noise in Comments:** Comment only to explain *why* a complex decision was made (e.g., specific Qdrant filter mechanics, threshold values, or API quirks). Do not comment obvious code.

## 6. Documentation
- **Keep README in sync:** When making code changes that affect architecture, configuration, pipeline flows, or local dev workflow, update `README.md` to reflect those changes in the same commit.
