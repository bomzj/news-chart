# News Chart

Maps news events to price candlesticks to display market-response correlation.
Crypto news collector that ingests news, deduplicates via semantic vectors, analyzes sentiment with AI agents, snapshots price, and tracks realized price movements over time.
**NOTE:** This MVP is 100% vibe-coded and support BTC only.

## What It Does

1. **Ingests news** from MarketAux API for configured crypto tickers (e.g. BTC)
2. **Deduplicates** using cosine similarity on embeddings — both within the current batch and against the last 24h in the vector DB
3. **Analyzes & filters** via a LangGraph `StateGraph` (conditional routing):
   - Junior analyst uses the Lite model to evaluate each news item in parallel
   - Analyst labels each item `noise`, `bullish`, `bearish`, or `uncertain`
   - Junior `uncertain` results route to Senior analyst using the Smart model
   - Noise and unresolved uncertainty are **discarded** (never stored) — only directional signals are kept
4. **Snapshots price** from Binance perpetual futures at ingestion time
5. **Stores enriched vectors** in Qdrant Cloud with full metadata
6. **Backfills realized price deltas** (1h, 24h, 7d, 30d) via a separate cron-triggered endpoint
7. **Exposes a dependency-free heartbeat endpoint** for keeping the Render service awake

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| Agent Orchestration | LangGraph |
| Observability | Langfuse |
| Vector Database | Qdrant Cloud |
| LLM Provider | Azure AI (Lite / Smart deployment aliases) |
| Embeddings | Azure AI text-embedding-3-large (256 dims) |
| Price Data | Binance USDⓈ-M Futures API with Spot API fallback |
| News Source | MarketAux API |
| Hosting | Render (back-end web service + front end) |
| Cron | https://cron-job.org |
| Language | Python 3.14 |
| Toolchain | uv |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Render Platform                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │ Cron: 15 min │────▶│ POST         │                         │
│  │ collect-news │     │ /api/collect- │                         │
│  │              │     │ news          │                         │
│  └──────────────┘     └──────┬───────┘                         │
│                              │                                  │
│  ┌──────────────┐     ┌──────┴────────┐                        │
│  │ Cron: 10 min │────▶│ GET          │                         │
│  │ heartbeat    │     │ /api/heartbeat│                        │
│  └──────────────┘     └──────┬────────┘                        │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         │            FastAPI App                   │            │
│         │                                         │            │
│         │  news_collector/   │   price_updater/   │            │
│         │  ┌─────────────┐   │   ┌─────────────┐  │            │
│         │  │ fetch_news  │   │   │ updater     │  │            │
│         │  │ dedup       │   │   │ (delta calc)│  │            │
│         │  │ graph (LG)  │   │   └──────┬──────┘  │            │
│         │  │ agents      │   │          │         │            │
│         │  │ store       │   │          │         │            │
│         │  └──────┬──────┘   │          │         │            │
│         │         │          │          │         │            │
│         │  shared/│          │          │         │            │
│         │  ┌──────┴──────────┴──────────┴──────┐  │            │
│         │  │ qdrant.py  embeddings.py  binance │  │            │
│         │  └──────────────────────────────────┘  │            │
│         └─────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
         │                    │                │
         ▼                    ▼                ▼
  ┌─────────────┐    ┌──────────────┐  ┌─────────────┐
  │ Qdrant Cloud│    │  Azure AI    │  │  Binance    │
  │ (vectors +  │    │  (embed +    │  │  Futures    │
  │  metadata)  │    │   LLM)       │  │  API        │
  └─────────────┘    └──────────────┘  └─────────────┘
         ▲
         │
  ┌─────────────┐
  │ MarketAux   │
  │ (news feed) │
  └─────────────┘
```
**Warning** Render on free tier shuts down a web service after 15 minutes without incoming requests, so call `GET /api/heartbeat` at least every 10 minutes via cron job. The heartbeat handler does not call Binance market-data endpoints, so it cannot trigger Binance rate limits or HTTP 418 responses.

Keep `/api/update-prices` on its own schedule when realized price deltas need to be backfilled; it is no longer used as the service heartbeat.

**Note** MarketAux API costs credits, to not exceed monthly budget we can call the API not often than 15 minutes.

The backend uses Binance Futures mark prices and klines when available. If the
Futures API is rate-limited or unavailable (including HTTP 418 responses from
an IP ban), it falls back to the equivalent public Spot ticker or kline
endpoint. Fallback prices are Spot last-trade/close prices rather than Futures
mark prices, but keep ingestion and price backfills running.

## Collector Flows

### `/api/collect-news` (every 15 minutes)

If a previous background collector run is still active, a new trigger is
skipped and returns `202 {"status":"already_running"}`. The legacy
`/api/read-news` route remains available as a compatibility alias.

Transient MarketAux request failures are retried once. If the request still
fails, that ticker is skipped and the background collector continues.

```
MarketAux API ──▶ Raw news articles (filtered by ticker)
       │
       ▼
Extract full text from article URLs (trafilatura)
       │
       ▼
Skip articles with unavailable URLs (403, 404, timeout → dropped from collector)
       │
       ▼
Condense oversized articles (>2000 chars) via the Lite model
       │
       ▼
Embed all articles (Azure AI batch) ──▶ 256-dim vectors
       │
       ▼
Langfuse deduplication observability:
  ├── deduplicate-news parent observation
  ├── dedup embedding generation
  ├── unfiltered Qdrant top-1 match for every fetched article
  └── best non-self intra-batch match for every article
       │
       ▼
Intra-batch dedup (pairwise cosine ≥ 0.90 → keep first)
       │
       ▼
RAG dedup vs Qdrant (last 24h, cosine ≥ 0.90 → discard)
       │
       ▼
Attach similar past news as context (with realized price data)
       │
       ▼
LangGraph StateGraph (graph.py):
  ┌─────────────────────────────────────────────────────────┐
  │ START → junior_analyst (Lite model)                    │
  │           │                                             │
  │           ├── noise → END (news dropped)                │
  │           ├── bullish/bearish → END (result kept)       │
  │           └── uncertain → senior_analyst (Smart model)  │
  │                              ├── bullish/bearish → keep │
  │                              └── noise/uncertain → drop │
  └─────────────────────────────────────────────────────────┘
  Batch: asyncio.gather over graph.ainvoke() per news item
       │
       ▼
Fetch Binance mark price (BTCUSDT perpetual) — only for kept news
       │
       ▼
Batch upsert to Qdrant (vector + full metadata payload)
```

### Langfuse Observability

The `deduplicate-news` parent includes the dedup embedding call, Qdrant
retrieval observations, and explicit `duplicate-check` observations.

Each analyzed news item also creates an `analyze-news` parent chain. The
Langfuse Azure OpenAI wrapper records each `classify-news` generation, including
the concrete Azure deployment model, prompt, response, latency, token usage,
and errors. Lite and Smart are internal aliases only; Langfuse continues to
record the configured concrete IDs (`gpt-5.4-nano` or `gpt-5.4-mini`). The
parent output records the Junior label, final label, whether escalation
occurred, and whether the item was stored or discarded.

The analyst label is a classification, not a probability. No numerical
confidence score or evaluator score is recorded. Human labels can be added
later for calibration and quality evaluation.

The collector records duplicate checks for later threshold and embedding
evaluation without changing the production deduplication decision. Qdrant audit
queries use the top-1 result with no payload filter and no score threshold; the
existing 24-hour filtered query remains responsible for deduplication and
context retrieval.

Each `duplicate-check` observation contains:

```json
{
  "input": {
    "left": {
       "published_at": "...",
       "description": "..."
    },
    "right": {
       "published_at": "...",
       "description": "..."
    }
  },
  "output": {
    "duplicate": false,
    "cosine_similarity": 0.85,
    "threshold": 0.9
  },
  "metadata": {
    "stage": "qdrant",
    "embedding_model": "text-embedding-3-large",
    "embedding_dimensions": 256
  }
}
```

`output.duplicate` is the current system classification (`cosine_similarity >= threshold`), not verified ground truth. Human labels should be added later as Langfuse scores or dataset expected outputs. For historical Qdrant matches, the logged `right.description` is populated from the existing `news_full_text` payload.

### `/api/update-prices` (scheduled separately)

```
Query Qdrant: news with null realized_price_delta fields
  where published_at is old enough for each delta window
       │
       ▼
Fetch historical prices from Binance klines at:
  published_at + 1h, +24h, +7d, +30d
       │
       ▼
Group requested timestamps into one-minute ranges of at most 1000 klines
       │
       ▼
Fetch each range in one Binance request; use Spot for remaining ranges after a Futures fallback
       │
       ▼
Calculate: (historical_price - price_at_ingestion) / price_at_ingestion
       │
       ▼
Batch update Qdrant payloads with realized deltas
```

The price updater coalesces historical lookups instead of making one Binance
request per news point. The HTTP endpoint also skips overlapping backfill runs,
so a slow run cannot multiply requests when the scheduler triggers again.

## Data Model (Qdrant Payload)

```json
{
  "ticker": "BTC",
  "source": "coindesk",
  "published_at": "2026-05-21T18:30:00Z",
  "news_summary": "The FED cuts rates by 50 bps, signaling aggressive easing...",
  "news_full_text": "Full article text...",
  "sentiment": "bullish",
  "impact": 3,
  "predicted_by_model": "gpt-5.4-nano",
  "price_at_ingestion": 68250.00,
  "realized_price_delta_pct_1h": null,
  "realized_price_delta_pct_24h": null,
  "realized_price_delta_pct_7d": null,
  "realized_price_delta_pct_30d": null
}
```

**Analyst label:** The internal four-way classifier returns `"noise"`,
`"bullish"`, `"bearish"`, or `"uncertain"`. Junior uncertainty is sent to the
Senior analyst. If Senior returns noise or remains uncertain, the article is
discarded. Only bullish and bearish results are stored, so the Qdrant payload
and chart API continue to expose binary `sentiment`.

**Impact Scale (1-3):**

| Level | Name | Timeframe | Examples |
|-------|------|-----------|----------|
| **1** | **Notable** | Moves price for hours | Coin listed on major exchange (Binance, Coinbase) · Significant token burn · Large whale movement · Successful protocol upgrade · Major partnership |
| **2** | **High** | Moves price for days | ETF approval/rejection · Major hack/exploit ($100M+) · Country bans/legalizes crypto · Institutional entry/exit (BlackRock, Fidelity) · Exchange withdrawal freeze |
| **3** | **Extreme** | Moves entire market for weeks | Fed rate decision · Major stablecoin depeg · Exchange collapse (FTX-level) · International regulatory crackdown · Global banking crisis |

## Configuration

### `config.yaml` — App Settings

All tunable parameters live here. No restart needed for threshold changes (reload on next request).

```yaml
tickers: [BTC]

dedup:
  cosine_threshold: 0.90    # similarity score to consider as duplicate
  context_similarity_threshold: 0.75  # minimum similarity for agent context
  lookback_hours: 168       # how far back to check for duplicates (one week)

agents:
  lite_model: gpt-5.4-nano    # Lite model
  smart_model: gpt-5.4-mini  # Smart model
  api_version: "2025-04-01-preview"
  reasoning_effort: high

embeddings:
  deployment: text-embedding-3-large
  dimensions: 256
  api_version: "2023-05-15"

collector:
  fetch_news_interval_minutes: 10
  max_full_text_chars: 2000    # articles longer than this get LLM-summarized

price_updater:
  deltas: [1h, 24h, 7d, 30d]
```

### Environment Variables — Secrets Only

```bash
MARKETAUX_API_KEY=...
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=...
AZURE_AI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_AI_API_KEY=...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
LANGFUSE_TRACING_ENABLED=true
```

Langfuse tracing is optional and disables itself when its API keys are not
configured. Each `/api/collect-news` execution records deduplication
observations and analyst classification traces; `/api/read-news` remains a
compatibility alias. The standard Azure OpenAI client remains the fallback
when tracing is disabled; the Langfuse OpenAI wrapper is opted into for analyst
generations, deduplication embeddings, and the dedup evaluation command.

## Evals

### News duplicate-detection evaluation

The first evaluation measures whether the embedding model and cosine-similarity
threshold correctly classify pairs of news articles as duplicates or distinct
articles. It runs as a Langfuse experiment and does not change the production
deduplication workflow or write anything to Qdrant.

Each experiment item in the `news-duplicate-pairs` dataset must contain two
article texts and a human-verified label:

```json
{
  "input": {
    "news_1": "First article text",
    "news_2": "Second article text"
  },
  "expected_output": {
    "duplicate": true
  }
}
```

`expected_output` may also be the boolean `true` or `false`. The evaluation
embeds both texts with the configured Azure embedding deployment and selected
embedding dimensions, calculates their cosine similarity, and predicts
`duplicate=true` when the score is at least the selected threshold. It emits
one score per item: `duplicate-correct` (`BOOLEAN`, `1` when the prediction
matches the label). Use manually verified labels for `expected_output`; the
similarity values recorded by the production audit are system predictions, not
ground truth.

Naming follows one rule: the decision is a **duplicate**.

#### Running the evaluation

Run the commands from `backend/` so the evaluator can load `.env` and
`config.yaml`:

```bash
cd backend
uv sync

# Uses the news-duplicate-pairs dataset plus dedup.cosine_threshold and
# embeddings.dimensions from config.yaml
uv run news-dedup-evals

# Evaluate a specific threshold
uv run news-dedup-evals --threshold 0.90

# Evaluate a specific threshold and embedding dimension
uv run news-dedup-evals --threshold 0.90 --dimensions 1024

# Evaluate another labeled dataset
uv run news-dedup-evals \
  --dataset another-news-dataset \
  --threshold 0.95 \
  --dimensions 512
```

Before running, configure the Azure AI and Langfuse variables in
`backend/.env`, including `AZURE_AI_ENDPOINT`, `AZURE_AI_API_KEY`,
`LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`. The command prints the
experiment result and records it in Langfuse. Each run is named
`dedup-threshold-<value>-dimensions-<value>`; run it with several threshold
and dimension combinations and compare the `duplicate-correct` scores in
Langfuse to choose settings that fit the labeled news pairs. Both options
default to `config.yaml` when omitted.

## Local Development

### Backend

```bash
cd backend

# Install dependencies with uv
uv sync

# Copy and fill in secrets
cp .env.example .env

# Run the API server
uv run start

# Trigger the news collector manually
curl -X POST http://localhost:8000/api/collect-news
# The legacy trigger remains available during the rename
curl -X POST http://localhost:8000/api/read-news
curl -X POST http://localhost:8000/api/update-prices

# Keep the backend awake without calling external APIs
curl http://localhost:8000/api/heartbeat

# Run tests
uv run pytest
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Copy env (defaults to localhost:8000 backend)
cp .env.example .env.local

# Run dev server (http://localhost:3000)
npm run dev

# Build static export
npm run build
```


### Frontend Data Flow

1. User selects ticker (e.g., `BTCUSDT`) and timeframe (e.g., `1d`)
2. Frontend fetches klines directly from Binance Futures public API
3. Chart renders candlesticks and subscribes to visible range changes
4. When visible range changes (scroll/zoom), frontend fetches news from backend for that time window
5. While the page remains open, the frontend refreshes the latest candles and visible news every five minutes and when the tab becomes visible
6. News items are aggregated per candle:
   - **Color** = sentiment (red = bearish, green = bullish)
   - **Size** = impact (1 = smallest circle, 3 = largest)
7. Impact slider filters out news below the threshold (client-side re-aggregation)
8. Click on a marker shows popup with news items sorted by impact (most impactful first)

### Frontend API Endpoint

```
GET /api/news?ticker=BTC&from_ts=2026-01-01T00:00:00Z&to_ts=2026-05-28T00:00:00Z&min_impact=2
```

Response:
```json
{
  "items": [
    {
      "published_at": "2026-05-21T18:30:00Z",
      "sentiment": "bullish",
      "impact": 3,
      "news_summary": "BTC ETF inflows hit record...",
      "predicted_by_model": "gpt-5.4-nano",
      "price_at_ingestion": 68250.00
    }
  ],
  "count": 42
}
```

### Frontend Local Development

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Copy env file and configure API URL
cp .env.example .env.local

# Run dev server
npm run dev
# → http://localhost:3000

# Build for production (static export)
npm run build
# Output: frontend/out/
```


## Deployment (Render)


The `render.yaml` at the repo root defines:
- **Web service**: FastAPI app serving collector + chart API endpoints
- **Static site**: Next.js frontend (static export to `out/`)
- **Cron job (15 min)**: hits `/api/collect-news`
- **Cron job (10 min)**: sends `GET /api/heartbeat` to keep the backend awake
- **Price backfill cron**: sends `POST /api/update-prices` on the desired backfill schedule

Set all environment variables in the Render dashboard:
- Backend: `MARKETAUX_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `AZURE_AI_ENDPOINT`, `AZURE_AI_API_KEY`
- Optional Langfuse tracing: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
  `LANGFUSE_BASE_URL`, `LANGFUSE_TRACING_ENVIRONMENT`, and
  `LANGFUSE_TRACING_ENABLED`
- Frontend: `NEXT_PUBLIC_API_URL` (set to the backend service's external URL)

Render does not load `backend/.env`. To enable Langfuse tracing, enter the
Langfuse public and secret keys in the backend service's Environment settings
and redeploy. If the keys are omitted, set `LANGFUSE_TRACING_ENABLED=false`;
the backend will continue using the standard Azure client without Langfuse
tracing.

The backend reads `config.yaml` from the repo for non-secret settings.