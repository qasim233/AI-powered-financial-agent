# MEMORY.md — Buy or Wait? Implementation State

> **Purpose**: This file tracks all decisions, what has been implemented, what remains,
> and any deviations from SOLUTION.md. Updated after every component implementation.
> Use this to rebuild context if switching models mid-implementation.

---

## Stack Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Orchestration framework | LangGraph `StateGraph` | User mandated LangGraph; SOLUTION.md's pipeline maps to a linear graph |
| LLM abstraction | LangChain `ChatOpenAI` | User mandated LangChain; native OpenAI-compatible wrapper |
| Model | `gemma2:9b` via local Ollama | code/llm_client/client.py |
| OCR | EasyOCR | SOLUTION.md §5 |
| Data handling | pandas + csv stdlib | For CSV loading |
| Schemas | Pydantic v2 | For structured LLM output + data models |

## Deviations from SOLUTION.md (approved by user)

| SOLUTION.md says | Changed to | Reason |
|------------------|-----------|--------|
| Exclude all `pending` events from forecast (§3 line 80) | Include pending debits, exclude pending credits | problem_statement.md says "reserve pending debits"; user approved |
| Messages with blank `related_event_id` produce no claim (§5 point 6) | Process these messages per problem_statement.md rules | User said "Follow the problem_statement here too" |
| Handle only `settled`, `pending`, `cancelled`, `failed` statuses | Handle ALL status values per problem_statement.md | User said "Yes handle all status values" |

## Implementation Status

| # | Component | Status | Files |
|---|-----------|--------|-------|
| 0 | LLM Client + Usage Tracker | ✅ DONE | `llm_client/__init__.py`, `llm_client/client.py`, `llm_client/usage_tracker.py` |
| 1 | Data Loader & Currency Normalizer | ✅ DONE | `loaders/data_loader.py`, `currency/converter.py` |
| 2 | Unstructured Extraction (LLM + EasyOCR) | ⬜ TODO | `extraction/ocr.py`, `extraction/llm_extractor.py` |
| 3 | Conflict Resolution | ⬜ TODO | `resolution/conflict_resolver.py` |
| 4 | State Reconstruction / Recurrence Detection | ✅ DONE (Tested) | `state/recurrence.py` |
| 5 | 90-Day Forecast & Safety Engine | ⬜ TODO | `forecast/safety_engine.py` |
| 6 | Spending-Change Optimizer | ⬜ TODO | `optimizer/spending_optimizer.py` |
| 7 | Plan Ranker | ⬜ TODO | `ranking/plan_ranker.py` |
| 8 | Explanation Generation | ⬜ TODO | `explanation/explain_llm.py` |
| 9 | Output Assembler & Validator | ⬜ TODO | `output/validator.py` |
| 10 | Orchestrator (main.py + LangGraph) | ⬜ TODO | `main.py` |
| 11 | Evaluation Harness | ⬜ TODO | `evaluation/harness.py` |

## Component 0 Details — LLM Client + Usage Tracker

**Files created**:
- `code/llm_client/__init__.py` — package init, exports `LLMClient`, `UsageTracker`
- `code/llm_client/client.py` — Experiential gateway wrapper via `ChatOpenAI`
  - `extract()` — structured output with caching by `(source_id, prompt_version)`
  - `explain()` — free-text explanation, not cached
  - Error handling for API failures and parsing errors
  - Token usage extraction from `AIMessage.usage_metadata`
- `code/llm_client/usage_tracker.py` — call logging and report generation
  - `log_call()` — records timestamp, call_type, target_id, input/output tokens
  - `generate_report()` — writes `evaluation/usage_report.md` with per-model totals

**Also created**:
- `code/requirements.txt` — all project dependencies

## Dataset Summary

| File | Rows | Key for joining |
|------|------|-----------------|
| requests.csv | 250 | request_id, user_id |
| financial_profiles.csv | 275 | user_id |
| financial_events.csv | ~25,342 | event_id, user_id |
| exchange_rates.csv | 134 | rate_date, from/to_currency |
| request_payment_options.csv | 790 | payment_option_id, request_id |
| messages.csv | 215 | message_id, user_id, request_id, related_event_id |
| images.csv | 16 | image_id, user_id, request_id, related_event_id |
| Image files | 16 PNGs | dataset/media/images/ |
| Currencies | INR, ZAR, IDR, USD, EUR | — |

## Critical Dataset Discoveries (differs from SOLUTION.md assumptions)

| Field | SOLUTION.md assumed | Actual values in dataset |
|-------|---------------------|--------------------------|
| `flexibility` | `fixed`, `flexible` | `fixed`, `reducible`, `reducible_or_stoppable`, `stoppable` |
| `direction` | `debit`, `credit` | `debit`, `credit`, `non_cash` |
| `event_type` | unspecified | `debt_payment`, `expense`, `income`, `investment_purchase`, `investment_sale`, `investment_valuation`, `refund`, `subscription` |
| `status` | `settled`, `pending`, `cancelled`, `failed` | + `scheduled`, `unrealized` |

**Flexibility mapping for spending changes**:
- `reducible` → eligible for `reduce_to` only (category must be in user's `_reduce` list)
- `stoppable` → eligible for `stop` only (category must be in user's `_stop` list)
- `reducible_or_stoppable` → eligible for either (category checked against the matching list)
- `fixed` → not eligible for any spending change

**Exchange rate pairs**: EUR↔USD, EUR→ZAR, USD→IDR, USD→INR (+ inverses). BFS needed for cross-pairs like ZAR→INR.

## Component 1 Details — Data Loader & Currency Normalizer

**Files created**:
- `code/loaders/__init__.py` — exports all data models and DataLoader
- `code/loaders/data_loader.py` — CSV loading with typed dataclass records
  - 7 dataclasses: FinancialProfile, FinancialEvent, Request, PaymentOption, Message, ImageRecord, ExchangeRate
  - Primary indices: keyed by natural ID (event_id, request_id, etc.)
  - Secondary indices: events_by_user, messages_by_user, messages_by_request, images_by_user, images_by_event, payment_options_by_request
  - Handles blank fields (amount, linked_event_id, etc.) as None
  - Uses utf-8-sig encoding for BOM handling
- `code/currency/__init__.py` — exports CurrencyConverter
- `code/currency/converter.py` — BFS-based multi-hop currency converter
  - Per-date adjacency graphs with forward + inverse (1/rate) edges
  - BFS shortest path by hop count
  - Date fallback: nearest date with complete path (unbounded distance)
  - Hard failure if no path on any date
  - Fallback log for eval visibility

**Test results (verified)**:
- All 7 CSVs loaded successfully with correct counts
- All 16 blank-amount events correctly linked to images via images_by_event index
- Currency conversion: same-currency passthrough ✓, direct pair ✓, multi-hop BFS ✓, date fallback ✓

## Component 4 Details — State Reconstruction & Recurrence Heuristic Fix

**Fix applied directly to `code/state/recurrence.py`**:
1. **Description-Aware Grouping**: For variable categories (`groceries`, `transport`, `dining`, `shopping`, `entertainment`), grouped by `(category, description, direction)` to prevent artificial multi-series phantom debits. Fixed commitments remain grouped by `(category, direction)`.
2. **Series Liveness & 70% Delta Strictness**: Enforces that the most recent occurrence must have happened within `1.5 * interval` of `request_date` to prevent zombie projections from lapsed 2023 events, and requires $\ge 70\%$ of deltas to match the interval.
3. **Confirmed Salary Continuation**: Explicit confirmed salary anchors ongoing monthly cycles across the remainder of the 90-day window per SOLUTION.md §7.

**Verification Results against original baseline**:
- `request_01`: **100% PERFECT PASS** across all 8 fields (was 0 ZAR safe, now exactly 25,256 ZAR, `affordable_now`, `full_payment`, `plan = 2024-03-03:25256`).
- `request_12`: **100% PERFECT PASS** (`installments`, `2026-04-19:22590.19|...`).
- `request_16`: **100% PERFECT PASS** (`affordable_now`, `full_payment`, `2023-08-12:122500`).
- `request_09`: **100% PERFECT PASS** (`not_affordable`, `not_recommended`, `none`).
- Total passing sample requests immediately jumped from 0/25 (0%) to 4/25 (16%) with zero changes to any other component.
