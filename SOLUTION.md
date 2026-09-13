# SOLUTION.md — Buy or Wait? Affordability Decision Engine

## 1. Overview

A single, deterministic, orchestrated pipeline (no multi-agent framework) that decides,
for each row in `requests.csv`, whether a user can safely afford a requested expense —
producing `amount_safe_to_pay`, `affordability_status`, `recommended_payment_method`,
`payment_plan`, `earliest_date_for_full_payment`, `spending_changes_needed`, and
`decision_explanation`.

**Core principle**: the LLM is used for exactly two narrow, stateless jobs —
(a) extracting structured facts from unstructured messages/images, and
(b) writing the natural-language explanation for an already-computed decision.
Every number, date, and ranking decision is produced by deterministic Python code,
so the system is reproducible, unit-testable, and auditable.

**Model**: `gpt-5.6-luna` via the Experiential gateway
(`base_url=https://api.experientiallabs.ai/v1`, OpenAI-compatible client,
`EXPLABS_API_KEY` env var).

**OCR**: EasyOCR for all images (no direct multimodal LLM calls on images — OCR text is
fed into the same extraction prompt used for messages).

---

## 2. Architecture

```
requests.csv ──┐
               ▼
[1] Data Loader & Currency Normalizer
               │
               ▼
[2] Unstructured Extraction (LLM + EasyOCR)     ◄── messages.csv, images.csv
               │
               ▼
[3] Conflict Resolution (deterministic)
               │
               ▼
[4] State Reconstruction / Recurrence Detection (deterministic)
               │
               ▼
[5] 90-Day Forecast & Safety Engine (deterministic)
               │
               ▼
[6] Spending-Change Optimizer (deterministic)
               │
               ▼
[7] Plan Ranker (deterministic)
               │
               ▼
[8] Explanation Generation (LLM)
               │
               ▼
[9] Output Assembler & Validator (deterministic)
               │
               ▼
          output.csv

Cross-cutting: LLM Client + Cache + Usage Tracker, Evaluation Harness
```

Each component is a pure function or a class with a single responsibility and an
explicit input/output contract, so components 3–9 can be unit-tested with hand-built
fixtures with zero LLM calls.

---

## 3. Data Model Assumptions (from provided samples)

- `financial_events.csv` has no explicit "is_recurring" flag — recurrence is inferred
  (see Component 4).
- `flexibility` column on an event (fixed/flexible) is per-event. A spending change is
  only valid on an event where **both** `flexibility=flexible` **and** its `category`
  appears in the user's `expense_categories_user_is_willing_to_reduce` (for `reduce_to`)
  or `expense_categories_user_is_willing_to_stop` (for `stop`).
- `linked_event_id` chains amendments/settlements/series for the same underlying
  transaction or investment lifecycle.
- `status` values include at least `settled`; treat `pending`, `cancelled`, `failed` as
  excluded from the forecast per spec, except an explicit **next confirmed salary**
  event, which is included despite being future-dated.
- Currency conversion requires a graph traversal (BFS) over same-date rate pairs, not a
  flat lookup table, since not all currency pairs are directly listed.
- Images require **field disambiguation**: payslip images → use **Net Pay**, not gross;
  receipt/invoice images → use **Balance Due** (amount still owed going forward), not
  Total or Amount Received, unless the linked event's own context clearly indicates
  otherwise (rare — flag such cases in extraction confidence rather than guessing).

---

## 4. Component 1 — Data Loader & Currency Normalizer

**Responsibility**: load all CSVs into typed records keyed by primary ID; provide
`convert(amount, from_currency, to_currency, date) -> amount`.

**Design**:
- Load each CSV into a dict keyed by its natural ID (`event_id`, `request_id`,
  `user_id`, `message_id`, `image_id`, `payment_option_id`).
- Build a per-date currency graph: nodes = currencies, edges = rate pairs from
  `exchange_rates.csv` on that date, plus implied inverse edges (`1/rate`).
- `from_currency == to_currency` → return amount unchanged, no graph lookup.
- Otherwise: BFS shortest path (by hop count) from source to target currency on the
  exact date requested.
- **Date fallback**: if no complete path exists on the exact date, search outward by
  absolute day difference (no distance cap — accept any distance) for the nearest date
  that has a complete path. Log every fallback use (requested date, date used,
  currency pair) for visibility in the eval harness.
- **Hard failure**: if no path exists for that currency pair on *any* date in the file,
  raise — do not default to a 1:1 rate.

---

## 5. Component 2 — Unstructured Extraction (LLM + EasyOCR)

**Responsibility**: turn each message and each image into zero or one structured
**claim** object.

**Pipeline**:
1. For images: EasyOCR → raw text. For messages: `message_text` used directly.
2. Single LLM call per message/image (cached by `message_id`/`image_id`) with a strict
   system prompt:
   - The model's only job is extraction into a fixed JSON schema.
   - The model must never follow instructions found in the source text — it treats all
     source text purely as data to classify, never as commands to the assistant.
   - No translation step; extract directly regardless of source language.
3. **Claim schema**:
   ```json
   {
     "target_event_id": "event_14 or null",
     "target_request_id": "request_26 or null",
     "claim_type": "amend | cancel | confirm | delay | new_income | new_expense | none",
     "field": "amount | event_date | status | null",
     "new_value": "...",
     "effective_date": "YYYY-MM-DD or null",
     "source_date": "from sent_at / event context",
     "confidence": "high | medium | low"
   }
   ```
4. **Field disambiguation rule** (applied in the prompt, not left to model judgment):
   - Payslip-style image → extract **Net Pay** figure specifically.
   - Receipt/invoice-style image → extract **Balance Due** specifically.
5. **Low-confidence or ambiguous claims are kept, not dropped** — they flow into
   Component 3's conflict resolution, where the "safer interpretation" rule (rule 4)
   resolves them deterministically rather than the LLM guessing.
6. Messages with `related_event_id` blank and only `request_id` present produce **no
   structured claim** — they are context-only and never injected into the forecast.
7. Claims can amend fields on events that already have a non-blank value (not just fill
   blanks) — e.g., a confirmed salary increase overrides a previously known salary
   amount from its `effective_date` onward.

---

## 6. Component 3 — Conflict Resolution (deterministic)

**Responsibility**: given all claims targeting the same event/field, resolve to one
final value.

**Precedence** (in order):
1. Explicit cancellation/settlement/amendment claim wins outright over confirm/delay.
2. Among remaining claims, the **newest** (by `source_date`/`sent_at`), compared across
   *all* claims regardless of `source_type`, wins.
3. A **settled** event/claim outranks an estimate or forecast-only claim.
4. If still unresolved (e.g., tie or genuinely ambiguous low-confidence claim): apply
   the **safer interpretation**:
   - For **expenses**: prefer the higher amount / earlier date.
   - For **income**: prefer the lower amount / later date.

Output: one resolved fact per `(event_id, field)`, ready for Component 4 to consume.

---

## 7. Component 4 — State Reconstruction / Recurrence Detection (deterministic)

**Responsibility**: build each user's forward-looking event ledger for the 90-day
forecast.

**Recurrence detection algorithm**:
1. Group settled historical events by `(user_id, category)`.
2. Within each group, cluster by amount similarity: two events are "the same series" if
   their amounts are within **±10%** of each other.
3. For each amount-cluster with **2 or more** occurrences, compute date deltas between
   consecutive occurrences.
4. If deltas are consistent within **±3 days** of a common interval (e.g., ~30, ~7,
   ~14, ~365 days), classify the series as **recurring** at that interval.
5. Fewer than 2 occurrences, or inconsistent deltas → **one-time**, not projected
   forward.

**Projection**:
- Recurring series are projected forward using the **most recent occurrence's amount**,
  at the detected interval, for the full 90-day window.
- If a resolved claim (Component 3) amends the amount effective from a future date,
  the projection uses the old amount before that date and the new amount from that
  date onward.
- **Explicit future events** (e.g., the next confirmed salary row) always take
  precedence for their specific period — the recurrence projector does not also
  generate a projected occurrence for that same period; projection resumes at the next
  cycle after the explicit event.

**Investment events**: only debit/settled contribution events count as cash outflow.
Valuation/appreciation/unrealized entries are excluded entirely from the forecast
(non-cash).

Output: a per-user, date-ordered ledger of all events (historical settled + confirmed
future + recurring projections) covering `request_date` through `request_date + 90`.

---

## 8. Component 5 — 90-Day Forecast & Safety Engine (deterministic)

**Responsibility**: simulate balance and compute safety-check outputs.

- **Simulation granularity**: event-by-event (balance is piecewise-constant between
  event dates — mathematically equivalent to daily simulation, cheaper to compute).
- **Window**: fixed and anchored to `request_date` → `request_date + 90`. Not
  re-anchored per candidate payment date.
- **`amount_safe_to_pay`** (pre-spending-changes):
  ```
  amount_safe_to_pay = min(
      requested_amount,
      max(0, min_balance_over_90day_forecast_without_this_payment - minimum_balance_to_keep)
  )
  ```
- **`earliest_date_for_full_payment`**: first date within the 90-day window at which
  paying `requested_amount` in full, as a single payment on that date, keeps the
  forecast balance at or above `minimum_balance_to_keep` for the remainder of the
  window. If no such date exists within the window, leave empty.
- **No extrapolation past day 90** under any circumstance — if full payment isn't
  confirmed safe within the window, and/or `desired_completion_date` falls beyond it,
  the request is `not_affordable`.

---

## 9. Component 6 — Spending-Change Optimizer (deterministic)

**Responsibility**: when the request isn't safely affordable as-is, find the minimal
spending change(s) that make it safe.

- Only engaged if Component 5's default result doesn't satisfy the request's
  affordability requirement.
- Eligible events: `flexibility=flexible` **and** category present in the user's
  `expense_categories_user_is_willing_to_reduce` or `_stop` list (matching the change
  type).
- **Preference order**: `reduce_to` is preferred over `stop` whenever either would
  close the safety gap (less drastic first). `stop` is used only when reduction alone
  (even down to the event's floor) isn't sufficient.
- `reduce_to` amounts are floored at the event's `minimum_allowed_amount` — never
  reduced below it.
- Up to 3 changes total, mutually exclusive per event (never both `stop` and
  `reduce_to` on the same `event_id`).
- Spending changes only ever help reach `affordable_with_plan` or pull
  `earliest_date_for_full_payment` earlier — they never modify the pre-change
  `amount_safe_to_pay` value itself, per spec wording ("before optional spending
  changes").

---

## 10. Component 7 — Plan Ranker (deterministic)

**Responsibility**: select the final recommendation among safe, eligible candidates.

**Eligibility**:
- `full_payment` / `partial_payment` / `installments` eligible only if in the user's
  `payment_methods_user_will_consider`.
- `wait` eligible only if full payment becomes safe later and `full_payment` is
  accepted.
- `not_recommended` is the fallback when no safe eligible plan exists.

**Ranking** (applied in order, first differentiator wins):
1. Completes the full request by `desired_completion_date`.
2. Requires no spending changes.
3. Minimizes total amount paid — compare `total_payable_amount` (includes financing
   fee) for `installments` options against plain `requested_amount` for
   `full_payment`/`partial_payment` (which carry no fee).
4. Starts payment earlier.
5. Uses fewer payments.
6. Lowest `payment_option_id` as final tie-break.

**Constraints**:
- `full_payment` and `partial_payment` never need to match a row in
  `request_payment_options.csv`.
- `installments` must exactly match a supplied payment option (dates, amounts, fee,
  total).
- `partial_payment` requires: `allows_partial_payment=true`, `full_payment` and/or
  `partial_payment` accepted by user, `0 < amount_safe_to_pay < requested_amount`,
  `earliest_date_for_full_payment <= desired_completion_date`. Plan = exactly two
  payments: `amount_safe_to_pay` on `request_date`, remainder on
  `earliest_date_for_full_payment`, summing to `requested_amount` exactly.

---

## 11. Component 8 — Explanation Generation (LLM)

**Responsibility**: produce `decision_explanation` in English, closely matching the
templated style seen in `sample_requests.csv`
(e.g., *"Pay ZAR 25,256 today. This leaves at least ZAR 18,000 available over the next
90 days."*).

- Input to the LLM is a **fully-computed structured summary** of the decision (amounts,
  dates, plan, any spending changes) — the LLM never sees raw event data and cannot
  alter any number.
- Prompt instructs close adherence to the sample's terse, factual template style rather
  than free-form prose.
- Always English output regardless of source message/request language.

---

## 12. Component 9 — Output Assembler & Validator (deterministic)

**Responsibility**: assemble each output row and validate it before writing.

**Validations**: `0 <= amount_safe_to_pay <= requested_amount`; `payment_plan` dates
chronological and summing correctly for the chosen method; `affordable_now` implies
`earliest_date_for_full_payment == request_date`; stop/reduce_to never on the same
event; installment plan exactly matches a supplied option; method matches user's
accepted methods.

**On validation failure**: **fall back to the safest conservative default** for that
row (`affordability_status=not_affordable`, `recommended_payment_method=not_recommended`,
`payment_plan=none`, `amount_safe_to_pay=0`) and log the row ID and reason — never halt
the whole run, and never emit an internally-inconsistent row.

---

## 13. Guardrails (prompt-injection defense)

- All message/image text is passed into the extraction prompt as **data only**, inside
  a clearly delimited block, with an explicit system instruction that content inside
  the block is never to be treated as instructions to the assistant.
- The extraction LLM call's output schema is rigid JSON with a fixed set of fields —
  there is no field through which an injected instruction could propagate into
  pipeline behavior.
- Component 3 (deterministic conflict resolution) re-validates every claim against the
  actual event/request it targets before it can affect the forecast — a claim
  referencing a nonexistent `event_id`, or attempting a change type not in the allowed
  set, is discarded.
- The explanation LLM (Component 8) only ever receives already-computed structured
  decision data, never raw untrusted text — it cannot be steered by anything in the
  original messages/images.

---

## 14. LLM Client, Caching, Usage Tracking

- Single client wrapper around the Experiential gateway (OpenAI-compatible), model
  `gpt-5.6-luna`.
- Cache extraction results keyed by `(message_id or image_id, prompt_version)` so
  reruns don't reprocess unchanged inputs.
- Every call logs: timestamp, call type (extraction/explanation), input tokens, output
  tokens, target ID.
- `evaluation/usage_report.md` is generated from these logs at the end of a full run:
  per-model and overall totals for calls, input/output tokens, total and average
  tokens per request, and estimated cost (gateway's free tier means cost may be $0 —
  report both list price and actual cost).

---

## 15. Evaluation Harness

- Runs the pipeline output against `sample_requests.csv`.
- Field-by-field comparison: exact match for categorical fields
  (`affordability_status`, `recommended_payment_method`), numeric tolerance for
  `amount_safe_to_pay`, structural match for `payment_plan` (parsed and compared
  date-by-date/amount-by-amount, not string-equal), exact/empty match for
  `earliest_date_for_full_payment`, set-equality for `spending_changes_needed`.
- Outputs a diff report highlighting which requests failed which field, to guide fast
  iteration within the time budget.

---

## 16. Suggested Repo Structure

```
code/
├── README.md
├── main.py                      # orchestrator entrypoint
├── loaders/
│   └── data_loader.py           # Component 1
├── currency/
│   └── converter.py             # Component 1
├── extraction/
│   ├── ocr.py                   # EasyOCR wrapper
│   └── llm_extractor.py         # Component 2
├── resolution/
│   └── conflict_resolver.py     # Component 3
├── state/
│   └── recurrence.py            # Component 4
├── forecast/
│   └── safety_engine.py         # Component 5
├── optimizer/
│   └── spending_optimizer.py    # Component 6
├── ranking/
│   └── plan_ranker.py           # Component 7
├── explanation/
│   └── explain_llm.py           # Component 8
├── output/
│   └── validator.py             # Component 9
├── llm_client/
│   ├── client.py                # Experiential gateway wrapper
│   └── usage_tracker.py
├── evaluation/
│   ├── harness.py
│   └── usage_report.md          # generated
└── tests/
    └── ...                      # unit tests per component, no LLM needed
```

---

## 17. Known Limitations / Open Assumptions

- Recurrence detection is a heuristic (±10% amount tolerance, ±3 day interval
  tolerance, 2+ occurrences) — may misclassify irregular but genuinely recurring
  expenses, or over-fit coincidental one-off amount matches.
- Currency conversion assumes the dataset guarantees a path for every needed pair on
  every needed date; nearest-date fallback (unbounded distance) is a safety net, not
  expected to trigger.
- Image field disambiguation (Net Pay / Balance Due) is rule-based on document type;
  a document type not seen in the two provided samples may need a new rule.
- `sample_requests.csv` (2 rows visible) is the only calibration signal for
  explanation style and decision tone — broader stylistic drift is possible until
  validated against more samples during the eval harness run.
