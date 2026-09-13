# Buy or Wait? Financial Decision Agent

Production-grade AI-powered financial decision system for the HackerRank Orchestrate (September 2026) challenge.

## Architecture

The system is built as an orchestrated pipeline using **LangGraph** (`StateGraph`) as the orchestration backbone and **LangChain** (`ChatOpenAI`) for LLM interactions.

### Pipeline Flow

1. **Data Loader & Currency Normalizer** (`loaders/`, `currency/`):
   Loads structured profiles, events, requests, options, and exchange rates. Performs multi-hop BFS currency conversion across dated exchange rates.
2. **Unstructured Extraction** (`extraction/`):
   Extracts financial claims from messages and images using EasyOCR + LLM with strict prompt injection guardrails (`[DATA_START]` / `[DATA_END]` delimiters).
3. **Conflict Resolution** (`resolution/`):
   Deterministically resolves conflicting claims per 4-level precedence: explicit amendment/cancellation > newest record > settled event > safer financial interpretation.
4. **State Reconstruction & Recurrence Detection** (`state/`):
   Reconstructs the forward-looking cash flow ledger for the 90-day window. Automatically detects recurring expenses/income (±10% amount clustering, ±3-day cycle consistency). Reserves pending debits.
5. **90-Day Forecast & Safety Engine** (`forecast/`):
   Simulates daily balance trajectory; computes `amount_safe_to_pay` and `earliest_date_for_full_payment` ensuring balance never breaches `minimum_balance_to_keep`.
6. **Spending-Change Optimizer** (`optimizer/`):
   Searches for minimal spending reductions (`reduce_to`) or stops (`stop`) on eligible flexible categories to unlock plan affordability.
7. **Plan Ranker** (`ranking/`):
   Generates and ranks candidate plans (full payment, wait, partial payment, installments) against user preferences and the 6-rule ranking hierarchy.
8. **Explanation Generation** (`explanation/`):
   Produces terse, factual 1-2 sentence explanations matching the calibration style of `sample_requests.csv`.
9. **Output Assembler & Validator** (`output/`):
   Enforces all challenge output contract invariants and produces `output.csv`.

## Requirements & Setup

```bash
# 1. Create Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama locally and pull the model
ollama pull gemma2:9b

# The pipeline will use the local Ollama server at http://localhost:11434
```

## Running the Pipeline

```bash
# Run full pipeline on dataset/requests.csv -> produces dataset/output.csv
python main.py --dataset-dir ../dataset --output-csv ../dataset/output.csv

# Run pipeline and evaluate against sample_requests.csv
python main.py --dataset-dir ../dataset --output-csv ../dataset/output.csv --evaluate

# Run in pure deterministic mode (no LLM calls)
python main.py --dataset-dir ../dataset --output-csv ../dataset/output.csv --no-llm
```
