"""
Main Orchestrator — Buy or Wait? Financial Decision Engine.

Orchestrated using LangGraph StateGraph, per the primary stack mandate.
Coordinates Components 1 through 9:
  1. Data Loader & Currency Normalizer
  2. Unstructured Extraction (LLM + EasyOCR)
  3. Conflict Resolution
  4. State Reconstruction & Recurrence Detection
  5. 90-Day Forecast & Safety Engine
  6. Spending-Change Optimizer
  7. Plan Ranker
  8. Explanation Generation (LLM)
  9. Output Assembler & Validator
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")

# LangGraph
from langgraph.graph import END, START, StateGraph

# Pipeline components
from currency.converter import CurrencyConverter
from evaluation.harness import EvaluationHarness
from explanation.explain_llm import ExplanationGenerator
from extraction.llm_extractor import Claim, LLMExtractor
from extraction.ocr import OCRProcessor
from forecast.safety_engine import ForecastResult, SafetyEngine
from llm_client.client import LLMClient
from loaders.data_loader import DataLoader, FinancialProfile, Request
from optimizer.spending_optimizer import SpendingChangeOptimizer
from output.validator import OutputAssembler, OutputRow
from ranking.plan_ranker import CandidatePlan, PlanRanker
from resolution.conflict_resolver import ConflictResolver, ResolvedFact
from state.recurrence import LedgerEntry, StateReconstructor


# ---------------------------------------------------------------------------
# LangGraph State Schema
# ---------------------------------------------------------------------------

class RequestPipelineGraphState(TypedDict):
    request: Request
    profile: FinancialProfile
    claims: List[Claim]
    resolved_facts: Dict[Any, ResolvedFact]
    ledger: List[LedgerEntry]
    forecast: Optional[ForecastResult]
    best_plan: Optional[CandidatePlan]
    spending_changes_str: str
    explanation: str
    output_row: Optional[OutputRow]


# ---------------------------------------------------------------------------
# Pipeline Engine
# ---------------------------------------------------------------------------

class BuyOrWaitPipeline:
    """
    High-level engine that sets up all components and compiles the LangGraph graph.
    """

    def __init__(
        self,
        dataset_dir: str,
        requests_filename: str = "requests.csv",
        enable_llm: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.requests_filename = requests_filename
        self.enable_llm = enable_llm

        # 1. DataLoader & CurrencyConverter
        logger.info("Initializing Data Loader (requests from %s)...", requests_filename)
        self.data_loader = DataLoader(str(self.dataset_dir), requests_filename=requests_filename)
        self.data_loader.load_all()
        self.converter = CurrencyConverter(self.data_loader.exchange_rates)

        # LLM & OCR
        self.llm_client = None
        if self.enable_llm and os.environ.get("EXPLABS_API_KEY"):
            try:
                self.llm_client = LLMClient()
                logger.info("LLMClient initialized with model %s", self.llm_client.llm.model_name)
            except Exception:
                logger.warning("Could not initialize LLMClient. Running in deterministic mode.")
        else:
            logger.info("Running in deterministic/template mode without LLM calls.")

        self.ocr_processor = OCRProcessor()

        # Components
        self.extractor = LLMExtractor(self.llm_client, self.data_loader, self.ocr_processor)
        self.conflict_resolver = ConflictResolver(self.data_loader)
        self.state_reconstructor = StateReconstructor(self.data_loader, self.converter)
        self.safety_engine = SafetyEngine(forecast_days=90)
        self.spending_optimizer = SpendingChangeOptimizer(self.safety_engine)
        self.plan_ranker = PlanRanker(self.data_loader, self.safety_engine, self.spending_optimizer)
        self.explanation_gen = ExplanationGenerator(self.llm_client, self.data_loader)
        self.output_assembler = OutputAssembler()

        # Compile LangGraph graph
        self.graph = self._build_langgraph_workflow()

    def _build_langgraph_workflow(self):
        """Construct the LangGraph StateGraph orchestration workflow."""
        builder = StateGraph(RequestPipelineGraphState)

        # Node 1: Unstructured claim extraction
        def node_extract(state: RequestPipelineGraphState) -> Dict[str, Any]:
            req = state["request"]
            claims = []
            if self.llm_client is not None:
                claims = self.extractor.extract_claims_for_user(req.user_id)
            return {"claims": claims}

        # Node 2: Deterministic conflict resolution
        def node_resolve(state: RequestPipelineGraphState) -> Dict[str, Any]:
            resolved = self.conflict_resolver.resolve_claims(state["claims"])
            return {"resolved_facts": resolved}

        # Node 3: State reconstruction & recurrence detection
        def node_reconstruct(state: RequestPipelineGraphState) -> Dict[str, Any]:
            req = state["request"]
            ledger = self.state_reconstructor.build_forecast_ledger(
                user_id=req.user_id,
                request_date=req.request_date,
                resolved_facts=state["resolved_facts"],
                forecast_days=90,
            )
            return {"ledger": ledger}

        # Node 4: 90-day forecast and safety evaluation
        def node_forecast(state: RequestPipelineGraphState) -> Dict[str, Any]:
            req = state["request"]
            prof = state["profile"]
            forecast = self.safety_engine.evaluate_request(req, prof, state["ledger"])
            return {"forecast": forecast}

        # Node 5: Plan candidate generation, spending optimization, and ranking
        def node_rank_and_optimize(state: RequestPipelineGraphState) -> Dict[str, Any]:
            req = state["request"]
            prof = state["profile"]
            best_plan = self.plan_ranker.select_best_plan(
                request=req,
                profile=prof,
                ledger=state["ledger"],
                forecast=state["forecast"],
            )
            sc_str = best_plan.format_spending_changes_str(self.data_loader)
            return {"best_plan": best_plan, "spending_changes_str": sc_str}

        # Node 6: Explanation generation
        def node_explain(state: RequestPipelineGraphState) -> Dict[str, Any]:
            req = state["request"]
            prof = state["profile"]
            plan = state["best_plan"]
            fc = state["forecast"]
            earliest_str = fc.earliest_date_for_full_payment.isoformat() if fc.earliest_date_for_full_payment else None
            explanation = self.explanation_gen.generate_explanation(
                request=req,
                profile=prof,
                plan=plan,
                safe_today=fc.amount_safe_to_pay,
                earliest_full_date=earliest_str,
                spending_changes_str=state["spending_changes_str"],
            )
            return {"explanation": explanation}

        # Node 7: Output row assembly and contract validation
        def node_assemble(state: RequestPipelineGraphState) -> Dict[str, Any]:
            req = state["request"]
            prof = state["profile"]
            plan = state["best_plan"]
            fc = state["forecast"]
            row = self.output_assembler.assemble_and_validate_row(
                request=req,
                profile=prof,
                plan=plan,
                safe_today=fc.amount_safe_to_pay,
                earliest_full_date=fc.earliest_date_for_full_payment,
                spending_changes_str=state["spending_changes_str"],
                explanation=state["explanation"],
            )
            return {"output_row": row}

        # Register nodes
        builder.add_node("extract_unstructured", node_extract)
        builder.add_node("resolve_conflicts", node_resolve)
        builder.add_node("reconstruct_ledger", node_reconstruct)
        builder.add_node("forecast_safety", node_forecast)
        builder.add_node("rank_and_optimize", node_rank_and_optimize)
        builder.add_node("explain_decision", node_explain)
        builder.add_node("assemble_output", node_assemble)

        # Wire linear sequential flow
        builder.add_edge(START, "extract_unstructured")
        builder.add_edge("extract_unstructured", "resolve_conflicts")
        builder.add_edge("resolve_conflicts", "reconstruct_ledger")
        builder.add_edge("reconstruct_ledger", "forecast_safety")
        builder.add_edge("forecast_safety", "rank_and_optimize")
        builder.add_edge("rank_and_optimize", "explain_decision")
        builder.add_edge("explain_decision", "assemble_output")
        builder.add_edge("assemble_output", END)

        return builder.compile()

    def process_all_requests(self, output_csv_path: str) -> List[OutputRow]:
        """Execute the LangGraph pipeline for every request in requests.csv."""
        requests_list = list(self.data_loader.requests.values())
        logger.info("Processing %d requests through LangGraph pipeline...", len(requests_list))

        rows: List[OutputRow] = []
        for i, req in enumerate(requests_list, start=1):
            prof = self.data_loader.profiles.get(req.user_id)
            if not prof:
                logger.error("Missing profile for user %s (request %s)", req.user_id, req.request_id)
                continue

            init_state: RequestPipelineGraphState = {
                "request": req,
                "profile": prof,
                "claims": [],
                "resolved_facts": {},
                "ledger": [],
                "forecast": None,
                "best_plan": None,
                "spending_changes_str": "none",
                "explanation": "",
                "output_row": None,
            }

            final_state = self.graph.invoke(init_state)
            row = final_state["output_row"]
            if row:
                rows.append(row)

            if i % 25 == 0 or i == len(requests_list):
                logger.info("Processed %d / %d requests", i, len(requests_list))

        # Write output.csv
        self.output_assembler.write_output_csv(rows, output_csv_path)

        # Generate usage report
        if self.llm_client:
            report_path = Path("evaluation/usage_report.md")
            self.llm_client.tracker.generate_report(str(report_path), num_requests=len(rows))

        return rows


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Buy or Wait? Decision Engine")
    parser.add_argument(
        "--dataset-dir",
        default="../dataset",
        help="Path to dataset directory (default: ../dataset)",
    )
    parser.add_argument(
        "--requests-file",
        default="requests.csv",
        help="Path or filename of requests CSV to process (default: requests.csv)",
    )
    parser.add_argument(
        "--output-csv",
        default="../dataset/output.csv",
        help="Path for output.csv (default: ../dataset/output.csv)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation against sample_requests.csv after processing",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run pure deterministic pipeline without LLM calls",
    )
    args = parser.parse_args()

    pipeline = BuyOrWaitPipeline(
        dataset_dir=args.dataset_dir,
        requests_filename=args.requests_file,
        enable_llm=not args.no_llm,
    )

    rows = pipeline.process_all_requests(args.output_csv)
    print(f"Successfully generated {len(rows)} predictions in {args.output_csv}")

    if args.evaluate:
        samples_path = Path(args.dataset_dir) / "sample_requests.csv"
        harness = EvaluationHarness(str(samples_path))
        acc, diffs = harness.evaluate_predictions(args.output_csv)
        harness.print_report(acc, diffs)


if __name__ == "__main__":
    main()
