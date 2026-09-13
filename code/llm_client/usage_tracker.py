"""
Usage Tracker — Logs every LLM call and generates evaluation/usage_report.md.

Per SOLUTION.md §14: every call logs timestamp, call type (extraction/explanation),
input tokens, output tokens, and target ID. The report is generated at the end of
a full run with per-model and overall totals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """Single LLM call record."""

    timestamp: str
    call_type: str  # "extraction" or "explanation"
    target_id: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageTracker:
    """
    Tracks all LLM calls and generates the final usage report.

    Accumulates CallRecord entries during the pipeline run.
    At the end, ``generate_report`` writes ``evaluation/usage_report.md``
    with per-model and overall totals, as required by SOLUTION.md §14 and
    AGENTS.md §6.5.
    """

    def __init__(
        self,
        model: str = "gemma2:9b",
        provider: str = "Ollama",
    ) -> None:
        self.model = model
        self.provider = provider
        self.calls: List[CallRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def log_call(
        self,
        call_type: str,
        target_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Append a call record."""
        record = CallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            call_type=call_type,
            target_id=target_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.calls.append(record)
        logger.info(
            "LLM call: type=%s target=%s in_tokens=%d out_tokens=%d",
            call_type,
            target_id,
            input_tokens,
            output_tokens,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, output_path: str, num_requests: int) -> None:
        """
        Write ``evaluation/usage_report.md`` summarising the full-dataset run.

        Parameters
        ----------
        output_path:
            Filesystem path for the report (e.g. ``code/evaluation/usage_report.md``).
        num_requests:
            Number of requests processed (for per-request averages).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        total_calls = len(self.calls)
        total_input = sum(c.input_tokens for c in self.calls)
        total_output = sum(c.output_tokens for c in self.calls)
        total_tokens = total_input + total_output

        extraction_calls = [c for c in self.calls if c.call_type == "extraction"]
        explanation_calls = [c for c in self.calls if c.call_type == "explanation"]

        ext_input = sum(c.input_tokens for c in extraction_calls)
        ext_output = sum(c.output_tokens for c in extraction_calls)
        ext_total = ext_input + ext_output

        exp_input = sum(c.input_tokens for c in explanation_calls)
        exp_output = sum(c.output_tokens for c in explanation_calls)
        exp_total = exp_input + exp_output

        avg_tokens = total_tokens / num_requests if num_requests > 0 else 0
        avg_calls = total_calls / num_requests if num_requests > 0 else 0

        # Cost estimation — gateway is free-tier; list-price estimates are
        # included for reference only as required by AGENTS.md §6.5.
        est_input_cost_per_1k = 0.005
        est_output_cost_per_1k = 0.015
        est_total_cost = (
            (total_input / 1000) * est_input_cost_per_1k
            + (total_output / 1000) * est_output_cost_per_1k
        )
        est_per_request = est_total_cost / num_requests if num_requests > 0 else 0

        report_lines = [
            "# Usage Report",
            "",
            "## Model Information",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Provider | {self.provider} |",
            f"| Model | {self.model} |",
            "| Base URL | http://localhost:11434/v1 |",
            "",
            "## Call Summary",
            "",
            "| Metric | Extraction | Explanation | Total |",
            "|--------|-----------|-------------|-------|",
            f"| Calls | {len(extraction_calls)} | {len(explanation_calls)} | {total_calls} |",
            f"| Input Tokens | {ext_input} | {exp_input} | {total_input} |",
            f"| Output Tokens | {ext_output} | {exp_output} | {total_output} |",
            f"| Total Tokens | {ext_total} | {exp_total} | {total_tokens} |",
            "",
            "## Per-Request Averages",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Requests Processed | {num_requests} |",
            f"| Average Tokens per Request | {avg_tokens:.1f} |",
            f"| Average Calls per Request | {avg_calls:.2f} |",
            "",
            "## Cost Estimate",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Estimated Total Cost (list price) | ${est_total_cost:.4f} |",
            f"| Estimated Per-Request Cost (list price) | ${est_per_request:.6f} |",
            "| Actual Cost (free-tier gateway) | $0.00 |",
            "",
            "> Note: Ollama runs locally, so these are list-price reference estimates only.",
            "> Actual runtime cost depends on your local hardware and setup.",
            "",
        ]

        path.write_text("\n".join(report_lines), encoding="utf-8")
        logger.info("Usage report written to %s", path)
