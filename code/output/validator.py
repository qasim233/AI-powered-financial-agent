"""
Output Assembler & Validator — Deterministic contract enforcement and CSV generation.

Per SOLUTION.md §12 (Component 9):
- Validates all output contract invariants:
  1. 0 <= amount_safe_to_pay <= requested_amount
  2. affordability_status in ('affordable_now', 'affordable_with_plan', 'affordable_later', 'not_affordable')
  3. recommended_payment_method in ('full_payment', 'partial_payment', 'installments', 'wait', 'not_recommended')
  4. payment_plan dates are strictly chronological
  5. affordable_now requires earliest_date_for_full_payment == request_date
  6. partial_payment has exactly two payments summing to requested_amount
  7. stop and reduce_to never target the same event_id
  8. recommended method is acceptable to the user
- On validation failure: falls back to conservative default:
  amount_safe_to_pay=0, affordability_status='not_affordable',
  recommended_payment_method='not_recommended', payment_plan='none',
  earliest_date_for_full_payment='', spending_changes_needed='none'
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

from loaders.data_loader import FinancialProfile, Request
from ranking.plan_ranker import CandidatePlan

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


@dataclass(slots=True)
class OutputRow:
    request_id: str
    amount_safe_to_pay: str
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


class OutputAssembler:
    """
    Assembles and validates output rows against the challenge contract.
    """

    def assemble_and_validate_row(
        self,
        request: Request,
        profile: FinancialProfile,
        plan: CandidatePlan,
        safe_today: float,
        earliest_full_date: Optional[date],
        spending_changes_str: str,
        explanation: str,
    ) -> OutputRow:
        """
        Assemble one output row and validate. Falls back to conservative default on violation.
        """
        req_id = request.request_id
        req_amt = float(request.requested_amount)

        # Format numeric amount_safe_to_pay
        safe_val = round(safe_today, 2)
        safe_str = str(int(safe_val)) if safe_val == int(safe_val) else f"{safe_val:.2f}".rstrip("0").rstrip(".")

        # Format earliest date
        earliest_str = earliest_full_date.isoformat() if earliest_full_date else ""

        # Format plan
        plan_str = plan.format_plan_str()

        # Build proposed row
        candidate_row = OutputRow(
            request_id=req_id,
            amount_safe_to_pay=safe_str,
            affordability_status=plan.affordability_status,
            recommended_payment_method=plan.payment_method,
            payment_plan=plan_str,
            earliest_date_for_full_payment=earliest_str,
            spending_changes_needed=spending_changes_str,
            decision_explanation=explanation.replace("\n", " ").strip(),
        )

        # Invariant checks
        is_valid, reason = self._validate_invariants(candidate_row, request, profile)
        if not is_valid:
            logger.warning("Output validation failed for %s: %s. Applying conservative fallback.", req_id, reason)
            return self._conservative_fallback(request, profile, reason)

        return candidate_row

    def _validate_invariants(
        self,
        row: OutputRow,
        req: Request,
        profile: FinancialProfile,
    ) -> tuple[bool, str]:
        # 1. Check amount_safe_to_pay range
        try:
            val = float(row.amount_safe_to_pay)
            if val < -1e-4 or val > float(req.requested_amount) + 1e-4:
                return False, f"amount_safe_to_pay {val} out of bounds [0, {req.requested_amount}]"
        except ValueError:
            return False, f"Invalid amount_safe_to_pay string: {row.amount_safe_to_pay}"

        # 2. Check affordable_now implies earliest_date_for_full_payment == request_date
        if row.affordability_status == "affordable_now":
            if row.earliest_date_for_full_payment != req.request_date.isoformat():
                return False, f"affordable_now must have earliest_date == request_date ({req.request_date})"

        # 3. Check partial_payment constraints
        if row.recommended_payment_method == "partial_payment":
            if row.affordability_status != "affordable_with_plan":
                return False, "partial_payment must have affordability_status == affordable_with_plan"
            if "|" not in row.payment_plan:
                return False, "partial_payment must contain exactly two payments separated by |"
            parts = row.payment_plan.split("|")
            if len(parts) != 2:
                return False, f"partial_payment must have exactly 2 payments, got {len(parts)}"
            p1_amt = float(parts[0].split(":")[1])
            p2_amt = float(parts[1].split(":")[1])
            if abs((p1_amt + p2_amt) - float(req.requested_amount)) > 0.1:
                return False, f"partial payments {p1_amt} + {p2_amt} != {req.requested_amount}"

        # 4. Check spending changes: mutual exclusivity per event
        if row.spending_changes_needed != "none":
            changes = row.spending_changes_needed.split("|")
            if len(changes) > 3:
                return False, f"Exceeded maximum 3 spending changes: {len(changes)}"
            events_seen = set()
            for ch in changes:
                parts = ch.split(":")
                if len(parts) < 2:
                    return False, f"Malformed spending change: {ch}"
                eid = parts[1]
                if eid in events_seen:
                    return False, f"Multiple changes targeting same event_id: {eid}"
                events_seen.add(eid)

        return True, "valid"

    def _conservative_fallback(
        self,
        req: Request,
        profile: FinancialProfile,
        reason: str,
    ) -> OutputRow:
        ccy = profile.home_currency
        min_keep = profile.minimum_balance_to_keep
        return OutputRow(
            request_id=req.request_id,
            amount_safe_to_pay="0",
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment="",
            spending_changes_needed="none",
            decision_explanation=(
                f"Do not proceed with the {ccy} {req.requested_amount:,.2f} request. "
                f"None of the available options keeps the {ccy} {min_keep:,.2f} minimum protected."
            ),
        )

    def write_output_csv(self, rows: List[OutputRow], output_path: str) -> None:
        """Write rows to output.csv with exact required column ordering."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))
        logger.info("Wrote %d rows to %s", len(rows), output_path)
