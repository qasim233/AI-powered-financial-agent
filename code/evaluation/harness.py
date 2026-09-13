"""
Evaluation Harness — Field-by-field comparison against sample_requests.csv.

Per SOLUTION.md §15:
- Compares pipeline outputs against ground-truth sample requests.
- Metrics:
  - amount_safe_to_pay: numeric tolerance (within 1.0)
  - affordability_status: exact match
  - recommended_payment_method: exact match
  - payment_plan: parsed structural match (set of date:amount tuples)
  - earliest_date_for_full_payment: exact string match
  - spending_changes_needed: set-equality of changes
- Generates a clear terminal diff report.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FieldDiff:
    field_name: str
    expected: str
    actual: str


@dataclass
class RequestDiff:
    request_id: str
    matches: bool
    diffs: List[FieldDiff]


class EvaluationHarness:
    """
    Evaluates predictions against sample ground truth.
    """

    def __init__(self, sample_requests_path: str) -> None:
        self.sample_path = Path(sample_requests_path)
        self.samples: Dict[str, Dict[str, str]] = {}
        self._load_samples()

    def _load_samples(self) -> None:
        if not self.sample_path.exists():
            logger.warning("Sample requests file not found: %s", self.sample_path)
            return
        with open(self.sample_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rid = row["request_id"].strip()
                self.samples[rid] = {k.strip(): v.strip() for k, v in row.items()}

    def evaluate_predictions(self, predictions_path: str) -> Tuple[float, List[RequestDiff]]:
        """
        Compare predictions.csv against sample_requests.csv.
        Returns overall accuracy score (0.0 to 1.0) and list of diffs.
        """
        preds_path = Path(predictions_path)
        if not preds_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {preds_path}")

        preds: Dict[str, Dict[str, str]] = {}
        with open(preds_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rid = row["request_id"].strip()
                preds[rid] = {k.strip(): v.strip() for k, v in row.items()}

        results: List[RequestDiff] = []
        matching_count = 0
        total_evaluated = 0

        for rid, expected in self.samples.items():
            if rid not in preds:
                continue
            total_evaluated += 1
            actual = preds[rid]
            diffs = self._compare_row(expected, actual)
            is_match = (len(diffs) == 0)
            if is_match:
                matching_count += 1
            results.append(RequestDiff(request_id=rid, matches=is_match, diffs=diffs))

        accuracy = (matching_count / total_evaluated) if total_evaluated > 0 else 0.0
        return accuracy, results

    def print_report(self, accuracy: float, diffs: List[RequestDiff]) -> None:
        """Print formatted evaluation report."""
        print("\n" + "=" * 60)
        if len(diffs) == 0:
            print("EVALUATION REPORT — NO MATCHING SAMPLES FOUND (0.0%)")
            print("=" * 60)
            print("⚠️  WARNING: 0 matching request IDs found between predictions and sample_requests.csv.")
            print("   The predictions file contains IDs that do not exist in sample_requests.csv.")
            print("   To benchmark against ground truth, run with:")
            print("   python main.py --requests-file ../dataset/sample_requests.csv --evaluate")
        else:
            passing = len([d for d in diffs if d.matches])
            print(f"EVALUATION REPORT — Accuracy on Samples: {accuracy * 100:.1f}% ({passing}/{len(diffs)} Passing)")
            print("=" * 60)

            for d in diffs:
                status = "✅ PASS" if d.matches else "❌ FAIL"
                print(f"\n{status} [{d.request_id}]")
                for fd in d.diffs:
                    print(f"   Field '{fd.field_name}':")
                    print(f"     Expected: {fd.expected}")
                    print(f"     Actual:   {fd.actual}")

        print("\n" + "=" * 60)

    def _compare_row(self, exp: Dict[str, str], act: Dict[str, str]) -> List[FieldDiff]:
        diffs: List[FieldDiff] = []

        # 1. amount_safe_to_pay (numeric comparison within 1.0 tolerance)
        e_safe = float(exp.get("amount_safe_to_pay", "0") or "0")
        a_safe = float(act.get("amount_safe_to_pay", "0") or "0")
        if abs(e_safe - a_safe) > 1.0:
            diffs.append(FieldDiff("amount_safe_to_pay", str(e_safe), str(a_safe)))

        # 2. affordability_status
        if exp.get("affordability_status") != act.get("affordability_status"):
            diffs.append(FieldDiff("affordability_status", exp.get("affordability_status", ""), act.get("affordability_status", "")))

        # 3. recommended_payment_method
        if exp.get("recommended_payment_method") != act.get("recommended_payment_method"):
            diffs.append(FieldDiff("recommended_payment_method", exp.get("recommended_payment_method", ""), act.get("recommended_payment_method", "")))

        # 4. earliest_date_for_full_payment
        e_ed = exp.get("earliest_date_for_full_payment", "")
        a_ed = act.get("earliest_date_for_full_payment", "")
        if e_ed != a_ed:
            diffs.append(FieldDiff("earliest_date_for_full_payment", e_ed, a_ed))

        # 5. payment_plan (structural comparison)
        if not self._plans_match(exp.get("payment_plan", ""), act.get("payment_plan", "")):
            diffs.append(FieldDiff("payment_plan", exp.get("payment_plan", ""), act.get("payment_plan", "")))

        # 6. spending_changes_needed (set comparison)
        e_sc = set(exp.get("spending_changes_needed", "").split("|"))
        a_sc = set(act.get("spending_changes_needed", "").split("|"))
        if e_sc != a_sc:
            diffs.append(FieldDiff("spending_changes_needed", exp.get("spending_changes_needed", ""), act.get("spending_changes_needed", "")))

        return diffs

    def _plans_match(self, plan1: str, plan2: str) -> bool:
        if plan1 == plan2:
            return True
        if plan1 == "none" or plan2 == "none":
            return False

        def parse_items(p_str: str) -> Set[Tuple[str, float]]:
            res = set()
            for item in p_str.split("|"):
                if ":" in item:
                    d, a = item.split(":", 1)
                    res.add((d.strip(), round(float(a), 1)))
            return res

        try:
            return parse_items(plan1) == parse_items(plan2)
        except Exception:
            return plan1.strip() == plan2.strip()
