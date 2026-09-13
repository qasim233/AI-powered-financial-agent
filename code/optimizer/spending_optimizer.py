"""
Spending-Change Optimizer — Minimal adjustment search to unlock safe plan affordability.

Per SOLUTION.md §9 (Component 6):
- Engaged when a plan is not safe as-is.
- Eligible events:
  - debit / expense with event_id
  - category NOT in protected categories
  - for reduce_to: flexibility in ('reducible', 'reducible_or_stoppable') and
    category in user's expense_categories_user_is_willing_to_reduce
  - for stop: flexibility in ('stoppable', 'reducible_or_stoppable') and
    category in user's expense_categories_user_is_willing_to_stop
- Preference order:
  - reduce_to is preferred over stop whenever either would close the safety gap.
  - Fewer changes preferred (up to 3 changes maximum).
  - Mutually exclusive per event_id (never both stop and reduce_to for the same event).
  - reduce_to is floored at event's minimum_allowed_amount.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

from forecast.safety_engine import SafetyEngine
from loaders.data_loader import FinancialProfile, Request
from state.recurrence import LedgerEntry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SpendingChange:
    action: str        # "stop" or "reduce_to"
    event_id: str
    new_amount_home: float
    original_currency_amount: Optional[float] = None

    def to_output_str(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"
        else:
            # Format nicely: integer if whole number, else 2 decimal places
            val = self.original_currency_amount if self.original_currency_amount is not None else self.new_amount_home
            if val == int(val):
                formatted = str(int(val))
            else:
                formatted = f"{val:.2f}".rstrip("0").rstrip(".")
            return f"reduce_to:{self.event_id}:{formatted}"


class SpendingChangeOptimizer:
    """
    Finds minimal spending adjustments to make a payment plan safe.
    """

    def __init__(self, safety_engine: SafetyEngine) -> None:
        self.safety_engine = safety_engine

    def find_minimal_changes(
        self,
        candidate_payments: List[Tuple[date, float]],
        ledger: List[LedgerEntry],
        profile: FinancialProfile,
        request: Request,
    ) -> Tuple[bool, List[SpendingChange]]:
        """
        Check if candidate_payments can be made safe using 0 to 3 spending changes.

        Returns
        -------
        (is_possible, list_of_changes)
        If safe without changes, returns (True, []).
        """
        req_date = request.request_date
        init_bal = profile.current_available_balance
        min_keep = profile.minimum_balance_to_keep

        # 1. First check if safe as-is
        if self.safety_engine.is_plan_safe(
            payments=candidate_payments,
            ledger=ledger,
            initial_balance=init_bal,
            minimum_balance_to_keep=min_keep,
            request_date=req_date,
        ):
            return True, []

        # 2. Collect candidate modifications per distinct event_id in ledger
        candidates = self._find_candidate_modifications(ledger, profile, req_date)
        if not candidates:
            return False, []

        # Search combinations of sizes 1, 2, 3
        # We sort candidates to try reduce_to before stop
        candidates.sort(key=lambda c: (0 if c.action == "reduce_to" else 1, c.event_id))

        for k in (1, 2, 3):
            if k > len(candidates):
                break
            for combo in itertools.combinations(candidates, k):
                # Ensure mutual exclusivity: at most 1 action per event_id
                event_ids = [c.event_id for c in combo]
                if len(set(event_ids)) < len(event_ids):
                    continue

                # Build spending adjustments dictionary: entry_id -> new_amount_home
                adjustments = self._build_adjustments_dict(ledger, combo)

                if self.safety_engine.is_plan_safe(
                    payments=candidate_payments,
                    ledger=ledger,
                    initial_balance=init_bal,
                    minimum_balance_to_keep=min_keep,
                    request_date=req_date,
                    spending_adjustments=adjustments,
                ):
                    return True, list(combo)

        return False, []

    def _find_candidate_modifications(
        self,
        ledger: List[LedgerEntry],
        profile: FinancialProfile,
        request_date: date,
    ) -> List[SpendingChange]:
        """Find all valid single-event spending change actions."""
        protected = set(profile.expense_categories_to_protect)
        willing_reduce = set(profile.expense_categories_user_is_willing_to_reduce)
        willing_stop = set(profile.expense_categories_user_is_willing_to_stop)

        seen_events: Set[str] = set()
        actions: List[SpendingChange] = []

        for entry in ledger:
            if entry.direction != "debit":
                continue
            if not entry.event_id:
                continue
            if entry.event_id in seen_events:
                continue
            if entry.category in protected:
                continue

            seen_events.add(entry.event_id)

            can_reduce = (
                entry.flexibility in ("reducible", "reducible_or_stoppable")
                and entry.category in willing_reduce
            )
            can_stop = (
                entry.flexibility in ("stoppable", "reducible_or_stoppable")
                and entry.category in willing_stop
            )

            if can_reduce and entry.minimum_allowed_amount_home is not None:
                if entry.minimum_allowed_amount_home < entry.amount_home:
                    actions.append(
                        SpendingChange(
                            action="reduce_to",
                            event_id=entry.event_id,
                            new_amount_home=entry.minimum_allowed_amount_home,
                            original_currency_amount=None,  # will map from data
                        )
                    )

            if can_stop:
                actions.append(
                    SpendingChange(
                        action="stop",
                        event_id=entry.event_id,
                        new_amount_home=0.0,
                    )
                )

        return actions

    def _build_adjustments_dict(
        self,
        ledger: List[LedgerEntry],
        changes: Tuple[SpendingChange, ...],
    ) -> Dict[str, float]:
        """Map all ledger entry_ids corresponding to the modified event_ids to their new amount."""
        change_map = {c.event_id: c.new_amount_home for c in changes}
        adjustments: Dict[str, float] = {}
        for entry in ledger:
            if entry.event_id and entry.event_id in change_map:
                adjustments[entry.entry_id] = change_map[entry.event_id]
        return adjustments
