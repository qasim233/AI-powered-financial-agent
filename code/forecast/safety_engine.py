"""
90-Day Forecast & Safety Engine — Deterministic balance simulation and safety evaluation.

Per SOLUTION.md §8 (Component 5):
- Event-by-event balance simulation anchored to [request_date, request_date + 90].
- amount_safe_to_pay = min(
      requested_amount,
      max(0, min_balance_over_90day_forecast_without_this_payment - minimum_balance_to_keep)
  )
- earliest_date_for_full_payment: first date within the 90-day window where paying
  requested_amount in full keeps the forecast balance >= minimum_balance_to_keep
  for the remainder of the window.
- Simulation safety check method for candidate payment schedules (installments, partial payments).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from loaders.data_loader import FinancialProfile, Request
from state.recurrence import LedgerEntry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BalancePoint:
    on_date: date
    balance: float


@dataclass(slots=True)
class ForecastResult:
    request_id: str
    request_date: date
    window_end_date: date
    initial_balance: float
    minimum_balance_to_keep: float
    min_balance_in_window: float
    amount_safe_to_pay: float
    earliest_date_for_full_payment: Optional[date]
    balance_trajectory: List[BalancePoint]


class SafetyEngine:
    """
    Simulates user balances over a 90-day horizon and checks financial safety.
    """

    def __init__(self, forecast_days: int = 90) -> None:
        self.forecast_days = forecast_days

    def evaluate_request(
        self,
        request: Request,
        profile: FinancialProfile,
        ledger: List[LedgerEntry],
    ) -> ForecastResult:
        """
        Run the baseline 90-day forecast simulation without the candidate purchase.
        Computes amount_safe_to_pay and earliest_date_for_full_payment.
        """
        req_date = request.request_date
        window_end = req_date + timedelta(days=self.forecast_days)
        init_bal = profile.current_available_balance
        min_keep = profile.minimum_balance_to_keep

        # Build trajectory of (date, balance)
        trajectory = self._build_trajectory(init_bal, ledger, req_date, window_end)

        # Minimum balance over the entire 90-day window
        min_bal = min(pt.balance for pt in trajectory)
        headroom = max(0.0, min_bal - min_keep)
        safe_to_pay = min(float(request.requested_amount), headroom)

        # Earliest date for full payment
        earliest_full = self._find_earliest_date_for_full_payment(
            trajectory=trajectory,
            requested_amount=request.requested_amount,
            min_keep=min_keep,
            request_date=req_date,
            window_end=window_end,
        )

        return ForecastResult(
            request_id=request.request_id,
            request_date=req_date,
            window_end_date=window_end,
            initial_balance=init_bal,
            minimum_balance_to_keep=min_keep,
            min_balance_in_window=min_bal,
            amount_safe_to_pay=round(safe_to_pay, 2),
            earliest_date_for_full_payment=earliest_full,
            balance_trajectory=trajectory,
        )

    def is_plan_safe(
        self,
        payments: List[Tuple[date, float]],
        ledger: List[LedgerEntry],
        initial_balance: float,
        minimum_balance_to_keep: float,
        request_date: date,
        window_end_date: Optional[date] = None,
        spending_adjustments: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Check whether a proposed schedule of payments [ (date, amount), ... ]
        maintains balance >= minimum_balance_to_keep throughout the window.

        spending_adjustments: optional dict mapping entry_id -> new_amount_home (or 0.0 for stopped)
        """
        if window_end_date is None:
            window_end_date = request_date + timedelta(days=self.forecast_days)

        spending_adjustments = spending_adjustments or {}

        # Collect all daily cash flow delta on each date
        flows_by_date: Dict[date, float] = {}

        # 1. Base ledger flows
        for entry in ledger:
            if not (request_date <= entry.entry_date <= window_end_date):
                continue
            amt = entry.amount_home
            if entry.entry_id in spending_adjustments:
                amt = spending_adjustments[entry.entry_id]

            delta = amt if entry.direction == "credit" else -amt
            flows_by_date[entry.entry_date] = flows_by_date.get(entry.entry_date, 0.0) + delta

        # 2. Add candidate plan payments (outflows)
        for pay_date, pay_amt in payments:
            if pay_date < request_date or pay_date > window_end_date:
                # Any payment outside the 90-day window or before request date cannot be certified safe
                return False
            flows_by_date[pay_date] = flows_by_date.get(pay_date, 0.0) - pay_amt

        # Sort dates
        unique_dates = sorted(flows_by_date.keys())
        if not unique_dates or unique_dates[0] > request_date:
            unique_dates.insert(0, request_date)

        curr_bal = initial_balance
        if curr_bal < minimum_balance_to_keep - 1e-4:
            return False

        for d in unique_dates:
            curr_bal += flows_by_date.get(d, 0.0)
            if curr_bal < minimum_balance_to_keep - 1e-4:
                return False

        return True

    def _build_trajectory(
        self,
        initial_balance: float,
        ledger: List[LedgerEntry],
        start_date: date,
        end_date: date,
    ) -> List[BalancePoint]:
        """Compute piecewise-constant balance points from start_date to end_date."""
        trajectory: List[BalancePoint] = [BalancePoint(on_date=start_date, balance=initial_balance)]

        # Group ledger entries by date
        daily_delta: Dict[date, float] = {}
        for entry in ledger:
            if start_date <= entry.entry_date <= end_date:
                delta = entry.amount_home if entry.direction == "credit" else -entry.amount_home
                daily_delta[entry.entry_date] = daily_delta.get(entry.entry_date, 0.0) + delta

        curr_balance = initial_balance
        for d in sorted(daily_delta.keys()):
            curr_balance += daily_delta[d]
            trajectory.append(BalancePoint(on_date=d, balance=curr_balance))

        return trajectory

    def _find_earliest_date_for_full_payment(
        self,
        trajectory: List[BalancePoint],
        requested_amount: float,
        min_keep: float,
        request_date: date,
        window_end: date,
    ) -> Optional[date]:
        """
        Find the first date T in [request_date, window_end] such that
        for all trajectory points with on_date >= T, balance - requested_amount >= min_keep.
        """
        # Collect candidate check dates: request_date and all event dates
        candidate_dates = sorted({pt.on_date for pt in trajectory if request_date <= pt.on_date <= window_end})

        for cand_date in candidate_dates:
            # Check if for all points at or after cand_date, balance - requested_amount >= min_keep
            is_safe = True
            for pt in trajectory:
                if pt.on_date >= cand_date:
                    if pt.balance - requested_amount < min_keep - 1e-4:
                        is_safe = False
                        break
            if is_safe:
                return cand_date

        return None
