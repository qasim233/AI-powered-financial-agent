"""
Plan Ranker — Deterministic candidate generation and multi-criteria ranking.

Per SOLUTION.md §10 (Component 7):
- Generates candidate plans across eligible payment methods:
  - full_payment (on request_date)
  - wait (full payment on earliest_date_for_full_payment)
  - partial_payment (exactly 2 payments: safe_today on request_date, remainder on earliest_date)
  - installments (derived from request_payment_options.csv)
- Tests each candidate for safety (first without spending changes, then with spending changes via Optimizer).
- Ranks safe eligible candidates strictly in order:
  1. Completes by desired_completion_date.
  2. Requires no spending changes.
  3. Minimizes total payable amount.
  4. Starts payment earlier.
  5. Uses fewer payments.
  6. Lowest payment_option_id tie-breaker.
- Falls back to not_recommended / not_affordable if no safe plan exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from forecast.safety_engine import ForecastResult, SafetyEngine
from loaders.data_loader import DataLoader, FinancialProfile, PaymentOption, Request
from optimizer.spending_optimizer import SpendingChange, SpendingChangeOptimizer
from state.recurrence import LedgerEntry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CandidatePlan:
    payment_method: str              # "full_payment", "partial_payment", "installments", "wait"
    payment_option_id: Optional[str] # for installments
    payments: List[Tuple[date, float]]
    total_amount_paid: float
    start_date: date
    completion_date: date
    completes_by_deadline: bool
    spending_changes: List[SpendingChange]
    affordability_status: str        # "affordable_now", "affordable_with_plan", "affordable_later"

    def format_plan_str(self) -> str:
        """Format payment plan string e.g. 'YYYY-MM-DD:amount|YYYY-MM-DD:amount'."""
        if not self.payments or self.payment_method == "not_recommended":
            return "none"
        parts = []
        for p_date, p_amt in self.payments:
            if p_amt == int(p_amt):
                amt_str = str(int(p_amt))
            else:
                amt_str = f"{p_amt:.2f}".rstrip("0").rstrip(".")
            parts.append(f"{p_date.isoformat()}:{amt_str}")
        return "|".join(parts)

    def format_spending_changes_str(self, data_loader: Optional[DataLoader] = None) -> str:
        """Format spending changes needed e.g. 'stop:event_14|reduce_to:event_21:100' or 'none'."""
        if not self.spending_changes:
            return "none"
        formatted = []
        for sc in self.spending_changes:
            if sc.action == "stop":
                formatted.append(f"stop:{sc.event_id}")
            else:
                amt = sc.new_amount_home
                if data_loader and sc.event_id in data_loader.events:
                    ev = data_loader.events[sc.event_id]
                    if ev.minimum_allowed_amount is not None:
                        amt = ev.minimum_allowed_amount
                if amt == int(amt):
                    amt_str = str(int(amt))
                else:
                    amt_str = f"{amt:.2f}".rstrip("0").rstrip(".")
                formatted.append(f"reduce_to:{sc.event_id}:{amt_str}")
        return "|".join(formatted)


class PlanRanker:
    """
    Evaluates and ranks candidate payment plans deterministically.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        safety_engine: SafetyEngine,
        spending_optimizer: SpendingChangeOptimizer,
    ) -> None:
        self.data = data_loader
        self.safety_engine = safety_engine
        self.optimizer = spending_optimizer

    def select_best_plan(
        self,
        request: Request,
        profile: FinancialProfile,
        ledger: List[LedgerEntry],
        forecast: ForecastResult,
    ) -> CandidatePlan:
        """
        Evaluate all candidate plans, filter to safe options, and rank.
        """
        candidates: List[CandidatePlan] = []
        user_methods = set(profile.payment_methods_user_will_consider)

        # 1. Candidate: Full Payment on request_date
        if "full_payment" in user_methods:
            plan = self._eval_full_payment_now(request, profile, ledger, forecast)
            if plan:
                candidates.append(plan)

        # 2. Candidate: Wait (Full Payment on earliest_date_for_full_payment)
        if "full_payment" in user_methods:
            plan = self._eval_wait(request, profile, ledger, forecast)
            if plan:
                candidates.append(plan)

        # 3. Candidate: Partial Payment
        # Allowed if user considers partial_payment (or full_payment) and request allows partial payment
        if request.allows_partial_payment and ("partial_payment" in user_methods or "full_payment" in user_methods):
            plan = self._eval_partial_payment(request, profile, ledger, forecast)
            if plan:
                candidates.append(plan)

        # 4. Candidates: Installments from request_payment_options
        if "installments" in user_methods:
            installment_plans = self._eval_installments(request, profile, ledger)
            candidates.extend(installment_plans)

        if not candidates:
            # Fallback
            return CandidatePlan(
                payment_method="not_recommended",
                payment_option_id=None,
                payments=[],
                total_amount_paid=0.0,
                start_date=request.request_date,
                completion_date=request.request_date,
                completes_by_deadline=False,
                spending_changes=[],
                affordability_status="not_affordable",
            )

        # Rank candidates strictly per SOLUTION.md §10
        candidates.sort(key=lambda c: self._rank_key(c, request))

        return candidates[0]

    def _eval_full_payment_now(
        self,
        request: Request,
        profile: FinancialProfile,
        ledger: List[LedgerEntry],
        forecast: ForecastResult,
    ) -> Optional[CandidatePlan]:
        req_date = request.request_date
        amt = float(request.requested_amount)
        payments = [(req_date, amt)]

        # Check if safe without spending changes
        if forecast.amount_safe_to_pay >= amt:
            return CandidatePlan(
                payment_method="full_payment",
                payment_option_id=None,
                payments=payments,
                total_amount_paid=amt,
                start_date=req_date,
                completion_date=req_date,
                completes_by_deadline=(req_date <= request.desired_completion_date),
                spending_changes=[],
                affordability_status="affordable_now",
            )

        # Check if safe with spending changes
        is_safe, changes = self.optimizer.find_minimal_changes(payments, ledger, profile, request)
        if is_safe and changes:
            return CandidatePlan(
                payment_method="full_payment",
                payment_option_id=None,
                payments=payments,
                total_amount_paid=amt,
                start_date=req_date,
                completion_date=req_date,
                completes_by_deadline=(req_date <= request.desired_completion_date),
                spending_changes=changes,
                affordability_status="affordable_with_plan",
            )

        return None

    def _eval_wait(
        self,
        request: Request,
        profile: FinancialProfile,
        ledger: List[LedgerEntry],
        forecast: ForecastResult,
    ) -> Optional[CandidatePlan]:
        earliest_date = forecast.earliest_date_for_full_payment
        if not earliest_date or earliest_date <= request.request_date:
            return None

        amt = float(request.requested_amount)
        payments = [(earliest_date, amt)]

        # Wait implies paying in full on earliest_date without optional spending changes
        if self.safety_engine.is_plan_safe(
            payments=payments,
            ledger=ledger,
            initial_balance=profile.current_available_balance,
            minimum_balance_to_keep=profile.minimum_balance_to_keep,
            request_date=request.request_date,
        ):
            return CandidatePlan(
                payment_method="wait",
                payment_option_id=None,
                payments=payments,
                total_amount_paid=amt,
                start_date=earliest_date,
                completion_date=earliest_date,
                completes_by_deadline=(earliest_date <= request.desired_completion_date),
                spending_changes=[],
                affordability_status="affordable_later",
            )

        return None

    def _eval_partial_payment(
        self,
        request: Request,
        profile: FinancialProfile,
        ledger: List[LedgerEntry],
        forecast: ForecastResult,
    ) -> Optional[CandidatePlan]:
        safe_today = forecast.amount_safe_to_pay
        total_amt = float(request.requested_amount)
        earliest_date = forecast.earliest_date_for_full_payment

        # Constraint: 0 < amount_safe_to_pay < requested_amount
        if not (0 < safe_today < total_amt):
            return None

        # Constraint: earliest_date_for_full_payment <= desired_completion_date
        if not earliest_date or earliest_date > request.desired_completion_date:
            return None

        remainder = round(total_amt - safe_today, 2)
        payments = [
            (request.request_date, safe_today),
            (earliest_date, remainder),
        ]

        # Verify safety
        if self.safety_engine.is_plan_safe(
            payments=payments,
            ledger=ledger,
            initial_balance=profile.current_available_balance,
            minimum_balance_to_keep=profile.minimum_balance_to_keep,
            request_date=request.request_date,
        ):
            return CandidatePlan(
                payment_method="partial_payment",
                payment_option_id=None,
                payments=payments,
                total_amount_paid=total_amt,
                start_date=request.request_date,
                completion_date=earliest_date,
                completes_by_deadline=True,
                spending_changes=[],
                affordability_status="affordable_with_plan",
            )

        return None

    def _eval_installments(
        self,
        request: Request,
        profile: FinancialProfile,
        ledger: List[LedgerEntry],
    ) -> List[CandidatePlan]:
        options = self.data.payment_options_by_request.get(request.request_id, [])
        valid_plans: List[CandidatePlan] = []

        for opt in options:
            if opt.payment_method != "installments":
                continue

            # Respect max_installment_months if specified
            if profile.max_installment_months is not None:
                # Estimate duration in months: (number_of_payments - 1) * freq_days / 30
                num_months = ((opt.number_of_payments - 1) * (opt.payment_frequency_days or 30)) / 30.0
                if num_months > profile.max_installment_months + 0.1:
                    continue

            # Build payments list
            payments = self._build_option_schedule(opt, request.request_date)
            if not payments:
                continue

            start_d = payments[0][0]
            end_d = payments[-1][0]
            completes_by_deadline = (end_d <= request.desired_completion_date)

            # Test safety without spending changes
            is_safe, changes = self.optimizer.find_minimal_changes(payments, ledger, profile, request)
            if is_safe:
                valid_plans.append(
                    CandidatePlan(
                        payment_method="installments",
                        payment_option_id=opt.payment_option_id,
                        payments=payments,
                        total_amount_paid=float(opt.total_payable_amount),
                        start_date=start_d,
                        completion_date=end_d,
                        completes_by_deadline=completes_by_deadline,
                        spending_changes=changes,
                        affordability_status="affordable_with_plan",
                    )
                )

        return valid_plans

    def _build_option_schedule(self, opt: PaymentOption, default_start: date) -> List[Tuple[date, float]]:
        start = opt.first_payment_date or default_start
        freq = opt.payment_frequency_days or 30
        schedule = []
        for i in range(opt.number_of_payments):
            d = start + timedelta(days=i * freq)
            schedule.append((d, opt.payment_amount))
        return schedule

    def _rank_key(self, c: CandidatePlan, req: Request) -> Tuple:
        """
        Ranking criteria (lower is better):
        1. Completes by desired_completion_date (0 = Yes, 1 = No)
        2. Requires no spending changes (0 = No changes, 1 = Has changes)
        3. Total payable amount (lower is better)
        4. Starts payment earlier (earlier start_date is better)
        5. Uses fewer payments (fewer payments is better)
        6. Lowest payment_option_id as final tie-breaker
        """
        rule1 = 0 if c.completes_by_deadline else 1
        rule2 = 0 if len(c.spending_changes) == 0 else 1
        rule3 = c.total_amount_paid
        rule4 = c.start_date.toordinal()
        rule5 = len(c.payments)
        rule6 = c.payment_option_id or "zzzzzz"

        return (rule1, rule2, rule3, rule4, rule5, rule6)
