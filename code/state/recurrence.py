"""
State Reconstruction & Recurrence Detection — Deterministic ledger projection.

Per SOLUTION.md §7 (Component 4) and user approved rules:
- Builds a per-user, date-ordered ledger of cash flows covering [request_date, request_date + 90].
- Applies resolved claims from Component 3 to amend event amounts, dates, or cancellations.
- Detects recurring expenses and income from historical settled events using:
  1. Group by (user_id, category).
  2. Cluster by amount similarity (±10%).
  3. Consistent delta between occurrences within ±3 days of common intervals (7, 14, 30, 365 days).
  4. Projects forward using the most recent occurrence's amount.
- Incorporates explicit future events:
  - Next confirmed salary (counted on its settlement date).
  - Reserved pending debits (committed cash outflow).
  - Excludes pending credits (except confirmed salary).
  - Excludes cancelled/failed transactions and unrealized investment valuations.
- Converts all amounts to the user's home_currency on the relevant date.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple

from currency.converter import CurrencyConverter
from loaders.data_loader import DataLoader, FinancialEvent, FinancialProfile
from resolution.conflict_resolver import ResolvedFact

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LedgerEntry:
    entry_id: str
    event_id: Optional[str]  # None for projected recurring occurrences
    entry_date: date
    direction: str           # "credit" (inflow) or "debit" (outflow)
    category: str
    description: str
    amount_home: float       # Amount converted to user's home currency (> 0)
    flexibility: str         # "fixed", "reducible", "stoppable", "reducible_or_stoppable"
    minimum_allowed_amount_home: Optional[float]
    is_projected: bool


class StateReconstructor:
    """
    Deterministic forward-looking ledger generator.
    """

    def __init__(self, data_loader: DataLoader, currency_converter: CurrencyConverter) -> None:
        self.data = data_loader
        self.converter = currency_converter

    def build_forecast_ledger(
        self,
        user_id: str,
        request_date: date,
        resolved_facts: Optional[Dict[Tuple[str, str], ResolvedFact]] = None,
        forecast_days: int = 90,
    ) -> List[LedgerEntry]:
        """
        Build the chronological cash-flow ledger for [request_date, request_date + forecast_days].
        """
        profile = self.data.profiles.get(user_id)
        if not profile:
            raise ValueError(f"Profile not found for user {user_id}")

        home_ccy = profile.home_currency
        end_date = request_date + timedelta(days=forecast_days)
        resolved_facts = resolved_facts or {}

        events = self.data.events_by_user.get(user_id, [])

        # Step 1: Materialize updated events with resolved facts applied
        effective_events = self._apply_resolved_facts(events, resolved_facts)

        # Step 2: Identify and collect explicit future events in [request_date, end_date]
        explicit_entries, explicit_event_signatures = self._collect_explicit_future_entries(
            effective_events, request_date, end_date, home_ccy
        )

        # Step 3: Recurrence detection on historical settled events (before request_date)
        recurring_entries = self._detect_and_project_recurrence(
            effective_events, request_date, end_date, home_ccy, explicit_event_signatures, explicit_entries=explicit_entries
        )

        # Combine and sort chronologically
        combined = explicit_entries + recurring_entries
        combined.sort(key=lambda e: (e.entry_date, 0 if e.direction == "credit" else 1, e.entry_id))

        logger.info(
            "Built forecast ledger for user %s: %d explicit entries, %d projected entries",
            user_id,
            len(explicit_entries),
            len(recurring_entries),
        )
        return combined

    def _apply_resolved_facts(
        self, events: List[FinancialEvent], resolved_facts: Dict[Tuple[str, str], ResolvedFact]
    ) -> List[FinancialEvent]:
        """Return shallow copies of events with resolved facts applied."""
        updated: List[FinancialEvent] = []
        for e in events:
            eid = e.event_id
            amt = e.amount
            edate = e.event_date
            sdate = e.settlement_date
            status = e.status

            if (eid, "amount") in resolved_facts:
                try:
                    amt = float(resolved_facts[(eid, "amount")].value)
                except ValueError:
                    pass

            if (eid, "event_date") in resolved_facts:
                try:
                    raw = resolved_facts[(eid, "event_date")].value
                    parsed_d = date.fromisoformat(raw[:10])
                    edate = parsed_d
                    if sdate:
                        sdate = parsed_d
                except Exception:
                    pass

            if (eid, "status") in resolved_facts:
                status = resolved_facts[(eid, "status")].value

            updated.append(
                FinancialEvent(
                    event_id=e.event_id,
                    user_id=e.user_id,
                    event_type=e.event_type,
                    description=e.description,
                    category=e.category,
                    direction=e.direction,
                    amount=amt,
                    currency=e.currency,
                    event_date=edate,
                    settlement_date=sdate,
                    status=status,
                    linked_event_id=e.linked_event_id,
                    flexibility=e.flexibility,
                    minimum_allowed_amount=e.minimum_allowed_amount,
                )
            )
        return updated

    def _collect_explicit_future_entries(
        self,
        events: List[FinancialEvent],
        start_date: date,
        end_date: date,
        home_ccy: str,
    ) -> Tuple[List[LedgerEntry], Set[Tuple[str, date]]]:
        """Collect explicit future cash flows falling in the window."""
        entries: List[LedgerEntry] = []
        signatures: Set[Tuple[str, date]] = set()

        for e in events:
            if e.status in ("cancelled", "failed", "unrealized"):
                continue
            if e.direction == "non_cash":
                continue

            event_target_date = e.settlement_date or e.event_date
            if not event_target_date:
                continue

            if not (start_date <= event_target_date <= end_date):
                continue

            # Cash-state rules
            if e.direction == "debit":
                # Settled, scheduled, or pending debits are counted as committed outflows
                if e.status not in ("settled", "scheduled", "pending"):
                    continue
            elif e.direction == "credit":
                # Settled credits count.
                # Scheduled/pending salary counts ONLY if confirmed salary
                is_confirmed_salary = (e.event_type == "income" or e.category == "salary")
                if e.status == "settled":
                    pass
                elif e.status in ("scheduled", "pending") and is_confirmed_salary:
                    pass
                else:
                    continue
            else:
                continue

            # Ensure amount is valid
            if e.amount is None or e.amount <= 0:
                continue

            try:
                amt_home = self.converter.convert(e.amount, e.currency, home_ccy, event_target_date)
            except Exception:
                amt_home = e.amount

            min_amt_home = None
            if e.minimum_allowed_amount is not None:
                try:
                    min_amt_home = self.converter.convert(
                        e.minimum_allowed_amount, e.currency, home_ccy, event_target_date
                    )
                except Exception:
                    min_amt_home = e.minimum_allowed_amount

            entries.append(
                LedgerEntry(
                    entry_id=f"exp_{e.event_id}",
                    event_id=e.event_id,
                    entry_date=event_target_date,
                    direction=e.direction,
                    category=e.category,
                    description=e.description,
                    amount_home=amt_home,
                    flexibility=e.flexibility,
                    minimum_allowed_amount_home=min_amt_home,
                    is_projected=False,
                )
            )
            # Track signature to prevent recurrence double-counting for that month/week
            signatures.add((e.category, event_target_date))

        return entries, signatures

    def _detect_and_project_recurrence(
        self,
        events: List[FinancialEvent],
        start_date: date,
        end_date: date,
        home_ccy: str,
        explicit_signatures: Set[Tuple[str, date]],
        explicit_entries: Optional[List[LedgerEntry]] = None,
    ) -> List[LedgerEntry]:
        """Detect recurring patterns from historical settled events and project forward."""
        # Variable categories use (category, description, direction); fixed commitments use (category, direction)
        variable_categories = {"groceries", "transport", "dining", "shopping", "entertainment"}

        history = [
            e for e in events
            if e.status == "settled"
            and e.amount is not None
            and e.amount > 0
            and e.direction in ("credit", "debit")
            and (e.settlement_date or e.event_date or date.min) < start_date
        ]

        groups: Dict[Tuple, List[FinancialEvent]] = {}
        for e in history:
            if e.category in variable_categories:
                key = (e.category, e.description.strip().lower(), e.direction)
            else:
                key = (e.category, e.direction)
            groups.setdefault(key, []).append(e)

        projected_entries: List[LedgerEntry] = []

        # 1. Project standard recurring series from history
        for key, cat_events in groups.items():
            category = key[0]
            direction = key[-1]

            if len(cat_events) < 2:
                continue

            # Sort by event_date
            cat_events.sort(key=lambda e: e.settlement_date or e.event_date or date.min)

            # Amount clustering: within ±10%
            clusters = self._cluster_by_amount(cat_events)

            for cluster in clusters:
                if len(cluster) < 2:
                    continue

                # Compute consecutive date deltas
                deltas = []
                dates_in_cluster = [e.settlement_date or e.event_date for e in cluster if (e.settlement_date or e.event_date)]
                for i in range(len(dates_in_cluster) - 1):
                    d1 = dates_in_cluster[i]
                    d2 = dates_in_cluster[i + 1]
                    deltas.append((d2 - d1).days)

                interval = self._detect_common_interval(deltas)
                if not interval:
                    continue

                last_event = cluster[-1]
                last_date = dates_in_cluster[-1]

                # Liveness check: series must still be active leading up to start_date
                if (start_date - last_date).days > max(45, int(interval * 1.5)):
                    continue

                last_amount = last_event.amount
                last_currency = last_event.currency

                # Project forward cycles into [start_date, end_date]
                curr_date = last_date + timedelta(days=interval)
                cycle = 1
                while curr_date <= end_date:
                    if curr_date >= start_date:
                        is_duplicate = False
                        for offset in range(-4, 5):
                            if (category, curr_date + timedelta(days=offset)) in explicit_signatures:
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            try:
                                amt_home = self.converter.convert(last_amount, last_currency, home_ccy, curr_date)
                            except Exception:
                                amt_home = last_amount

                            min_amt_home = None
                            if last_event.minimum_allowed_amount is not None:
                                try:
                                    min_amt_home = self.converter.convert(
                                        last_event.minimum_allowed_amount, last_currency, home_ccy, curr_date
                                    )
                                except Exception:
                                    min_amt_home = last_event.minimum_allowed_amount

                            projected_entries.append(
                                LedgerEntry(
                                    entry_id=f"rec_{last_event.event_id}_{cycle}",
                                    event_id=last_event.event_id,
                                    entry_date=curr_date,
                                    direction=direction,
                                    category=category,
                                    description=f"Projected {last_event.description}",
                                    amount_home=amt_home,
                                    flexibility=last_event.flexibility,
                                    minimum_allowed_amount_home=min_amt_home,
                                    is_projected=True,
                                )
                            )

                    curr_date += timedelta(days=interval)
                    cycle += 1

        # 2. Confirmed Salary Continuation across 90-day window
        # Per SOLUTION.md §7: projection resumes at the next cycle after the explicit event
        explicit_entries = explicit_entries or []
        salary_explicit = [
            e for e in explicit_entries
            if e.direction == "credit" and (e.category == "salary" or "salary" in e.description.lower())
        ]
        if salary_explicit:
            # Anchor to the latest explicit confirmed salary
            salary_explicit.sort(key=lambda e: e.entry_date)
            last_sal = salary_explicit[-1]
            sal_date = last_sal.entry_date + timedelta(days=30)
            sal_cycle = 1
            while sal_date <= end_date:
                is_duplicate = any(abs((sal_date - s.entry_date).days) <= 4 for s in salary_explicit)
                if not is_duplicate:
                    projected_entries.append(
                        LedgerEntry(
                            entry_id=f"rec_salary_cont_{sal_cycle}",
                            event_id=last_sal.event_id,
                            entry_date=sal_date,
                            direction="credit",
                            category="salary",
                            description="Confirmed ongoing salary",
                            amount_home=last_sal.amount_home,
                            flexibility=last_sal.flexibility,
                            minimum_allowed_amount_home=None,
                            is_projected=True,
                        )
                    )
                sal_date += timedelta(days=30)
                sal_cycle += 1

        return projected_entries

    def _cluster_by_amount(self, events: List[FinancialEvent]) -> List[List[FinancialEvent]]:
        """Cluster events where amounts are within ±10%."""
        clusters: List[List[FinancialEvent]] = []
        for e in events:
            placed = False
            for c in clusters:
                ref_amt = c[0].amount
                if abs(e.amount - ref_amt) <= 0.10 * ref_amt:
                    c.append(e)
                    placed = True
                    break
            if not placed:
                clusters.append([e])
        return clusters

    def _detect_common_interval(self, deltas: List[int]) -> Optional[int]:
        """Detect if consecutive deltas match standard recurrence intervals (~7, ~14, ~30, ~365) within ±3 days."""
        if not deltas:
            return None

        candidate_intervals = [7, 14, 30, 365]
        for base in candidate_intervals:
            matching_vals = [d for d in deltas if abs(d - base) <= 3]
            # Require consistent periodicity: at least 70% of deltas must match the interval
            if len(matching_vals) >= max(1, int(len(deltas) * 0.7)):
                return round(sum(matching_vals) / len(matching_vals))

        return None
