"""
Conflict Resolution — Deterministic resolution across multiple claims targeting the same event/field.

Per SOLUTION.md §6 (Component 3):
Given all claims targeting the same (event_id, field), resolve to one final value.

Precedence (in order):
1. Explicit cancellation/settlement/amendment claim wins outright over confirm/delay.
2. Among remaining claims, the newest (by source_date/sent_at), compared across all claims regardless of source_type, wins.
3. A settled event/claim outranks an estimate or forecast-only claim.
4. If still unresolved (e.g. tie or ambiguous low-confidence claim): apply the safer interpretation:
   - For expenses: prefer higher amount / earlier date.
   - For income: prefer lower amount / later date.

Guardrails (SOLUTION.md §13):
- Discards claims referencing nonexistent event_ids (when target_event_id is populated).
- Validates field names to allowed set ('amount', 'event_date', 'status').
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from extraction.claim import Claim
from loaders.data_loader import DataLoader, FinancialEvent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResolvedFact:
    event_id: str
    field: str
    value: str
    source_claim: Claim
    effective_date: Optional[str] = None


class ConflictResolver:
    """
    Deterministic conflict resolver for extracted claims.
    """

    def __init__(self, data_loader: DataLoader) -> None:
        self.data = data_loader

    def resolve_claims(self, claims: List[Claim]) -> Dict[Tuple[str, str], ResolvedFact]:
        """
        Group claims by (target_event_id, field) and resolve conflicts deterministically.

        Returns
        -------
        Dict mapping (event_id, field) -> ResolvedFact
        """
        valid_claims: List[Claim] = []
        for claim in claims:
            if not self._is_valid_claim(claim):
                continue
            valid_claims.append(claim)

        # Group by (event_id, field)
        grouped: Dict[Tuple[str, str], List[Claim]] = {}
        for claim in valid_claims:
            target_event_id = claim.target_event_id
            if not target_event_id:
                continue

            field_name = claim.field
            if claim.claim_type == "cancel":
                field_name = "status"
                claim.new_value = "cancelled"

            if not field_name:
                continue

            key = (target_event_id, field_name)
            grouped.setdefault(key, []).append(claim)

        resolved: Dict[Tuple[str, str], ResolvedFact] = {}
        for (event_id, field_name), claim_list in grouped.items():
            best_claim = self._resolve_group(event_id, field_name, claim_list)
            if best_claim and best_claim.new_value:
                resolved[(event_id, field_name)] = ResolvedFact(
                    event_id=event_id,
                    field=field_name,
                    value=best_claim.new_value,
                    source_claim=best_claim,
                    effective_date=best_claim.effective_date,
                )

        logger.info("ConflictResolver resolved %d facts across %d claims", len(resolved), len(claims))
        return resolved

    def _is_valid_claim(self, claim: Claim) -> bool:
        """Apply guardrails per SOLUTION.md §13."""
        if claim.claim_type == "none":
            return False

        if claim.target_event_id:
            if claim.target_event_id not in self.data.events:
                logger.warning("Discarding claim referencing nonexistent event_id: %s", claim.target_event_id)
                return False

        if claim.claim_type != "cancel" and claim.field not in ("amount", "event_date", "status"):
            logger.warning("Discarding claim with invalid field: %s", claim.field)
            return False

        return True

    def _resolve_group(self, event_id: str, field_name: str, claims: List[Claim]) -> Optional[Claim]:
        if not claims:
            return None
        if len(claims) == 1:
            return claims[0]

        event = self.data.events.get(event_id)
        is_income = event.direction == "credit" if event else False

        def get_type_rank(c: Claim) -> int:
            # 1. Explicit cancellation / settlement / amendment claim wins outright over confirm / delay
            if c.claim_type in ("cancel", "amend"):
                return 2
            if c.claim_type == "confirm" and c.field == "status" and c.new_value == "settled":
                return 2
            return 1

        def parse_ts(ts_str: Optional[str]) -> float:
            if not ts_str:
                return 0.0
            clean = ts_str.replace("Z", "+00:00")
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(clean[:19], fmt[: len(clean[:19])]).timestamp()
                except Exception:
                    pass
            return 0.0

        def is_settled(c: Claim) -> int:
            return 1 if (c.field == "status" and c.new_value == "settled") else 0

        # Step 1: Claim type precedence
        max_type_rank = max(get_type_rank(c) for c in claims)
        candidates = [c for c in claims if get_type_rank(c) == max_type_rank]
        if len(candidates) == 1:
            return candidates[0]

        # Step 2: Newest by source_date / sent_at
        timestamps = [parse_ts(c.source_date) for c in candidates]
        max_ts = max(timestamps)
        candidates = [c for c, ts in zip(candidates, timestamps) if ts == max_ts]
        if len(candidates) == 1:
            return candidates[0]

        # Step 3: Settled event/claim outranks estimate
        settled_ranks = [is_settled(c) for c in candidates]
        max_settled = max(settled_ranks)
        candidates = [c for c, s in zip(candidates, settled_ranks) if s == max_settled]
        if len(candidates) == 1:
            return candidates[0]

        # Step 4: Safer interpretation
        # For expenses: higher amount / earlier date
        # For income: lower amount / later date
        if field_name == "amount":
            def get_num(c: Claim) -> float:
                try:
                    return float(c.new_value) if c.new_value else 0.0
                except ValueError:
                    return 0.0

            if is_income:
                return min(candidates, key=get_num)
            else:
                return max(candidates, key=get_num)

        elif field_name == "event_date":
            def get_date_val(c: Claim) -> str:
                return c.new_value or "9999-99-99"

            if is_income:
                # Later date for income is safer
                return max(candidates, key=get_date_val)
            else:
                # Earlier date for expense is safer
                return min(candidates, key=get_date_val)

        # Fallback to first candidate
        return candidates[0]
