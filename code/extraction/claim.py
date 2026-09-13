"""Claim schema — Pydantic data model for extracted financial facts."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Claim(BaseModel):
    """
    A single structured financial claim extracted from a message or image.

    Per SOLUTION.md §5, claim schema fields.
    """

    target_event_id: Optional[str] = Field(
        default=None,
        description=(
            "The event_id this claim targets (e.g. 'event_14'), "
            "or null if the claim does not reference a specific known event."
        ),
    )
    target_request_id: Optional[str] = Field(
        default=None,
        description=(
            "The request_id this claim relates to (e.g. 'request_26'), "
            "or null if not request-specific."
        ),
    )
    claim_type: Literal[
        "amend", "cancel", "confirm", "delay",
        "new_income", "new_expense", "none",
    ] = Field(
        description=(
            "Type of claim: "
            "'amend' = changes a field on an existing event, "
            "'cancel' = cancels an existing event, "
            "'confirm' = confirms details of an existing event, "
            "'delay' = postpones an event to a later date, "
            "'new_income' = reports new income not in financial records, "
            "'new_expense' = reports a new expense not in financial records, "
            "'none' = no actionable financial claim found."
        ),
    )
    field: Optional[Literal["amount", "event_date", "status"]] = Field(
        default=None,
        description=(
            "Which event field this claim modifies: "
            "'amount', 'event_date', or 'status'. "
            "Required for 'amend', 'confirm', 'delay'; null otherwise."
        ),
    )
    new_value: Optional[str] = Field(
        default=None,
        description=(
            "The new value for the field, as a string. "
            "For amounts: numeric only, no currency symbols or commas. "
            "For dates: YYYY-MM-DD format. "
            "For status: one of settled/pending/cancelled/failed/scheduled."
        ),
    )
    effective_date: Optional[str] = Field(
        default=None,
        description=(
            "Date from which this claim takes effect (YYYY-MM-DD). "
            "For salary changes, this is when the new amount starts."
        ),
    )
    source_date: Optional[str] = Field(
        default=None,
        description=(
            "Date/timestamp when this information was communicated "
            "(from sent_at or document date)."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "Confidence in the extraction: "
            "'high' = clear, unambiguous fact; "
            "'medium' = likely correct but some ambiguity; "
            "'low' = uncertain, may need conflict resolution."
        ),
    )
