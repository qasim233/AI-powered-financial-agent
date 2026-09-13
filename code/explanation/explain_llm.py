"""
Explanation Generation — Concise natural language explanation grounded in computed facts.

Per SOLUTION.md §11 (Component 8):
- Translates the fully-computed decision into concise English text.
- Follows the terse, factual style calibrated on sample_requests.csv.
- Guardrails: Only receives pre-computed structured summaries; cannot change any numbers or dates.
- Includes a template-based fallback if the LLM call fails.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loaders.data_loader import DataLoader, FinancialProfile, Request
from ranking.plan_ranker import CandidatePlan

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert financial advisor writing a concise, grounded explanation \
for an affordability decision.

Write 1-2 factual sentences explaining the decision in English, exactly matching \
the style seen in the examples below.

CRITICAL RULES:
- Never alter any numbers, dates, currency symbols, or event names given in the input.
- Keep the tone direct, factual, and supportive.
- Do NOT use filler phrases like "Based on our analysis" or "I recommend".
- Follow the specific style template that matches the decision.

STYLE EXAMPLES:
1. Full payment affordable today:
   "Pay ZAR 25,256 today. This leaves at least ZAR 18,000 available over the next 90 days."
2. Full payment with spending change:
   "Stop the family streaming plan, then pay EUR 620.40 today. This leaves at least EUR 800 available."
3. Installments:
   "Use 3 installments of INR 68,432, starting 12 September 2024. This leaves at least INR 93,000 available."
4. Wait:
   "Pay EUR 996.60 in full on 15 April 2025. Paying earlier would take the balance below the EUR 800 minimum."
5. Partial payment:
   "Pay INR 28,820 today and the remaining INR 10,840 on 15 September 2024. This completes the full request and keeps the INR 92,800 minimum protected."
6. Not affordable:
   "Do not make this payment by 10 February 2025. None of the available options keeps the INR 225,400 minimum protected."
   OR
   "Do not proceed with the EUR 5,414.20 request. Although EUR 597.74 is available today, the full amount cannot be completed safely within 90 days."
"""


class ExplanationGenerator:
    """
    Generates decision explanations using LLM or structured template fallback.
    """

    def __init__(self, llm_client=None, data_loader: Optional[DataLoader] = None) -> None:
        self.llm = llm_client
        self.data = data_loader

    def generate_explanation(
        self,
        request: Request,
        profile: FinancialProfile,
        plan: CandidatePlan,
        safe_today: float,
        earliest_full_date: Optional[str],
        spending_changes_str: str,
    ) -> str:
        """
        Generate the decision explanation. Uses LLM if available, falls back to deterministic template.
        """
        if self.llm is not None:
            try:
                prompt_content = self._build_input_summary(
                    request, profile, plan, safe_today, earliest_full_date, spending_changes_str
                )
                messages = [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=prompt_content),
                ]
                explanation = self.llm.explain(messages, target_id=request.request_id)
                clean = explanation.strip().strip('"')
                if len(clean) > 10:
                    return clean
            except Exception:
                logger.exception("LLM explanation failed for %s; using template fallback", request.request_id)

        # Fallback template
        return self._template_fallback(request, profile, plan, safe_today, earliest_full_date)

    def _build_input_summary(
        self,
        req: Request,
        profile: FinancialProfile,
        plan: CandidatePlan,
        safe_today: float,
        earliest_full_date: Optional[str],
        spending_changes_str: str,
    ) -> str:
        ccy = profile.home_currency
        min_keep = profile.minimum_balance_to_keep

        # Describe spending changes if any
        change_desc = "none"
        if plan.spending_changes and self.data:
            descs = []
            for sc in plan.spending_changes:
                ev = self.data.events.get(sc.event_id)
                ev_desc = ev.description.lower() if ev else sc.event_id
                if sc.action == "stop":
                    descs.append(f"Stop {ev_desc}")
                else:
                    amt = sc.new_amount_home
                    descs.append(f"Reduce {ev_desc} to {ccy} {amt:,.2f}".rstrip("0").rstrip("."))
            change_desc = ", and ".join(descs)

        return (
            f"User Currency: {ccy}\n"
            f"Minimum Balance to Keep: {ccy} {min_keep:,.2f}\n"
            f"Requested Amount: {ccy} {req.requested_amount:,.2f}\n"
            f"Request Date: {req.request_date.isoformat()}\n"
            f"Desired Completion Date: {req.desired_completion_date.isoformat()}\n"
            f"Safe to Pay Today: {ccy} {safe_today:,.2f}\n"
            f"Earliest Date for Full Payment: {earliest_full_date or 'none'}\n"
            f"Affordability Status: {plan.affordability_status}\n"
            f"Recommended Method: {plan.payment_method}\n"
            f"Payment Schedule: {plan.format_plan_str()}\n"
            f"Spending Changes: {change_desc}\n"
        )

    def _template_fallback(
        self,
        req: Request,
        profile: FinancialProfile,
        plan: CandidatePlan,
        safe_today: float,
        earliest_full_date: Optional[str],
    ) -> str:
        ccy = profile.home_currency
        min_keep = profile.minimum_balance_to_keep

        def fmt(n: float) -> str:
            return f"{n:,.2f}".rstrip("0").rstrip(".") if n != int(n) else f"{int(n):,}"

        if plan.affordability_status == "affordable_now":
            return f"Pay {ccy} {fmt(req.requested_amount)} today. This leaves at least {ccy} {fmt(min_keep)} available over the next 90 days."

        if plan.payment_method == "wait":
            d_str = plan.start_date.strftime("%d %B %Y")
            return f"Pay {ccy} {fmt(req.requested_amount)} in full on {d_str}. Paying earlier would take the balance below the {ccy} {fmt(min_keep)} minimum."

        if plan.payment_method == "partial_payment":
            p1 = plan.payments[0]
            p2 = plan.payments[1]
            d2_str = p2[0].strftime("%d %B %Y")
            return f"Pay {ccy} {fmt(p1[1])} today and the remaining {ccy} {fmt(p2[1])} on {d2_str}. This completes the full request and keeps the {ccy} {fmt(min_keep)} minimum protected."

        if plan.payment_method == "installments":
            count = len(plan.payments)
            inst_amt = plan.payments[0][1]
            start_str = plan.start_date.strftime("%d %B %Y")
            return f"Use {count} installments of {ccy} {fmt(inst_amt)}, starting {start_str}. This leaves at least {ccy} {fmt(min_keep)} available."

        if plan.affordability_status == "not_affordable":
            d_str = req.desired_completion_date.strftime("%d %B %Y")
            if safe_today > 0:
                return f"Do not proceed with the {ccy} {fmt(req.requested_amount)} request. Although {ccy} {fmt(safe_today)} is available today, the full amount cannot be completed safely within 90 days."
            return f"Do not make this payment by {d_str}. None of the available options keeps the {ccy} {fmt(min_keep)} minimum protected."

        return f"Payment plan recommended. This keeps the {ccy} {fmt(min_keep)} minimum protected."
