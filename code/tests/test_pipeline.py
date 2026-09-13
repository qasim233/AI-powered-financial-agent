"""
Unit Tests for Buy or Wait? Decision Engine.

Tests components 1-9 using deterministic fixtures with zero LLM calls.
"""

from datetime import date, timedelta
import unittest

from currency.converter import CurrencyConverter
from forecast.safety_engine import SafetyEngine
from loaders.data_loader import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    PaymentOption,
    Request,
)
from optimizer.spending_optimizer import SpendingChange, SpendingChangeOptimizer
from output.validator import OutputAssembler, OutputRow
from ranking.plan_ranker import PlanRanker
from resolution.conflict_resolver import ConflictResolver
from state.recurrence import LedgerEntry, StateReconstructor


class TestDecisionEngine(unittest.TestCase):

    def setUp(self):
        # Sample exchange rates
        self.rates = [
            ExchangeRate(rate_date=date(2025, 1, 1), from_currency="USD", to_currency="EUR", rate=0.9),
            ExchangeRate(rate_date=date(2025, 1, 1), from_currency="EUR", to_currency="ZAR", rate=20.0),
        ]
        self.converter = CurrencyConverter(self.rates)
        self.safety_engine = SafetyEngine(forecast_days=90)
        self.optimizer = SpendingChangeOptimizer(self.safety_engine)
        self.output_assembler = OutputAssembler()

    def test_currency_conversion(self):
        # Direct
        amt_eur = self.converter.convert(100.0, "USD", "EUR", date(2025, 1, 1))
        self.assertAlmostEqual(amt_eur, 90.0)

        # Inverse
        amt_usd = self.converter.convert(90.0, "EUR", "USD", date(2025, 1, 1))
        self.assertAlmostEqual(amt_usd, 100.0)

        # Multi-hop USD -> EUR -> ZAR
        amt_zar = self.converter.convert(100.0, "USD", "ZAR", date(2025, 1, 1))
        self.assertAlmostEqual(amt_zar, 1800.0)

    def test_safety_engine_headroom(self):
        req = Request(
            request_id="req_test_01",
            user_id="user_test",
            request_date=date(2025, 1, 1),
            request_type="purchase",
            requested_amount=500.0,
            desired_completion_date=date(2025, 2, 1),
            allows_partial_payment=False,
            request_text="Buy item",
        )
        profile = FinancialProfile(
            user_id="user_test",
            home_currency="USD",
            current_available_balance=1000.0,
            minimum_balance_to_keep=200.0,
            financial_priorities=[],
            expense_categories_to_protect=[],
            expense_categories_user_is_willing_to_reduce=[],
            expense_categories_user_is_willing_to_stop=[],
            payment_methods_user_will_consider=["full_payment"],
            max_installment_months=None,
        )
        ledger = [
            LedgerEntry(
                entry_id="e1",
                event_id="ev1",
                entry_date=date(2025, 1, 15),
                direction="debit",
                category="utilities",
                description="Power bill",
                amount_home=300.0,
                flexibility="fixed",
                minimum_allowed_amount_home=None,
                is_projected=False,
            )
        ]
        res = self.safety_engine.evaluate_request(req, profile, ledger)
        # Min balance in window is 1000 - 300 = 700.
        # Headroom = 700 - 200 = 500.
        # amount_safe_to_pay = min(500, 500) = 500.
        self.assertEqual(res.amount_safe_to_pay, 500.0)
        self.assertEqual(res.earliest_date_for_full_payment, date(2025, 1, 1))

    def test_output_validation_contract(self):
        req = Request(
            request_id="req_test_02",
            user_id="user_test",
            request_date=date(2025, 1, 1),
            request_type="purchase",
            requested_amount=100.0,
            desired_completion_date=date(2025, 2, 1),
            allows_partial_payment=False,
            request_text="Buy item",
        )
        profile = FinancialProfile(
            user_id="user_test",
            home_currency="USD",
            current_available_balance=500.0,
            minimum_balance_to_keep=100.0,
            financial_priorities=[],
            expense_categories_to_protect=[],
            expense_categories_user_is_willing_to_reduce=[],
            expense_categories_user_is_willing_to_stop=[],
            payment_methods_user_will_consider=["full_payment"],
            max_installment_months=None,
        )
        from ranking.plan_ranker import CandidatePlan
        plan = CandidatePlan(
            payment_method="full_payment",
            payment_option_id=None,
            payments=[(date(2025, 1, 1), 100.0)],
            total_amount_paid=100.0,
            start_date=date(2025, 1, 1),
            completion_date=date(2025, 1, 1),
            completes_by_deadline=True,
            spending_changes=[],
            affordability_status="affordable_now",
        )
        row = self.output_assembler.assemble_and_validate_row(
            request=req,
            profile=profile,
            plan=plan,
            safe_today=100.0,
            earliest_full_date=date(2025, 1, 1),
            spending_changes_str="none",
            explanation="Pay USD 100 today.",
        )
        self.assertEqual(row.affordability_status, "affordable_now")
        self.assertEqual(row.recommended_payment_method, "full_payment")
        self.assertEqual(row.earliest_date_for_full_payment, "2025-01-01")


if __name__ == "__main__":
    unittest.main()
