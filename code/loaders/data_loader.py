"""
Data Loader — Load all dataset CSVs into typed records keyed by primary ID.

Per SOLUTION.md §4 (Component 1): load each CSV into a dict keyed by its
natural ID (``event_id``, ``request_id``, ``user_id``, ``message_id``,
``image_id``, ``payment_option_id``).  Also builds secondary indices
(by ``user_id``, by ``request_id``, by ``related_event_id``) for efficient
downstream lookups.

Actual field values discovered in the dataset
----------------------------------------------
- **status**: settled, pending, scheduled, cancelled, failed, unrealized
- **flexibility**: fixed, reducible, reducible_or_stoppable, stoppable
- **direction**: credit, debit, non_cash
- **event_type**: debt_payment, expense, income, investment_purchase,
  investment_sale, investment_valuation, refund, subscription
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> Optional[date]:
    """Parse ``YYYY-MM-DD`` string to ``date``, or ``None`` if blank."""
    s = s.strip() if s else ""
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_float(s: str) -> Optional[float]:
    """Parse a numeric string to ``float``, or ``None`` if blank."""
    s = s.strip() if s else ""
    if not s:
        return None
    return float(s)


def _parse_int(s: str) -> Optional[int]:
    """Parse a numeric string to ``int``, or ``None`` if blank."""
    s = s.strip() if s else ""
    if not s:
        return None
    return int(float(s))  # handles "3.0" etc.


def _parse_bool(s: str) -> bool:
    """Parse boolean-ish string (``true``/``false``)."""
    return s.strip().lower() in ("true", "yes", "1") if s else False


def _parse_pipe_list(s: str) -> List[str]:
    """Parse a ``|``-delimited string into a list of stripped tokens."""
    if not s or not s.strip():
        return []
    return [item.strip() for item in s.split("|") if item.strip()]


def _opt_str(s: str) -> Optional[str]:
    """Return stripped string or ``None`` if blank."""
    s = s.strip() if s else ""
    return s if s else None


# ---------------------------------------------------------------------------
# Data models (dataclasses, one per CSV)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: List[str]
    expense_categories_to_protect: List[str]
    expense_categories_user_is_willing_to_reduce: List[str]
    expense_categories_user_is_willing_to_stop: List[str]
    payment_methods_user_will_consider: List[str]
    max_installment_months: Optional[int]


@dataclass(slots=True)
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str        # "debit", "credit", or "non_cash"
    amount: Optional[float]
    currency: str
    event_date: Optional[date]
    settlement_date: Optional[date]
    status: str           # settled / pending / scheduled / cancelled / failed / unrealized
    linked_event_id: Optional[str]
    flexibility: str      # fixed / reducible / reducible_or_stoppable / stoppable
    minimum_allowed_amount: Optional[float]


@dataclass(slots=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(slots=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str      # full_payment / partial_payment / installments
    payment_amount: float    # per-instalment amount
    number_of_payments: int
    first_payment_date: Optional[date]
    payment_frequency_days: Optional[int]
    financing_fee: float
    total_payable_amount: float


@dataclass(slots=True)
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: str              # ISO-8601 timestamp string
    source_type: str          # bank / employer / merchant / financial_service / service_provider
    message_text: str


@dataclass(slots=True)
class ImageRecord:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]


@dataclass(slots=True)
class ExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: float


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Load all dataset CSVs into typed records keyed by primary ID.

    After calling :py:meth:`load_all`, the following dicts and indices are
    populated for downstream components:

    Primary indices (keyed by natural ID):
        ``profiles``, ``events``, ``requests``, ``payment_options``,
        ``messages``, ``images``

    Secondary indices (for efficient per-user / per-request lookups):
        ``events_by_user``, ``messages_by_user``, ``images_by_user``,
        ``images_by_event``, ``payment_options_by_request``,
        ``messages_by_request``
    """

    def __init__(self, dataset_dir: str, requests_filename: str = "requests.csv") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.requests_filename = requests_filename

        # Primary indices ─ keyed by natural ID
        self.profiles: Dict[str, FinancialProfile] = {}
        self.events: Dict[str, FinancialEvent] = {}
        self.requests: Dict[str, Request] = {}
        self.payment_options: Dict[str, PaymentOption] = {}
        self.messages: Dict[str, Message] = {}
        self.images: Dict[str, ImageRecord] = {}
        self.exchange_rates: List[ExchangeRate] = []

        # Secondary indices
        self.events_by_user: Dict[str, List[FinancialEvent]] = {}
        self.messages_by_user: Dict[str, List[Message]] = {}
        self.messages_by_request: Dict[str, List[Message]] = {}
        self.images_by_user: Dict[str, List[ImageRecord]] = {}
        self.images_by_event: Dict[str, List[ImageRecord]] = {}
        self.payment_options_by_request: Dict[str, List[PaymentOption]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load every dataset CSV and build all indices."""
        self._load_profiles()
        self._load_events()
        self._load_requests()
        self._load_payment_options()
        self._load_messages()
        self._load_images()
        self._load_exchange_rates()
        logger.info(
            "DataLoader complete — %d profiles, %d events, %d requests, "
            "%d payment_options, %d messages, %d images, %d exchange_rates",
            len(self.profiles),
            len(self.events),
            len(self.requests),
            len(self.payment_options),
            len(self.messages),
            len(self.images),
            len(self.exchange_rates),
        )

    def get_image_path(self, image_id: str) -> Path:
        """Resolve an image_id to its PNG file path under ``dataset/media/images/``."""
        return self.dataset_dir / "media" / "images" / f"{image_id}.png"

    # ------------------------------------------------------------------
    # CSV reading helper
    # ------------------------------------------------------------------

    def _read_csv(self, filename: str) -> List[Dict[str, str]]:
        """Read a CSV file and return a list of row dicts."""
        return self._read_csv_path(self.dataset_dir / filename)

    def _read_csv_path(self, path: Path) -> List[Dict[str, str]]:
        """Read a CSV path and return a list of row dicts."""
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    # ------------------------------------------------------------------
    # Per-file loaders
    # ------------------------------------------------------------------

    def _load_profiles(self) -> None:
        for row in self._read_csv("financial_profiles.csv"):
            uid = row["user_id"].strip()
            self.profiles[uid] = FinancialProfile(
                user_id=uid,
                home_currency=row["home_currency"].strip(),
                current_available_balance=float(row["current_available_balance"]),
                minimum_balance_to_keep=float(row["minimum_balance_to_keep"]),
                financial_priorities=_parse_pipe_list(
                    row.get("financial_priorities", "")
                ),
                expense_categories_to_protect=_parse_pipe_list(
                    row.get("expense_categories_to_protect", "")
                ),
                expense_categories_user_is_willing_to_reduce=_parse_pipe_list(
                    row.get("expense_categories_user_is_willing_to_reduce", "")
                ),
                expense_categories_user_is_willing_to_stop=_parse_pipe_list(
                    row.get("expense_categories_user_is_willing_to_stop", "")
                ),
                payment_methods_user_will_consider=_parse_pipe_list(
                    row.get("payment_methods_user_will_consider", "")
                ),
                max_installment_months=_parse_int(
                    row.get("max_installment_months", "")
                ),
            )
        logger.info("Loaded %d financial profiles", len(self.profiles))

    def _load_events(self) -> None:
        for row in self._read_csv("financial_events.csv"):
            eid = row["event_id"].strip()
            uid = row["user_id"].strip()
            event = FinancialEvent(
                event_id=eid,
                user_id=uid,
                event_type=row.get("event_type", "").strip(),
                description=row.get("description", "").strip(),
                category=row.get("category", "").strip(),
                direction=row.get("direction", "").strip(),
                amount=_parse_float(row.get("amount", "")),
                currency=row.get("currency", "").strip(),
                event_date=_parse_date(row.get("event_date", "")),
                settlement_date=_parse_date(row.get("settlement_date", "")),
                status=row.get("status", "").strip(),
                linked_event_id=_opt_str(row.get("linked_event_id", "")),
                flexibility=row.get("flexibility", "").strip(),
                minimum_allowed_amount=_parse_float(
                    row.get("minimum_allowed_amount", "")
                ),
            )
            self.events[eid] = event
            self.events_by_user.setdefault(uid, []).append(event)
        logger.info("Loaded %d financial events", len(self.events))

    def _load_requests(self) -> None:
        path = Path(self.requests_filename)
        if not path.is_file():
            path = self.dataset_dir / self.requests_filename
        for row in self._read_csv_path(path):
            rid = row["request_id"].strip()
            self.requests[rid] = Request(
                request_id=rid,
                user_id=row["user_id"].strip(),
                request_date=_parse_date(row["request_date"]),
                request_type=row.get("request_type", "").strip(),
                requested_amount=float(row["requested_amount"]),
                desired_completion_date=_parse_date(row["desired_completion_date"]),
                allows_partial_payment=_parse_bool(
                    row.get("allows_partial_payment", "false")
                ),
                request_text=row.get("request_text", "").strip(),
            )
        logger.info("Loaded %d requests", len(self.requests))

    def _load_payment_options(self) -> None:
        for row in self._read_csv("request_payment_options.csv"):
            pid = row["payment_option_id"].strip()
            rid = row["request_id"].strip()
            opt = PaymentOption(
                payment_option_id=pid,
                request_id=rid,
                payment_method=row.get("payment_method", "").strip(),
                payment_amount=float(row.get("payment_amount", "0")),
                number_of_payments=_parse_int(row.get("number_of_payments", "1"))
                or 1,
                first_payment_date=_parse_date(
                    row.get("first_payment_date", "")
                ),
                payment_frequency_days=_parse_int(
                    row.get("payment_frequency_days", "")
                ),
                financing_fee=float(row.get("financing_fee", "0")),
                total_payable_amount=float(
                    row.get("total_payable_amount", "0")
                ),
            )
            self.payment_options[pid] = opt
            self.payment_options_by_request.setdefault(rid, []).append(opt)
        logger.info("Loaded %d payment options", len(self.payment_options))

    def _load_messages(self) -> None:
        for row in self._read_csv("messages.csv"):
            mid = row["message_id"].strip()
            uid = row["user_id"].strip()
            rid = _opt_str(row.get("request_id", ""))
            msg = Message(
                message_id=mid,
                user_id=uid,
                request_id=rid,
                related_event_id=_opt_str(row.get("related_event_id", "")),
                sent_at=row.get("sent_at", "").strip(),
                source_type=row.get("source_type", "").strip(),
                message_text=row.get("message_text", "").strip(),
            )
            self.messages[mid] = msg
            self.messages_by_user.setdefault(uid, []).append(msg)
            if rid:
                self.messages_by_request.setdefault(rid, []).append(msg)
        logger.info("Loaded %d messages", len(self.messages))

    def _load_images(self) -> None:
        for row in self._read_csv("images.csv"):
            iid = row["image_id"].strip()
            uid = row["user_id"].strip()
            eid = _opt_str(row.get("related_event_id", ""))
            img = ImageRecord(
                image_id=iid,
                user_id=uid,
                request_id=_opt_str(row.get("request_id", "")),
                related_event_id=eid,
            )
            self.images[iid] = img
            self.images_by_user.setdefault(uid, []).append(img)
            if eid:
                self.images_by_event.setdefault(eid, []).append(img)
        logger.info("Loaded %d images", len(self.images))

    def _load_exchange_rates(self) -> None:
        for row in self._read_csv("exchange_rates.csv"):
            self.exchange_rates.append(
                ExchangeRate(
                    rate_date=_parse_date(row["rate_date"]),
                    from_currency=row["from_currency"].strip(),
                    to_currency=row["to_currency"].strip(),
                    rate=float(row["rate"]),
                )
            )
        logger.info("Loaded %d exchange rates", len(self.exchange_rates))
