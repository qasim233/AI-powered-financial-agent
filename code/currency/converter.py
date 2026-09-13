"""
Currency Converter — BFS-based multi-hop conversion using exchange_rates.csv.

Per SOLUTION.md §4 (Component 1):

- Per-date currency graph with nodes = currencies, edges = rate pairs plus
  implied inverse edges (``1/rate``).
- ``from_currency == to_currency`` → return amount unchanged.
- BFS shortest path (by hop count) from source to target currency.
- **Date fallback**: if no complete path on the exact date, search outward
  by absolute day difference (no distance cap) for the nearest date with a
  complete path.  Every fallback is logged for eval-harness visibility.
- **Hard failure**: if no path exists on *any* date, raise ``ValueError``.

Actual dataset observations
---------------------------
Pairs in ``exchange_rates.csv``:
  EUR→USD, EUR→ZAR, USD→EUR, USD→IDR, USD→INR

With inverse edges the graph can reach e.g. ZAR→EUR→USD→INR via BFS.
39 distinct dates cover 2023-10-15 to 2026-11-15.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CurrencyConverter:
    """
    BFS-based currency converter.

    Initialised with a list of :class:`ExchangeRate` records (from
    :class:`DataLoader`).  Call :py:meth:`convert` to convert an amount
    between any two currencies on a given date.
    """

    def __init__(self, exchange_rates: List[Any]) -> None:
        # Per-date adjacency list:  date → { from_ccy → { to_ccy: rate } }
        self._graphs: Dict[date, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._all_dates: List[date] = []
        self._fallback_log: List[Dict[str, str]] = []

        for er in exchange_rates:
            d = er.rate_date
            # Forward edge
            self._graphs[d][er.from_currency][er.to_currency] = er.rate
            # Implied inverse edge
            if er.rate != 0:
                self._graphs[d][er.to_currency][er.from_currency] = 1.0 / er.rate

        # Pre-sort dates for the fallback search
        self._all_dates = sorted(self._graphs.keys())

        logger.info(
            "CurrencyConverter: %d dates, %d rate entries",
            len(self._all_dates),
            len(exchange_rates),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        on_date: date,
    ) -> float:
        """
        Convert *amount* from *from_currency* to *to_currency* on *on_date*.

        Returns the converted amount.

        Raises
        ------
        ValueError
            If no conversion path exists on any date in the exchange-rate file.
        """
        if from_currency == to_currency:
            return amount

        # 1. Try exact date
        rate = self._bfs_rate(from_currency, to_currency, on_date)
        if rate is not None:
            return amount * rate

        # 2. Date fallback — nearest date with a complete path
        rate, fallback_date = self._find_nearest_date_rate(
            from_currency, to_currency, on_date
        )
        if rate is not None and fallback_date is not None:
            self._fallback_log.append(
                {
                    "requested_date": on_date.isoformat(),
                    "used_date": fallback_date.isoformat(),
                    "from": from_currency,
                    "to": to_currency,
                }
            )
            logger.warning(
                "Currency fallback: %s→%s on %s — used rate from %s",
                from_currency,
                to_currency,
                on_date,
                fallback_date,
            )
            return amount * rate

        # 3. Hard failure
        raise ValueError(
            f"No currency conversion path found for "
            f"{from_currency}→{to_currency} on any date in the exchange-rate file."
        )

    # ------------------------------------------------------------------
    # BFS implementation
    # ------------------------------------------------------------------

    def _bfs_rate(
        self,
        from_currency: str,
        to_currency: str,
        on_date: date,
    ) -> Optional[float]:
        """
        BFS shortest-path (by hop count) on the currency graph for *on_date*.

        Returns the cumulative conversion rate, or ``None`` if no path.
        """
        graph = self._graphs.get(on_date)
        if graph is None:
            return None

        # from_currency might not appear as a source in any edge
        if from_currency not in graph:
            return None

        visited = {from_currency}
        queue: deque[Tuple[str, float]] = deque()
        queue.append((from_currency, 1.0))

        while queue:
            current, cumulative_rate = queue.popleft()
            if current == to_currency:
                return cumulative_rate

            for neighbour, edge_rate in graph.get(current, {}).items():
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, cumulative_rate * edge_rate))

        return None

    def _find_nearest_date_rate(
        self,
        from_currency: str,
        to_currency: str,
        target_date: date,
    ) -> Tuple[Optional[float], Optional[date]]:
        """
        Search all available dates, closest-first, for one that has a
        BFS path between the requested currencies.
        """
        if not self._all_dates:
            return None, None

        # Sort candidates by absolute distance from target_date
        candidates = sorted(
            self._all_dates,
            key=lambda d: abs((d - target_date).days),
        )

        for candidate in candidates:
            if candidate == target_date:
                continue  # already tried in convert()
            rate = self._bfs_rate(from_currency, to_currency, candidate)
            if rate is not None:
                return rate, candidate

        return None, None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def fallback_log(self) -> List[Dict[str, str]]:
        """All date-fallback conversions — for eval-harness visibility."""
        return list(self._fallback_log)

    @property
    def available_dates(self) -> List[date]:
        """Sorted list of dates present in the exchange-rate file."""
        return list(self._all_dates)
