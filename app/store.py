"""Operational state (customers, orders, cases, returns, pickups, refunds, follow-ups, handoffs).

This stands in for Cosmic Mart's order-management / ticketing systems. It is a single
JSON file initialised from data/seed_data.json; swap this class for real API clients
to go to production.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

COLLECTIONS = ("cases", "returns", "pickups", "refunds", "follow_ups", "handoffs")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class StateStore:
    def __init__(self, state_path: Path, seed_path: Path):
        self.state_path = Path(state_path)
        self.seed_path = Path(seed_path)
        self._lock = threading.RLock()
        if self.state_path.exists():
            self._data = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self._data = self._from_seed()
            self._save()

    @contextmanager
    def transaction(self) -> Iterator[dict]:
        """Mutable access to the state. Changes are saved on success and rolled back on any exception."""
        with self._lock:
            backup = copy.deepcopy(self._data)
            try:
                yield self._data
            except BaseException:
                self._data = backup
                raise
            self._save()

    @contextmanager
    def read(self) -> Iterator[dict]:
        """Read-only access (returns a deep copy so callers cannot mutate state by accident)."""
        with self._lock:
            yield copy.deepcopy(self._data)

    def next_id(self, data: dict, prefix: str) -> str:
        counters = data.setdefault("counters", {})
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]:05d}"

    def reset(self) -> None:
        with self._lock:
            self._data = self._from_seed()
            self._save()

    # ------------------------------------------------------------------ internals
    def _from_seed(self) -> dict:
        seed = json.loads(self.seed_path.read_text(encoding="utf-8"))
        now = utcnow()
        orders = {}
        for raw in seed["orders"]:
            order = {k: v for k, v in raw.items() if not k.endswith("_days_ago")}
            for key in ("placed", "shipped", "delivered"):
                days = raw.get(f"{key}_days_ago")
                order[f"{key}_at"] = iso(now - timedelta(days=days)) if days is not None else None
            order["total"] = round(sum(i["unit_price"] * i["quantity"] for i in order["items"]), 2)
            for item in order["items"]:
                item.setdefault("returned_quantity", 0)
            orders[order["order_id"]] = order
        data = {
            "customers": {c["customer_id"]: c for c in seed["customers"]},
            "orders": orders,
            "counters": {},
        }
        for name in COLLECTIONS:
            data[name] = {}
        return data

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)
