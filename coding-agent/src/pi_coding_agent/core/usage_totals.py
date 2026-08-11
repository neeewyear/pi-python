from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi_ai.types import Usage

    from .session_manager import SessionEntry


@dataclass
class UsageTotals:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0


def create_usage_totals() -> UsageTotals:
    return UsageTotals()


def add_usage_to_totals(totals: UsageTotals, usage: Usage) -> None:
    totals.input += usage.input
    totals.output += usage.output
    totals.cache_read += usage.cache_read
    totals.cache_write += usage.cache_write
    totals.cost += usage.cost.total


@dataclass
class UsageCostBreakdownEntry:
    key: str = ""
    cost: float = 0.0
    tokens: int = 0


def get_usage_cost_breakdown(
    entries: list[SessionEntry],
) -> list[UsageCostBreakdownEntry]:
    """Group attributable assistant usage by model and all other usage into a separate bucket."""
    totals_by_key: dict[str, UsageTotals] = {}

    for entry in entries:
        key: str | None = None
        usage: Usage | None = None
        if entry.type == "message" and entry.message.role == "assistant":
            response_model = getattr(entry.message, "response_model", None)
            key = f"{entry.message.provider}/{response_model or entry.message.model}"
            usage = entry.message.usage
        elif (
            entry.type == "message"
            and entry.message.role == "toolResult"
            and entry.message.usage
        ):
            key = "Tools/summaries"
            usage = entry.message.usage
        elif entry.type in ("branch_summary", "compaction") and entry.usage:
            key = "Tools/summaries"
            usage = entry.usage

        if not key or not usage:
            continue

        totals = totals_by_key.get(key)
        if not totals:
            totals = create_usage_totals()
            totals_by_key[key] = totals
        add_usage_to_totals(totals, usage)

    result = [
        UsageCostBreakdownEntry(
            key=key,
            cost=totals.cost,
            tokens=totals.input
            + totals.output
            + totals.cache_read
            + totals.cache_write,
        )
        for key, totals in totals_by_key.items()
    ]
    result = [e for e in result if e.cost > 0 or e.tokens > 0]
    result.sort(key=lambda e: e.cost, reverse=True)
    return result
