from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pi_agent.types import AssistantMessage

    from .session_manager import SessionEntry

CACHE_TTL_MS = 5 * 60 * 1000
NOISE_FLOOR_TOKENS = 1024


class CacheMiss:
    """A counted cache miss on a single assistant message."""

    def __init__(
        self,
        missed_tokens: int,
        missed_cost: float,
        idle_ms: int,
        model_changed: bool,
    ) -> None:
        self.missed_tokens = missed_tokens
        self.missed_cost = missed_cost
        self.idle_ms = idle_ms
        self.model_changed = model_changed


class CacheWasteTotals:
    def __init__(
        self,
        missed_tokens: int = 0,
        missed_cost: float = 0.0,
        miss_count: int = 0,
    ) -> None:
        self.missed_tokens = missed_tokens
        self.missed_cost = missed_cost
        self.miss_count = miss_count


class ModelPriceSource:
    """Minimal pricing lookup."""

    def get_model(self, provider: str, model_id: str) -> dict[str, Any] | None: ...


class PreviousRequest:
    """The last request seen by the scan."""

    def __init__(
        self,
        prompt_tokens: int,
        model_key: str,
        timestamp: int,
        reported_cache: bool,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.model_key = model_key
        self.timestamp = timestamp
        self.reported_cache = reported_cache


def _detect_miss(
    prev: PreviousRequest | None,
    message: AssistantMessage,
    models: ModelPriceSource,
) -> CacheMiss | None:
    usage = message.usage
    if usage is None:
        return None
    prompt_tokens = usage.input + usage.cache_read + usage.cache_write
    if (
        not prev
        or prompt_tokens <= 0
        or (usage.cache_read + usage.cache_write == 0 and not prev.reported_cache)
    ):
        return None

    missed_tokens = min(prev.prompt_tokens, prompt_tokens) - usage.cache_read
    if missed_tokens <= NOISE_FLOOR_TOKENS:
        return None

    paid_tokens = usage.input + usage.cache_write
    paid_per_token = (
        (usage.cost.input + usage.cost.cache_write) / paid_tokens
        if paid_tokens > 0
        else 0
    )
    if usage.cache_read > 0:
        read_per_token = usage.cost.cache_read / usage.cache_read
    else:
        model_data = models.get_model(message.provider, message.model)
        read_per_token = (
            (model_data.get("cost", {}).get("cache_read", 0) / 1_000_000)
            if model_data
            else 0
        )

    return CacheMiss(
        missed_tokens=missed_tokens,
        missed_cost=missed_tokens * max(0.0, paid_per_token - read_per_token),
        idle_ms=max(0, message.timestamp - prev.timestamp),
        model_changed=f"{message.provider}/{message.model}" != prev.model_key,
    )


def _as_previous_request(
    message: AssistantMessage, reported_cache: bool
) -> PreviousRequest | None:
    usage = message.usage
    if usage is None:
        return None
    prompt_tokens = usage.input + usage.cache_read + usage.cache_write
    if prompt_tokens <= 0:
        return None
    return PreviousRequest(
        prompt_tokens=prompt_tokens,
        model_key=f"{message.provider}/{message.model}",
        timestamp=message.timestamp,
        reported_cache=reported_cache or (usage.cache_read + usage.cache_write > 0),
    )


def _scan(
    entries: list[SessionEntry],
    models: ModelPriceSource,
) -> tuple[PreviousRequest | None, CacheWasteTotals, dict[AssistantMessage, CacheMiss]]:
    prev: PreviousRequest | None = None
    totals = CacheWasteTotals()
    misses: dict[AssistantMessage, CacheMiss] = {}

    for entry in entries:
        if entry.type in ("compaction", "branch_summary"):
            prev = None
            continue
        if entry.type == "message" and entry.message.role == "assistant":
            miss = _detect_miss(prev, entry.message, models)
            if miss:
                totals.missed_tokens += miss.missed_tokens
                totals.missed_cost += miss.missed_cost
                totals.miss_count += 1
                misses[entry.message] = miss
            prev = (
                _as_previous_request(
                    entry.message, prev.reported_cache if prev else False
                )
                or prev
            )

    return prev, totals, misses


def compute_cache_waste(
    entries: list[SessionEntry], models: ModelPriceSource
) -> CacheWasteTotals:
    """Cumulative cache waste across a session."""
    _, totals, _ = _scan(entries, models)
    return totals


def collect_cache_misses(
    entries: list[SessionEntry],
    models: ModelPriceSource,
) -> dict[AssistantMessage, CacheMiss]:
    """All counted cache misses across a session, keyed by the assistant message."""
    _, _, misses = _scan(entries, models)
    return misses


def detect_cache_miss(
    entries: list[SessionEntry],
    message: AssistantMessage,
    models: ModelPriceSource,
) -> CacheMiss | None:
    """Detect a cache miss on a just-completed assistant message."""
    prev, _, _ = _scan(entries, models)
    return _detect_miss(prev, message, models)
