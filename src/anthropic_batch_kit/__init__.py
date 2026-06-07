"""anthropic-batch-kit - submit + poll + retrieve Anthropic Message Batches.

The official SDK exposes the Message Batches API, but you still need the
boilerplate to turn that into a "give me a list of dicts, get back a list
of results when they're done" flow with a progress bar and a cost tally.
This is that boilerplate.

    from anthropic_batch_kit import BatchRunner
    import anthropic

    runner = BatchRunner(anthropic.Anthropic())

    requests = [
        {"custom_id": f"row-{i}", "params": {
            "model": "claude-haiku-4-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": row.text}],
        }}
        for i, row in enumerate(rows)
    ]

    result = runner.run(
        requests,
        poll_interval=10.0,
        on_progress=lambda r: print(r.status, r.succeeded, "/", r.total),
    )

    for custom_id, msg in result.results.items():
        ...

The batch API price tier is 50% off the synchronous rates, so `BatchResult`
applies that multiplier when it computes `cost_usd`. Override the
`prices` dict or pass `batch_discount=1.0` if you don't want the discount.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import sleep, perf_counter
from typing import Any, Callable, Iterable

__version__ = "0.1.0"
__all__ = [
    "BatchRunner",
    "BatchProgress",
    "BatchResult",
    "BatchTimeoutError",
    "PriceTable",
    "DEFAULT_PRICES",
    "BATCH_DISCOUNT",
]


# ---- pricing ---------------------------------------------------------------


PriceTable = dict[str, dict[str, float]]


DEFAULT_PRICES: PriceTable = {
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}

BATCH_DISCOUNT = 0.50  # 50% off the synchronous rates


# ---- result types ----------------------------------------------------------


@dataclass(frozen=True)
class BatchProgress:
    batch_id: str
    status: str  # in_progress / canceling / ended
    succeeded: int
    errored: int
    canceled: int
    expired: int
    processing: int
    total: int


@dataclass
class BatchResult:
    batch_id: str
    status: str
    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    duration_s: float = 0.0


class BatchTimeoutError(TimeoutError):
    """Raised when a poll loop exceeds the configured timeout."""

    def __init__(self, batch_id: str, last_progress: BatchProgress):
        self.batch_id = batch_id
        self.last_progress = last_progress
        super().__init__(f"batch {batch_id} did not finish before timeout")


# ---- accessors -------------------------------------------------------------


def _g(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _counts(batch: Any) -> dict[str, int]:
    rc = _g(batch, "request_counts") or {}
    return {
        "succeeded": int(_g(rc, "succeeded", 0)),
        "errored": int(_g(rc, "errored", 0)),
        "canceled": int(_g(rc, "canceled", 0)),
        "expired": int(_g(rc, "expired", 0)),
        "processing": int(_g(rc, "processing", 0)),
    }


# ---- runner ----------------------------------------------------------------


class BatchRunner:
    """Submit a list of Message Batch requests, poll, return results.

    Args:
        client: an `anthropic.Anthropic()` instance, or any object that exposes
            `.messages.batches` with `.create`, `.retrieve`, and `.results`.
            For tests, see `tests/test_batch.py` for the minimal fake shape.
        prices: optional price table override.
        batch_discount: multiplier applied to per-token rates. Default 0.50.
    """

    def __init__(
        self,
        client: Any,
        *,
        prices: PriceTable | None = None,
        batch_discount: float = BATCH_DISCOUNT,
    ) -> None:
        batches = _g(_g(client, "messages"), "batches")
        if batches is None:
            raise TypeError(
                "BatchRunner requires a client with .messages.batches "
                "(create/retrieve/results); got %r" % (client,)
            )
        self._batches = batches
        self._prices: PriceTable = dict(prices) if prices else dict(DEFAULT_PRICES)
        self._discount = float(batch_discount)

    # ---- submission --------------------------------------------------

    def submit(self, requests: Iterable[dict]) -> str:
        """Submit a batch; return its batch_id."""
        batch = self._batches.create(requests=list(requests))
        return _g(batch, "id")

    # ---- progress / polling -----------------------------------------

    def progress(self, batch_id: str) -> BatchProgress:
        batch = self._batches.retrieve(batch_id)
        c = _counts(batch)
        total = sum(c.values())
        return BatchProgress(
            batch_id=batch_id,
            status=str(_g(batch, "processing_status", _g(batch, "status", "unknown"))),
            succeeded=c["succeeded"],
            errored=c["errored"],
            canceled=c["canceled"],
            expired=c["expired"],
            processing=c["processing"],
            total=total,
        )

    # ---- retrieval ---------------------------------------------------

    def retrieve(self, batch_id: str) -> BatchResult:
        """Pull all results for a finished batch and tally cost."""
        results: dict[str, Any] = {}
        errors: dict[str, Any] = {}
        in_tok = 0
        out_tok = 0

        model_for_cost: str | None = None

        for row in self._batches.results(batch_id):
            cid = _g(row, "custom_id")
            r = _g(row, "result")
            rtype = _g(r, "type")
            if rtype == "succeeded":
                msg = _g(r, "message")
                results[cid] = msg
                usage = _g(msg, "usage")
                in_tok += int(_g(usage, "input_tokens", 0))
                out_tok += int(_g(usage, "output_tokens", 0))
                if model_for_cost is None:
                    model_for_cost = _g(msg, "model")
            else:
                errors[cid] = r

        cost = self._estimate_cost(model_for_cost, in_tok, out_tok)
        return BatchResult(
            batch_id=batch_id,
            status="ended",
            results=results,
            errors=errors,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )

    def _estimate_cost(
        self, model: str | None, in_tok: int, out_tok: int
    ) -> float | None:
        if model is None:
            return None
        price = self._prices.get(model)
        if price is None:
            return None
        per_in = price["input"] / 1_000_000 * self._discount
        per_out = price["output"] / 1_000_000 * self._discount
        return in_tok * per_in + out_tok * per_out

    # ---- full pipeline ----------------------------------------------

    def run(
        self,
        requests: Iterable[dict],
        *,
        poll_interval: float = 10.0,
        timeout_s: float | None = None,
        on_progress: Callable[[BatchProgress], None] | None = None,
        _sleep: Callable[[float], None] = sleep,
        _now: Callable[[], float] = perf_counter,
    ) -> BatchResult:
        """Submit + poll + retrieve in one call.

        Returns a `BatchResult`. Raises `BatchTimeoutError` if the batch
        does not finish within `timeout_s` seconds.
        """
        batch_id = self.submit(requests)
        started = _now()

        while True:
            prog = self.progress(batch_id)
            if on_progress:
                on_progress(prog)
            if prog.status == "ended":
                break
            if timeout_s is not None and (_now() - started) >= timeout_s:
                raise BatchTimeoutError(batch_id, prog)
            _sleep(poll_interval)

        result = self.retrieve(batch_id)
        result.duration_s = _now() - started
        return result
