"""Tests for anthropic_batch_kit.BatchRunner."""

from __future__ import annotations

import pytest

from anthropic_batch_kit import (
    BATCH_DISCOUNT,
    BatchProgress,
    BatchResult,
    BatchRunner,
    BatchTimeoutError,
)


# ---- fake Anthropic shape -------------------------------------------------


class _FakeBatch:
    def __init__(self, id_: str, status: str, counts: dict[str, int]):
        self.id = id_
        self.processing_status = status
        self.request_counts = counts


class _FakeResult:
    def __init__(self, custom_id: str, result: dict):
        self.custom_id = custom_id
        self.result = result


class _FakeMessage:
    def __init__(self, model: str, in_tok: int, out_tok: int):
        self.model = model
        self.usage = {"input_tokens": in_tok, "output_tokens": out_tok}


class _FakeBatches:
    def __init__(self):
        self.created: list[list[dict]] = []
        self._progress: list[_FakeBatch] = []
        self._results: dict[str, list[_FakeResult]] = {}
        self._next_id = 0

    def create(self, *, requests):
        self.created.append(list(requests))
        bid = f"batch_{self._next_id:03}"
        self._next_id += 1
        # default progress: scripted via _progress queue if set, otherwise
        # a single "ended" entry with all-succeeded counts.
        return _FakeBatch(
            bid,
            "in_progress",
            {
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
                "processing": len(requests),
            },
        )

    def script_progress(self, *batches: _FakeBatch):
        self._progress = list(batches)

    def script_results(self, batch_id: str, rows: list[_FakeResult]):
        self._results[batch_id] = rows

    def retrieve(self, batch_id):
        if self._progress:
            b = self._progress.pop(0)
            b.id = batch_id
            return b
        # Default: finished
        return _FakeBatch(
            batch_id,
            "ended",
            {
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
                "processing": 0,
            },
        )

    def results(self, batch_id):
        return iter(self._results.get(batch_id, []))


class _FakeMessages:
    def __init__(self, batches: _FakeBatches):
        self.batches = batches


class _FakeClient:
    def __init__(self):
        self.batches = _FakeBatches()
        self.messages = _FakeMessages(self.batches)


# ---- submit / progress / retrieve ----------------------------------------


def test_submit_returns_batch_id_and_records_requests():
    client = _FakeClient()
    runner = BatchRunner(client)
    bid = runner.submit(
        [
            {
                "custom_id": "a",
                "params": {
                    "model": "claude-haiku-4-5",
                    "max_tokens": 1,
                    "messages": [],
                },
            },
        ]
    )
    assert bid == "batch_000"
    assert client.batches.created[0][0]["custom_id"] == "a"


def test_progress_reports_request_counts():
    client = _FakeClient()
    client.batches.script_progress(
        _FakeBatch(
            "batch_000",
            "in_progress",
            {
                "succeeded": 2,
                "errored": 1,
                "canceled": 0,
                "expired": 0,
                "processing": 7,
            },
        ),
    )
    runner = BatchRunner(client)
    prog = runner.progress("batch_000")
    assert isinstance(prog, BatchProgress)
    assert prog.succeeded == 2
    assert prog.errored == 1
    assert prog.processing == 7
    assert prog.total == 10


def test_retrieve_collects_succeeded_and_errored():
    client = _FakeClient()
    msg_ok = _FakeMessage("claude-haiku-4-5", 100, 50)
    client.batches.script_results(
        "batch_X",
        [
            _FakeResult("row-1", {"type": "succeeded", "message": msg_ok}),
            _FakeResult(
                "row-2", {"type": "errored", "error": {"type": "invalid_request_error"}}
            ),
        ],
    )
    runner = BatchRunner(client)
    result = runner.retrieve("batch_X")
    assert isinstance(result, BatchResult)
    assert list(result.results.keys()) == ["row-1"]
    assert list(result.errors.keys()) == ["row-2"]
    assert result.input_tokens == 100
    assert result.output_tokens == 50


def test_retrieve_cost_applies_batch_discount():
    client = _FakeClient()
    # haiku: $1.00 / $5.00 per 1M, batch discount 0.5
    # 1M in tokens => 0.50 USD; 1M out tokens => 2.50 USD; total 3.00
    msg = _FakeMessage("claude-haiku-4-5", 1_000_000, 1_000_000)
    client.batches.script_results(
        "b",
        [
            _FakeResult("r1", {"type": "succeeded", "message": msg}),
        ],
    )
    runner = BatchRunner(client)
    result = runner.retrieve("b")
    assert result.cost_usd is not None
    assert 2.99 <= result.cost_usd <= 3.01


def test_retrieve_cost_none_for_unknown_model():
    client = _FakeClient()
    msg = _FakeMessage("some-unknown", 100, 100)
    client.batches.script_results(
        "b",
        [
            _FakeResult("r1", {"type": "succeeded", "message": msg}),
        ],
    )
    runner = BatchRunner(client)
    result = runner.retrieve("b")
    assert result.cost_usd is None


def test_retrieve_cost_opus_pricing():
    client = _FakeClient()
    # opus: $5.00 / $25.00 per 1M, batch discount 0.5
    # 1M in tokens => 2.50 USD; 1M out tokens => 12.50 USD; total 15.00
    msg = _FakeMessage("claude-opus-4-8", 1_000_000, 1_000_000)
    client.batches.script_results(
        "b",
        [
            _FakeResult("r1", {"type": "succeeded", "message": msg}),
        ],
    )
    runner = BatchRunner(client)
    result = runner.retrieve("b")
    assert result.cost_usd is not None
    assert 14.99 <= result.cost_usd <= 15.01


# ---- run() end-to-end -----------------------------------------------------


def test_run_submits_polls_until_ended_then_retrieves():
    client = _FakeClient()
    client.batches.script_progress(
        _FakeBatch(
            "x",
            "in_progress",
            {
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
                "processing": 1,
            },
        ),
        _FakeBatch(
            "x",
            "ended",
            {
                "succeeded": 1,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
                "processing": 0,
            },
        ),
    )
    msg = _FakeMessage("claude-haiku-4-5", 10, 5)
    client.batches.script_results(
        "batch_000",
        [
            _FakeResult("row-1", {"type": "succeeded", "message": msg}),
        ],
    )

    seen: list[BatchProgress] = []
    runner = BatchRunner(client)
    result = runner.run(
        [
            {
                "custom_id": "row-1",
                "params": {
                    "model": "claude-haiku-4-5",
                    "max_tokens": 1,
                    "messages": [],
                },
            }
        ],
        poll_interval=0.0,
        on_progress=seen.append,
        _sleep=lambda _t: None,
    )
    assert result.status == "ended"
    assert "row-1" in result.results
    assert [p.status for p in seen] == ["in_progress", "ended"]


def test_run_times_out_if_never_ends():
    client = _FakeClient()
    client.batches.script_progress(
        *[
            _FakeBatch(
                "x",
                "in_progress",
                {
                    "succeeded": 0,
                    "errored": 0,
                    "canceled": 0,
                    "expired": 0,
                    "processing": 1,
                },
            )
            for _ in range(20)
        ]
    )
    runner = BatchRunner(client)
    fake_clock = iter([0.0, 0.0, 5.0, 11.0])

    with pytest.raises(BatchTimeoutError) as exc:
        runner.run(
            [
                {
                    "custom_id": "r",
                    "params": {
                        "model": "claude-haiku-4-5",
                        "max_tokens": 1,
                        "messages": [],
                    },
                }
            ],
            poll_interval=0.0,
            timeout_s=10.0,
            _sleep=lambda _t: None,
            _now=lambda: next(fake_clock),
        )
    assert exc.value.batch_id == "batch_000"
    assert exc.value.last_progress.status == "in_progress"


# ---- guards ---------------------------------------------------------------


def test_rejects_client_without_batches_surface():
    class _Bad:
        pass

    with pytest.raises(TypeError):
        BatchRunner(_Bad())


def test_batch_discount_constant_is_half():
    assert BATCH_DISCOUNT == 0.50


def test_custom_discount_overrides():
    client = _FakeClient()
    msg = _FakeMessage("claude-haiku-4-5", 1_000_000, 0)
    client.batches.script_results(
        "b",
        [
            _FakeResult("r", {"type": "succeeded", "message": msg}),
        ],
    )
    runner = BatchRunner(client, batch_discount=1.0)
    result = runner.retrieve("b")
    # 1M in tokens at $1.00/M with no discount = $1.00
    assert result.cost_usd is not None
    assert 0.99 <= result.cost_usd <= 1.01
