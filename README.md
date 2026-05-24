# anthropic-batch-kit

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/anthropic-batch-kit.svg)](https://pypi.org/project/anthropic-batch-kit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Submit, poll, and retrieve Anthropic Message Batches with a progress callback and a cost tally.** Zero runtime deps; you supply the `anthropic` SDK.

```python
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
    on_progress=lambda p: print(f"{p.status} {p.succeeded}/{p.total}"),
)

print(f"cost: ${result.cost_usd:.4f}")
for cid, msg in result.results.items():
    handle(cid, msg)
```

## Why

The Anthropic Message Batches API is 50% cheaper than the synchronous API but the SDK leaves you to write all the polling, results-streaming, error bucketing, and cost math yourself. That code is the same in every project. This is that code, with the batch discount applied automatically and a clean `BatchResult` return type.

## Install

```bash
pip install anthropic-batch-kit
pip install "anthropic-batch-kit[anthropic]"   # if you don't have the SDK yet
```

## API

```python
runner = BatchRunner(
    client,
    prices=None,             # override the default price table
    batch_discount=0.50,     # set to 1.0 to disable the batch-tier discount
)

batch_id = runner.submit(requests)
progress = runner.progress(batch_id)   # BatchProgress(status, succeeded, errored, ...)
result   = runner.retrieve(batch_id)   # BatchResult(results, errors, cost_usd, ...)

# or do all three in one call
result = runner.run(
    requests,
    poll_interval=10.0,
    timeout_s=3600,
    on_progress=lambda p: ...,
)
```

`BatchResult.results` is `{custom_id: message}`; `BatchResult.errors` is `{custom_id: result_block}`. `BatchResult.cost_usd` already includes the 50% batch discount when the model is in the price table.

If a batch does not finish before `timeout_s`, `run` raises `BatchTimeoutError`. The exception carries `.batch_id` and `.last_progress` so you can decide whether to cancel, keep polling, or surface partial results.

## Companion libraries

- [`prompt-cache-warmer`](https://github.com/MukundaKatta/prompt-cache-warmer) — pre-warm the system prompt before submitting a batch.
- [`claude-cost`](https://github.com/MukundaKatta/claude-cost) — same price-table model, Rust.

## License

MIT
