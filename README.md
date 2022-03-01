# ddp-amp-activation-checkpointing-memory-bench

Technique from 2022, reimplemented in 2026. This scaffold contains the package, CPU smoke test and CI harness for DDP, automatic mixed precision and activation checkpointing on a BERT-style encoder.

## Reproduce

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## License

MIT
