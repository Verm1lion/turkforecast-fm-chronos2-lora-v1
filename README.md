# TurkForecast-FM

**Turkish-vertical time-series foundation model.** Fine-tune of [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) (120M T5 encoder) with LoRA on Turkish synthetic + real macroeconomic data.

> **Status:** Active development through 2026-05-27 — `v1.0` release on submission day.
> Pre-release builds tagged `v0.x-pre` track milestone progress.

## Goals

- **GIFT-Eval leaderboard** submission targeting Top-15 (stretch: Top-10) on the 55-config short-horizon benchmark
- **First-mover** Turkish-vertical TSFM with measurable transfer on real EPIAS electricity 2025 OOT holdout (−13.4% MASE vs Chronos-2 zero-shot)
- **Reproducible** Tier-1.5 deterministic training stack (math-SDP + adamw_torch + manual seeding) — no FlashAttention/tf32 leakage

## Roadmap

| Date | Milestone | Tag |
|---|---|---|
| 2026-05-17 | Release scaffold + WIP model card | `v0.1-pre` |
| ~2026-05-24 | Per-bucket WiSE-FT routing decision matrix (P5.5) | `v0.2-pre` |
| ~2026-05-26 | Final pipeline architecture + isotonic recalibration (P5.7) | `v0.3-pre` |
| **2026-05-27** | **v1.0 release** — LoRA adapter + clean inference + GIFT-Eval submission | `v1.0` |

## Architecture (planned for v1.0)

- **Base:** `amazon/chronos-2` (120M params, T5 encoder, Apache-2.0)
- **Adapter:** LoRA r=32 α=64 dropout=0 on 5 canonical paths (`self_attention.{q,k,v,o}` + `output_patch_embedding.output_layer`)
- **Per-(domain, freq) router:** 22 buckets across 7 domains × 5 freq classes; bucket-wise route to fine-tune vs zero-shot vs blended
- **WiSE-FT** ([Wortsman 2022](https://arxiv.org/abs/2109.01903)) per-bucket α-interpolation between FT and ZS weights
- **Isotonic recalibration** ([Kuleshov 2018](https://proceedings.mlr.press/v80/kuleshov18a.html)) for per-quantile monotonicity

## Locked dependency versions (Path B, reproducibility-critical)

See [`requirements.txt`](requirements.txt). Recommended:

```bash
pip install -r requirements.txt
```

## Usage (v1.0 placeholder)

```python
# Available on 2026-05-27 with v1.0 release.
# See inference.py for the planned API surface.
```

## License

[Apache-2.0](LICENSE) — matches Chronos-2 base license for derivative compatibility.

## Citation

Citation BibTeX will be available with the v1.0 release. Until then please reference the GIFT-Eval submission once published.

## Acknowledgements

- Amazon Science for [Chronos-2](https://arxiv.org/abs/2510.15821)
- Salesforce AI Research for [GIFT-Eval](https://arxiv.org/abs/2410.10393)
- EPIAS (Energy Exchange Istanbul) for the public Turkish electricity market data used for OOT holdout validation
