# TurkForecast-FM-Chronos2-LoRA-v1

**Turkish-vertical LoRA fine-tune of [amazon/chronos-2](https://huggingface.co/amazon/chronos-2)** (120M T5 encoder, Apache-2.0). First publicly-released **Apache-2.0 Turkish-vertical time-series foundation model (TSFM)** fine-tune.

> **v1.0 released 2026-05-27.** Adapter weights, V4 router, and bucket calibrators are hosted on HuggingFace Hub:
> [`Verm1ion/turkforecast-fm-chronos2-lora-v1`](https://huggingface.co/Verm1ion/turkforecast-fm-chronos2-lora-v1).

## Distinction from prior Turkish ML work

TurkForecast-FM is the **first publicly-released Apache-2.0 Turkish-vertical time-series foundation model (TSFM) fine-tune**. We explicitly distinguish from three related but non-overlapping prior works:

1. **TabiBERT** (arXiv:2512.23065, Boğaziçi University / VNGRS-AI, December 2025) is a Turkish ModernBERT **language** encoder pretrained from scratch on text — different modality (NLP, not time series).
2. **Ertürk et al.** (MDPI Applied Sciences 16(6):2760, March 2026) train a supervised LSTM+BLSTM+GRU+Transformer hybrid from scratch on EPİAŞ electricity data — not a TSFM fine-tune (no pre-trained foundation backbone).
3. **NIEXCHE/chronos-t5-small-fine-tuned-v1** (Fevzi Kılaş, HuggingFace) is a Chronos-T5 fine-tune on a 15M-row multi-domain proprietary dataset — **not** claimed as Turkish-vertical.

*Özet (TR):* TurkForecast-FM, Türkçe dikey alana özgü, açık-kaynak (Apache-2.0) ilk TSFM ince-ayarıdır.

## GIFT-Eval Results

Evaluated on the full GIFT-Eval benchmark (97 canonical (dataset, term) entries across 7 domains × 3 forecast horizons):

| Metric | Mean | Median |
|--------|------|--------|
| MASE[0.5] | **1.1486** | 0.8550 |
| mean wQL | **0.2054** | 0.1091 |

### By forecast term

| Term | n | Mean MASE | Mean wQL |
|------|---|-----------|----------|
| short | 55 | 1.1881 | 0.1918 |
| medium | 21 | 1.0728 | 0.2201 |
| long | 21 | 1.1207 | 0.2262 |

### By domain

| Domain | Mean MASE | Mean wQL |
|--------|-----------|----------|
| Econ/Fin | 2.3419 | 0.0444 |
| Energy | 0.9657 | 0.1603 |
| Healthcare | 1.8100 | 0.0408 |
| Nature | 1.1891 | 0.2198 |
| Sales | 0.7856 | 0.5592 |
| Transport | 0.6820 | 0.1010 |
| Web/CloudOps | 1.3100 | 0.3638 |

Full per-dataset metrics: [`all_results.csv`](all_results.csv).

## Architecture

A **3-stage inference pipeline** (not a monolithic adapter):

1. **Base + LoRA adapter** — `amazon/chronos-2` + LoRA `r=32, α=64, dropout=0` on all-linear `target_modules`
2. **V4 router with WiSE-FT α-blend** ([`router_v5_with_alphas_and_calibration_v4.json`](router_v5_with_alphas_and_calibration_v4.json)) — per-(domain, freq) bucket selects FT vs ZS vs blend α ∈ {0, 0.25, 0.5, 0.75, 1.0}; **MASE-aware adoption rule**: `adopt = (wql_delta<0) OR (mase_delta<−3 AND wql_delta<5)` over the FT-only baseline
3. **Bucket isotonic calibrators** (downloaded from HF Hub) — per-bucket Kuleshov ([ICML 2018](https://proceedings.mlr.press/v80/kuleshov18a.html)) PIT-CDF isotonic regression on 9 quantile levels, with per-domain fallback for low-cardinality buckets

## Quick Start

```bash
pip install -r requirements.txt
python inference.py
```

See [`inference.py`](inference.py) for a minimal working example loading the adapter + V4 router + calibrators from HF Hub.

## Replication

End-to-end GIFT-Eval reproduction (load test windows → run inference per bucket → apply V4 router + bucket calibration → write 97-row submission CSV):

```bash
python replicate.py
```

Generates `all_results.csv` matching the released submission file row-for-row (deterministic seed=42, Tier 1.5 reproducibility stack).

**Tier 1.5 reproducibility settings** (locked in [`requirements.txt`](requirements.txt)):

- `torch==2.4.1+cu124`, `transformers==4.46.3`, `peft==0.13.2`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `seed=42`
- `torch.use_deterministic_algorithms(True)`
- SDP backend: math-only (`enable_flash_sdp(False) + enable_mem_efficient_sdp(False) + enable_math_sdp(True)`)
- `torch.backends.cuda.matmul.allow_tf32=False`, `cudnn.deterministic=True`

## Repository Layout

```
├── README.md                                       (this file)
├── LICENSE                                         (Apache-2.0)
├── requirements.txt                                (Path B locked stack)
├── inference.py                                    (minimal forecasting example)
├── replicate.py                                    (full GIFT-Eval reproduction)
├── all_results.csv                                 (97-row GIFT-Eval submission)
├── router_v5_with_alphas_and_calibration_v4.json   (V4 router with MASE-aware adoption)
└── bucket_adoption_summary_v4.jsonl                (per-bucket adoption decision matrix)
```

LoRA adapter weights and bucket isotonic calibrator pickle (3.7 MB) are hosted on HF Hub: [`Verm1ion/turkforecast-fm-chronos2-lora-v1`](https://huggingface.co/Verm1ion/turkforecast-fm-chronos2-lora-v1).

## License

[Apache-2.0](LICENSE) — matches `amazon/chronos-2` base license. No NC-restricted ancestor models in the training corpus.

## Citation

```bibtex
@misc{turkforecastfm2026,
  title  = {TurkForecast-FM-Chronos2-LoRA-v1: First Apache-2.0 Turkish-vertical Time-Series Foundation Model Fine-Tune},
  author = {Karatay, Mert},
  year   = {2026},
  url    = {https://huggingface.co/Verm1ion/turkforecast-fm-chronos2-lora-v1},
  note   = {GIFT-Eval submission, 2026-05-27}
}
```

## Acknowledgements

- **Amazon Science** for [Chronos-2](https://arxiv.org/abs/2510.15821) (base model + KernelSynth generator)
- **Salesforce AI Research** for [GIFT-Eval](https://arxiv.org/abs/2410.10393) (benchmark)
- **EPİAŞ** (Energy Exchange Istanbul) for public Turkish electricity market data used in out-of-time validation
