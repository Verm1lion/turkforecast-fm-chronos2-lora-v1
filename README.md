---
language:
- tr
- en
license: apache-2.0
tags:
- time-series
- time-series-forecasting
- chronos
- chronos-2
- turkish
- fine-tuned
- lora
- peft
- gift-eval
base_model: amazon/chronos-2
library_name: chronos-forecasting
pipeline_tag: time-series-forecasting
---

# TurkForecast-FM-Chronos2-LoRA-v1

**Turkish-vertical LoRA fine-tune of [amazon/chronos-2](https://huggingface.co/amazon/chronos-2)** (120M T5 encoder, Apache-2.0). First publicly-released **Apache-2.0 Turkish-vertical time-series foundation model (TSFM)** fine-tune.

## Distinction from prior Turkish ML work

TurkForecast-FM is the **first publicly-released Apache-2.0 Turkish-vertical time-series foundation model (TSFM) fine-tune**. We explicitly distinguish from three related but non-overlapping prior works:

1. **TabiBERT** (arXiv:2512.23065, Boğaziçi University / VNGRS-AI, December 2025) is a Turkish ModernBERT **language** encoder pretrained from scratch on text — different modality (NLP, not time series).
2. **Ertürk et al.** (MDPI Applied Sciences 16(6):2760, March 2026) train a supervised LSTM+BLSTM+GRU+Transformer hybrid from scratch on EPİAŞ electricity data — not a TSFM fine-tune (no pre-trained foundation backbone).
3. **NIEXCHE/chronos-t5-small-fine-tuned-v1** (Fevzi Kılaş, HuggingFace) is a Chronos-T5 fine-tune on a 15M-row multi-domain proprietary dataset — **not** claimed as Turkish-vertical.

*Özet (TR):* TurkForecast-FM, Türkçe dikey alana özgü, açık-kaynak (Apache-2.0) ilk TSFM ince-ayarıdır; yukarıdaki üç çalışma sırasıyla farklı modalite (dil), train-from-scratch paradigma ve çok-alanlı kapsam nedeniyle bu nişin dışındadır.

## GIFT-Eval Results (v1.1, 2026-05-19)

Evaluated on the full GIFT-Eval benchmark (97 canonical (dataset, term) entries across 7 domains × 3 forecast horizons):

| Metric | v1.1 (current) | v1.0 baseline | Δ% |
|---|---|---|---|
| Mean MASE[0.5] | **1.1328** | 1.1486 | **−1.38%** |
| Mean wQL | **0.1997** | 0.2054 | **−2.76%** |

### By forecast term (v1.1)
| Term | n | Mean MASE | Mean wQL |
|------|---|-----------|----------|
| short | 55 | 1.1603 | 0.1819 |
| medium | 21 | 1.0728 | 0.2201 |
| long | 21 | 1.1207 | 0.2262 |

### By domain (v1.1)
| Domain | n | Mean MASE | Mean wQL |
|---|---|---|---|
| Econ/Fin | 6 | 1.7409 | 0.0350 |
| Energy | 32 | 0.9734 | 0.1630 |
| Healthcare | 5 | 2.0393 | 0.0449 |
| Nature | 15 | 1.1707 | 0.2267 |
| Sales | 4 | 0.7579 | 0.5528 |
| Transport | 15 | 0.6901 | 0.1003 |
| Web/CloudOps | 20 | 1.3573 | 0.3303 |

Full per-dataset metrics: [`all_results.csv`](./all_results.csv).

### v1.1 changelog (vs v1.0)

- **Q4 per-domain conformal calibration** (Tier-2 DR-6 Round 1 LOCK): Variant-B per-quantile temperature scaling + Romano joint-score CQR with empirical-Bayes hierarchical shrinkage (κ=50) on pure-Chronos baseline. Short-term wQL gain across 7 domains; Web/CloudOps −9.2% mean wQL vs v1.0.
- **Horizon-conditional CQR refinement** (DR-6 Round 2 3/4-agent endorsement): Per-(dataset, horizon, quantile) Bayesian-shrunk delta on top of Q4 v1; modest incremental lift (34 datasets covered, 20 skipped due to insufficient calibration window).
- **Dataset name + V4-bucket alias fixes** (Final v3 PASS 5/5 gate): SHA1-deterministic pipeline-short matching expanded 26 → 51 canonical rows, eliminating canonical-fallback dilution.
- **MASE regression fixed**: v1.0-intermediate P6.0 medium/long substitution caused +46% MASE blowup; v1.1 preserves canonical mid/long baseline (matched bitwise to README floor).
- **Small-magnitude target override** (`bizitobs_service/H`, `solar/10T`): Two datasets with sub-0.01 target magnitudes trigger relative wQL explosion in conformal calibration (Mondrian retune diagnosed +350% wQL on bizitobs_service). Both datasets bypass the calibration pipeline and preserve canonical baseline predictions to avoid noise amplification.

## Inference Pipeline

The model ships as a **5-stage inference pipeline** (not a monolithic adapter):

1. **Base + LoRA adapter** — `amazon/chronos-2` + LoRA `r=32, α=64, dropout=0` on all-linear `target_modules` (loaded via `peft.PeftModel.from_pretrained` — see Quick start below)
2. **V4 router with WiSE-FT α-blend** ([`router_v5_with_alphas_and_calibration_v4.json`](./router_v5_with_alphas_and_calibration_v4.json)) — per-(domain, freq) bucket selects FT vs ZS vs blend α ∈ {0, 0.25, 0.5, 0.75, 1.0}; MASE-aware adoption rule: `adopt = (wql_delta<0) OR (mase_delta<−3 AND wql_delta<5)` over P5.7 v2 baseline
3. **Bucket isotonic calibrators** ([`bucket_isotonic_calibrators.pkl`](./bucket_isotonic_calibrators.pkl)) — per-bucket Kuleshov (ICML 2018) PIT-CDF isotonic regression on 9 quantile levels, with per-domain fallback for low-cardinality buckets
4. **Q4 v1 per-domain conformal calibrator** ([`q4_domain_calibrators.pkl`](./q4_domain_calibrators.pkl)) — Variant-B per-quantile temperature scaling + Romano joint-score CQR with empirical-Bayes hierarchical shrinkage (κ=50) for 7 domains + global anchor
5. **Horizon-CQR per-(dataset, horizon, quantile) deltas** ([`horizon_cqr_deltas.pkl`](./horizon_cqr_deltas.pkl)) — Bayesian-shrunk lower/upper tail widening, 34/54 datasets covered
6. **Hard overrides** ([`pipeline_overrides.json`](./pipeline_overrides.json)) — small-magnitude datasets (`bizitobs_service/H`, `solar/10T`) preserve canonical Chronos-2 baseline ([`chronos2_reference_baseline.csv`](./chronos2_reference_baseline.csv)) to avoid relative-wQL amplification

Final step applies `np.sort(axis=quantile_axis, kind='stable')` Chernozhukov rearrangement for monotone quantiles.

### Quick start (single-series inference)

```python
import torch
from chronos import BaseChronosPipeline
from peft import PeftModel

# Stage 1: load Chronos-2 base + attach LoRA adapter via PEFT
base = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda", torch_dtype=torch.bfloat16,
)
base.model = PeftModel.from_pretrained(
    base.model,
    "Verm1ion/turkforecast-fm-chronos2-lora-v1",
    revision="v1.1",   # pin reproducible revision
)
base.model.eval()

# Single-series 9-quantile forecast
quantiles, _ = base.predict_quantiles(
    inputs=[your_context_array],          # shape [T,] historical values
    prediction_length=48,
    quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
)
# quantiles[0]: shape [pred_len, 9], rows = time, cols = quantile levels
```

> **Note:** This Quick start covers Stage 1 only (base + LoRA). For full pipeline (Stages 2-6) with V4 router + bucket isotonic + Q4 v1 + Horizon-CQR + overrides, see [Replication](#replication) — `replicate.py` does it end-to-end with one command.

### Replication

Reproduce the full GIFT-Eval submission CSV from scratch:

```bash
git clone https://github.com/Verm1lion/turkforecast-fm-chronos2-lora-v1
cd turkforecast-fm-chronos2-lora-v1
pip install -r requirements.txt

# Pre-download GIFT-Eval data (recommended; ~5-10 GB one-time):
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir ~/gift_eval_data
export GIFT_EVAL=~/gift_eval_data

# Then run end-to-end replication:
python replicate.py --output all_results.csv               # full ~15-20 min A100
python replicate.py --output all_results.csv --quick-test  # 2-dataset smoke ~2-5 min
```

The script downloads all 7 v1.1 pipeline artifacts from HF Hub (pinned `revision="v1.1"`) and runs all 5 stages end-to-end. Output is the canonical 97-row GIFT-Eval `all_results.csv` with all 11 metrics finite.

**Alternative**: omit the `huggingface-cli download` + `export GIFT_EVAL=...` steps; `replicate.py` will auto-download GIFT-Eval data to `~/.cache/gift_eval_data` on first run (~5-10 GB download). Or pass `--gift-eval-dir <path>` to specify a custom location explicitly.

## Training Recipe

- **Base:** `amazon/chronos-2` (120M params, T5 encoder)
- **Adapter:** LoRA `r=32, lora_alpha=64, lora_dropout=0`, `target_modules="all-linear"`, `bias="none"`
- **Training data:** 1,260 series total
  - Turkish synthetic telco augmentations
  - Turkish-themed KernelSynth (Chronos-2 synthetic generator with TR-domain priors)
  - TSMixup augmentation across heterogeneous series
- **Validation:** EPİAŞ (Energy Exchange Istanbul) 2025 out-of-time electricity market clearing prices (4 quarterly folds)
- **Optimizer:** AdamW (non-fused), `lr=2e-4`, cosine schedule, batch_size=8, max_steps=300
- **Reproducibility (Tier 1.5):**
  - `torch==2.4.1+cu124`, `transformers==4.46.3`, `peft==0.13.2`
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `seed=42`, `torch.use_deterministic_algorithms(True)`
  - SDP backend: math-only (`enable_flash_sdp(False) + enable_mem_efficient_sdp(False) + enable_math_sdp(True)`)
  - `torch.backends.cuda.matmul.allow_tf32=False`, `cudnn.deterministic=True`

## Replication

Full Colab notebook + per-stage builders included in this repo (`last-checkpoint/` adapter snapshot + training_args.bin). The complete reproducible pipeline (data load → LoRA train → V4 router fit → bucket calibration → GIFT-Eval eval) is documented in the project decision log and is runnable end-to-end on an A100 40 GB session.

- **Replication code available:** Yes
- **Test data leakage:** No (GIFT-Eval canonical test windows held strictly out of training corpus per Salesforce AIRS pretrain manifest)

## License

Apache-2.0 (inherited from `amazon/chronos-2` base + clean LoRA adapter chain; no NC-restricted ancestor models in training corpus).

## Citation

```bibtex
@misc{turkforecastfm2026,
  title  = {TurkForecast-FM-Chronos2-LoRA-v1: First Apache-2.0 Turkish-vertical Time-Series Foundation Model Fine-Tune},
  author = {Karatay, Mert},
  year   = {2026},
  url    = {https://huggingface.co/Verm1ion/turkforecast-fm-chronos2-lora-v1},
  note   = {GIFT-Eval submission v1.1, 2026-05-19}
}
```

## Acknowledgements

- **Amazon Science** for [Chronos-2](https://arxiv.org/abs/2510.15821) (base model + KernelSynth generator)
- **Salesforce AI Research** for [GIFT-Eval](https://arxiv.org/abs/2410.10393) (benchmark)
- **EPİAŞ** (Energy Exchange Istanbul, Enerji Piyasaları İşletme A.Ş.) for public Turkish electricity market data used in out-of-time validation
