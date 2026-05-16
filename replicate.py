"""TurkForecast-FM replication entry point (placeholder for v1.0 release).

The v1.0 release will include a single-file end-to-end replication script:
1. Download GIFT-Eval benchmark (Salesforce/GiftEval, ~1.6 GB)
2. Load the fine-tuned LoRA adapter from HuggingFace Hub
3. Run the per-(domain, freq) router across all 55 short-horizon configurations
4. Produce the GIFT-Eval submission CSV (98 rows x 15 columns)

Intermediate artifacts (router decision matrix, per-bucket WiSE-FT alphas,
isotonic recalibration parameters) ship as JSON alongside this script.
"""

raise NotImplementedError(
    "v1.0 replication script ships on 2026-05-27. Track progress at "
    "https://github.com/Verm1lion/turkforecast-fm-chronos2-lora-v1"
)
