"""TurkForecast-FM inference API (placeholder for v1.0 release).

Final API will follow the Chronos-2 pipeline pattern:

    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(
        "Verm1ion/turkforecast-fm-chronos2-lora-v1",
        device_map="cuda", torch_dtype="bfloat16",
    )
    pipe.model.eval()
    quantiles, _ = pipe.predict_quantiles(
        inputs=[context_array],
        prediction_length=48,
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )

The v1.0 release on 2026-05-27 will include the per-(domain, freq) router that
selects between fine-tuned, zero-shot, or blended forecasts based on the input
characteristics. See README.md roadmap.
"""

raise NotImplementedError(
    "v1.0 inference API ships on 2026-05-27. Track progress at "
    "https://github.com/Verm1lion/turkforecast-fm-chronos2-lora-v1"
)
