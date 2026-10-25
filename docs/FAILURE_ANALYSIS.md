# Visually reviewed failure cases
Selected from 8-shot, seed 0, full memory; these 24 examples are illustrative, not a prevalence estimate.

- candle_1 (FP): Hotspots on intact candle surfaces and boundaries; normal wick/shape variation is a plausible trigger.
- candle_2 (FP): Several intact rims are highlighted, consistent with normal shape/illumination variation.
- candle_3 (FN): The small marked wick-region defect receives a local response, but image score remains below the normal-calibrated threshold.
- candle_4 (FN): The narrow marked rim defect is visible in the heatmap; the image score narrowly misses the decision threshold.
- capsules_1 (FP): Largest false hotspot lies near the right image boundary among differently oriented normal capsules.
- capsules_2 (FP): False response is concentrated near the left boundary of a normal multi-instance layout.
- capsules_3 (FN): Tiny marked capsule defect is weak at the 16x16 patch grid and does not trigger image-level rejection.
- capsules_4 (FN): Small lower-capsule annotation has some local response but insufficient image score.
- cashew_1 (FN): The small lower surface blemish is localized, but rejected by the conservative image threshold.
- cashew_2 (FN): Small left-edge annotation competes with broader normal surface responses.
- cashew_3 (FN): A localized left-side defect response remains below the image-level threshold.
- cashew_4 (FN): The lower brown blemish is visible and localized; image score still falls just below threshold.
- chewinggum_1 (FP): Strong responses extend across the upper textured background outside the normal product.
- chewinggum_2 (FP): A normal top product edge produces the dominant response.
- chewinggum_3 (FN): The missing lower-right corner is localized but the image-level score is below threshold.
- chewinggum_4 (FN): Two small edge defects produce separated hotspots, neither yielding sufficient image score.
- fryum_1 (FP): An intact upper rim is highlighted, plausibly reflecting normal texture or lighting variation.
- fryum_2 (FP): Broad false responses follow normal spokes and right rim.
- fryum_3 (FN): Several marked structural rim defects merge into a coarse response below the image threshold.
- fryum_4 (FN): A thin outer-boundary defect is weakly localized at coarse patch resolution.
- pipe_fryum_1 (FP): Normal surface variation creates a right-side hotspot just above the image threshold.
- pipe_fryum_2 (FP): A normal central texture produces a localized false hotspot.
- pipe_fryum_3 (FN): A thin left-edge defect is weak relative to central normal-texture responses; score narrowly misses threshold.
- pipe_fryum_4 (FN): Small pale surface mark is weakly localized and remains below image threshold.