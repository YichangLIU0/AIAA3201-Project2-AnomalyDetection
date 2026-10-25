# Reviewed coreset transition examples

Examples are the first category/filename for each transition at seed 0, not a random prevalence sample. Maps are divided by each method's normal-calibrated pixel threshold before display.
- FP fixed: normal candle image; random-bank response is strong around the lower-left candle's wick/surface. Coreset reduces this relative response; image score drops from 0.269/0.248 to 0.160/0.230 (score/threshold).
- FP introduced: normal candle image; both maps respond near upper-left and lower-right candle regions. The new image score barely exceeds its lower threshold: 0.247/0.248 becomes 0.231/0.230. Lower raw score can still introduce a false alarm because the operating threshold changes.
- FN fixed: annotated lower-left candle defect. Both maps localize a response, but score/threshold changes from 0.235/0.248 to 0.238/0.230, changing OK to NG.
- FN introduced: annotated upper-left candle defect still produces a visible local response; 0.258/0.248 becomes 0.228/0.230, changing NG to OK. A visible hotspot alone does not establish a correct image decision.

These observations describe displayed evidence, not proof that a particular patch selection caused a semantic failure mechanism.
