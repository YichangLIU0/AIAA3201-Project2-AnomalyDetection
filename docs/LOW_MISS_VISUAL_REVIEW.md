# High-resolution additional-category failure review

Twelve seed-0 errors were visually inspected from three contact sheets. Selection is first FP and first FN by filename per category; these are illustrative, not prevalence estimates. Facts below describe visible maps and annotations. Pose sensitivity, foreground masking and representation limitations are hypotheses, not proven causes.

- macaroni1 FP (macaroni1\test\good\0688.JPG): Response near the lower edge of the upper-left normal piece; GT is empty.
- macaroni1 FN (macaroni1\test\bad\000.JPG): Tiny annotation on the lower-right piece; diffuse object responses do not cross the image threshold.
- macaroni2 FP (macaroni2\test\good\0107.JPG): Bright response at a normal piece tip in a rotated arrangement; GT is empty.
- macaroni2 FN (macaroni2\test\bad\003.JPG): Tiny annotated region on the right-hand piece; the map is broadly responsive over the pieces rather than sharply isolated at that region.
- pcb1 FP (pcb1\test\good\0159.JPG): Response at the left board edge; the image score is only slightly above the normal-calibrated threshold.
- pcb1 FN (pcb1\test\bad\000.JPG): Thin annotated pin below the board has weak response; board responses remain below the image threshold.
- pcb2 FP (pcb2\test\good\0011.JPG): Strong response at the right board edge on an image with empty GT.
- pcb2 FN (pcb2\test\bad\005.JPG): Two thin upper-pin annotations have some response, but the image score remains below threshold.
- pcb3 FP (pcb3\test\good\0046.JPG): Hotspots on the right-side normal components; GT is empty.
- pcb3 FN (pcb3\test\bad\002.JPG): Thin left-pin annotation has weak local response and is missed at the calibrated threshold.
- pcb4 FP (pcb4\test\good\0552.JPG): Very bright response along the upper normal board edge with empty GT.
- pcb4 FN (pcb4\test\bad\005.JPG): A small lower-left annotation is weakly represented while a stronger hotspot lies higher near the left connector; the image remains below threshold.

Across the selected boards, thin pins and border components remain challenging. A follow-up could audit PCA masking around these structures, but masking was deliberately held fixed in the reported resolution comparison. Corrected cases do not eliminate these remaining failures.