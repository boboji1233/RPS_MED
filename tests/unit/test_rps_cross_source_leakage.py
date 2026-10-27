from __future__ import annotations

from collections import Counter

import numpy as np

from rps_med.cross_source_leakage import SourceFingerprints, summarize_overlap


def _fingerprints(encoded: tuple[str, ...], pixels: tuple[str, ...]) -> SourceFingerprints:
    count = len(encoded)
    descriptors = np.zeros((count, 4), dtype=np.float32)
    descriptors[:, 0] = 1.0
    return SourceFingerprints(
        encoded_exact=encoded,
        pixel_exact=pixels,
        dhash=tuple(range(count)),
        descriptors=descriptors,
        sizes=Counter(),
    )


def test_overlap_distinguishes_encoded_and_decoded_pixel_identity() -> None:
    left = _fingerprints(("encoded_a", "encoded_b"), ("pixels_a", "pixels_b"))
    right = _fingerprints(("encoded_x", "encoded_b"), ("pixels_a", "pixels_x"))

    result = summarize_overlap(left, right)

    assert result["encoded_exact"]["duplicate_pair_count"] == 1
    assert result["decoded_pixel_exact"]["duplicate_pair_count"] == 1
    assert "encoded_a" not in str(result)
