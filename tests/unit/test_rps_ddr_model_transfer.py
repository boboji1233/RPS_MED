from __future__ import annotations

import numpy as np
import pytest

from rps_med.ddr_model_transfer import _resize_rgb


def test_resize_rgb_enforces_frozen_shape_contract() -> None:
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    assert _resize_rgb(image, 768).shape == (768, 768, 3)
    with pytest.raises(ValueError, match="contract"):
        _resize_rgb(image, 750)
