"""Bootstrap patches for eval scripts (watermarker stub, etc.)."""
from __future__ import annotations

import perth


class _NoWatermark:
    def apply_watermark(self, wav, sample_rate=24000):
        return wav


if getattr(perth, "PerthImplicitWatermarker", None) is None:
    perth.PerthImplicitWatermarker = _NoWatermark
