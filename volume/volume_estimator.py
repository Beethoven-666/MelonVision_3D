from __future__ import annotations

import math


def estimate_volume_by_ellipsoid(axes_length_m: dict[str, float]) -> dict[str, float | str]:
    major = float(axes_length_m["major"])
    middle = float(axes_length_m["middle"])
    minor = float(axes_length_m["minor"])

    a = major / 2.0
    b = middle / 2.0
    c = minor / 2.0
    volume_m3 = 4.0 / 3.0 * math.pi * a * b * c
    return {
        "volume_m3": float(volume_m3),
        "volume_liter": float(volume_m3 * 1000.0),
        "method": "ellipsoid_pca",
        "confidence": 0.65,
    }
