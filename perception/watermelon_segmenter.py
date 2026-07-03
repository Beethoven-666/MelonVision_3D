from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WatermelonDetection:
    mask: np.ndarray
    bbox_xywh: list[int]
    score: float
    class_name: str
    area: float

    def as_dict(self) -> dict[str, object]:
        return {
            "mask": self.mask,
            "bbox_xywh": self.bbox_xywh,
            "score": self.score,
            "class_name": self.class_name,
            "area": self.area,
        }


class WatermelonSegmenter:
    def __init__(
        self,
        min_area: float = 1000.0,
        hsv_lower: tuple[int, int, int] = (25, 40, 30),
        hsv_upper: tuple[int, int, int] = (95, 255, 255),
        depth_min_m: float = 0.2,
        depth_max_m: float = 3.0,
    ) -> None:
        self.min_area = min_area
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m

    def segment(self, color_bgr: np.ndarray, depth_mm: np.ndarray | None = None) -> list[dict[str, object]]:
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        if depth_mm is not None and depth_mm.shape[:2] == mask.shape[:2]:
            depth_m = depth_mm.astype(np.float32) / 1000.0
            valid_depth = (depth_m >= self.depth_min_m) & (depth_m <= self.depth_max_m)
            mask = np.where(valid_depth, mask, 0).astype(np.uint8)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, object]] = []
        image_area = max(1, mask.shape[0] * mask.shape[1])

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            one_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(one_mask, [contour], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(contour)
            score = min(0.95, 0.55 + area / image_area)
            detections.append(
                WatermelonDetection(
                    mask=one_mask,
                    bbox_xywh=[int(x), int(y), int(w), int(h)],
                    score=float(score),
                    class_name="watermelon",
                    area=area,
                ).as_dict()
            )

        detections.sort(key=lambda item: float(item["area"]), reverse=True)
        return detections
