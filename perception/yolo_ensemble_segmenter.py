from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perception.watermelon_segmenter import WatermelonDetection


@dataclass(frozen=True)
class YoloMergePolicy:
    confidence_threshold: float = 0.35
    min_area_ratio: float = 0.0004
    min_support: int = 1
    support_iou_threshold: float = 0.35
    nms_iou_threshold: float = 0.3


@dataclass(frozen=True)
class YoloInstancePrediction:
    model_name: str
    confidence: float
    points: np.ndarray
    mask: np.ndarray
    area: int
    area_ratio: float
    bbox: tuple[float, float, float, float]


class DualYoloSegmentationPolicySegmenter:
    """In-memory two-model YOLO segmentation ensemble for watermelon instances."""

    def __init__(
        self,
        model_a: str | Path,
        model_b: str | Path,
        policy_json: str | Path | None = None,
        policy_source: str = "default",
        model_a_name: str = "runs8",
        model_b_name: str = "runs24",
        imgsz: int = 768,
        device: str | None = "0",
        inference_conf: float = 0.02,
        inference_iou: float = 0.7,
        max_det: int = 300,
        raster_size: int = 768,
        class_ids: list[int] | None = None,
        min_area_pixels: float = 0.0,
        class_name: str = "watermelon",
        confidence_threshold: float | None = None,
        min_area_ratio: float | None = None,
        min_support: int | None = None,
        support_iou_threshold: float | None = None,
        nms_iou_threshold: float | None = None,
    ) -> None:
        self.model_a_path = Path(model_a)
        self.model_b_path = Path(model_b)
        self._validate_path(self.model_a_path, "model_a")
        self._validate_path(self.model_b_path, "model_b")

        self.policy = self._load_policy(policy_json, policy_source)
        self.policy = self._apply_policy_overrides(
            self.policy,
            confidence_threshold=confidence_threshold,
            min_area_ratio=min_area_ratio,
            min_support=min_support,
            support_iou_threshold=support_iou_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        self.policy_source = policy_source
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.imgsz = imgsz
        self.device = device
        self.inference_conf = inference_conf
        self.inference_iou = inference_iou
        self.max_det = max_det
        self.raster_size = raster_size
        self.class_ids = class_ids
        self.min_area_pixels = min_area_pixels
        self.class_name = class_name

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for --segmenter dual-yolo.") from exc

        self.model_a = YOLO(str(self.model_a_path))
        self.model_b = YOLO(str(self.model_b_path))

    def segment(self, color_bgr: np.ndarray, depth_mm: np.ndarray | None = None) -> list[dict[str, object]]:
        del depth_mm
        height, width = color_bgr.shape[:2]
        pred_a = self._predict(color_bgr, self.model_a, self.model_a_name)
        pred_b = self._predict(color_bgr, self.model_b, self.model_b_name)
        kept = self._apply_policy(pred_a, pred_b)

        detections: list[dict[str, object]] = []
        for pred in kept:
            mask = _image_mask_from_points(pred.points, width, height)
            area = float(np.count_nonzero(mask))
            if area < self.min_area_pixels:
                continue
            bbox_xywh = _mask_bbox_xywh(mask)
            if bbox_xywh is None:
                continue
            detections.append(
                WatermelonDetection(
                    mask=mask,
                    bbox_xywh=bbox_xywh,
                    score=float(pred.confidence),
                    class_name=self.class_name,
                    area=area,
                ).as_dict()
                | {
                    "model_name": pred.model_name,
                    "area_ratio": float(pred.area_ratio),
                    "policy_source": self.policy_source,
                    "policy": {
                        "confidence_threshold": self.policy.confidence_threshold,
                        "min_area_ratio": self.policy.min_area_ratio,
                        "min_support": self.policy.min_support,
                        "support_iou_threshold": self.policy.support_iou_threshold,
                        "nms_iou_threshold": self.policy.nms_iou_threshold,
                    },
                }
            )

        detections.sort(key=lambda item: float(item["score"]), reverse=True)
        return detections

    def _predict(self, color_bgr: np.ndarray, model: Any, model_name: str) -> list[YoloInstancePrediction]:
        results = model.predict(
            source=color_bgr,
            imgsz=self.imgsz,
            conf=self.inference_conf,
            iou=self.inference_iou,
            max_det=self.max_det,
            device=self.device,
            classes=self.class_ids,
            save=False,
            verbose=False,
        )
        if not results:
            return []
        return _predictions_from_result(results[0], model_name, self.raster_size)

    def _apply_policy(
        self,
        pred_a: list[YoloInstancePrediction],
        pred_b: list[YoloInstancePrediction],
    ) -> list[YoloInstancePrediction]:
        selected_a = [
            pred
            for pred in pred_a
            if pred.confidence >= self.policy.confidence_threshold
            and pred.area_ratio >= self.policy.min_area_ratio
        ]
        selected_b = [
            pred
            for pred in pred_b
            if pred.confidence >= self.policy.confidence_threshold
            and pred.area_ratio >= self.policy.min_area_ratio
        ]
        if self.policy.min_support == 2:
            candidates = [
                pred
                for pred in selected_a
                if _supported_by_other_model(pred, selected_b, self.policy.support_iou_threshold)
            ] + [
                pred
                for pred in selected_b
                if _supported_by_other_model(pred, selected_a, self.policy.support_iou_threshold)
            ]
        else:
            candidates = selected_a + selected_b
        return _nms(candidates, self.policy.nms_iou_threshold)

    @staticmethod
    def _validate_path(path: Path, name: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{name} was not found: {path}")

    @staticmethod
    def _load_policy(policy_json: str | Path | None, policy_source: str) -> YoloMergePolicy:
        if not policy_json:
            return YoloMergePolicy()
        path = Path(policy_json)
        if not path.exists():
            raise FileNotFoundError(f"policy_json was not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw_policies = raw.get("policies", raw)
        selected = raw_policies.get(policy_source) or raw_policies.get("default")
        if selected is None:
            return YoloMergePolicy()
        return YoloMergePolicy(
            confidence_threshold=float(selected["confidence_threshold"]),
            min_area_ratio=float(selected["min_area_ratio"]),
            min_support=int(selected["min_support"]),
            support_iou_threshold=float(selected["support_iou_threshold"]),
            nms_iou_threshold=float(selected["nms_iou_threshold"]),
        )

    @staticmethod
    def _apply_policy_overrides(policy: YoloMergePolicy, **overrides: Any) -> YoloMergePolicy:
        values = {key: value for key, value in overrides.items() if value is not None}
        if not values:
            return policy
        return replace(policy, **values)


def _predictions_from_result(
    result: Any,
    model_name: str,
    raster_size: int,
) -> list[YoloInstancePrediction]:
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return []

    points_rows = getattr(masks, "xyn", [])
    confidences = boxes.conf.detach().cpu().numpy().tolist()
    predictions: list[YoloInstancePrediction] = []
    for points_raw, confidence in zip(points_rows, confidences):
        points = np.asarray(points_raw, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        points = np.clip(points, 0.0, 1.0)
        mask = _raster_mask_from_points(points, raster_size)
        area = int(mask.sum())
        if area <= 0:
            continue
        predictions.append(
            YoloInstancePrediction(
                model_name=model_name,
                confidence=float(confidence),
                points=points,
                mask=mask,
                area=area,
                area_ratio=area / float(raster_size * raster_size),
                bbox=_normalized_bbox(points),
            )
        )
    return predictions


def _normalized_bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _bboxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _raster_mask_from_points(points: np.ndarray, raster_size: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= raster_size - 1
    scaled[:, 1] *= raster_size - 1
    polygon = np.rint(scaled).astype(np.int32)
    mask = np.zeros((raster_size, raster_size), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def _image_mask_from_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= width - 1
    scaled[:, 1] *= height - 1
    polygon = np.rint(scaled).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def _mask_bbox_xywh(mask: np.ndarray) -> list[int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return [int(x), int(y), int(w), int(h)]


def _mask_iou(a: YoloInstancePrediction, b: YoloInstancePrediction) -> float:
    if a.area <= 0 or b.area <= 0 or not _bboxes_overlap(a.bbox, b.bbox):
        return 0.0
    intersection = int(np.logical_and(a.mask, b.mask).sum())
    union = a.area + b.area - intersection
    return float(intersection / union) if union else 0.0


def _supported_by_other_model(
    pred: YoloInstancePrediction,
    others: list[YoloInstancePrediction],
    support_iou_threshold: float,
) -> bool:
    return any(_mask_iou(pred, other) >= support_iou_threshold for other in others)


def _nms(
    predictions: list[YoloInstancePrediction],
    nms_iou_threshold: float,
) -> list[YoloInstancePrediction]:
    kept: list[YoloInstancePrediction] = []
    for pred in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        if any(_mask_iou(pred, kept_pred) > nms_iou_threshold for kept_pred in kept):
            continue
        kept.append(pred)
    return kept
