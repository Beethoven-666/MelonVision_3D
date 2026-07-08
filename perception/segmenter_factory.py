from __future__ import annotations

import argparse

from perception.watermelon_segmenter import WatermelonSegmenter
from perception.yolo_ensemble_segmenter import DualYoloSegmentationPolicySegmenter


DEFAULT_YOLO_MODEL_A = "D:/MelonDataset/watermelon_seg/runs/runs8/segment/yolo11m_768_industrial/weights/best.pt"
DEFAULT_YOLO_MODEL_B = "D:/MelonDataset/watermelon_seg/runs/runs24/segment/team128_hardneg_from_runs15/weights/best.pt"
DEFAULT_YOLO_POLICY_JSON = (
    "D:/MelonDataset/watermelon_seg/runs/runs26/segment/source_aware_approx_p90r90_policy.json"
)


def add_segmenter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--segmenter",
        choices=("dual-yolo", "hsv"),
        default="dual-yolo",
        help="2D watermelon instance segmentation backend.",
    )
    parser.add_argument("--yolo-model-a", default=DEFAULT_YOLO_MODEL_A)
    parser.add_argument("--yolo-model-b", default=DEFAULT_YOLO_MODEL_B)
    parser.add_argument("--yolo-policy-json", default=DEFAULT_YOLO_POLICY_JSON)
    parser.add_argument("--yolo-policy-source", default="default")
    parser.add_argument("--yolo-model-a-name", default="runs8")
    parser.add_argument("--yolo-model-b-name", default="runs24")
    parser.add_argument("--yolo-imgsz", type=int, default=768)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-inference-conf", type=float, default=0.02)
    parser.add_argument("--yolo-inference-iou", type=float, default=0.7)
    parser.add_argument("--yolo-max-det", type=int, default=300)
    parser.add_argument("--yolo-raster-size", type=int, default=768)
    parser.add_argument("--yolo-class-ids", type=int, nargs="*", default=None)
    parser.add_argument("--yolo-min-area-pixels", type=float, default=0.0)
    parser.add_argument("--yolo-confidence-threshold", type=float, default=None)
    parser.add_argument("--yolo-min-area-ratio", type=float, default=None)
    parser.add_argument("--yolo-min-support", type=int, choices=(1, 2), default=None)
    parser.add_argument("--yolo-support-iou-threshold", type=float, default=None)
    parser.add_argument("--yolo-nms-iou-threshold", type=float, default=None)


def build_segmenter_from_args(args: argparse.Namespace) -> object:
    if args.segmenter == "hsv":
        return WatermelonSegmenter()

    return DualYoloSegmentationPolicySegmenter(
        model_a=args.yolo_model_a,
        model_b=args.yolo_model_b,
        policy_json=args.yolo_policy_json,
        policy_source=args.yolo_policy_source,
        model_a_name=args.yolo_model_a_name,
        model_b_name=args.yolo_model_b_name,
        imgsz=args.yolo_imgsz,
        device=args.yolo_device,
        inference_conf=args.yolo_inference_conf,
        inference_iou=args.yolo_inference_iou,
        max_det=args.yolo_max_det,
        raster_size=args.yolo_raster_size,
        class_ids=args.yolo_class_ids,
        min_area_pixels=args.yolo_min_area_pixels,
        confidence_threshold=args.yolo_confidence_threshold,
        min_area_ratio=args.yolo_min_area_ratio,
        min_support=args.yolo_min_support,
        support_iou_threshold=args.yolo_support_iou_threshold,
        nms_iou_threshold=args.yolo_nms_iou_threshold,
    )
