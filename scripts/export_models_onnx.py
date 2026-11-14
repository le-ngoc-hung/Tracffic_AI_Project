"""
Script tiện ích xuất các mô hình YOLOv8 (.pt) sang định dạng ONNX (.onnx)
để tăng tốc suy luận thực tế. UnifiedTrafficDetector sẽ tự động ưu tiên nạp .onnx nếu có.
"""

from __future__ import annotations

import os
from ultralytics import YOLO


MODELS_TO_EXPORT = [
    "yolov8s.pt",
    "models/vehicle_model.pt",
    "models/sign_model.pt",
    "models/plate_model.pt",
]


def main() -> None:
    for pt_path in MODELS_TO_EXPORT:
        if not os.path.exists(pt_path):
            print(f"[SKIP] Không tìm thấy {pt_path}")
            continue
        onnx_path = os.path.splitext(pt_path)[0] + ".onnx"
        if os.path.exists(onnx_path):
            print(f"[OK] Đã tồn tại {onnx_path}")
            continue
        print(f"[EXPORT] Đang xuất {pt_path} -> {onnx_path} ...")
        model = YOLO(pt_path)
        exported = model.export(format="onnx", imgsz=640, simplify=True)
        print(f"[DONE] Đã xuất: {exported}")


if __name__ == "__main__":
    main()
