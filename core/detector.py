import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectedVehicle:
    """Represents a single tracked or supplemental detected vehicle."""
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    track_id: int
    vehicle_type: str
    conf: float = 0.0


@dataclass
class DetectedSign:
    """Represents a detected Vietnamese traffic sign."""
    x1: int
    y1: int
    x2: int
    y2: int
    sign_code: str
    conf: float


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(areaA + areaB - inter_area)


class UnifiedTrafficDetector:
    """
    Multi-model AI inference engine combining:
    1. YOLOv8s (COCO international vehicles: Car, Motorbike, Bus, Truck) + ByteTrack
    2. Custom VN Vehicle Model (models/vehicle_model.pt) for distant/dense motorbikes
    3. Custom VN Traffic Sign Model (models/sign_model.pt) for traffic rules (P.106a, P.102, P.127...)
    """

    COCO_MAP: Dict[int, str] = {
        2: "Car",
        3: "Motorbike",
        5: "Bus",
        7: "Truck",
    }

    @staticmethod
    def _resolve_accelerated_model(pt_path: str) -> str:
        """Auto-selects TensorRT (.engine) or ONNX (.onnx) beside .pt when available."""
        stem, _ = os.path.splitext(pt_path)
        for ext in (".engine", ".onnx"):
            candidate = f"{stem}{ext}"
            if os.path.exists(candidate):
                return candidate
        return pt_path

    def __init__(
        self,
        base_model_path: str = "yolov8s.pt",
        vn_model_path: str = "models/vehicle_model.pt",
        sign_model_path: str = "models/sign_model.pt",
        prefer_accelerated: bool = True,
    ):
        print("[AI Core] Đang nạp hệ thống mô hình AI đa năng...")
        b_path = self._resolve_accelerated_model(base_model_path) if prefer_accelerated else base_model_path
        v_path = self._resolve_accelerated_model(vn_model_path) if prefer_accelerated else vn_model_path
        s_path = self._resolve_accelerated_model(sign_model_path) if prefer_accelerated else sign_model_path
        self.base_model = YOLO(b_path)
        self.vn_model = YOLO(v_path)
        self.sign_model = YOLO(s_path)
        self.backend_summary = f"base={os.path.basename(b_path)}, vn={os.path.basename(v_path)}, sign={os.path.basename(s_path)}"
        print(f"[AI Core] Đã nạp thành công 3 mô hình ({self.backend_summary})!")

    def reset_tracker(self) -> None:
        """Resets ByteTrack state when switching between cameras."""
        if hasattr(self.base_model, "predictor") and self.base_model.predictor is not None:
            if hasattr(self.base_model.predictor, "trackers"):
                for tracker in self.base_model.predictor.trackers:
                    tracker.reset()

    def detect_signs(self, frame: np.ndarray, conf: float = 0.18) -> List[DetectedSign]:
        """Scans frame for Vietnamese traffic signs."""
        results = self.sign_model.predict(frame, conf=conf, verbose=False)
        signs: List[DetectedSign] = []
        if not results:
            return signs

        for box in results[0].boxes:
            cls_idx = int(box.cls[0])
            sign_code = str(results[0].names[cls_idx])
            sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
            s_conf = float(box.conf[0])
            signs.append(DetectedSign(sx1, sy1, sx2, sy2, sign_code, s_conf))
        return signs

    def detect_and_track_vehicles(self, frame: np.ndarray) -> List[DetectedVehicle]:
        """
        Runs primary ByteTrack tracking with base_model and supplements distant
        motorbikes using vn_model with IoU deduplication.
        """
        results = self.base_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[2, 3, 5, 7],
            conf=0.22,
            imgsz=640,
            iou=0.45,
            verbose=False,
        )

        vehicles: List[DetectedVehicle] = []
        base_boxes: List[Tuple[int, int, int, int]] = []

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy().astype(float)

            for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                v_type = self.COCO_MAP.get(int(cls_id), "Car")
                base_boxes.append((x1, y1, x2, y2))
                vehicles.append(
                    DetectedVehicle(x1, y1, x2, y2, cx, cy, int(track_id), v_type, float(conf))
                )

        # Supplemental scan for small/distant motorbikes using VN model
        vn_res = self.vn_model.predict(frame, conf=0.15, imgsz=640, verbose=False)
        extra_id_counter = 7000

        if vn_res:
            for vbox in vn_res[0].boxes:
                v_cls = int(vbox.cls[0])
                v_raw_name = str(self.vn_model.names[v_cls]).lower()
                vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
                v_conf = float(vbox.conf[0])

                if "motor" in v_raw_name or "xe" in v_raw_name or v_cls == 6:
                    is_duplicate = any(
                        compute_iou((vx1, vy1, vx2, vy2), b) > 0.20 for b in base_boxes
                    )
                    if not is_duplicate:
                        extra_id_counter += 1
                        cx = int((vx1 + vx2) / 2)
                        cy = int((vy1 + vy2) / 2)
                        vehicles.append(
                            DetectedVehicle(
                                vx1, vy1, vx2, vy2, cx, cy, extra_id_counter, "Motorbike", v_conf
                            )
                        )

        return vehicles
