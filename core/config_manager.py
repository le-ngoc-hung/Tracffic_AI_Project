import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple
import cv2
import numpy as np

from core.logger import get_app_logger

logger = get_app_logger("ConfigManager")


@dataclass
class CameraConfig:
    """Configuration parameters specific to a single camera view (normalized to 854x480 canvas)."""
    cam_key: str
    name: str
    video_path: str
    # Counting line defined by 2 points: [[x1, y1], [x2, y2]]
    counting_line: List[List[int]] = field(default_factory=lambda: [[40, 350], [810, 350]])
    # ROI polygon defined by N points: [[x1, y1], [x2, y2], ...]
    roi_polygon: List[List[int]] = field(
        default_factory=lambda: [[40, 160], [810, 160], [840, 470], [10, 470]]
    )
    # Lane divider line [[x1, y1], [x2, y2]] for 2-way roads
    divider_line: List[List[int]] = field(default_factory=lambda: [[427, 150], [427, 470]])
    # Allowed traffic direction mode: "DOWN", "UP", "HIGHWAY_2WAY", "URBAN_2WAY", "ANY"
    direction_mode: str = "ANY"

    # Perspective / Homography Speed Calibration
    # 4 points on road surface: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    calibration_quad: List[List[int]] = field(
        default_factory=lambda: [[200, 180], [650, 180], [800, 450], [60, 450]]
    )
    real_world_width_m: float = 10.5
    real_world_length_m: float = 28.0
    is_speed_calibrated: bool = False
    speed_limit_kmh: float = 60.0
    meters_per_pixel: float = 0.120
    congestion_threshold: int = 9

    # Multi-frame confirmation & confidence thresholds (Decoupling detection from violation)
    min_observation_frames: int = 4
    min_violation_frames: int = 3
    min_vehicle_conf: float = 0.45
    min_sign_conf: float = 0.35
    min_plate_conf: float = 0.65

    # Rule toggles
    enable_no_truck_rule: bool = True
    enable_speed_rule: bool = True
    enable_wrong_way_rule: bool = True
    enable_illegal_parking_rule: bool = False

    def validate(self, canvas_w: int = 854, canvas_h: int = 480) -> Tuple[bool, str]:
        """Validates ROI polygon, counting line, and numeric thresholds."""
        if len(self.counting_line) != 2:
            return False, "Vạch đếm phải gồm đúng 2 điểm đầu và cuối."
        p1, p2 = self.counting_line[0], self.counting_line[1]
        line_len = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if line_len < 30.0:
            return False, f"Vạch đếm quá ngắn ({line_len:.1f}px < 30px). Hãy vẽ vạch dài hơn."

        if len(self.roi_polygon) < 3:
            return False, "Vùng ROI phải có ít nhất 3 đỉnh đa giác."
        pts = np.array(self.roi_polygon, dtype=np.float32)
        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] > canvas_w) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] > canvas_h):
            return False, "Tọa độ đỉnh ROI vượt ra ngoài khung hình (854x480)."

        area = float(cv2.contourArea(pts))
        if area < 2500.0:
            return False, f"Diện tích vùng ROI quá nhỏ ({area:.0f} px² < 2500 px²)."

        if not (10.0 <= self.speed_limit_kmh <= 200.0):
            return False, "Giới hạn tốc độ phải nằm trong khoảng 10 - 200 km/h."

        if self.real_world_length_m <= 1.0 or self.real_world_width_m <= 1.0:
            return False, "Kích thước mặt đường thực tế (m) phải lớn hơn 1.0m."

        return True, "Cấu hình hợp lệ."


DEFAULT_CAMERA_CONFIGS: Dict[str, CameraConfig] = {
    "1": CameraConfig(
        cam_key="1",
        name="CAM 1: Giao Thông Đô Thị VN (Normal)",
        video_path="videos/cam1_normal.mp4",
        counting_line=[[55, 345], [815, 345]],
        roi_polygon=[[50, 175], [805, 175], [840, 465], [15, 465]],
        divider_line=[[412, 175], [405, 465]],
        direction_mode="URBAN_2WAY",
        calibration_quad=[[180, 185], [680, 185], [820, 455], [40, 455]],
        real_world_width_m=12.0,
        real_world_length_m=24.0,
        is_speed_calibrated=True,
        speed_limit_kmh=50.0,
        meters_per_pixel=0.085,
        congestion_threshold=14,
        min_observation_frames=4,
        min_violation_frames=3,
        enable_no_truck_rule=False,
        enable_speed_rule=True,
        enable_wrong_way_rule=True,
    ),
    "2": CameraConfig(
        cam_key="2",
        name="CAM 2: Cảnh Báo Kẹt Xe (Jam)",
        video_path="videos/cam2_jam.mp4",
        counting_line=[[60, 360], [780, 360]],
        roi_polygon=[[80, 140], [740, 140], [820, 470], [20, 470]],
        divider_line=[[410, 140], [410, 470]],
        direction_mode="UP",
        calibration_quad=[[160, 160], [690, 160], [790, 460], [50, 460]],
        real_world_width_m=14.0,
        real_world_length_m=32.0,
        is_speed_calibrated=False,
        speed_limit_kmh=50.0,
        meters_per_pixel=0.110,
        congestion_threshold=9,
        min_observation_frames=4,
        min_violation_frames=3,
        enable_no_truck_rule=False,
        enable_speed_rule=True,
        enable_wrong_way_rule=True,
    ),
    "3": CameraConfig(
        cam_key="3",
        name="CAM 3: Bắt Lỗi Biển Báo & Phạt Nguội (Violation)",
        video_path="videos/cam3_violation.mp4",
        counting_line=[[260, 355], [835, 355]],
        roi_polygon=[[280, 160], [770, 160], [845, 470], [240, 470]],
        divider_line=[[320, 160], [360, 470]],
        direction_mode="DOWN",
        calibration_quad=[[340, 175], [720, 175], [835, 460], [260, 460]],
        real_world_width_m=14.0,
        real_world_length_m=36.0,
        is_speed_calibrated=True,
        speed_limit_kmh=60.0,
        meters_per_pixel=0.135,
        congestion_threshold=9,
        min_observation_frames=4,
        min_violation_frames=3,
        enable_no_truck_rule=True,
        enable_speed_rule=True,
        enable_wrong_way_rule=True,
    ),
    "4": CameraConfig(
        cam_key="4",
        name="CAM 4: Cao Tốc 2 Chiều (Highway)",
        video_path="videos/cam4_highway.mp4",
        counting_line=[[60, 340], [825, 340]],
        roi_polygon=[[210, 175], [765, 175], [845, 470], [25, 470]],
        divider_line=[[640, 175], [475, 470]],
        direction_mode="HIGHWAY_2WAY",
        calibration_quad=[[250, 185], [750, 185], [835, 460], [45, 460]],
        real_world_width_m=22.0,
        real_world_length_m=45.0,
        is_speed_calibrated=True,
        speed_limit_kmh=80.0,
        meters_per_pixel=0.165,
        congestion_threshold=10,
        min_observation_frames=4,
        min_violation_frames=3,
        enable_no_truck_rule=False,
        enable_speed_rule=True,
        enable_wrong_way_rule=True,
        enable_illegal_parking_rule=False,
    ),
    "5": CameraConfig(
        cam_key="5",
        name="CAM 5: Camera IP Cận Cảnh (ANPR & Đỗ Xe)",
        video_path="videos/Traffic IP Camera video.mp4",
        counting_line=[[360, 195], [830, 195]],
        roi_polygon=[[110, 80], [830, 80], [840, 460], [60, 460]],
        divider_line=[[430, 80], [210, 460]],
        direction_mode="DOWN",
        calibration_quad=[[380, 100], [810, 100], [835, 450], [220, 450]],
        real_world_width_m=9.5,
        real_world_length_m=18.0,
        is_speed_calibrated=True,
        speed_limit_kmh=40.0,
        meters_per_pixel=0.065,
        congestion_threshold=8,
        min_observation_frames=4,
        min_violation_frames=3,
        min_plate_conf=0.60,
        enable_no_truck_rule=False,
        enable_speed_rule=True,
        enable_wrong_way_rule=True,
        enable_illegal_parking_rule=True,
    ),
}


class ConfigManager:
    """Loads, validates, and saves per-camera ROI, counting line, and rule configurations."""

    def __init__(self, config_path: str = "configs/camera_configs.json"):
        self.config_path = config_path
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        self.configs: Dict[str, CameraConfig] = {}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # 1. Load standard cameras 1..4
                template_fields = asdict(DEFAULT_CAMERA_CONFIGS["1"])
                for key, default_cfg in DEFAULT_CAMERA_CONFIGS.items():
                    if key in raw:
                        valid_fields = {k: v for k, v in raw[key].items() if k in template_fields}
                        merged = {**asdict(default_cfg), **valid_fields}
                        cfg = CameraConfig(**merged)
                        ok, msg = cfg.validate()
                        if ok:
                            self.configs[key] = cfg
                        else:
                            logger.warning(f"Config cam {key} không hợp lệ ({msg}), khôi phục mặc định.")
                            self.configs[key] = CameraConfig(**asdict(default_cfg))
                    else:
                        self.configs[key] = CameraConfig(**asdict(default_cfg))

                # 2. Load any user-added custom cameras (e.g., "6", "7", ...)
                for key, cam_data in raw.items():
                    if key.isdigit() and key not in DEFAULT_CAMERA_CONFIGS and isinstance(cam_data, dict):
                        valid_fields = {k: v for k, v in cam_data.items() if k in template_fields}
                        merged = {**template_fields, **valid_fields, "cam_key": key}
                        cfg = CameraConfig(**merged)
                        ok, _ = cfg.validate()
                        if ok:
                            self.configs[key] = cfg

                self.save()
                return
            except Exception as e:
                logger.error(f"Lỗi đọc cấu hình {self.config_path}: {e}. Dùng mặc định.")

        self.reset_all_to_default()

    def get_next_cam_key(self) -> str:
        """Returns the next available numeric camera key ('5', '6', ...)."""
        numeric_keys = [int(k) for k in self.configs.keys() if k.isdigit()]
        next_num = max(numeric_keys, default=4) + 1
        return str(next_num)

    def add_camera(self, cfg: CameraConfig) -> CameraConfig:
        """Registers a new camera configuration and persists it to JSON."""
        self.configs[cfg.cam_key] = cfg
        self.save()
        logger.info(f"Đã thêm Camera mới: [{cfg.cam_key}] {cfg.name} -> {cfg.video_path}")
        return cfg

    def remove_camera(self, cam_key: str) -> bool:
        """Removes a custom camera (keys '1'..'4' are protected standard cameras)."""
        if cam_key in DEFAULT_CAMERA_CONFIGS:
            return False
        if cam_key in self.configs:
            del self.configs[cam_key]
            self.save()
            logger.info(f"Đã xóa Camera [{cam_key}] khỏi cấu hình.")
            return True
        return False

    def save(self) -> None:
        try:
            data = {k: asdict(v) for k, v in self.configs.items()}
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Không thể lưu cấu hình: {e}")

    def get_config(self, cam_key: str) -> CameraConfig:
        if cam_key not in self.configs:
            self.configs[cam_key] = CameraConfig(
                cam_key=cam_key,
                name=f"CAM Custom ({os.path.basename(cam_key)})",
                video_path=cam_key,
                is_speed_calibrated=False,
            )
        return self.configs[cam_key]

    def reset_camera_to_default(self, cam_key: str) -> CameraConfig:
        if cam_key in DEFAULT_CAMERA_CONFIGS:
            self.configs[cam_key] = CameraConfig(**asdict(DEFAULT_CAMERA_CONFIGS[cam_key]))
        else:
            self.configs[cam_key] = CameraConfig(
                cam_key=cam_key,
                name=f"CAM Custom ({os.path.basename(cam_key)})",
                video_path=cam_key,
                is_speed_calibrated=False,
            )
        self.save()
        return self.configs[cam_key]

    def reset_all_to_default(self) -> None:
        self.configs = {
            k: CameraConfig(**asdict(v)) for k, v in DEFAULT_CAMERA_CONFIGS.items()
        }
        self.save()
