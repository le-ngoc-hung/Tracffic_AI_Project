import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


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
    # Allowed traffic direction mode:
    # "DOWN" (towards camera, dy > 0)
    # "UP" (away from camera, dy < 0)
    # "HIGHWAY_2WAY" (left of divider: UP, right of divider: DOWN)
    # "URBAN_2WAY" (left of divider: UP/DOWN with oncoming lane check on far-left)
    # "ANY" (both directions allowed)
    direction_mode: str = "ANY"
    # Speed estimation & enforcement
    speed_limit_kmh: float = 60.0
    meters_per_pixel: float = 0.045
    congestion_threshold: int = 9
    # Rule toggles
    enable_no_truck_rule: bool = True
    enable_speed_rule: bool = True
    enable_wrong_way_rule: bool = True


DEFAULT_CAMERA_CONFIGS: Dict[str, CameraConfig] = {
    "1": CameraConfig(
        cam_key="1",
        name="CAM 1: Giao Thông Đô Thị VN (Normal)",
        video_path="videos/cam1_normal.mp4",
        counting_line=[[55, 345], [815, 345]],
        roi_polygon=[[50, 175], [805, 175], [840, 465], [15, 465]],
        divider_line=[[412, 175], [405, 465]],
        direction_mode="URBAN_2WAY",
        speed_limit_kmh=50.0,
        meters_per_pixel=0.085,
        congestion_threshold=14,
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
        speed_limit_kmh=50.0,
        meters_per_pixel=0.110,
        congestion_threshold=9,
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
        speed_limit_kmh=60.0,
        meters_per_pixel=0.140,
        congestion_threshold=9,
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
        speed_limit_kmh=80.0,
        meters_per_pixel=0.175,
        congestion_threshold=10,
        enable_no_truck_rule=False,
        enable_speed_rule=True,
        enable_wrong_way_rule=True,
    ),
}


class ConfigManager:
    """Loads and saves per-camera ROI, counting line, and rule configurations."""

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
                for key, default_cfg in DEFAULT_CAMERA_CONFIGS.items():
                    if key in raw:
                        merged = {**asdict(default_cfg), **raw[key]}
                        self.configs[key] = CameraConfig(**merged)
                    else:
                        self.configs[key] = CameraConfig(**asdict(default_cfg))
                return
            except Exception as e:
                print(f"[CẢNH BÁO] Lỗi đọc cấu hình {self.config_path}: {e}. Dùng mặc định.")

        self.reset_all_to_default()

    def save(self) -> None:
        data = {k: asdict(v) for k, v in self.configs.items()}
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_config(self, cam_key: str) -> CameraConfig:
        if cam_key not in self.configs:
            self.configs[cam_key] = CameraConfig(
                cam_key=cam_key,
                name=f"CAM Custom ({ os.path.basename(cam_key) })",
                video_path=cam_key,
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
            )
        self.save()
        return self.configs[cam_key]

    def reset_all_to_default(self) -> None:
        self.configs = {
            k: CameraConfig(**asdict(v)) for k, v in DEFAULT_CAMERA_CONFIGS.items()
        }
        self.save()
