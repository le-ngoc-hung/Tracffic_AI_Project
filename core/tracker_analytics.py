from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from core.config_manager import CameraConfig
from core.database import ViolationDatabase, ViolationRecord
from core.detector import DetectedSign, DetectedVehicle
from core.plate_anpr import LicensePlateANPR


@dataclass
class EnrichedVehicle:
    """Vehicle state enriched with speed, trajectory, plate number, and violation flags."""
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    track_id: int
    vehicle_type: str
    speed_kmh: float = 0.0
    direction_label: str = ""
    is_violated: bool = False
    violation_reasons: List[str] = field(default_factory=list)
    plate_number: str = ""
    trail: List[Tuple[int, int]] = field(default_factory=list)


def _ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    """Returns True if line segment p1->p2 intersects line segment q1->q2."""
    return (_ccw(p1, q1, q2) != _ccw(p2, q1, q2)) and (_ccw(p1, p2, q1) != _ccw(p1, p2, q2))


class TrafficAnalyticsEngine:
    """
    Stateful per-camera traffic analytics engine:
    - Line Crossing vehicle counter (supports arbitrary 2-point lines drawn by mouse)
    - ROI polygon density & congestion detector
    - Perspective-compensated speed estimation (km/h)
    - Wrong-way / lane invasion detector
    - Automated ANPR & violation evidence logging
    """

    def __init__(
        self,
        cam_config: CameraConfig,
        anpr: LicensePlateANPR,
        db: ViolationDatabase,
        fps: float = 30.0,
    ):
        self.config = cam_config
        self.anpr = anpr
        self.db = db
        self.fps = max(15.0, min(fps, 60.0))
        self.reset_state()

    def reset_state(self) -> None:
        self.counts: Dict[str, int] = {"Car": 0, "Motorbike": 0, "Bus": 0, "Truck": 0}
        self.counted_ids: Set[int] = set()
        self.violation_ids: Set[int] = set()
        self.violation_details: Dict[int, Tuple[str, str]] = {}  # track_id -> (reason, plate)
        # History per track_id: deque of (frame_idx, cx, cy)
        self.track_history: Dict[int, Deque[Tuple[int, int, int]]] = {}
        self.smoothed_speeds: Dict[int, float] = {}
        self.overspeed_streaks: Dict[int, int] = {}
        self.wrong_way_streaks: Dict[int, int] = {}
        self.traffic_status: str = "NORMAL"
        self.rule_no_truck_detected: bool = False
        self.active_signs: List[DetectedSign] = []
        self.frame_idx: int = 0
        self.vehicles_in_roi: int = 0
        self.avg_speed_kmh: float = 0.0
        self.last_enriched_vehicles: List[EnrichedVehicle] = []
        self.new_violations: List[ViolationRecord] = []

    def update_signs(self, signs: List[DetectedSign]) -> None:
        for s in signs:
            if "106" in s.sign_code or "P.106a" in s.sign_code:
                self.rule_no_truck_detected = True
                self.active_signs = [s]

    def _estimate_speed_kmh(self, track_id: int, history: Deque[Tuple[int, int, int]]) -> float:
        if len(history) < 2:
            return 0.0

        f_old, x_old, y_old = history[0]
        f_now, x_now, y_now = history[-1]
        df = f_now - f_old
        if df <= 0:
            return self.smoothed_speeds.get(track_id, 0.0)

        dx = float(x_now - x_old)
        dy = float(y_now - y_old)

        # Perspective compensation: objects higher up (smaller Y) travel more meters per pixel
        mean_y = max(120.0, min(460.0, (y_old + y_now) / 2.0))
        perspective_factor = 1.0 + 1.35 * ((460.0 - mean_y) / 340.0)

        pixel_dist = np.hypot(dx * 0.85, dy * perspective_factor)
        meters = pixel_dist * self.config.meters_per_pixel
        elapsed_sec = df / self.fps
        raw_kmh = (meters / max(elapsed_sec, 0.001)) * 3.6

        # Ignore tiny jitter when stationary
        if pixel_dist / max(df, 1) < 0.35:
            raw_kmh = 0.0

        raw_kmh = float(np.clip(raw_kmh, 0.0, 145.0))
        prev_speed = self.smoothed_speeds.get(track_id, raw_kmh)
        smoothed = 0.65 * prev_speed + 0.35 * raw_kmh
        self.smoothed_speeds[track_id] = smoothed
        return smoothed

    def _get_divider_x_at_y(self, y: int) -> float:
        div = self.config.divider_line
        if len(div) < 2:
            return 427.0
        (x1, y1), (x2, y2) = div[0], div[1]
        if abs(y2 - y1) < 1:
            return float(x1)
        t = (y - y1) / float(y2 - y1)
        return x1 + t * (x2 - x1)

    def _check_wrong_way(self, track_id: int, history: Deque[Tuple[int, int, int]]) -> bool:
        if not self.config.enable_wrong_way_rule or len(history) < 3:
            return False

        _, x_old, y_old = history[0]
        _, x_now, y_now = history[-1]
        dy = y_now - y_old
        if abs(dy) < 12:
            return False

        mode = self.config.direction_mode
        div_x = self._get_divider_x_at_y(y_now)

        is_wrong = False
        if mode == "DOWN":
            # Valid traffic goes DOWN (dy > 0); moving UP (dy < -12) is wrong way
            is_wrong = dy < -12
        elif mode == "UP":
            # Valid traffic goes UP (dy < 0); moving DOWN (dy > 12) is wrong way
            is_wrong = dy > 12
        elif mode == "HIGHWAY_2WAY":
            # Left of divider moves UP (dy < 0), right of divider moves DOWN (dy > 0)
            if x_now < div_x:
                is_wrong = dy > 12
            else:
                is_wrong = dy < -12
        elif mode == "URBAN_2WAY":
            # Right of yellow line moves UP (dy < 0); far left oncoming lane moves DOWN (dy > 0)
            if x_now >= div_x:
                is_wrong = dy > 14
            elif x_now < div_x - 85 and y_now < 330:
                is_wrong = dy < -15

        if is_wrong:
            self.wrong_way_streaks[track_id] = self.wrong_way_streaks.get(track_id, 0) + 1
        else:
            self.wrong_way_streaks[track_id] = max(0, self.wrong_way_streaks.get(track_id, 0) - 1)

        return self.wrong_way_streaks.get(track_id, 0) >= 2

    def process_detections(
        self,
        vehicles: List[DetectedVehicle],
        raw_frame: np.ndarray,
        canvas_frame: np.ndarray,
    ) -> List[EnrichedVehicle]:
        """
        Updates trajectories, counts vehicles crossing the counting line,
        computes ROI congestion, checks violations, and triggers ANPR evidence capture.
        """
        self.new_violations.clear()
        roi_poly = np.array(self.config.roi_polygon, dtype=np.int32)
        line_pts = self.config.counting_line
        l_p1 = (float(line_pts[0][0]), float(line_pts[0][1]))
        l_p2 = (float(line_pts[1][0]), float(line_pts[1][1]))

        enriched_list: List[EnrichedVehicle] = []
        vehicles_in_roi = 0
        active_speeds: List[float] = []

        no_truck_active = self.config.enable_no_truck_rule and (
            self.rule_no_truck_detected or self.config.cam_key == "3"
        )

        for v in vehicles:
            tid = v.track_id
            in_roi = cv2.pointPolygonTest(roi_poly, (float(v.cx), float(v.cy)), False) >= 0
            if in_roi:
                vehicles_in_roi += 1

            speed_kmh = 0.0
            direction_str = ""
            trail_pts: List[Tuple[int, int]] = []

            if tid < 7000:
                if tid not in self.track_history:
                    self.track_history[tid] = deque(maxlen=8)
                hist = self.track_history[tid]

                prev_pt: Optional[Tuple[float, float]] = None
                if len(hist) > 0:
                    prev_pt = (float(hist[-1][1]), float(hist[-1][2]))

                hist.append((self.frame_idx, v.cx, v.cy))
                trail_pts = [(pt[1], pt[2]) for pt in hist]

                # 1. Estimate vehicle speed (km/h)
                speed_kmh = self._estimate_speed_kmh(tid, hist)
                if speed_kmh > 3.0:
                    active_speeds.append(speed_kmh)

                if len(hist) >= 2:
                    dy_total = hist[-1][2] - hist[0][2]
                    if dy_total > 4:
                        direction_str = "v"
                    elif dy_total < -4:
                        direction_str = "^"

                # 2. Check Line Crossing (supports any custom line angle)
                crossed = False
                if prev_pt is not None:
                    curr_pt = (float(v.cx), float(v.cy))
                    crossed = segments_intersect(prev_pt, curr_pt, l_p1, l_p2)

                if crossed and tid not in self.counted_ids:
                    self.counted_ids.add(tid)
                    self.counts[v.vehicle_type] = self.counts.get(v.vehicle_type, 0) + 1

                # 3. Check Violations (No-Truck Sign, Overspeeding, Wrong-Way)
                if tid not in self.violation_ids and (in_roi or crossed):
                    violation_reason: Optional[str] = None

                    # Rule A: Forbidden Truck Sign (P.106a)
                    if no_truck_active and v.vehicle_type == "Truck" and crossed:
                        violation_reason = "Cam xe tai (P.106a)"

                    # Rule B: Wrong-Way Driving
                    elif in_roi and self._check_wrong_way(tid, hist):
                        violation_reason = "Di nguoc chieu / Lan lan"

                    # Rule C: Overspeeding
                    elif in_roi and self.config.enable_speed_rule and speed_kmh > self.config.speed_limit_kmh:
                        self.overspeed_streaks[tid] = self.overspeed_streaks.get(tid, 0) + 1
                        if self.overspeed_streaks[tid] >= 2:
                            violation_reason = f"Qua toc do ({speed_kmh:.0f}/{self.config.speed_limit_kmh:.0f} km/h)"
                    else:
                        self.overspeed_streaks[tid] = 0

                    if violation_reason is not None:
                        self.violation_ids.add(tid)
                        plate_text, veh_crop, plate_crop = self.anpr.recognize_plate(
                            raw_frame=raw_frame,
                            box_canvas=(v.x1, v.y1, v.x2, v.y2),
                            canvas_size=(canvas_frame.shape[1], canvas_frame.shape[0]),
                            cam_key=self.config.cam_key,
                            track_id=tid,
                            vehicle_type=v.vehicle_type,
                        )
                        self.violation_details[tid] = (violation_reason, plate_text)

                        ev_path, pl_path = self.anpr.save_violation_evidence(
                            annotated_canvas=canvas_frame,
                            vehicle_crop=veh_crop,
                            plate_crop=plate_crop,
                            cam_key=self.config.cam_key,
                            track_id=tid,
                            vehicle_type=v.vehicle_type,
                            plate_text=plate_text,
                            speed_kmh=speed_kmh,
                            violation_label=violation_reason,
                            box_canvas=(v.x1, v.y1, v.x2, v.y2),
                        )

                        rec = self.db.insert_violation(
                            camera_id=f"CAM {self.config.cam_key}",
                            track_id=tid,
                            vehicle_type=v.vehicle_type,
                            plate_number=plate_text,
                            speed_kmh=speed_kmh,
                            violation_type=violation_reason,
                            image_path=ev_path,
                            plate_crop_path=pl_path,
                        )
                        self.new_violations.append(rec)
                        print(
                            f"--> [PHẠT NGUỘI ANPR] {rec.camera_id} | Xe: {v.vehicle_type} #{tid} | "
                            f"Biển số: {plate_text} | Lỗi: {violation_reason} | Đã lưu: {ev_path}"
                        )

            is_violated = tid in self.violation_ids
            reasons: List[str] = []
            plate_num = ""
            if is_violated and tid in self.violation_details:
                reason_str, plate_num = self.violation_details[tid]
                reasons.append(reason_str)

            enriched_list.append(
                EnrichedVehicle(
                    x1=v.x1,
                    y1=v.y1,
                    x2=v.x2,
                    y2=v.y2,
                    cx=v.cx,
                    cy=v.cy,
                    track_id=tid,
                    vehicle_type=v.vehicle_type,
                    speed_kmh=speed_kmh,
                    direction_label=direction_str,
                    is_violated=is_violated,
                    violation_reasons=reasons,
                    plate_number=plate_num,
                    trail=trail_pts,
                )
            )

        self.vehicles_in_roi = vehicles_in_roi
        self.traffic_status = (
            "CONGESTED" if vehicles_in_roi >= self.config.congestion_threshold else "NORMAL"
        )
        self.avg_speed_kmh = float(np.mean(active_speeds)) if active_speeds else 0.0
        self.last_enriched_vehicles = enriched_list
        return enriched_list
