from collections import deque
from dataclasses import dataclass, field
import json
from typing import Deque, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from core.config_manager import CameraConfig
from core.database import ViolationDatabase, ViolationRecord
from core.detector import DetectedSign, DetectedVehicle
from core.logger import get_app_logger
from core.plate_anpr import LicensePlateANPR

logger = get_app_logger("TrackerAnalytics")

MODEL_VERSION_TAG = "YOLOv8s+VN_Veh+VN_Sign+RapidOCR_v2.0"


@dataclass
class EnrichedVehicle:
    """Vehicle state enriched with calibrated/estimated speed, confidence, ANPR status, and audit trail."""
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    track_id: int
    vehicle_type: str
    conf: float = 0.0
    speed_kmh: float = 0.0
    is_speed_calibrated: bool = False
    direction_label: str = ""
    is_violated: bool = False
    verification_status: str = ""
    violation_reasons: List[str] = field(default_factory=list)
    plate_number: str = ""
    plate_conf: float = 0.0
    plate_status: str = "UNREADABLE"
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
    Stateful per-camera traffic analytics engine with:
    - Homography-calibrated Bird's-Eye-View speed measurement + fallback estimation flag
    - Multi-frame violation confirmation decoupled from single-frame detection
    - Before/During/After frame ring buffer for verifiable event evidence
    - Automatic stale-track memory cleanup
    """

    MAX_TRACK_AGE_FRAMES = 180

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
        self._homography_matrix: Optional[np.ndarray] = None
        self._update_homography()
        self.reset_state()

    def _update_homography(self) -> None:
        """Computes 3x3 Homography matrix from 4 road calibration points to real-world meters."""
        self._homography_matrix = None
        if len(self.config.calibration_quad) == 4:
            try:
                src = np.array(self.config.calibration_quad, dtype=np.float32)
                w_m = float(max(2.0, self.config.real_world_width_m))
                l_m = float(max(5.0, self.config.real_world_length_m))
                dst = np.array(
                    [[0.0, 0.0], [w_m, 0.0], [w_m, l_m], [0.0, l_m]], dtype=np.float32
                )
                self._homography_matrix = cv2.getPerspectiveTransform(src, dst)
            except Exception as e:
                logger.warning(f"Không thể tính ma trận Homography: {e}")

    def reset_state(self) -> None:
        self._update_homography()
        self.counts: Dict[str, int] = {"Car": 0, "Motorbike": 0, "Bus": 0, "Truck": 0}
        self.counted_ids: Set[int] = set()
        self.violation_ids: Set[int] = set()
        # track_id -> (reason, plate_text, plate_conf, plate_status, verification_status)
        self.violation_details: Dict[int, Tuple[str, str, float, str, str]] = {}
        # History per track_id: deque of (frame_idx, cx, cy, bottom_y, conf, timestamp_sec)
        self.track_history: Dict[int, Deque[Tuple[int, int, int, int, float, float]]] = {}
        self.smoothed_speeds: Dict[int, float] = {}
        self.overspeed_streaks: Dict[int, int] = {}
        self.wrong_way_streaks: Dict[int, int] = {}
        self.no_truck_streaks: Dict[int, int] = {}
        self.stationary_streaks: Dict[int, int] = {}

        # Sign multi-frame confirmation
        self.sign_confirm_count: int = 0
        self.max_sign_conf: float = 0.0
        self.rule_no_truck_detected: bool = False
        self.active_signs: List[DetectedSign] = []

        # Ring buffers for Before/During/After timeline & short MP4 evidence clips
        self.frame_ring_buffer: Deque[np.ndarray] = deque(maxlen=12)
        self.clip_ring_buffer: Deque[np.ndarray] = deque(maxlen=45)

        self.traffic_status: str = "NORMAL"
        self.frame_idx: int = 0
        self.vehicles_in_roi: int = 0
        self.avg_speed_kmh: float = 0.0
        self.last_enriched_vehicles: List[EnrichedVehicle] = []
        self.new_violations: List[ViolationRecord] = []

    def update_signs(self, signs: List[DetectedSign]) -> None:
        """Requires multi-frame or above-threshold sign detection to confirm P.106a rule."""
        for s in signs:
            if ("106" in s.sign_code or "P.106a" in s.sign_code) and s.conf >= 0.18:
                self.sign_confirm_count += 1
                self.max_sign_conf = max(self.max_sign_conf, s.conf)
                self.active_signs = [s]
                if self.sign_confirm_count >= 1 or s.conf >= self.config.min_sign_conf:
                    self.rule_no_truck_detected = True

    def _estimate_speed_kmh(
        self, track_id: int, history: Deque[Tuple[int, int, int, int, float, float]]
    ) -> Tuple[float, bool]:
        """
        Computes vehicle speed in km/h using real frame timestamps (supports variable FPS RTSP).
        Uses Homography projection to road meters when calibrated; otherwise uses
        perspective-compensated pixel approximation and returns is_calibrated=False.
        """
        if len(history) < 2:
            return 0.0, self.config.is_speed_calibrated

        p_old = history[0]
        p_now = history[-1]
        f_old, x_old, y_old, by_old = int(p_old[0]), int(p_old[1]), int(p_old[2]), int(p_old[3])
        f_now, x_now, y_now, by_now = int(p_now[0]), int(p_now[1]), int(p_now[2]), int(p_now[3])
        t_old = float(p_old[5]) if len(p_old) > 5 else (f_old / max(self.fps, 1.0))
        t_now = float(p_now[5]) if len(p_now) > 5 else (f_now / max(self.fps, 1.0))
        df = f_now - f_old
        if df <= 0:
            return self.smoothed_speeds.get(track_id, 0.0), self.config.is_speed_calibrated

        pixel_move = float(np.hypot(x_now - x_old, by_now - by_old))
        if pixel_move / max(df, 1) < 0.35:
            # Stationary vehicle
            prev = self.smoothed_speeds.get(track_id, 0.0)
            smoothed = 0.5 * prev
            self.smoothed_speeds[track_id] = smoothed
            return smoothed, self.config.is_speed_calibrated

        dt = t_now - t_old
        elapsed_sec = dt if dt > 0.005 else (df / self.fps)
        meters = 0.0
        used_homography = False

        if self.config.is_speed_calibrated and self._homography_matrix is not None:
            pts = np.array([[[float(x_old), float(by_old)], [float(x_now), float(by_now)]]], dtype=np.float32)
            projected = cv2.perspectiveTransform(pts, self._homography_matrix)[0]
            hm_dist = float(np.hypot(projected[1][0] - projected[0][0], projected[1][1] - projected[0][1]))
            # Guard against extreme vanishing-point distortion outside the road quad
            if 0.05 <= hm_dist <= 60.0:
                meters = hm_dist
                used_homography = True

        if not used_homography:
            dx = float(x_now - x_old)
            dy = float(by_now - by_old)
            mean_y = max(120.0, min(460.0, (by_old + by_now) / 2.0))
            perspective_factor = 1.0 + 1.35 * ((460.0 - mean_y) / 340.0)
            pixel_dist = np.hypot(dx * 0.85, dy * perspective_factor)
            meters = pixel_dist * self.config.meters_per_pixel

        raw_kmh = (meters / max(elapsed_sec, 0.001)) * 3.6
        raw_kmh = float(np.clip(raw_kmh, 0.0, 145.0))
        prev_speed = self.smoothed_speeds.get(track_id, raw_kmh)
        smoothed = 0.65 * prev_speed + 0.35 * raw_kmh
        self.smoothed_speeds[track_id] = smoothed
        return smoothed, (self.config.is_speed_calibrated and used_homography)

    def _get_divider_x_at_y(self, y: int) -> float:
        div = self.config.divider_line
        if len(div) < 2:
            return 427.0
        (x1, y1), (x2, y2) = div[0], div[1]
        if abs(y2 - y1) < 1:
            return float(x1)
        t = (y - y1) / float(y2 - y1)
        return x1 + t * (x2 - x1)

    def _check_wrong_way(
        self, track_id: int, history: Deque[Tuple[int, int, int, int, float, float]]
    ) -> bool:
        if not self.config.enable_wrong_way_rule or len(history) < self.config.min_observation_frames:
            return False

        x_old, y_old = int(history[0][1]), int(history[0][2])
        x_now, y_now = int(history[-1][1]), int(history[-1][2])
        dy = y_now - y_old
        if abs(dy) < 14:
            return False

        mode = self.config.direction_mode
        div_x = self._get_divider_x_at_y(y_now)

        is_wrong = False
        if mode == "DOWN":
            is_wrong = dy < -14
        elif mode == "UP":
            is_wrong = dy > 14
        elif mode == "HIGHWAY_2WAY":
            if x_now < div_x:
                is_wrong = dy > 14
            else:
                is_wrong = dy < -14
        elif mode == "URBAN_2WAY":
            if x_now >= div_x:
                is_wrong = dy > 15
            elif x_now < div_x - 85 and y_now < 330:
                is_wrong = dy < -16

        if is_wrong:
            self.wrong_way_streaks[track_id] = self.wrong_way_streaks.get(track_id, 0) + 1
        else:
            self.wrong_way_streaks[track_id] = max(0, self.wrong_way_streaks.get(track_id, 0) - 1)

        return self.wrong_way_streaks.get(track_id, 0) >= self.config.min_violation_frames

    def _cleanup_stale_tracks(self) -> None:
        """Removes tracks not seen for MAX_TRACK_AGE_FRAMES to bound memory usage."""
        stale_ids = [
            tid
            for tid, hist in self.track_history.items()
            if not hist or (self.frame_idx - hist[-1][0]) > self.MAX_TRACK_AGE_FRAMES
        ]
        for tid in stale_ids:
            self.track_history.pop(tid, None)
            self.smoothed_speeds.pop(tid, None)
            self.overspeed_streaks.pop(tid, None)
            self.wrong_way_streaks.pop(tid, None)
            self.no_truck_streaks.pop(tid, None)

    def process_detections(
        self,
        vehicles: List[DetectedVehicle],
        raw_frame: np.ndarray,
        canvas_frame: np.ndarray,
        frame_time_sec: Optional[float] = None,
    ) -> List[EnrichedVehicle]:
        """
        Processes detections with multi-frame confirmation, Homography speed calibration,
        honest ANPR, and Before/During/After + MP4 clip + SHA-256 audit logging.
        """
        self.new_violations.clear()
        if self.frame_idx % 60 == 0:
            self._cleanup_stale_tracks()
        current_ts = float(frame_time_sec) if frame_time_sec is not None else (self.frame_idx / max(self.fps, 1.0))

        roi_poly = np.array(self.config.roi_polygon, dtype=np.int32)
        line_pts = self.config.counting_line
        l_p1 = (float(line_pts[0][0]), float(line_pts[0][1]))
        l_p2 = (float(line_pts[1][0]), float(line_pts[1][1]))

        enriched_list: List[EnrichedVehicle] = []
        vehicles_in_roi = 0
        active_speeds: List[float] = []

        no_truck_active = self.config.enable_no_truck_rule and self.rule_no_truck_detected

        for v in vehicles:
            tid = v.track_id
            in_roi = cv2.pointPolygonTest(roi_poly, (float(v.cx), float(v.cy)), False) >= 0
            if in_roi:
                vehicles_in_roi += 1

            speed_kmh = 0.0
            is_calibrated = self.config.is_speed_calibrated
            direction_str = ""
            trail_pts: List[Tuple[int, int]] = []

            if tid < 7000:
                if tid not in self.track_history:
                    self.track_history[tid] = deque(maxlen=10)
                hist = self.track_history[tid]

                prev_pt: Optional[Tuple[float, float]] = None
                if len(hist) > 0:
                    prev_pt = (float(hist[-1][1]), float(hist[-1][2]))

                hist.append((self.frame_idx, v.cx, v.cy, v.y2, v.conf, current_ts))
                trail_pts = [(pt[1], pt[2]) for pt in hist]
                obs_count = len(hist)
                mean_veh_conf = float(np.mean([pt[4] for pt in hist]))

                # 1. Estimate speed (km/h) with calibration status
                speed_kmh, is_calibrated = self._estimate_speed_kmh(tid, hist)
                if speed_kmh > 3.0:
                    active_speeds.append(speed_kmh)

                if obs_count >= 2:
                    dy_total = hist[-1][2] - hist[0][2]
                    if dy_total > 4:
                        direction_str = "v"
                    elif dy_total < -4:
                        direction_str = "^"

                # 2. Check Line Crossing (requires >= 2 observations)
                crossed = False
                if prev_pt is not None:
                    curr_pt = (float(v.cx), float(v.cy))
                    crossed = segments_intersect(prev_pt, curr_pt, l_p1, l_p2)

                if crossed and tid not in self.counted_ids:
                    self.counted_ids.add(tid)
                    self.counts[v.vehicle_type] = self.counts.get(v.vehicle_type, 0) + 1

                # 3. Multi-frame Violation Evaluation (Decoupled from single-frame detection)
                if v.vehicle_type == "Truck" and in_roi:
                    self.no_truck_streaks[tid] = self.no_truck_streaks.get(tid, 0) + 1

                if in_roi and obs_count >= 2:
                    total_disp = float(np.hypot(hist[-1][1] - hist[0][1], hist[-1][2] - hist[0][2]))
                    if total_disp < 6.0:
                        self.stationary_streaks[tid] = self.stationary_streaks.get(tid, 0) + 1
                    else:
                        self.stationary_streaks[tid] = 0

                if tid not in self.violation_ids and obs_count >= self.config.min_observation_frames and (in_roi or crossed):
                    violation_reason: Optional[str] = None
                    trigger_detail: str = ""

                    # Rule A: Forbidden Truck Sign (P.106a) - requires truck observed >= min_violation_frames & crossing line
                    if (
                        no_truck_active
                        and v.vehicle_type == "Truck"
                        and crossed
                        and self.no_truck_streaks.get(tid, 0) >= self.config.min_violation_frames
                    ):
                        violation_reason = "Cam xe tai (P.106a)"
                        trigger_detail = (
                            f"Quan sat {obs_count} chu ky; cat vach tai frame #{self.frame_idx}; "
                            f"Truck conf={mean_veh_conf:.2f}; Sign P.106a conf={self.max_sign_conf:.2f} ({self.sign_confirm_count} lan)"
                        )

                    # Rule B: Wrong-Way Driving - requires >= min_violation_frames consecutive wrong-way steps
                    elif in_roi and self._check_wrong_way(tid, hist):
                        violation_reason = "Di nguoc chieu / Lan lan"
                        dy_val = hist[-1][2] - hist[0][2]
                        trigger_detail = (
                            f"Di nguoc chieu {self.wrong_way_streaks.get(tid, 0)} chu ky lien tiep (dy={dy_val}px) "
                            f"trong che do {self.config.direction_mode}; Veh conf={mean_veh_conf:.2f}"
                        )

                    # Rule C: Overspeeding - requires >= min_violation_frames consecutive overspeed steps
                    elif in_roi and self.config.enable_speed_rule and speed_kmh > self.config.speed_limit_kmh:
                        self.overspeed_streaks[tid] = self.overspeed_streaks.get(tid, 0) + 1
                        if self.overspeed_streaks[tid] >= self.config.min_violation_frames:
                            est_str = "" if is_calibrated else " ~Uoc tinh"
                            violation_reason = f"Qua toc do ({speed_kmh:.0f}/{self.config.speed_limit_kmh:.0f} km/h{est_str})"
                            trigger_detail = (
                                f"Vuot nguong {self.config.speed_limit_kmh:.0f}km/h trong "
                                f"{self.overspeed_streaks[tid]} chu ky ({speed_kmh:.1f}km/h, "
                                f"calibrated={is_calibrated}); Veh conf={mean_veh_conf:.2f}"
                            )

                    # Rule D: Illegal Parking / Stationary in No-Parking ROI
                    elif (
                        in_roi
                        and getattr(self.config, "enable_illegal_parking_rule", False)
                        and self.traffic_status == "NORMAL"
                        and self.stationary_streaks.get(tid, 0) >= 6
                    ):
                        violation_reason = "Dung do trai phep"
                        trigger_detail = (
                            f"Dung do bat dong trong ROI {self.stationary_streaks[tid]} chu ky lien tiep "
                            f"(dich chuyen < 6px); Veh conf={mean_veh_conf:.2f}"
                        )
                    else:
                        self.overspeed_streaks[tid] = max(0, self.overspeed_streaks.get(tid, 0) - 1)

                    if violation_reason is not None:
                        self.violation_ids.add(tid)
                        anpr_res = self.anpr.recognize_plate(
                            raw_frame=raw_frame,
                            box_canvas=(v.x1, v.y1, v.x2, v.y2),
                            canvas_size=(canvas_frame.shape[1], canvas_frame.shape[0]),
                            cam_key=self.config.cam_key,
                            track_id=tid,
                            vehicle_type=v.vehicle_type,
                            min_plate_conf=self.config.min_plate_conf,
                        )

                        # Determine verification status based on evidence quality
                        is_high_conf = (
                            anpr_res.plate_status == "READ"
                            and mean_veh_conf >= self.config.min_vehicle_conf
                            and (is_calibrated or "Qua toc do" not in violation_reason)
                        )
                        ver_status = (
                            "TỰ ĐỘNG XÁC NHẬN" if is_high_conf else "CẦN KIỂM TRA THỦ CÔNG"
                        )
                        ver_code = "AUTO_CONFIRMED" if is_high_conf else "NEEDS_REVIEW"

                        self.violation_details[tid] = (
                            violation_reason,
                            anpr_res.plate_text,
                            anpr_res.plate_conf,
                            anpr_res.plate_status,
                            ver_status,
                        )

                        before_f = self.frame_ring_buffer[0] if len(self.frame_ring_buffer) > 0 else canvas_frame
                        after_f = canvas_frame
                        clip_frames = list(self.clip_ring_buffer) + [canvas_frame.copy()]

                        ev_path, pl_path, strip_path, clip_path, ev_hash = self.anpr.save_violation_evidence(
                            annotated_canvas=canvas_frame,
                            anpr_res=anpr_res,
                            cam_key=self.config.cam_key,
                            track_id=tid,
                            vehicle_type=v.vehicle_type,
                            speed_kmh=speed_kmh,
                            is_speed_calibrated=is_calibrated,
                            violation_label=violation_reason,
                            verification_status=ver_code,
                            trigger_reason=trigger_detail,
                            box_canvas=(v.x1, v.y1, v.x2, v.y2),
                            before_frame=before_f,
                            after_frame=after_f,
                            clip_frames=clip_frames,
                            clip_fps=max(8.0, self.fps / 3.0),
                        )

                        cfg_snap = json.dumps(
                            {
                                "cam": self.config.cam_key,
                                "speed_limit": self.config.speed_limit_kmh,
                                "calibrated": is_calibrated,
                                "direction_mode": self.config.direction_mode,
                                "min_obs": self.config.min_observation_frames,
                            },
                            ensure_ascii=False,
                        )

                        rec = self.db.insert_violation(
                            camera_id=f"CAM {self.config.cam_key}",
                            track_id=tid,
                            vehicle_type=v.vehicle_type,
                            plate_number=anpr_res.plate_text,
                            plate_conf=anpr_res.plate_conf,
                            plate_status=anpr_res.plate_status,
                            vehicle_conf=mean_veh_conf,
                            sign_conf=self.max_sign_conf if "106" in violation_reason else 0.0,
                            speed_kmh=speed_kmh,
                            is_speed_calibrated=is_calibrated,
                            violation_type=violation_reason,
                            verification_status=ver_status,
                            trigger_reason=trigger_detail,
                            model_version=MODEL_VERSION_TAG,
                            config_snapshot=cfg_snap,
                            image_path=ev_path,
                            plate_crop_path=pl_path,
                            timeline_path=strip_path,
                            clip_path=clip_path,
                            evidence_hash=ev_hash,
                        )
                        self.new_violations.append(rec)
                        logger.info(
                            f"[VIOLATION] {rec.camera_id} | {v.vehicle_type} #{tid} | "
                            f"Plate: {anpr_res.plate_text} ({anpr_res.plate_status}, conf={anpr_res.plate_conf:.2f}) | "
                            f"Status: {ver_status} | Reason: {trigger_detail}"
                        )

            is_violated = tid in self.violation_ids
            reasons: List[str] = []
            plate_num = ""
            plate_conf = 0.0
            plate_status = "UNREADABLE"
            ver_status = ""
            if is_violated and tid in self.violation_details:
                reason_str, plate_num, plate_conf, plate_status, ver_status = self.violation_details[tid]
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
                    conf=v.conf,
                    speed_kmh=speed_kmh,
                    is_speed_calibrated=is_calibrated,
                    direction_label=direction_str,
                    is_violated=is_violated,
                    verification_status=ver_status,
                    violation_reasons=reasons,
                    plate_number=plate_num,
                    plate_conf=plate_conf,
                    plate_status=plate_status,
                    trail=trail_pts,
                )
            )

        frame_copy = canvas_frame.copy()
        self.frame_ring_buffer.append(frame_copy)
        self.clip_ring_buffer.append(frame_copy)
        self.vehicles_in_roi = vehicles_in_roi
        self.traffic_status = (
            "CONGESTED" if vehicles_in_roi >= self.config.congestion_threshold else "NORMAL"
        )
        self.avg_speed_kmh = float(np.mean(active_speeds)) if active_speeds else 0.0
        self.last_enriched_vehicles = enriched_list
        return enriched_list
