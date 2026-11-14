from typing import Dict, Tuple
import cv2
import numpy as np

from core.config_manager import CameraConfig
from core.tracker_analytics import TrafficAnalyticsEngine


VEHICLE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Car": (0, 230, 64),         # Green
    "Motorbike": (255, 235, 0),  # Cyan-Yellow
    "Bus": (255, 64, 255),       # Magenta
    "Truck": (0, 165, 255),      # Orange
}


def render_annotated_canvas(
    base_canvas: np.ndarray,
    engine: TrafficAnalyticsEngine,
    config: CameraConfig,
    show_hud_overlay: bool = True,
) -> np.ndarray:
    """
    Renders ROI polygon, counting line, lane divider, trajectories, bounding boxes,
    calibrated/estimated speeds (km/h), honest ANPR status, and HUD onto the 854x480 canvas.
    """
    canvas = base_canvas.copy()
    overlay = canvas.copy()

    # 1. Draw translucent ROI Polygon
    roi_pts = np.array(config.roi_polygon, dtype=np.int32)
    is_congested = engine.traffic_status == "CONGESTED"
    roi_border_color = (0, 0, 255) if is_congested else (255, 160, 30)
    roi_fill_color = (0, 0, 180) if is_congested else (160, 90, 10)

    if len(roi_pts) >= 3:
        cv2.fillPoly(overlay, [roi_pts], roi_fill_color)
        cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0, canvas)
        cv2.polylines(canvas, [roi_pts], True, roi_border_color, 2, cv2.LINE_AA)

    # 2. Draw Lane Divider Line if 2-way mode is active
    if config.direction_mode in ("HIGHWAY_2WAY", "URBAN_2WAY") and len(config.divider_line) == 2:
        d1 = tuple(config.divider_line[0])
        d2 = tuple(config.divider_line[1])
        cv2.line(canvas, d1, d2, (0, 200, 255), 2, cv2.LINE_AA)
        mid_x = int((d1[0] + d2[0]) / 2)
        mid_y = int((d1[1] + d2[1]) / 2)
        cv2.putText(
            canvas,
            "Lane Divider",
            (mid_x - 42, mid_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. Draw Counting Line with endpoint handles
    if len(config.counting_line) == 2:
        p1 = tuple(config.counting_line[0])
        p2 = tuple(config.counting_line[1])
        cv2.line(canvas, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, p1, 6, (0, 180, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 6, (0, 180, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Counting Line",
            (p1[0] + 8, max(20, p1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 4. Draw Detected Traffic Signs with confidence
    for sign in engine.active_signs:
        cv2.rectangle(canvas, (sign.x1, sign.y1), (sign.x2, sign.y2), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"Sign: {sign.sign_code} ({sign.conf*100:.0f}%)",
            (max(10, sign.x1 - 35), max(18, sign.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # 5. Draw Vehicles, Trajectories, Calibrated/Estimated Speeds, and Violations
    for v in engine.last_enriched_vehicles:
        box_color = (0, 0, 255) if v.is_violated else VEHICLE_COLORS.get(v.vehicle_type, (0, 255, 0))

        if len(v.trail) >= 2:
            for i in range(1, len(v.trail)):
                cv2.line(canvas, v.trail[i - 1], v.trail[i], box_color, 2, cv2.LINE_AA)

        thickness = 3 if v.is_violated else 2
        cv2.rectangle(canvas, (v.x1, v.y1), (v.x2, v.y2), box_color, thickness)
        cv2.circle(canvas, (v.cx, v.cy), 3, (0, 0, 255), -1)

        if v.track_id < 7000:
            if v.speed_kmh >= 3.0:
                prefix = "" if v.is_speed_calibrated else "~"
                speed_tag = f" {prefix}{v.speed_kmh:.0f}km/h"
            else:
                speed_tag = ""
            label = f"#{v.track_id} {v.vehicle_type}{speed_tag}"
        else:
            label = f"{v.vehicle_type}"

        if v.is_violated:
            if v.plate_status in ("READ", "LOW_CONFIDENCE") and v.plate_number != "KHÔNG ĐỌC ĐƯỢC":
                label += f" [{v.plate_number}]"
            else:
                label += " [VIOLATION - Can KT]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        ly = max(v.y1 - 6, th + 6)
        cv2.rectangle(canvas, (v.x1, ly - th - 4), (v.x1 + tw + 6, ly + 2), (20, 20, 25), -1)
        cv2.putText(
            canvas,
            label,
            (v.x1 + 3, ly - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            box_color,
            1,
            cv2.LINE_AA,
        )

    # 6. Optional Compact HUD Pill on top-left of video (leaves road & vehicles unobstructed)
    if show_hud_overlay:
        st_color = (60, 80, 255) if is_congested else (80, 225, 110)
        cal_tag = "Calibrated" if config.is_speed_calibrated else "~Est"
        no_truck_on = config.enable_no_truck_rule and engine.rule_no_truck_detected
        line1 = (
            f"CAM {config.cam_key} | {engine.traffic_status} "
            f"(ROI: {engine.vehicles_in_roi}/{config.congestion_threshold}) | Viol: {len(engine.violation_ids)}"
        )
        line2 = f"P.106a: {'ON' if no_truck_on else 'OFF'} | Max: {config.speed_limit_kmh:.0f}km/h ({cal_tag})"

        (w1, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        (w2, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        box_w = min(base_canvas.shape[1] - 20, max(w1, w2) + 22)
        box_h = 48

        hud = canvas.copy()
        cv2.rectangle(hud, (10, 10), (10 + box_w, 10 + box_h), (16, 20, 28), -1)
        cv2.addWeighted(hud, 0.72, canvas, 0.28, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (10 + box_w, 10 + box_h), st_color, 1)

        cv2.putText(
            canvas,
            line1,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            st_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line2,
            (20, 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (215, 225, 240),
            1,
            cv2.LINE_AA,
        )

    return canvas
