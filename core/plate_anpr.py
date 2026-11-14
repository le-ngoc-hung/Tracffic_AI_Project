from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
import re
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from core.logger import get_app_logger

logger = get_app_logger("ANPR")


@dataclass
class ANPRResult:
    """Structured result from the ANPR pipeline without any synthetic fallback."""
    plate_text: str            # Actual OCR string or "KHÔNG ĐỌC ĐƯỢC"
    plate_conf: float          # Confidence in [0.0, 1.0]
    plate_status: str          # "READ" | "LOW_CONFIDENCE" | "UNREADABLE"
    vehicle_crop: np.ndarray   # High-res crop of the vehicle
    plate_crop: np.ndarray     # High-res crop of the plate region with OCR status strip


# Valid license plate regex:
# Matches Vietnamese 1-line/2-line plates (e.g. 51C-482.19, 59-S2 123.45, 30A12345)
# and international plates present in highway test videos (e.g. LS18HFM, AB12CDE)
PLATE_ALNUM_PATTERN = re.compile(r"^[A-Z0-9\-\.]{4,12}$")


class LicensePlateANPR:
    """
    Honest Automatic Number Plate Recognition (ANPR) using RapidOCR (PaddleOCR ONNX).
    - Never fabricates or guesses fake plate numbers.
    - Reports exact OCR confidence and marks unreadable/blurry plates as 'KHÔNG ĐỌC ĐƯỢC'.
    - Generates multi-frame Before/During/After composite evidence reports.
    """

    def __init__(self, output_dir: str = "violations"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ocr_engine = RapidOCR()
        self.plate_detector = self._load_optional_plate_detector()
        # Cache genuine readings and multi-frame weighted votes per (cam_key, track_id)
        self._best_reads: Dict[Tuple[str, int], Tuple[str, float, str]] = {}
        self._plate_votes: Dict[Tuple[str, int], Dict[str, float]] = {}

    @staticmethod
    def _load_optional_plate_detector():
        """Loads a dedicated YOLO plate detector if present in models/ (optional real-world upgrade)."""
        for candidate in ("models/plate_model.onnx", "models/plate_model.pt"):
            if os.path.exists(candidate):
                try:
                    from ultralytics import YOLO

                    logger.info("Đã kích hoạt model detect biển số chuyên dụng: %s", candidate)
                    return YOLO(candidate)
                except Exception as exc:
                    logger.warning("Không thể nạp %s (%s), dùng bộ dò Sobel.", candidate, exc)
        return None

    @staticmethod
    def compute_image_sharpness(img: np.ndarray) -> float:
        if img is None or img.size == 0:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_file_sha256(file_path: str) -> str:
        if not file_path or not os.path.exists(file_path):
            return ""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def deskew_plate_crop(plate_crop: np.ndarray) -> np.ndarray:
        """Corrects slight camera tilt angle (+/- 15 deg) on cropped license plates."""
        if plate_crop is None or plate_crop.shape[0] < 14 or plate_crop.shape[1] < 24:
            return plate_crop
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=25, minLineLength=max(16, plate_crop.shape[1] // 3), maxLineGap=6)
        if lines is None:
            return plate_crop
        angles = []
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4)[:12]:
            dx = int(x2) - int(x1)
            dy = int(y2) - int(y1)
            if abs(dx) > 8:
                ang = float(np.degrees(np.arctan2(dy, dx)))
                if -18.0 <= ang <= 18.0:
                    angles.append(ang)
        if not angles:
            return plate_crop
        median_angle = float(np.median(angles))
        if abs(median_angle) < 2.0:
            return plate_crop
        h, w = plate_crop.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), median_angle, 1.0)
        return cv2.warpAffine(plate_crop, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def locate_license_plate(
        self, vehicle_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Locates the candidate license plate bounding box inside a high-res vehicle crop
        using dedicated YOLO plate detector (if present) or Sobel vertical edge density.
        """
        vh, vw = vehicle_bgr.shape[:2]
        if vh < 20 or vw < 20:
            return vehicle_bgr.copy(), (0, 0, vw, vh)

        if self.plate_detector is not None:
            try:
                det_res = self.plate_detector.predict(vehicle_bgr, conf=0.28, verbose=False)
                if det_res and len(det_res[0].boxes) > 0:
                    best_det = max(det_res[0].boxes, key=lambda b: float(b.conf[0]))
                    dx1, dy1, dx2, dy2 = map(int, best_det.xyxy[0])
                    dx1, dy1 = max(0, dx1), max(0, dy1)
                    dx2, dy2 = min(vw, dx2), min(vh, dy2)
                    if (dx2 - dx1) >= 14 and (dy2 - dy1) >= 7:
                        return vehicle_bgr[dy1:dy2, dx1:dx2].copy(), (dx1, dy1, dx2, dy2)
            except Exception as exc:
                logger.debug("Lỗi plate_detector fallback về Sobel: %s", exc)

        # Search in the lower 60% of the vehicle (covers both front/rear bumpers & motorbike tail plates)
        y_start = int(vh * 0.40)
        x_start = int(vw * 0.12)
        x_end = int(vw * 0.88)
        roi = vehicle_bgr[y_start:, x_start:x_end]
        rh, rw = roi.shape[:2]
        if rh < 10 or rw < 16:
            return vehicle_bgr.copy(), (0, 0, vw, vh)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, rect_kernel)

        sobel_x = cv2.Sobel(tophat, cv2.CV_32F, 1, 0, ksize=3)
        sobel_x = np.absolute(sobel_x)
        max_val = np.max(sobel_x)
        if max_val > 0:
            sobel_x = np.uint8(255 * (sobel_x / max_val))
        else:
            sobel_x = np.zeros_like(gray_eq)

        blurred = cv2.GaussianBlur(sobel_x, (5, 5), 0)
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, rect_kernel)
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        best_score = -1.0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(max(h, 1))
            area_ratio = (w * h) / float(max(rw * rh, 1))
            if 1.2 <= aspect <= 5.5 and 0.015 <= area_ratio <= 0.50 and w >= 16 and h >= 7:
                center_dist = abs((x + w / 2.0) - (rw / 2.0)) / float(max(rw, 1))
                patch_gray = gray_eq[y : y + h, x : x + w]
                contrast = float(np.std(patch_gray)) if patch_gray.size > 0 else 0.0
                score = contrast * (1.0 - 0.65 * center_dist)
                if score > best_score:
                    best_score = score
                    best_box = (x + x_start, y + y_start, x + x_start + w, y + y_start + h)

        if best_box is None:
            px1 = int(vw * 0.28)
            px2 = int(vw * 0.74)
            py1 = int(vh * 0.68)
            py2 = int(vh * 0.94)
            best_box = (px1, py1, px2, py2)
        else:
            px1, py1, px2, py2 = best_box
            pad_x = max(5, int((px2 - px1) * 0.16))
            pad_y = max(4, int((py2 - py1) * 0.22))
            px1 = max(0, px1 - pad_x)
            py1 = max(0, py1 - pad_y)
            px2 = min(vw, px2 + pad_x)
            py2 = min(vh, py2 + pad_y)
            best_box = (px1, py1, px2, py2)

        px1, py1, px2, py2 = best_box
        plate_crop = vehicle_bgr[py1:py2, px1:px2].copy()
        return plate_crop, best_box

    def _enhance_plate_crop(self, plate_crop: np.ndarray) -> np.ndarray:
        """Upscales and sharpens the license plate patch for OCR and visual inspection."""
        if plate_crop is None or plate_crop.size == 0:
            return np.zeros((60, 200, 3), dtype=np.uint8)
        upscaled = cv2.resize(plate_crop, (240, 76), interpolation=cv2.INTER_LANCZOS4)
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 1.8)
        sharpened = cv2.addWeighted(upscaled, 1.55, gaussian, -0.55, 0)
        return sharpened

    @staticmethod
    def clean_and_validate_plate_text(raw_text: str) -> Optional[str]:
        """
        Cleans raw OCR output and validates if it represents a plausible license plate.
        Returns formatted plate string or None if invalid/noise.
        """
        if not raw_text:
            return None
        # Keep only uppercase ASCII letters, digits, hyphen, dot
        cleaned = re.sub(r"[^A-Z0-9\-\.]", "", raw_text.upper())
        cleaned = cleaned.strip("-.")
        if not PLATE_ALNUM_PATTERN.match(cleaned):
            return None

        alnum_only = re.sub(r"[^A-Z0-9]", "", cleaned)
        if len(alnum_only) < 5 or len(alnum_only) > 10:
            return None

        has_digit = any(c.isdigit() for c in alnum_only)
        has_alpha = any(c.isalpha() for c in alnum_only)
        if not (has_digit and has_alpha):
            return None

        # Format standard Vietnamese 1-line/2-line plate if matching e.g. 51C48219 -> 51C-482.19
        vn_match = re.match(r"^(\d{2}[A-Z]{1,2}\d?)(\d{3})(\d{2})$", alnum_only)
        if vn_match:
            p1, p2, p3 = vn_match.groups()
            if len(p1) == 4 and p1[3].isdigit():
                return f"{p1[:2]}-{p1[2:4]} {p2}.{p3}"
            return f"{p1}-{p2}.{p3}"

        vn_4digit = re.match(r"^(\d{2}[A-Z]{1,2}\d?)(\d{4})$", alnum_only)
        if vn_4digit:
            p1, p2 = vn_4digit.groups()
            return f"{p1}-{p2}"

        return cleaned

    def run_ocr_on_plate_image(
        self, plate_bgr: np.ndarray, min_conf_threshold: float = 0.65
    ) -> Tuple[str, float, str]:
        """
        Runs RapidOCR on a plate image (supports both single-line and two-line plates).
        Returns (plate_text, confidence, status) where status is READ, LOW_CONFIDENCE, or UNREADABLE.
        Never fabricates a plate number.
        """
        if plate_bgr is None or plate_bgr.size == 0:
            return "KHÔNG ĐỌC ĐƯỢC", 0.0, "UNREADABLE"

        candidates: List[Tuple[str, float]] = []

        # Pass 1: Run on original/deskewed/sharpened image with detection + recognition
        deskewed = self.deskew_plate_crop(plate_bgr)
        variants = (plate_bgr, self._enhance_plate_crop(plate_bgr))
        if deskewed is not plate_bgr:
            variants = (plate_bgr, deskewed, self._enhance_plate_crop(deskewed))
        for variant in variants:
            try:
                res, _ = self.ocr_engine(variant)
            except Exception as e:
                logger.debug(f"RapidOCR exception: {e}")
                res = None

            if res:
                # Sort detected text boxes top-to-bottom, then left-to-right (handles 2-line VN plates)
                sorted_boxes = sorted(
                    res,
                    key=lambda item: (
                        round((item[0][0][1] + item[0][2][1]) / 50.0),
                        item[0][0][0],
                    ),
                )
                texts = [str(item[1]) for item in sorted_boxes]
                confs = [float(item[2]) for item in sorted_boxes]
                combined_raw = "".join(texts)
                mean_conf = float(np.mean(confs)) if confs else 0.0
                valid_plate = self.clean_and_validate_plate_text(combined_raw)
                if valid_plate:
                    candidates.append((valid_plate, mean_conf))

        # Pass 2: Direct recognition mode (use_det=False) for tightly cropped 1-line plates
        try:
            res_nodet, _ = self.ocr_engine(plate_bgr, use_det=False, use_cls=False, use_rec=True)
            if res_nodet:
                for item in res_nodet:
                    txt, score = str(item[0]), float(item[1])
                    valid_plate = self.clean_and_validate_plate_text(txt)
                    if valid_plate:
                        candidates.append((valid_plate, score))
        except Exception:
            pass

        if not candidates:
            return "KHÔNG ĐỌC ĐƯỢC", 0.0, "UNREADABLE"

        # Pick the candidate with the highest confidence
        best_plate, best_conf = max(candidates, key=lambda x: x[1])
        if best_conf >= min_conf_threshold:
            return best_plate, round(best_conf, 3), "READ"
        elif best_conf >= 0.35:
            return best_plate, round(best_conf, 3), "LOW_CONFIDENCE"
        else:
            return "KHÔNG ĐỌC ĐƯỢC", round(best_conf, 3), "UNREADABLE"

    def recognize_plate(
        self,
        raw_frame: np.ndarray,
        box_canvas: Tuple[int, int, int, int],
        canvas_size: Tuple[int, int],
        cam_key: str,
        track_id: int,
        vehicle_type: str,
        min_plate_conf: float = 0.65,
    ) -> ANPRResult:
        """
        Extracts high-resolution vehicle & license plate crops from the Full HD frame
        and runs real RapidOCR. Never returns synthetic plate strings.
        """
        raw_h, raw_w = raw_frame.shape[:2]
        can_w, can_h = canvas_size
        scale_x = raw_w / float(can_w)
        scale_y = raw_h / float(can_h)

        x1, y1, x2, y2 = box_canvas
        rx1 = max(0, int(x1 * scale_x))
        ry1 = max(0, int(y1 * scale_y))
        rx2 = min(raw_w, int(x2 * scale_x))
        ry2 = min(raw_h, int(y2 * scale_y))

        vehicle_crop = raw_frame[ry1:ry2, rx1:rx2].copy()
        if vehicle_crop.size == 0:
            vehicle_crop = np.zeros((120, 160, 3), dtype=np.uint8)

        raw_plate_crop, (px1, py1, px2, py2) = self.locate_license_plate(vehicle_crop)

        # Try OCR on the localized plate crop first; if unreadable, also try the lower half of vehicle_crop
        plate_text, plate_conf, plate_status = self.run_ocr_on_plate_image(
            raw_plate_crop, min_conf_threshold=min_plate_conf
        )
        if plate_status == "UNREADABLE" and vehicle_crop.shape[0] >= 40:
            lower_bumper = vehicle_crop[int(vehicle_crop.shape[0] * 0.50) :, :]
            alt_text, alt_conf, alt_status = self.run_ocr_on_plate_image(
                lower_bumper, min_conf_threshold=min_plate_conf
            )
            if alt_status != "UNREADABLE" and alt_conf > plate_conf:
                plate_text, plate_conf, plate_status = alt_text, alt_conf, alt_status

        # Multi-frame weighted voting + sharpness-aware cache across frames
        cache_key = (cam_key, track_id)
        if plate_status != "UNREADABLE":
            sharpness = self.compute_image_sharpness(raw_plate_crop)
            sharp_weight = 1.0 + min(0.4, sharpness / 450.0)
            votes = self._plate_votes.setdefault(cache_key, {})
            votes[plate_text] = votes.get(plate_text, 0.0) + float(plate_conf) * sharp_weight

        if cache_key in self._best_reads:
            prev_text, prev_conf, prev_status = self._best_reads[cache_key]
            votes = self._plate_votes.get(cache_key, {})
            voted_top = max(votes.items(), key=lambda kv: kv[1])[0] if votes else prev_text
            if plate_status != "UNREADABLE" and (plate_text == voted_top or plate_conf > prev_conf):
                best_conf = max(prev_conf, plate_conf) if plate_text == prev_text else plate_conf
                best_status = "READ" if best_conf >= min_plate_conf else plate_status
                self._best_reads[cache_key] = (plate_text, best_conf, best_status)
                plate_text, plate_conf, plate_status = self._best_reads[cache_key]
            elif prev_status != "UNREADABLE":
                plate_text, plate_conf, plate_status = prev_text, prev_conf, prev_status
        elif plate_status != "UNREADABLE":
            self._best_reads[cache_key] = (plate_text, plate_conf, plate_status)

        box_color = (0, 255, 0) if plate_status == "READ" else (0, 165, 255)
        cv2.rectangle(vehicle_crop, (px1, py1), (px2, py2), box_color, 2)

        badge_plate_crop = self._overlay_ocr_badge(raw_plate_crop, plate_text, plate_conf, plate_status)
        return ANPRResult(
            plate_text=plate_text,
            plate_conf=plate_conf,
            plate_status=plate_status,
            vehicle_crop=vehicle_crop,
            plate_crop=badge_plate_crop,
        )

    def _overlay_ocr_badge(
        self, plate_crop: np.ndarray, plate_text: str, plate_conf: float, plate_status: str
    ) -> np.ndarray:
        """Appends an honest OCR status & confidence strip below the optical plate crop."""
        optical = self._enhance_plate_crop(plate_crop)
        optical = cv2.resize(optical, (220, 48), interpolation=cv2.INTER_LANCZOS4)
        badge = np.full((24, 220, 3), (35, 38, 46), dtype=np.uint8)
        cv2.rectangle(badge, (0, 0), (219, 23), (90, 95, 110), 1)

        if plate_status == "UNREADABLE":
            display_str = "ANPR: KHONG DOC DUOC"
            txt_color = (100, 140, 255)
        else:
            display_str = f"{plate_text} ({plate_conf*100:.0f}%)"
            txt_color = (80, 255, 120) if plate_status == "READ" else (0, 215, 255)

        cv2.putText(
            badge,
            display_str,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            txt_color,
            1,
            cv2.LINE_AA,
        )
        return np.vstack([optical, badge])

    def save_violation_evidence(
        self,
        annotated_canvas: np.ndarray,
        anpr_res: ANPRResult,
        cam_key: str,
        track_id: int,
        vehicle_type: str,
        speed_kmh: float,
        is_speed_calibrated: bool,
        violation_label: str,
        verification_status: str,
        trigger_reason: str,
        box_canvas: Tuple[int, int, int, int],
        before_frame: Optional[np.ndarray] = None,
        after_frame: Optional[np.ndarray] = None,
        clip_frames: Optional[List[np.ndarray]] = None,
        clip_fps: float = 12.0,
    ) -> Tuple[str, str, str, str, str]:
        """
        Saves:
        1. Composite Violation Report image (Scene + HD crop + ANPR crop + Audit metadata)
        2. Cropped plate image
        3. Before/During/After 3-stage progression strip image
        4. Short MP4 evidence clip (when clip_frames is available)
        5. Computes SHA-256 integrity hash of the saved report image
        Returns (evidence_path, plate_path, strip_path, clip_path, evidence_hash).
        """
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        display_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base_name = f"cam{cam_key}_{vehicle_type.lower()}_id{track_id}_{time_str}"

        evidence_path = os.path.join(self.output_dir, f"{base_name}.jpg")
        plate_path = os.path.join(self.output_dir, f"{base_name}_plate.jpg")
        strip_path = os.path.join(self.output_dir, f"{base_name}_timeline.jpg")
        clip_path = ""

        # Export short MP4 evidence clip if buffered frames exist
        if clip_frames and len(clip_frames) >= 2:
            candidate_clip = os.path.join(self.output_dir, f"{base_name}_clip.mp4")
            try:
                ch, cw = clip_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(candidate_clip, fourcc, max(4.0, min(clip_fps, 30.0)), (cw, ch))
                if writer.isOpened():
                    for cf in clip_frames:
                        if cf.shape[0] != ch or cf.shape[1] != cw:
                            cf = cv2.resize(cf, (cw, ch))
                        writer.write(cf)
                    writer.release()
                    clip_path = candidate_clip
                else:
                    writer.release()
            except Exception as exc:
                logger.warning("Không thể xuất video clip bằng chứng: %s", exc)

        scene = annotated_canvas.copy()
        x1, y1, x2, y2 = box_canvas
        cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 0, 255), 3)

        ascii_plate = "UNREADABLE" if anpr_res.plate_status == "UNREADABLE" else anpr_res.plate_text
        speed_tag = f"{speed_kmh:.0f}km/h" if is_speed_calibrated else f"~{speed_kmh:.0f}km/h(Est)"
        cv2.putText(
            scene,
            f"VIOLATION #{track_id} | {ascii_plate} | {speed_tag}",
            (max(10, x1), max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (0, 0, 255),
            2,
        )

        # Build composite report (854 + 346 = 1200 x 480)
        sh, sw = scene.shape[:2]
        panel_w = 346
        composite = np.full((sh, sw + panel_w, 3), (24, 26, 34), dtype=np.uint8)
        composite[:, :sw] = scene

        header_color = (20, 110, 40) if verification_status == "AUTO_CONFIRMED" else (20, 75, 175)
        cv2.rectangle(composite, (sw, 0), (sw + panel_w, 42), header_color, -1)
        cv2.putText(
            composite,
            f"HO SO VI PHAM ({verification_status})",
            (sw + 12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
        )

        # Vehicle crop thumbnail (316 x 150)
        veh_thumb = cv2.resize(anpr_res.vehicle_crop, (316, 150), interpolation=cv2.INTER_AREA)
        composite[52 : 52 + 150, sw + 15 : sw + 15 + 316] = veh_thumb
        cv2.rectangle(composite, (sw + 15, 52), (sw + 331, 202), (100, 180, 255), 1)

        # License Plate crop thumbnail (220 x 72)
        plate_thumb = cv2.resize(anpr_res.plate_crop, (220, 72), interpolation=cv2.INTER_LANCZOS4)
        px_offset = sw + 63
        composite[214 : 214 + 72, px_offset : px_offset + 220] = plate_thumb
        cv2.rectangle(composite, (px_offset, 214), (px_offset + 220, 286), (0, 255, 255), 2)

        # Audit metadata lines
        plate_line = (
            "Bien so: KHONG DOC DUOC (Can KT thu cong)"
            if anpr_res.plate_status == "UNREADABLE"
            else f"Bien so: {anpr_res.plate_text} (Conf: {anpr_res.plate_conf*100:.0f}%)"
        )
        speed_line = (
            f"Toc do: {speed_kmh:.1f} km/h (Da hieu chuan)"
            if is_speed_calibrated
            else f"Toc do: ~{speed_kmh:.1f} km/h (Uoc tinh)"
        )
        info_lines = [
            (plate_line, (0, 255, 255) if anpr_res.plate_status == "READ" else (120, 180, 255), 0.46, 1),
            (f"Vi pham: {violation_label}", (80, 120, 255), 0.46, 1),
            (f"Trang thai: {verification_status}", (120, 255, 120) if verification_status == "AUTO_CONFIRMED" else (0, 210, 255), 0.45, 1),
            (f"Phuong tien: {vehicle_type} (Track #{track_id})", (230, 230, 230), 0.44, 1),
            (speed_line, (200, 230, 200), 0.44, 1),
            (f"Ly do: {trigger_reason[:44]}", (180, 180, 195), 0.40, 1),
            (f"Cam {cam_key} | {display_time}", (150, 150, 165), 0.40, 1),
        ]

        y_txt = 310
        for text, color, scale, thickness in info_lines:
            cv2.putText(
                composite, text, (sw + 14, y_txt), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
            )
            y_txt += 23

        cv2.imwrite(evidence_path, composite)
        cv2.imwrite(plate_path, plate_thumb)
        evidence_hash = self.compute_file_sha256(evidence_path)

        # Save 3-frame Before / During / After timeline strip
        f_before = before_frame if before_frame is not None else annotated_canvas
        f_after = after_frame if after_frame is not None else scene
        t1 = cv2.resize(f_before, (380, 214))
        t2 = cv2.resize(scene, (380, 214))
        t3 = cv2.resize(f_after, (380, 214))
        for img_t, title in ((t1, "1. TRUOC SU KIEN (BEFORE)"), (t2, "2. THOI DIEM VI PHAM (EVENT)"), (t3, "3. SAU SU KIEN (AFTER)")):
            cv2.rectangle(img_t, (0, 0), (380, 26), (20, 20, 25), -1)
            cv2.putText(img_t, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        timeline_strip = np.hstack([t1, t2, t3])
        cv2.imwrite(strip_path, timeline_strip)

        return evidence_path, plate_path, strip_path, clip_path, evidence_hash
