import time
from typing import List, Optional
import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal

from core.config_manager import CameraConfig
from core.database import ViolationDatabase
from core.detector import UnifiedTrafficDetector
from core.health_monitor import cleanup_expired_evidence, inspect_system_health
from core.logger import get_app_logger
from core.plate_anpr import LicensePlateANPR
from core.renderer import render_annotated_canvas
from core.stream_reader import ResilientVideoStream
from core.tracker_analytics import TrafficAnalyticsEngine

logger = get_app_logger("InferenceWorker")


class VideoInferenceWorker(QThread):
    """
    Background QThread that decouples video decoding, YOLO multi-model inference,
    ByteTrack tracking, and RapidOCR ANPR from the PyQt6 UI thread.
    Ensures the GUI never freezes during heavy inference.
    """

    frame_ready = pyqtSignal(np.ndarray, np.ndarray, dict, list)  # (annotated_canvas, raw_canvas, telemetry, new_violations)
    stream_error = pyqtSignal(str)
    status_message = pyqtSignal(str)

    SKIP_FRAMES = 2

    def __init__(
        self,
        detector: UnifiedTrafficDetector,
        anpr: LicensePlateANPR,
        db: ViolationDatabase,
        initial_config: CameraConfig,
        parent=None,
    ):
        super().__init__(parent)
        self.detector = detector
        self.anpr = anpr
        self.db = db
        self._mutex = QMutex()

        self._running = True
        self._paused = False
        self._pending_config: Optional[CameraConfig] = initial_config
        self._reset_requested = False

        self.stream: Optional[ResilientVideoStream] = None
        self.engine: Optional[TrafficAnalyticsEngine] = None
        self.current_config: CameraConfig = initial_config
        self._last_health_check: float = 0.0

    def request_camera_switch(self, new_config: CameraConfig) -> None:
        with QMutexLocker(self._mutex):
            self._pending_config = new_config

    def request_state_reset(self) -> None:
        with QMutexLocker(self._mutex):
            self._reset_requested = True

    def set_paused(self, paused: bool) -> None:
        with QMutexLocker(self._mutex):
            self._paused = paused

    def stop_worker(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(8000)

    def _init_stream(self, cfg: CameraConfig) -> bool:
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass

        self.current_config = cfg
        self.stream = ResilientVideoStream(cfg.video_path)
        if not self.stream.open():
            msg = f"Không thể mở nguồn video/RTSP: '{cfg.video_path}'. Vui lòng kiểm tra kết nối hoặc đường dẫn."
            logger.error(msg)
            self.stream_error.emit(msg)
            return False

        fps = self.stream.fps
        self.detector.reset_tracker()
        self.engine = TrafficAnalyticsEngine(cfg, self.anpr, self.db, fps=fps)
        kind_tag = "LIVE STREAM" if self.stream.is_live else "VIDEO FILE"
        info = f"--> Đã nạp thành công [{kind_tag}]: {cfg.name} ({cfg.video_path} | {fps:.0f} FPS)"
        logger.info(info)
        self.status_message.emit(info)
        return True

    def run(self) -> None:
        stream_ok = False
        try:
            cleanup_expired_evidence(self.db.db_path, self.anpr.output_dir)
        except Exception as exc:
            logger.debug("Retention startup check skipped: %s", exc)

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
                pending_cfg = self._pending_config
                self._pending_config = None
                do_reset = self._reset_requested
                self._reset_requested = False
                is_paused = self._paused

            if pending_cfg is not None:
                stream_ok = self._init_stream(pending_cfg)

            if do_reset and self.engine is not None:
                self.detector.reset_tracker()
                self.engine.reset_state()
                logger.info(f"Đã reset trạng thái bộ đếm & tracker cho {self.current_config.name}")

            if not stream_ok or is_paused or self.stream is None or self.engine is None:
                time.sleep(0.03)
                continue

            try:
                ret, raw_frame, frame_ts, did_loop = self.stream.read()
                if did_loop:
                    self.detector.reset_tracker()
                    self.engine.reset_state()
                    logger.info(f"Video '{self.current_config.video_path}' kết thúc, tự động phát lặp lại và reset tracker.")
                    continue
                if not ret or raw_frame is None:
                    if self.stream.is_live and self.stream.health_state == "RECONNECTING":
                        self.status_message.emit(
                            f"⚠️ Mất kết nối RTSP '{self.current_config.name}', đang thử kết nối lại (lần #{self.stream.reconnect_attempts})..."
                        )
                    time.sleep(0.02)
                    continue

                self.engine.frame_idx += 1
                canvas_frame = cv2.resize(raw_frame, (854, 480))

                if self.engine.frame_idx % 20 == 0 or self.engine.frame_idx == 1:
                    signs = self.detector.detect_signs(canvas_frame, conf=0.18)
                    self.engine.update_signs(signs)

                new_viols: List = []
                if self.engine.frame_idx % (self.SKIP_FRAMES + 1) == 0 or self.engine.frame_idx == 1:
                    vehicles = self.detector.detect_and_track_vehicles(canvas_frame)
                    self.engine.process_detections(vehicles, raw_frame, canvas_frame, frame_time_sec=frame_ts)
                    new_viols = list(self.engine.new_violations)

                now = time.monotonic()
                if now - self._last_health_check >= 60.0:
                    self._last_health_check = now
                    health = inspect_system_health(self.anpr.output_dir)
                    if health.low_disk_warning:
                        self.status_message.emit(
                            f"⚠️ Cảnh báo ổ đĩa còn trống {health.disk_free_gb:.1f} GB (Bằng chứng: {health.evidence_dir_mb:.0f} MB)"
                        )

                annotated = render_annotated_canvas(canvas_frame, self.engine, self.engine.config)
                telemetry = {
                    "traffic_status": self.engine.traffic_status,
                    "vehicles_in_roi": self.engine.vehicles_in_roi,
                    "congestion_threshold": self.engine.config.congestion_threshold,
                    "counts": dict(self.engine.counts),
                    "avg_speed_kmh": self.engine.avg_speed_kmh,
                    "is_speed_calibrated": self.engine.config.is_speed_calibrated,
                    "total_violations": len(self.engine.violation_ids),
                    "stream_health": self.stream.health_state,
                    "is_live": self.stream.is_live,
                }
                self.frame_ready.emit(annotated, canvas_frame, telemetry, new_viols)
                time.sleep(0.012)

            except Exception as e:
                logger.exception(f"Lỗi trong luồng xử lý AI: {e}")
                self.stream_error.emit(f"Lỗi xử lý khung hình: {e}")
                time.sleep(0.2)

        if self.stream is not None:
            self.stream.close()
