"""
Bộ đọc luồng video bền bỉ cho thực tế:
  - Hỗ trợ cả file video cục bộ (.mp4, .avi, .mkv) lẫn camera IP live (RTSP/HTTP/Webcam).
  - Với luồng RTSP/HTTP/Webcam: chạy luồng nền liên tục xả buffer cũ để giữ độ trễ thấp,
    tự động kết nối lại theo exponential backoff khi rớt mạng.
  - Trả về timestamp thực của khung hình để tính tốc độ chính xác ngay cả khi FPS dao động.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from core.logger import get_app_logger

logger = get_app_logger("StreamReader")

LIVE_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://", "rtmp://")


def is_live_stream(source: str) -> bool:
    clean = (source or "").strip().lower()
    if clean.isdigit():
        return True
    return clean.startswith(LIVE_PREFIXES)


def parse_video_source(source: str) -> Union[int, str]:
    clean = (source or "").strip()
    if clean.isdigit():
        return int(clean)
    return clean


class ResilientVideoStream:
    """
    Wrapper thống nhất cho VideoCapture cục bộ & RTSP/HTTP/Webcam thời gian thực.
    """

    def __init__(self, source: str, reconnect_max_sec: float = 15.0):
        self.source = (source or "").strip()
        self.is_live = is_live_stream(self.source)
        self.reconnect_max_sec = float(reconnect_max_sec)

        self._cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.health_state: str = "INIT"  # ONLINE | RECONNECTING | OFFLINE
        self.reconnect_attempts: int = 0
        self._file_frame_idx: int = 0

        # Background latest-frame grabber state (live RTSP/HTTP/Webcam only)
        self._lock = threading.Lock()
        self._reader_running = False
        self._reader_thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._latest_seq: int = 0
        self._consumed_seq: int = 0

    def open(self) -> bool:
        self.close()
        parsed = parse_video_source(self.source)

        if self.is_live and isinstance(parsed, str) and parsed.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

        cap = cv2.VideoCapture(parsed)
        if self.is_live:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        if not cap.isOpened():
            cap.release()
            self.health_state = "OFFLINE"
            return False

        raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if raw_fps <= 1.0 or raw_fps > 120.0:
            raw_fps = 30.0
        self.fps = float(raw_fps)
        self._cap = cap
        self._file_frame_idx = 0
        self.health_state = "ONLINE"
        self.reconnect_attempts = 0

        if self.is_live:
            self._start_live_reader()
        return True

    def _start_live_reader(self) -> None:
        self._reader_running = True
        self._reader_thread = threading.Thread(target=self._live_loop, daemon=True)
        self._reader_thread.start()

    def _live_loop(self) -> None:
        backoff = 1.0
        while self._reader_running:
            cap = self._cap
            if cap is None or not cap.isOpened():
                self.health_state = "RECONNECTING"
                self.reconnect_attempts += 1
                logger.warning(
                    "Mất tín hiệu live stream '%s', thử kết nối lại (lần #%d sau %.1fs)...",
                    self.source,
                    self.reconnect_attempts,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(self.reconnect_max_sec, backoff * 1.8)
                new_cap = cv2.VideoCapture(parse_video_source(self.source))
                try:
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if new_cap.isOpened():
                    with self._lock:
                        if self._cap is not None:
                            self._cap.release()
                        self._cap = new_cap
                        self.health_state = "ONLINE"
                    backoff = 1.0
                    logger.info("Đã khôi phục kết nối live stream: %s", self.source)
                else:
                    new_cap.release()
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                self._cap = None
                continue

            backoff = 1.0
            with self._lock:
                self._latest_frame = frame
                self._latest_ts = time.monotonic()
                self._latest_seq += 1
                self.health_state = "ONLINE"

    def read(self) -> Tuple[bool, Optional[np.ndarray], float, bool]:
        """
        Trả về: (thành_công, khung_hình_BGR, timestamp_giây, đã_tua_lại_file)
        """
        if not self.is_live:
            if self._cap is None:
                return False, None, 0.0, False
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Hết file video cục bộ -> tua lại từ đầu
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._file_frame_idx = 0
                return False, None, 0.0, True
            self._file_frame_idx += 1
            ts = self._file_frame_idx / max(self.fps, 1.0)
            return True, frame, ts, False

        # Live RTSP/HTTP/Webcam: trả frame mới nhất chưa xử lý
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_seq > self._consumed_seq and self._latest_frame is not None:
                    self._consumed_seq = self._latest_seq
                    return True, self._latest_frame.copy(), self._latest_ts, False
            time.sleep(0.008)
        return False, None, 0.0, False

    def close(self) -> None:
        self._reader_running = False
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None

        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._latest_frame = None
