from typing import List, Optional, Tuple
import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class InteractiveVideoCanvas(QLabel):
    """
    Custom QLabel widget that renders OpenCV video frames and supports interactive
    mouse drawing of:
    - Counting Line (drag left mouse from point A to point B)
    - ROI Polygon (left-click vertices, right-click to finish polygon)
    - Lane Divider Line (drag left mouse from point A to point B)
    Coordinates are automatically mapped between widget pixel space and 854x480 canvas space.
    """

    line_updated = pyqtSignal(list)      # Emits [[x1, y1], [x2, y2]]
    roi_updated = pyqtSignal(list)       # Emits [[x1, y1], [x2, y2], ...]
    divider_updated = pyqtSignal(list)   # Emits [[x1, y1], [x2, y2]]
    mode_finished = pyqtSignal(str)      # Emits status message when drawing finishes

    CANVAS_W = 854
    CANVAS_H = 480

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(720, 405)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background-color: #0d1117; border: 2px solid #30363d; border-radius: 8px;"
        )
        self.setMouseTracking(True)

        self.draw_mode: str = "NONE"  # "NONE" | "DRAW_LINE" | "DRAW_ROI" | "DRAW_DIVIDER"
        self._last_bgr_frame: Optional[np.ndarray] = None

        # Temporary interactive drawing state
        self._drag_start: Optional[Tuple[int, int]] = None
        self._current_mouse: Optional[Tuple[int, int]] = None
        self._temp_roi_points: List[List[int]] = []

    def set_draw_mode(self, mode: str) -> None:
        self.draw_mode = mode
        self._drag_start = None
        self._temp_roi_points.clear()
        if mode == "NONE":
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
        self._refresh_display()

    def update_frame(self, bgr_frame: np.ndarray) -> None:
        self._last_bgr_frame = bgr_frame.copy()
        self._refresh_display()

    def _refresh_display(self) -> None:
        if self._last_bgr_frame is None:
            return

        display = self._last_bgr_frame.copy()

        # Draw interactive overlays while user is actively drawing with mouse
        if self.draw_mode in ("DRAW_LINE", "DRAW_DIVIDER"):
            banner = (
                "CHE DO VE VACH DEM: Keo tha chuot trai tu diem dau den diem cuoi"
                if self.draw_mode == "DRAW_LINE"
                else "CHE DO VE PHAN LAN: Keo tha chuot trai doc theo vach phan cach"
            )
            cv2.rectangle(display, (140, 8), (740, 38), (15, 80, 180), -1)
            cv2.putText(
                display, banner, (152, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
            )
            if self._drag_start and self._current_mouse:
                color = (0, 255, 255) if self.draw_mode == "DRAW_LINE" else (0, 200, 255)
                cv2.line(display, self._drag_start, self._current_mouse, color, 3, cv2.LINE_AA)
                cv2.circle(display, self._drag_start, 6, (0, 0, 255), -1)
                cv2.circle(display, self._current_mouse, 6, (0, 255, 0), -1)

        elif self.draw_mode == "DRAW_ROI":
            banner = f"CHE DO VE VUNG ROI: Click chuot trai chon dinh ({len(self._temp_roi_points)} dinh) | Click CHUOT PHAI de hoan tat"
            cv2.rectangle(display, (85, 8), (795, 38), (140, 40, 140), -1)
            cv2.putText(
                display, banner, (95, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
            pts = [tuple(p) for p in self._temp_roi_points]
            for pt in pts:
                cv2.circle(display, pt, 5, (0, 255, 255), -1)
            for i in range(1, len(pts)):
                cv2.line(display, pts[i - 1], pts[i], (255, 100, 255), 2, cv2.LINE_AA)
            if len(pts) >= 1 and self._current_mouse:
                cv2.line(display, pts[-1], self._current_mouse, (200, 200, 200), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def _map_widget_to_canvas(self, wx: int, wy: int) -> Optional[Tuple[int, int]]:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None

        pw = pixmap.width()
        ph = pixmap.height()
        if pw <= 0 or ph <= 0:
            return None

        offset_x = (self.width() - pw) / 2.0
        offset_y = (self.height() - ph) / 2.0

        rel_x = wx - offset_x
        rel_y = wy - offset_y

        cx = int(np.clip((rel_x / pw) * self.CANVAS_W, 5, self.CANVAS_W - 5))
        cy = int(np.clip((rel_y / ph) * self.CANVAS_H, 5, self.CANVAS_H - 5))
        return cx, cy

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.draw_mode == "NONE":
            return

        mapped = self._map_widget_to_canvas(int(event.position().x()), int(event.position().y()))
        if mapped is None:
            return

        if self.draw_mode in ("DRAW_LINE", "DRAW_DIVIDER"):
            if event.button() == Qt.MouseButton.LeftButton:
                self._drag_start = mapped
                self._current_mouse = mapped
                self._refresh_display()
            elif event.button() == Qt.MouseButton.RightButton:
                self.set_draw_mode("NONE")
                self.mode_finished.emit("Đã hủy chế độ vẽ.")

        elif self.draw_mode == "DRAW_ROI":
            if event.button() == Qt.MouseButton.LeftButton:
                self._temp_roi_points.append([mapped[0], mapped[1]])
                self._refresh_display()
            elif event.button() == Qt.MouseButton.RightButton:
                if len(self._temp_roi_points) >= 3:
                    new_roi = [list(p) for p in self._temp_roi_points]
                    self.roi_updated.emit(new_roi)
                    self.set_draw_mode("NONE")
                    self.mode_finished.emit(f"Đã cập nhật vùng ROI mới ({len(new_roi)} đỉnh)!")
                else:
                    self.set_draw_mode("NONE")
                    self.mode_finished.emit("Đã hủy vẽ ROI (cần ít nhất 3 đỉnh).")

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.draw_mode == "NONE":
            return
        mapped = self._map_widget_to_canvas(int(event.position().x()), int(event.position().y()))
        if mapped is not None:
            self._current_mouse = mapped
            self._refresh_display()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.draw_mode in ("DRAW_LINE", "DRAW_DIVIDER") and event.button() == Qt.MouseButton.LeftButton:
            if self._drag_start is not None:
                mapped = self._map_widget_to_canvas(int(event.position().x()), int(event.position().y()))
                if mapped is not None:
                    dist = np.hypot(mapped[0] - self._drag_start[0], mapped[1] - self._drag_start[1])
                    if dist >= 25:
                        pts = [[self._drag_start[0], self._drag_start[1]], [mapped[0], mapped[1]]]
                        if self.draw_mode == "DRAW_LINE":
                            self.line_updated.emit(pts)
                            self.mode_finished.emit("Đã cập nhật Vạch đếm mới thành công!")
                        else:
                            self.divider_updated.emit(pts)
                            self.mode_finished.emit("Đã cập nhật Vạch phân làn mới thành công!")
                        self.set_draw_mode("NONE")
                    else:
                        self._drag_start = None
                        self._refresh_display()
