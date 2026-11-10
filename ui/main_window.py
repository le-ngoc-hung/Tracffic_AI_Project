import os
from typing import Dict, Optional
import cv2
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import CameraConfig, ConfigManager
from core.database import ViolationDatabase, ViolationRecord
from core.detector import UnifiedTrafficDetector
from core.plate_anpr import LicensePlateANPR
from core.renderer import render_annotated_canvas
from core.tracker_analytics import TrafficAnalyticsEngine
from ui.video_canvas import InteractiveVideoCanvas


DIRECTION_MODE_ITEMS = [
    ("Đô thị 2 làn (Phải Lên / Trái Xuống)", "URBAN_2WAY"),
    ("Một chiều: Đi lên (Xa Camera)", "UP"),
    ("Một chiều: Đi xuống (Về phía Camera)", "DOWN"),
    ("Cao tốc 2 chiều (Trái Lên / Phải Xuống)", "HIGHWAY_2WAY"),
    ("Tự do 2 chiều (Không bắt ngược chiều)", "ANY"),
]


class EvidenceDialog(QDialog):
    """Modal dialog to inspect a full composite violation report image."""

    def __init__(self, record: ViolationRecord, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Biên Bản Phạt Nguội ANPR - Biển số: {record.plate_number} ({record.vehicle_type} #{record.track_id})"
        )
        self.resize(1100, 540)
        self.setStyleSheet("background-color: #161b22; color: #e6edf3;")

        layout = QVBoxLayout(self)
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if os.path.exists(record.image_path):
            pix = QPixmap(record.image_path)
            img_label.setPixmap(
                pix.scaled(1060, 460, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            img_label.setText(f"Không tìm thấy file ảnh: {record.image_path}")

        info = QLabel(
            f"<b>Thời gian:</b> {record.timestamp} &nbsp;|&nbsp; "
            f"<b>Camera:</b> {record.camera_id} &nbsp;|&nbsp; "
            f"<b>Phương tiện:</b> {record.vehicle_type} (#{record.track_id}) &nbsp;|&nbsp; "
            f"<b>Biển số ANPR:</b> <span style='color:#ffea00;'>{record.plate_number}</span> &nbsp;|&nbsp; "
            f"<b>Tốc độ:</b> {record.speed_kmh:.1f} km/h &nbsp;|&nbsp; "
            f"<b>Vi phạm:</b> <span style='color:#ff6b6b;'>{record.violation_type}</span>"
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("font-size: 13px; padding: 8px; background-color: #21262d; border-radius: 6px;")

        layout.addWidget(img_label)
        layout.addWidget(info)


class TrafficMonitoringWindow(QMainWindow):
    """
    Main PyQt6 Application Window for Smart Urban Traffic Monitoring System (DACN2).
    Combines Multi-Model YOLOv8 + ByteTrack, ANPR, Speed Estimation, Wrong-Way Detection,
    and Interactive Mouse ROI/Line Drawing.
    """

    SKIP_FRAMES = 2

    def __init__(self, initial_cam_key: str = "3"):
        super().__init__()
        self.setWindowTitle("Hệ Thống Giám Sát Giao Thông Đô Thị Thông Minh - AI Core & ANPR (DACN2)")
        self.resize(1460, 860)
        self._apply_dark_theme()

        # Initialize core services
        self.config_mgr = ConfigManager("configs/camera_configs.json")
        self.db = ViolationDatabase()
        self.anpr = LicensePlateANPR("violations")
        self.detector = UnifiedTrafficDetector()

        self.current_cam_key = initial_cam_key
        self.cap: Optional[cv2.VideoCapture] = None
        self.engine: Optional[TrafficAnalyticsEngine] = None
        self.is_paused: bool = False
        self._last_raw_frame: Optional[np.ndarray] = None
        self._last_canvas_frame: Optional[np.ndarray] = None
        self._violations_by_row: Dict[int, ViolationRecord] = {}

        self._build_ui()
        self._load_existing_violations()
        self.switch_camera(self.current_cam_key)

        # Video processing loop timer (~30 FPS UI refresh)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_next_frame)
        self.timer.start(28)

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #0d1117;
                color: #e6edf3;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 12px;
                font-weight: bold;
                color: #58a6ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 6px 12px;
                color: #e6edf3;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #58a6ff;
            }
            QPushButton:checked {
                background-color: #1f6feb;
                border-color: #58a6ff;
                color: #ffffff;
            }
            QTableWidget {
                background-color: #0d1117;
                alternate-background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                gridline-color: #21262d;
            }
            QHeaderView::section {
                background-color: #21262d;
                color: #58a6ff;
                font-weight: bold;
                padding: 5px;
                border: none;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 4px 8px;
                color: #e6edf3;
            }
            """
        )

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(8)

        # ======================================================
        # 1. TOP BAR: CAMERA SWITCHER & PLAYBACK CONTROLS
        # ======================================================
        top_bar = QHBoxLayout()
        title_lbl = QLabel("🚦 HỆ THỐNG GIÁM SÁT GIAO THÔNG ĐÔ THỊ & PHẠT NGUỘI ANPR")
        title_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #58a6ff; padding-right: 12px;")
        top_bar.addWidget(title_lbl)

        self.cam_buttons: Dict[str, QPushButton] = {}
        cam_labels = [
            ("1", "📹 CAM 1: Đô thị VN"),
            ("2", "📹 CAM 2: Kẹt xe"),
            ("3", "📹 CAM 3: Biển cấm P.106a"),
            ("4", "📹 CAM 4: Cao tốc 2 chiều"),
        ]
        for key, text in cam_labels:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, k=key: self.switch_camera(k))
            self.cam_buttons[key] = btn
            top_bar.addWidget(btn)

        btn_open_file = QPushButton("📂 Mở Video...")
        btn_open_file.clicked.connect(self._open_custom_video)
        top_bar.addWidget(btn_open_file)

        top_bar.addStretch()

        self.btn_pause = QPushButton("⏸ Tạm dừng")
        self.btn_pause.setCheckable(True)
        self.btn_pause.clicked.connect(self._toggle_pause)
        top_bar.addWidget(self.btn_pause)

        root_layout.addLayout(top_bar)

        # ======================================================
        # 2. MAIN CONTENT: LEFT VIDEO CANVAS + RIGHT DASHBOARD
        # ======================================================
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)

        # --- LEFT COLUMN: Video Canvas & Interactive Mouse Drawing Toolbar ---
        left_col = QVBoxLayout()
        left_col.setSpacing(6)

        self.video_canvas = InteractiveVideoCanvas(self)
        self.video_canvas.line_updated.connect(self._on_mouse_line_updated)
        self.video_canvas.roi_updated.connect(self._on_mouse_roi_updated)
        self.video_canvas.divider_updated.connect(self._on_mouse_divider_updated)
        self.video_canvas.mode_finished.connect(self._on_draw_mode_finished)
        left_col.addWidget(self.video_canvas, stretch=1)

        # Interactive Drawing Toolbar
        draw_bar = QHBoxLayout()
        self.btn_draw_line = QPushButton("✏️ Vẽ Vạch Đếm (Kéo chuột)")
        self.btn_draw_line.setStyleSheet("background-color: #1f4b2c; border-color: #2ea043;")
        self.btn_draw_line.clicked.connect(lambda: self._activate_draw_mode("DRAW_LINE"))
        draw_bar.addWidget(self.btn_draw_line)

        self.btn_draw_roi = QPushButton("⬠ Vẽ Vùng ROI (Click chuột trái / Phải để chốt)")
        self.btn_draw_roi.setStyleSheet("background-color: #4b2c5e; border-color: #8957e5;")
        self.btn_draw_roi.clicked.connect(lambda: self._activate_draw_mode("DRAW_ROI"))
        draw_bar.addWidget(self.btn_draw_roi)

        self.btn_draw_divider = QPushButton("🛣️ Vẽ Vạch Phân Làn")
        self.btn_draw_divider.clicked.connect(lambda: self._activate_draw_mode("DRAW_DIVIDER"))
        draw_bar.addWidget(self.btn_draw_divider)

        btn_reset_cfg = QPushButton("🔄 Khôi phục Mặc định Cam")
        btn_reset_cfg.clicked.connect(self._reset_current_cam_config)
        draw_bar.addWidget(btn_reset_cfg)

        left_col.addLayout(draw_bar)

        self.status_banner = QLabel(
            "💡 Mẹo: Bạn có thể bấm 'Vẽ Vạch Đếm' hoặc 'Vẽ Vùng ROI' rồi thao tác chuột trực tiếp lên màn hình video."
        )
        self.status_banner.setStyleSheet("color: #8b949e; font-size: 11px; padding: 2px 4px;")
        left_col.addWidget(self.status_banner)

        content_layout.addLayout(left_col, stretch=6)

        # --- RIGHT COLUMN: Analytics Cards, Rule Config & ANPR Violation Log ---
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        # CARD A: Live Telemetry & Vehicle Counters
        telemetry_group = QGroupBox("📊 THỐNG KÊ LƯU LƯỢNG & TRẠNG THÁI GIAO THÔNG")
        tel_layout = QVBoxLayout(telemetry_group)

        self.lbl_traffic_status = QLabel("TRẠNG THÁI: NORMAL (BÌNH THƯỜNG)")
        self.lbl_traffic_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_traffic_status.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.lbl_traffic_status.setStyleSheet(
            "background-color: #123d24; color: #3fb950; border: 1px solid #2ea043; border-radius: 6px; padding: 6px;"
        )
        tel_layout.addWidget(self.lbl_traffic_status)

        grid_counts = QGridLayout()
        self.lbl_counts: Dict[str, QLabel] = {}
        cards_meta = [
            ("Car", "🚗 Ô tô (Car)", "#3fb950", 0, 0),
            ("Motorbike", "🛵 Xe máy (Motor)", "#d29922", 0, 1),
            ("Bus", "🚌 Xe buýt (Bus)", "#bc8cff", 0, 2),
            ("Truck", "🚛 Xe tải (Truck)", "#f0883e", 1, 0),
            ("AvgSpeed", "⚡ Tốc độ TB", "#58a6ff", 1, 1),
            ("InROI", "📍 Xe trong ROI", "#ff7b72", 1, 2),
        ]
        for key, title, color, r, c in cards_meta:
            frame = QFrame()
            frame.setStyleSheet("background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px;")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(8, 4, 8, 4)
            t_lbl = QLabel(title)
            t_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
            v_lbl = QLabel("0")
            v_lbl.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            v_lbl.setStyleSheet(f"color: {color};")
            fl.addWidget(t_lbl)
            fl.addWidget(v_lbl)
            self.lbl_counts[key] = v_lbl
            grid_counts.addWidget(frame, r, c)

        tel_layout.addLayout(grid_counts)
        right_col.addWidget(telemetry_group)

        # CARD B: Traffic Rules & Speed Configuration
        rules_group = QGroupBox("⚙️ CẤU HÌNH LUẬT GIAO THÔNG & TỐC ĐỘ (THEO CAMERA)")
        rules_layout = QGridLayout(rules_group)

        rules_layout.addWidget(QLabel("Tốc độ tối đa (km/h):"), 0, 0)
        self.spin_speed_limit = QDoubleSpinBox()
        self.spin_speed_limit.setRange(20.0, 140.0)
        self.spin_speed_limit.setSingleStep(5.0)
        self.spin_speed_limit.valueChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.spin_speed_limit, 0, 1)

        rules_layout.addWidget(QLabel("Ngưỡng báo kẹt xe (ROI):"), 0, 2)
        self.spin_congestion = QSpinBox()
        self.spin_congestion.setRange(3, 50)
        self.spin_congestion.valueChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.spin_congestion, 0, 3)

        rules_layout.addWidget(QLabel("Chiều làn đường:"), 1, 0)
        self.combo_direction = QComboBox()
        for label_txt, mode_val in DIRECTION_MODE_ITEMS:
            self.combo_direction.addItem(label_txt, mode_val)
        self.combo_direction.currentIndexChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.combo_direction, 1, 1, 1, 3)

        self.chk_no_truck = QCheckBox("Bắt lỗi biển cấm xe tải (P.106a)")
        self.chk_no_truck.stateChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.chk_no_truck, 2, 0, 1, 2)

        self.chk_speed = QCheckBox("Bắt lỗi quá tốc độ")
        self.chk_speed.stateChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.chk_speed, 2, 2, 1, 2)

        self.chk_wrong_way = QCheckBox("Bắt lỗi đi ngược chiều / Lấn làn")
        self.chk_wrong_way.stateChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.chk_wrong_way, 3, 0, 1, 4)

        right_col.addWidget(rules_group)

        # CARD C: Violation Log Table & ANPR License Plate Preview
        viol_group = QGroupBox("🚨 DANH SÁCH VI PHẠM PHẠT NGUỘI & NHẬN DIỆN BIỂN SỐ (ANPR)")
        viol_layout = QVBoxLayout(viol_group)

        self.table_violations = QTableWidget(0, 6)
        self.table_violations.setHorizontalHeaderLabels(
            ["Thời gian", "Cam", "Phương tiện", "Biển số (ANPR)", "Tốc độ", "Lỗi vi phạm"]
        )
        self.table_violations.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_violations.setAlternatingRowColors(True)
        self.table_violations.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_violations.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_violations.itemSelectionChanged.connect(self._on_violation_row_selected)
        self.table_violations.doubleClicked.connect(self._open_selected_evidence_dialog)
        viol_layout.addWidget(self.table_violations, stretch=1)

        # ANPR Plate Zoom & Evidence Action Bar
        anpr_box = QFrame()
        anpr_box.setStyleSheet("background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px;")
        anpr_layout = QHBoxLayout(anpr_box)
        anpr_layout.setContentsMargins(8, 6, 8, 6)

        self.lbl_plate_thumb = QLabel("Chưa chọn\nvi phạm")
        self.lbl_plate_thumb.setFixedSize(165, 56)
        self.lbl_plate_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_plate_thumb.setStyleSheet(
            "background-color: #161b22; border: 1px solid #d29922; border-radius: 4px; color: #8b949e;"
        )
        anpr_layout.addWidget(self.lbl_plate_thumb)

        self.lbl_anpr_details = QLabel(
            "<b>Biển số ANPR:</b> ---<br>"
            "<b>Lỗi:</b> Chọn 1 dòng để xem ảnh biển số & biên bản"
        )
        self.lbl_anpr_details.setStyleSheet("font-size: 11px;")
        anpr_layout.addWidget(self.lbl_anpr_details, stretch=1)

        btn_col = QVBoxLayout()
        self.btn_view_evidence = QPushButton("🔍 Xem Biên Bản")
        self.btn_view_evidence.clicked.connect(self._open_selected_evidence_dialog)
        btn_col.addWidget(self.btn_view_evidence)

        btn_export_csv = QPushButton("📥 Xuất CSV")
        btn_export_csv.clicked.connect(self._export_csv_report)
        btn_col.addWidget(btn_export_csv)

        anpr_layout.addLayout(btn_col)
        viol_layout.addWidget(anpr_box)

        right_col.addWidget(viol_group, stretch=1)
        content_layout.addLayout(right_col, stretch=4)
        root_layout.addLayout(content_layout, stretch=1)

    def _load_existing_violations(self) -> None:
        records = self.db.get_recent_violations(limit=40)
        for rec in reversed(records):
            self._prepend_violation_row(rec, select_row=False)

    def switch_camera(self, cam_key: str) -> None:
        if self.cap is not None:
            self.cap.release()

        self.current_cam_key = cam_key
        for k, btn in self.cam_buttons.items():
            btn.setChecked(k == cam_key)

        cfg = self.config_mgr.get_config(cam_key)
        self.cap = cv2.VideoCapture(cfg.video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1.0 or fps > 120.0:
            fps = 30.0

        self.detector.reset_tracker()
        self.engine = TrafficAnalyticsEngine(cfg, self.anpr, self.db, fps=fps)
        self._sync_ui_controls_from_config(cfg)
        self.status_banner.setText(f"--> Đã chuyển sang: {cfg.name} ({cfg.video_path} | {fps:.0f} FPS)")

    def _open_custom_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file Video giao thông", "videos", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            rel_path = os.path.relpath(file_path) if os.path.exists(file_path) else file_path
            self.switch_camera(rel_path)

    def _sync_ui_controls_from_config(self, cfg: CameraConfig) -> None:
        self._updating_ui = True
        self.spin_speed_limit.setValue(cfg.speed_limit_kmh)
        self.spin_congestion.setValue(cfg.congestion_threshold)

        idx = 0
        for i, (_, mode_val) in enumerate(DIRECTION_MODE_ITEMS):
            if mode_val == cfg.direction_mode:
                idx = i
                break
        self.combo_direction.setCurrentIndex(idx)

        self.chk_no_truck.setChecked(cfg.enable_no_truck_rule)
        self.chk_speed.setChecked(cfg.enable_speed_rule)
        self.chk_wrong_way.setChecked(cfg.enable_wrong_way_rule)
        self._updating_ui = False

    def _on_rule_param_changed(self) -> None:
        if getattr(self, "_updating_ui", False) or self.engine is None:
            return
        cfg = self.engine.config
        cfg.speed_limit_kmh = float(self.spin_speed_limit.value())
        cfg.congestion_threshold = int(self.spin_congestion.value())
        cfg.direction_mode = str(self.combo_direction.currentData())
        cfg.enable_no_truck_rule = self.chk_no_truck.isChecked()
        cfg.enable_speed_rule = self.chk_speed.isChecked()
        cfg.enable_wrong_way_rule = self.chk_wrong_way.isChecked()
        self.config_mgr.save()

    def _toggle_pause(self, checked: bool) -> None:
        self.is_paused = checked
        self.btn_pause.setText("▶ Tiếp tục" if checked else "⏸ Tạm dừng")

    def _activate_draw_mode(self, mode: str) -> None:
        self.video_canvas.set_draw_mode(mode)
        if mode == "DRAW_LINE":
            self.status_banner.setText("✏️ Kéo thả chuột trái trên video để vẽ Vạch đếm mới (Chuột phải để hủy).")
        elif mode == "DRAW_ROI":
            self.status_banner.setText("⬠ Click chuột trái chọn các đỉnh đa giác ROI, sau đó click CHUỘT PHẢI để hoàn tất.")
        elif mode == "DRAW_DIVIDER":
            self.status_banner.setText("🛣️ Kéo thả chuột trái dọc theo vạch phân làn giữa 2 chiều đường.")

    def _on_mouse_line_updated(self, points: list) -> None:
        if self.engine is None:
            return
        self.engine.config.counting_line = points
        self.config_mgr.save()
        self._rerender_paused_canvas()

    def _on_mouse_roi_updated(self, points: list) -> None:
        if self.engine is None:
            return
        self.engine.config.roi_polygon = points
        self.config_mgr.save()
        self._rerender_paused_canvas()

    def _on_mouse_divider_updated(self, points: list) -> None:
        if self.engine is None:
            return
        self.engine.config.divider_line = points
        self.config_mgr.save()
        self._rerender_paused_canvas()

    def _on_draw_mode_finished(self, msg: str) -> None:
        self.status_banner.setText(f"✅ {msg} (Đã lưu tự động vào configs/camera_configs.json)")

    def _reset_current_cam_config(self) -> None:
        cfg = self.config_mgr.reset_camera_to_default(self.current_cam_key)
        if self.engine is not None:
            self.engine.config = cfg
        self._sync_ui_controls_from_config(cfg)
        self._rerender_paused_canvas()
        self.status_banner.setText("🔄 Đã khôi phục cấu hình Vạch đếm & ROI mặc định cho Camera này.")

    def _rerender_paused_canvas(self) -> None:
        if self._last_canvas_frame is not None and self.engine is not None:
            annotated = render_annotated_canvas(self._last_canvas_frame, self.engine, self.engine.config)
            self.video_canvas.update_frame(annotated)

    def _process_next_frame(self) -> None:
        if self.is_paused or self.cap is None or self.engine is None:
            return

        ret, raw_frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.engine.reset_state()
            return

        self.engine.frame_idx += 1
        canvas_frame = cv2.resize(raw_frame, (854, 480))
        self._last_raw_frame = raw_frame
        self._last_canvas_frame = canvas_frame

        # 1. Detect Traffic Signs periodically
        if self.engine.frame_idx % 20 == 0 or self.engine.frame_idx == 1:
            signs = self.detector.detect_signs(canvas_frame, conf=0.18)
            self.engine.update_signs(signs)

        # 2. Detect & Track Vehicles + Run Analytics every (SKIP_FRAMES + 1)
        if self.engine.frame_idx % (self.SKIP_FRAMES + 1) == 0 or self.engine.frame_idx == 1:
            vehicles = self.detector.detect_and_track_vehicles(canvas_frame)
            self.engine.process_detections(vehicles, raw_frame, canvas_frame)

            for rec in self.engine.new_violations:
                self._prepend_violation_row(rec, select_row=True)

            self._update_telemetry_labels()

        # 3. Render annotated video canvas
        annotated = render_annotated_canvas(canvas_frame, self.engine, self.engine.config)
        self.video_canvas.update_frame(annotated)

    def _update_telemetry_labels(self) -> None:
        if self.engine is None:
            return
        is_congested = self.engine.traffic_status == "CONGESTED"
        if is_congested:
            self.lbl_traffic_status.setText(
                f"⚠️ TRẠNG THÁI: KẸT XE - CONGESTED ({self.engine.vehicles_in_roi} xe trong ROI)"
            )
            self.lbl_traffic_status.setStyleSheet(
                "background-color: #491217; color: #ff7b72; border: 1px solid #f85149; border-radius: 6px; padding: 6px;"
            )
        else:
            self.lbl_traffic_status.setText(
                f"✅ TRẠNG THÁI: THÔNG THOÁNG - NORMAL ({self.engine.vehicles_in_roi} xe trong ROI)"
            )
            self.lbl_traffic_status.setStyleSheet(
                "background-color: #123d24; color: #3fb950; border: 1px solid #2ea043; border-radius: 6px; padding: 6px;"
            )

        for v_type in ("Car", "Motorbike", "Bus", "Truck"):
            self.lbl_counts[v_type].setText(str(self.engine.counts.get(v_type, 0)))

        self.lbl_counts["AvgSpeed"].setText(f"{self.engine.avg_speed_kmh:.1f} km/h")
        self.lbl_counts["InROI"].setText(
            f"{self.engine.vehicles_in_roi}/{self.engine.config.congestion_threshold}"
        )

    def _prepend_violation_row(self, rec: ViolationRecord, select_row: bool = True) -> None:
        self.table_violations.insertRow(0)
        # Shift existing row mapping down by 1
        new_map: Dict[int, ViolationRecord] = {0: rec}
        for r_idx, old_rec in self._violations_by_row.items():
            new_map[r_idx + 1] = old_rec
        self._violations_by_row = new_map

        time_short = rec.timestamp.split(" ")[-1] if " " in rec.timestamp else rec.timestamp
        cells = [
            time_short,
            rec.camera_id,
            f"{rec.vehicle_type} #{rec.track_id}",
            rec.plate_number,
            f"{rec.speed_kmh:.1f} km/h",
            rec.violation_type,
        ]
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if col == 3:
                item.setForeground(QColor("#ffea00"))
                item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            elif col == 5:
                item.setForeground(QColor("#ff7b72"))
            self.table_violations.setItem(0, col, item)

        if select_row:
            self.table_violations.selectRow(0)
            self._show_violation_preview(rec)

    def _on_violation_row_selected(self) -> None:
        row = self.table_violations.currentRow()
        if row in self._violations_by_row:
            self._show_violation_preview(self._violations_by_row[row])

    def _show_violation_preview(self, rec: ViolationRecord) -> None:
        self.lbl_anpr_details.setText(
            f"<b>Biển số ANPR:</b> <span style='color:#ffea00; font-size:13px;'>{rec.plate_number}</span><br>"
            f"<b>Phương tiện:</b> {rec.vehicle_type} (#{rec.track_id}) | <b>Tốc độ:</b> {rec.speed_kmh:.1f} km/h<br>"
            f"<b>Lỗi:</b> <span style='color:#ff7b72;'>{rec.violation_type}</span>"
        )
        if rec.plate_crop_path and os.path.exists(rec.plate_crop_path):
            pix = QPixmap(rec.plate_crop_path)
            self.lbl_plate_thumb.setPixmap(
                pix.scaled(
                    self.lbl_plate_thumb.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        elif os.path.exists(rec.image_path):
            pix = QPixmap(rec.image_path)
            self.lbl_plate_thumb.setPixmap(
                pix.scaled(
                    self.lbl_plate_thumb.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _open_selected_evidence_dialog(self) -> None:
        row = self.table_violations.currentRow()
        if row in self._violations_by_row:
            dlg = EvidenceDialog(self._violations_by_row[row], self)
            dlg.exec()
        else:
            QMessageBox.information(self, "Thông báo", "Hãy chọn một dòng vi phạm trong bảng để xem biên bản.")

    def _export_csv_report(self) -> None:
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất Báo Cáo Vi Phạm CSV",
            "violations/bao_cao_phat_nguoi.csv",
            "CSV Files (*.csv)",
        )
        if save_path:
            count = self.db.export_to_csv(save_path)
            QMessageBox.information(
                self,
                "Xuất Báo Cáo Thành Công",
                f"Đã xuất {count} bản ghi vi phạm phạt nguội ra file:\n{save_path}",
            )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key_text = event.text().lower()
        if key_text in ("1", "2", "3", "4"):
            self.switch_camera(key_text)
        elif key_text == "q":
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)
