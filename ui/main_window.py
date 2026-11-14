import os
import shutil
from typing import Dict, List, Optional
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
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
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.ai_voice_narrator import AISettings, ViolationVoiceWorker, load_ai_settings, save_ai_settings
from core.config_manager import CameraConfig, ConfigManager
from core.database import ViolationDatabase, ViolationRecord
from core.detector import UnifiedTrafficDetector
from core.logger import get_app_logger
from core.plate_anpr import LicensePlateANPR
from core.renderer import render_annotated_canvas
from core.stream_reader import is_live_stream, parse_video_source
from ui.inference_worker import VideoInferenceWorker
from ui.video_canvas import InteractiveVideoCanvas

logger = get_app_logger("MainWindow")

DIRECTION_MODE_ITEMS = [
    ("Đô thị 2 làn (Phải Lên / Trái Xuống)", "URBAN_2WAY"),
    ("Một chiều: Đi lên (Xa Camera)", "UP"),
    ("Một chiều: Đi xuống (Về phía Camera)", "DOWN"),
    ("Cao tốc 2 chiều (Trái Lên / Phải Xuống)", "HIGHWAY_2WAY"),
    ("Tự do 2 chiều (Không bắt ngược chiều)", "ANY"),
]

VERIFICATION_STATUSES = [
    "CẦN KIỂM TRA THỦ CÔNG",
    "TỰ ĐỘNG XÁC NHẬN",
    "ĐÃ XÁC NHẬN THỦ CÔNG",
    "BÁO NHẦM (ĐÃ HỦY)",
]


class EvidenceAndReviewDialog(QDialog):
    """
    Modal dialog to inspect:
    1. Full composite violation report image
    2. 3-frame Before / During / After timeline progression strip
    3. Full audit metadata (Vehicle/Sign/Plate confidences, model version, trigger reason)
    4. Manual verification & editing of license plate, violation type, and confirmation status
    """

    def __init__(self, record: ViolationRecord, db: ViolationDatabase, parent=None):
        super().__init__(parent)
        self.record = record
        self.db = db
        self.updated_record: Optional[ViolationRecord] = None

        self.setWindowTitle(
            f"Hồ Sơ Bằng Chứng & Xác Minh Vi Phạm #{record.id} - {record.camera_id} (Track #{record.track_id})"
        )
        self.resize(1160, 720)
        self.setStyleSheet(
            """
            QDialog, QWidget { background-color: #161b22; color: #e6edf3; font-family: 'Segoe UI'; }
            QLineEdit, QComboBox { background-color: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 5px; color: #e6edf3; }
            QPushButton { background-color: #238636; border: 1px solid #2ea043; border-radius: 6px; padding: 7px 14px; color: white; font-weight: bold; }
            QPushButton:hover { background-color: #2ea043; }
            """
        )

        layout = QVBoxLayout(self)

        # Tabs for Composite Report vs Before-During-After Timeline
        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabBar::tab { background: #21262d; padding: 8px 16px; border-radius: 4px; margin-right: 4px; } "
            "QTabBar::tab:selected { background: #1f6feb; color: white; font-weight: bold; }"
        )

        lbl_composite = QLabel()
        lbl_composite.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.exists(record.image_path):
            pix = QPixmap(record.image_path)
            lbl_composite.setPixmap(
                pix.scaled(1100, 430, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            lbl_composite.setText(f"Không tìm thấy ảnh biên bản: {record.image_path}")
        tabs.addTab(lbl_composite, "📸 1. Ảnh Biên Bản Tổng Hợp (Full HD Crop + ANPR)")

        lbl_timeline = QLabel()
        lbl_timeline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if record.timeline_path and os.path.exists(record.timeline_path):
            pix_tl = QPixmap(record.timeline_path)
            lbl_timeline.setPixmap(
                pix_tl.scaled(1100, 430, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            lbl_timeline.setText("Không có ảnh chuỗi Trước - Trong - Sau sự kiện cho bản ghi cũ này.")
        tabs.addTab(lbl_timeline, "🎞️ 2. Chuỗi Khung Hình Trước - Trong - Sau Sự Kiện (Before / During / After)")

        layout.addWidget(tabs, stretch=1)

        # Audit Metadata Box
        speed_cal_str = "Đã hiệu chuẩn (Homography BEV)" if record.is_speed_calibrated else "Ước tính (~Chưa hiệu chuẩn)"
        hash_short = record.evidence_hash[:16] + "..." if record.evidence_hash else "Chưa tạo (bản ghi cũ)"
        clip_label = os.path.basename(record.clip_path) if record.clip_path else "Không có"
        audit_box = QLabel(
            f"<b>🔍 Truy vết hệ thống (Audit Trail & Pháp lý):</b><br>"
            f"• <b>Thời điểm:</b> {record.timestamp} | <b>Camera:</b> {record.camera_id} | <b>Model:</b> {record.model_version}<br>"
            f"• <b>Độ tin cậy AI:</b> Xe ({record.vehicle_type}): <b>{record.vehicle_conf*100:.1f}%</b> | "
            f"Biển báo: <b>{record.sign_conf*100:.1f}%</b> | "
            f"Biển số ANPR: <b>{record.plate_conf*100:.1f}%</b> (Trạng thái đọc: <b>{record.plate_status}</b>)<br>"
            f"• <b>Tốc độ ghi nhận:</b> {record.speed_kmh:.1f} km/h ({speed_cal_str}) | <b>Clip:</b> {clip_label} | <b>SHA-256:</b> <code>{hash_short}</code><br>"
            f"• <b>Lý do tạo cảnh báo:</b> <span style='color:#79c0ff;'>{record.trigger_reason}</span>"
        )
        audit_box.setWordWrap(True)
        audit_box.setStyleSheet(
            "background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-size: 12px;"
        )
        layout.addWidget(audit_box)

        # Manual Review & Edit Form
        edit_group = QGroupBox("✏️ KIỂM TRA && XÁC NHẬN THỦ CÔNG (MANUAL OPERATOR REVIEW)")
        edit_layout = QHBoxLayout(edit_group)

        edit_layout.addWidget(QLabel("Biển số xe:"))
        self.edit_plate = QLineEdit(record.plate_number)
        self.edit_plate.setMinimumWidth(140)
        edit_layout.addWidget(self.edit_plate)

        edit_layout.addWidget(QLabel("Loại vi phạm:"))
        self.edit_violation = QLineEdit(record.violation_type)
        self.edit_violation.setMinimumWidth(190)
        edit_layout.addWidget(self.edit_violation)

        edit_layout.addWidget(QLabel("Trạng thái xác minh:"))
        self.combo_status = QComboBox()
        for st in VERIFICATION_STATUSES:
            self.combo_status.addItem(st)
        if record.verification_status in VERIFICATION_STATUSES:
            self.combo_status.setCurrentText(record.verification_status)
        edit_layout.addWidget(self.combo_status)

        btn_clip = QPushButton("🎬 Mở Clip MP4")
        btn_clip.setStyleSheet("background-color: #1f6feb; border-color: #388bfd;")
        btn_clip.setEnabled(bool(record.clip_path and os.path.exists(record.clip_path)))
        btn_clip.clicked.connect(self._open_evidence_clip)
        edit_layout.addWidget(btn_clip)

        btn_save = QPushButton("💾 Lưu Xác Nhận")
        btn_save.clicked.connect(self._save_manual_review)
        edit_layout.addWidget(btn_save)

        btn_reject = QPushButton("🚫 Đánh Dấu Báo Nhầm")
        btn_reject.setStyleSheet("background-color: #da3633; border-color: #f85149;")
        btn_reject.clicked.connect(self._mark_false_positive)
        edit_layout.addWidget(btn_reject)

        layout.addWidget(edit_group)

    def _open_evidence_clip(self) -> None:
        if self.record.clip_path and os.path.exists(self.record.clip_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.record.clip_path)))
        else:
            QMessageBox.information(self, "Thông báo", "Không tìm thấy video clip bằng chứng cho bản ghi này.")

    def _save_manual_review(self) -> None:
        if self.record.id is None:
            return
        new_plate = self.edit_plate.text().strip() or "KHÔNG ĐỌC ĐƯỢC"
        new_viol = self.edit_violation.text().strip() or self.record.violation_type
        new_status = self.combo_status.currentText()
        if new_status == "CẦN KIỂM TRA THỦ CÔNG":
            new_status = "ĐÃ XÁC NHẬN THỦ CÔNG"
        self.updated_record = self.db.update_violation_manual(
            self.record.id, new_plate, new_viol, new_status
        )
        self.accept()

    def _mark_false_positive(self) -> None:
        if self.record.id is None:
            return
        new_plate = self.edit_plate.text().strip() or self.record.plate_number
        new_viol = self.edit_violation.text().strip() or self.record.violation_type
        self.updated_record = self.db.update_violation_manual(
            self.record.id, new_plate, new_viol, "BÁO NHẦM (ĐÃ HỦY)"
        )
        self.accept()


class AddCameraDialog(QDialog):
    """
    Dialog to register a new video source as a permanent Camera channel (e.g., CAM 5, CAM 6...)
    just like the built-in CAM 1..4 channels, with live thumbnail preview and optional copy to videos/.
    """

    def __init__(self, next_cam_key: str, parent=None):
        super().__init__(parent)
        self.next_cam_key = next_cam_key
        self.selected_video_path: str = ""
        self.created_config: Optional[CameraConfig] = None

        self.setWindowTitle(f"➕ Thêm Kênh Camera Mới (CAM {next_cam_key})")
        self.resize(620, 520)
        self.setStyleSheet(
            """
            QDialog, QWidget { background-color: #161b22; color: #e6edf3; font-family: 'Segoe UI'; font-size: 12px; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 5px; color: #e6edf3;
            }
            QPushButton {
                background-color: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 7px 14px; color: #e6edf3; font-weight: bold;
            }
            QPushButton:hover { background-color: #30363d; border-color: #58a6ff; }
            """
        )

        layout = QVBoxLayout(self)

        # 1. Video File / Live RTSP Source Picker & Preview
        file_group = QGroupBox(f"📹 1. CHỌN FILE VIDEO HOẶC LUỒNG RTSP / WEBCAM CHO CAM {next_cam_key}")
        fg_layout = QVBoxLayout(file_group)

        pick_row = QHBoxLayout()
        self.edit_file_path = QLineEdit()
        self.edit_file_path.setPlaceholderText("Nhập URL RTSP (rtsp://...) / ID Webcam (0) hoặc chọn file video (.mp4)...")
        pick_row.addWidget(self.edit_file_path, stretch=1)

        btn_browse = QPushButton("📂 Chọn File...")
        btn_browse.setStyleSheet("background-color: #1f6feb; color: white;")
        btn_browse.clicked.connect(self._browse_video)
        pick_row.addWidget(btn_browse)

        btn_test_stream = QPushButton("🔗 Thử Kết Nối")
        btn_test_stream.clicked.connect(self._preview_entered_source)
        pick_row.addWidget(btn_test_stream)
        fg_layout.addLayout(pick_row)

        self.lbl_preview = QLabel("Khung xem trước video sẽ hiển thị tại đây sau khi chọn file")
        self.lbl_preview.setFixedHeight(190)
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet(
            "background-color: #0d1117; border: 1px dashed #30363d; border-radius: 6px; color: #8b949e;"
        )
        fg_layout.addWidget(self.lbl_preview)

        self.chk_copy_to_videos = QCheckBox("Sao chép file video vào thư mục 'videos/' của dự án để lưu trữ cố định")
        self.chk_copy_to_videos.setChecked(True)
        fg_layout.addWidget(self.chk_copy_to_videos)

        layout.addWidget(file_group)

        # 2. Camera Info & Initial Rules
        cfg_group = QGroupBox("⚙️ 2. THÔNG TIN & CẤU HÌNH BAN ĐẦU CỦA CAMERA")
        grid = QGridLayout(cfg_group)

        grid.addWidget(QLabel("Tên hiển thị Camera:"), 0, 0)
        self.edit_cam_name = QLineEdit(f"CAM {next_cam_key}: Kênh Giám Sát Mới")
        grid.addWidget(self.edit_cam_name, 0, 1, 1, 3)

        grid.addWidget(QLabel("Tốc độ tối đa (km/h):"), 1, 0)
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(20.0, 160.0)
        self.spin_speed.setValue(60.0)
        self.spin_speed.setSingleStep(5.0)
        grid.addWidget(self.spin_speed, 1, 1)

        grid.addWidget(QLabel("Ngưỡng báo kẹt xe (ROI):"), 1, 2)
        self.spin_cong = QSpinBox()
        self.spin_cong.setRange(3, 50)
        self.spin_cong.setValue(9)
        grid.addWidget(self.spin_cong, 1, 3)

        grid.addWidget(QLabel("Chiều làn đường:"), 2, 0)
        self.combo_dir = QComboBox()
        for label_txt, mode_val in DIRECTION_MODE_ITEMS:
            self.combo_dir.addItem(label_txt, mode_val)
        grid.addWidget(self.combo_dir, 2, 1, 1, 3)

        self.chk_rule_truck = QCheckBox("Bắt cấm xe tải (P.106a)")
        self.chk_rule_truck.setChecked(True)
        grid.addWidget(self.chk_rule_truck, 3, 0, 1, 2)

        self.chk_rule_speed = QCheckBox("Bắt quá tốc độ")
        self.chk_rule_speed.setChecked(True)
        grid.addWidget(self.chk_rule_speed, 3, 2, 1, 2)

        self.chk_rule_wrong = QCheckBox("Bắt đi ngược chiều / Lấn làn")
        self.chk_rule_wrong.setChecked(True)
        grid.addWidget(self.chk_rule_wrong, 4, 0, 1, 2)

        self.chk_rule_parking = QCheckBox("Bắt dừng đỗ trái phép")
        self.chk_rule_parking.setChecked(False)
        grid.addWidget(self.chk_rule_parking, 4, 2, 1, 2)

        layout.addWidget(cfg_group)

        # 3. Dialog Action Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_cancel = QPushButton("Hủy")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_confirm = QPushButton(f"✅ Thêm vào Hệ Thống (CAM {next_cam_key})")
        btn_confirm.setStyleSheet("background-color: #238636; border-color: #2ea043; color: white;")
        btn_confirm.clicked.connect(self._confirm_add_camera)
        btn_row.addWidget(btn_confirm)

        layout.addLayout(btn_row)

    def _browse_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file Video cho Camera mới",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov)",
        )
        if not file_path:
            return
        self.edit_file_path.setText(file_path)
        self._load_source_preview(file_path)

    def _preview_entered_source(self) -> None:
        source_txt = self.edit_file_path.text().strip()
        if not source_txt:
            QMessageBox.warning(self, "Thiếu nguồn", "Hãy nhập URL RTSP / ID Webcam hoặc chọn file video trước.")
            return
        self._load_source_preview(source_txt)

    def _load_source_preview(self, source_txt: str) -> None:
        live = is_live_stream(source_txt)
        cap = cv2.VideoCapture(parse_video_source(source_txt))
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if not ret or frame is None:
            QMessageBox.warning(
                self, "Lỗi đọc Video / Stream", f"Không thể đọc khung hình từ nguồn:\n{source_txt}"
            )
            return

        self.selected_video_path = source_txt
        if live:
            self.chk_copy_to_videos.setChecked(False)
            self.chk_copy_to_videos.setEnabled(False)
            self.edit_cam_name.setText(f"CAM {self.next_cam_key}: IP Camera Live")
            badge_name = "LIVE STREAM"
        else:
            self.chk_copy_to_videos.setEnabled(True)
            base_stem = os.path.splitext(os.path.basename(source_txt))[0]
            self.edit_cam_name.setText(f"CAM {self.next_cam_key}: {base_stem}")
            badge_name = os.path.basename(source_txt)

        # Show preview thumbnail with resolution & FPS badge
        preview = cv2.resize(frame, (480, 270))
        cv2.rectangle(preview, (0, 0), (480, 28), (20, 22, 30), -1)
        cv2.putText(
            preview,
            f"Preview: {badge_name} ({w}x{h} @ {fps:.0f} FPS)",
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, 480, 270, 3 * 480, QImage.Format.Format_RGB888)
        self.lbl_preview.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.lbl_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _confirm_add_camera(self) -> None:
        source_txt = self.edit_file_path.text().strip() or self.selected_video_path
        live = is_live_stream(source_txt)
        if not source_txt or (not live and not os.path.exists(source_txt)):
            QMessageBox.warning(
                self,
                "Thiếu thông tin",
                "Vui lòng chọn file video tồn tại hoặc nhập URL RTSP/HTTP/Webcam hợp lệ trước khi thêm Camera!",
            )
            return

        cam_name = self.edit_cam_name.text().strip() or f"CAM {self.next_cam_key}: Custom"
        target_video_path = source_txt

        if not live and self.chk_copy_to_videos.isChecked():
            os.makedirs("videos", exist_ok=True)
            ext = os.path.splitext(source_txt)[1].lower() or ".mp4"
            safe_stem = "".join(
                c if c.isalnum() or c in ("_", "-") else "_"
                for c in os.path.splitext(os.path.basename(source_txt))[0]
            )[:24]
            dest_rel = f"videos/cam{self.next_cam_key}_{safe_stem}{ext}"
            try:
                if os.path.abspath(source_txt) != os.path.abspath(dest_rel):
                    shutil.copy2(source_txt, dest_rel)
                target_video_path = dest_rel
            except Exception as e:
                logger.warning(f"Không thể copy file vào videos/, dùng đường dẫn gốc: {e}")

        self.created_config = CameraConfig(
            cam_key=self.next_cam_key,
            name=cam_name,
            video_path=target_video_path,
            direction_mode=str(self.combo_dir.currentData()),
            speed_limit_kmh=float(self.spin_speed.value()),
            congestion_threshold=int(self.spin_cong.value()),
            is_speed_calibrated=False,
            enable_no_truck_rule=self.chk_rule_truck.isChecked(),
            enable_speed_rule=self.chk_rule_speed.isChecked(),
            enable_wrong_way_rule=self.chk_rule_wrong.isChecked(),
            enable_illegal_parking_rule=self.chk_rule_parking.isChecked(),
        )
        self.accept()


class SettingsDialog(QDialog):
    """
    Cửa sổ Cài đặt hệ thống:
      - Chọn ngôn ngữ giao diện (Tiếng Việt mặc định / English)
      - Chọn ngôn ngữ giọng đọc AI (Tiếng Việt mặc định / English)
      - Chọn tốc độ đọc lỗi (0.75x - 2.00x) giúp đọc nhanh khi nhiều xe vi phạm
      - Cấu hình bật/tắt tự động đọc, thời gian chống lặp & Google AI Studio API Key
    """

    SPEED_PRESETS = [
        ("0.85x — Chậm (Slow)", 0.85),
        ("1.00x — Bình thường (Normal)", 1.00),
        ("1.25x — Nhanh (Fast)", 1.25),
        ("1.50x — Rất nhanh (Very Fast)", 1.50),
        ("1.75x — Cấp tốc (Rapid)", 1.75),
        ("2.00x — Tối đa (Max)", 2.00),
    ]

    def __init__(self, settings: AISettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.saved_settings: Optional[AISettings] = None
        is_en = str(settings.ui_language).lower() == "en"

        self.setWindowTitle("System & AI Voice Settings" if is_en else "⚙️ Cài Đặt Hệ Thống & Giọng Đọc AI")
        self.resize(520, 420)
        self.setStyleSheet(
            """
            QDialog, QWidget { background-color: #161b22; color: #e6edf3; font-family: 'Segoe UI'; font-size: 12px; }
            QLineEdit, QComboBox, QDoubleSpinBox {
                background-color: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 5px 8px; color: #e6edf3;
            }
            QPushButton {
                background-color: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 7px 14px; color: #e6edf3; font-weight: bold;
            }
            QPushButton:hover { background-color: #30363d; border-color: #58a6ff; }
            """
        )

        layout = QVBoxLayout(self)

        # Group 1: Language Options
        lang_group = QGroupBox("🌐 NGÔN NGỮ HỆ THỐNG && GIỌNG ĐỌC" if not is_en else "🌐 SYSTEM && VOICE LANGUAGE")
        lg = QGridLayout(lang_group)

        lg.addWidget(QLabel("Ngôn ngữ giao diện hệ thống:" if not is_en else "System UI Language:"), 0, 0)
        self.combo_ui_lang = QComboBox()
        self.combo_ui_lang.addItem("Tiếng Việt (Mặc định)", "vi")
        self.combo_ui_lang.addItem("English", "en")
        self.combo_ui_lang.setCurrentIndex(1 if str(settings.ui_language).lower() == "en" else 0)
        lg.addWidget(self.combo_ui_lang, 0, 1)

        lg.addWidget(QLabel("Ngôn ngữ giọng đọc lỗi:" if not is_en else "Violation Voice Language:"), 1, 0)
        self.combo_voice_lang = QComboBox()
        self.combo_voice_lang.addItem("Tiếng Việt (Mặc định)", "vi")
        self.combo_voice_lang.addItem("English", "en")
        self.combo_voice_lang.setCurrentIndex(1 if str(settings.voice_language).lower() == "en" else 0)
        lg.addWidget(self.combo_voice_lang, 1, 1)

        layout.addWidget(lang_group)

        # Group 2: Speech Speed & AI Options
        voice_group = QGroupBox("🔊 TỐC ĐỘ ĐỌC LỖI && CẤU HÌNH AI" if not is_en else "🔊 READING SPEED && AI CONFIG")
        vg = QGridLayout(voice_group)

        vg.addWidget(QLabel("Tốc độ đọc nhanh/chậm:" if not is_en else "Speech Reading Speed:"), 0, 0)
        speed_row = QHBoxLayout()
        self.combo_speed_preset = QComboBox()
        for label, val in self.SPEED_PRESETS:
            self.combo_speed_preset.addItem(label, val)
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.75, 2.00)
        self.spin_speed.setSingleStep(0.05)
        self.spin_speed.setSuffix("x")
        cur_rate = float(getattr(settings, "speech_rate", 1.0))
        self.spin_speed.setValue(cur_rate)

        # Match closest preset
        best_idx = min(range(len(self.SPEED_PRESETS)), key=lambda i: abs(self.SPEED_PRESETS[i][1] - cur_rate))
        self.combo_speed_preset.setCurrentIndex(best_idx)
        self.combo_speed_preset.currentIndexChanged.connect(self._on_preset_changed)

        speed_row.addWidget(self.combo_speed_preset, stretch=1)
        speed_row.addWidget(self.spin_speed)
        vg.addLayout(speed_row, 0, 1)

        hint = QLabel(
            "💡 Tăng tốc độ (1.25x - 1.75x) giúp đọc kịp khi nhiều xe mắc lỗi liên tiếp."
            if not is_en
            else "💡 Higher speed (1.25x - 1.75x) helps read alerts faster during heavy violations."
        )
        hint.setStyleSheet("color: #8b949e; font-size: 11px;")
        vg.addWidget(hint, 1, 0, 1, 2)

        self.chk_voice_enabled = QCheckBox(
            "Tự động đọc thành tiếng khi phát hiện vi phạm mới"
            if not is_en
            else "Automatically speak aloud when new violations occur"
        )
        self.chk_voice_enabled.setChecked(bool(settings.voice_enabled))
        vg.addWidget(self.chk_voice_enabled, 2, 0, 1, 2)

        self.chk_ai_polish = QCheckBox(
            "Dùng AI Gemini chuẩn hóa câu && đọc giọng tự nhiên (Google AI Studio)"
            if not is_en
            else "Use Gemini AI sentence polish && neural TTS (Google AI Studio)"
        )
        self.chk_ai_polish.setChecked(bool(settings.use_ai_text_polish))
        vg.addWidget(self.chk_ai_polish, 3, 0, 1, 2)

        vg.addWidget(QLabel("Khoảng chống lặp (giây):" if not is_en else "Alert Cooldown (sec):"), 4, 0)
        self.spin_cooldown = QDoubleSpinBox()
        self.spin_cooldown.setRange(1.0, 60.0)
        self.spin_cooldown.setSingleStep(1.0)
        self.spin_cooldown.setValue(float(settings.cooldown_seconds))
        vg.addWidget(self.spin_cooldown, 4, 1)

        vg.addWidget(QLabel("Google AI Studio Key:"), 5, 0)
        self.edit_api_key = QLineEdit(settings.google_ai_api_key)
        self.edit_api_key.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        vg.addWidget(self.edit_api_key, 5, 1)

        layout.addWidget(voice_group)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel" if is_en else "Hủy")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_save = QPushButton("💾 Save Settings" if is_en else "💾 Lưu Cài Đặt")
        btn_save.setStyleSheet("background-color: #238636; border-color: #2ea043; color: white;")
        btn_save.clicked.connect(self._save)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

    def _on_preset_changed(self, idx: int) -> None:
        val = self.combo_speed_preset.itemData(idx)
        if val is not None:
            self.spin_speed.setValue(float(val))

    def _save(self) -> None:
        updated = AISettings(
            google_ai_api_key=self.edit_api_key.text().strip() or self.settings.google_ai_api_key,
            text_model=self.settings.text_model,
            tts_model=self.settings.tts_model,
            tts_voice=self.settings.tts_voice,
            voice_enabled=self.chk_voice_enabled.isChecked(),
            use_ai_text_polish=self.chk_ai_polish.isChecked(),
            use_ai_tts_audio=self.chk_ai_polish.isChecked(),
            cooldown_seconds=float(self.spin_cooldown.value()),
            ui_language=str(self.combo_ui_lang.currentData() or "vi"),
            voice_language=str(self.combo_voice_lang.currentData() or "vi"),
            speech_rate=round(float(self.spin_speed.value()), 2),
        )
        save_ai_settings(updated)
        self.saved_settings = updated
        self.accept()


class TrafficMonitoringWindow(QMainWindow):
    """
    Main PyQt6 Application Window for Smart Urban Traffic Monitoring System (DACN2).
    - Runs AI inference in a dedicated background QThread (VideoInferenceWorker)
    - Honest ANPR with confidence & UNREADABLE marking
    - Multi-frame confirmation & manual verification workflow
    - Search/filter & full CSV export
    """

    def __init__(self, initial_cam_key: str = "3"):
        super().__init__()
        self.setWindowTitle("Hệ Thống Giám Sát Giao Thông Đô Thị Thông Minh - AI Core & Honest ANPR (DACN2)")
        self._apply_dark_theme()

        self.config_mgr = ConfigManager("configs/camera_configs.json")
        self.db = ViolationDatabase()
        self.anpr = LicensePlateANPR("violations")
        self.detector = UnifiedTrafficDetector()
        self.ai_settings = load_ai_settings()

        self.current_cam_key = initial_cam_key
        self.is_paused: bool = False
        self._last_canvas_frame: Optional[np.ndarray] = None
        self._violations_by_row: Dict[int, ViolationRecord] = {}

        self._build_ui()
        self._apply_ui_language()
        self._fit_window_to_screen()

        self.voice_worker = ViolationVoiceWorker(self.ai_settings, self)
        self.voice_worker.narration_ready.connect(self._on_voice_narration_ready)
        self.voice_worker.start()

        initial_cfg = self.config_mgr.get_config(self.current_cam_key)
        self._sync_ui_controls_from_config(initial_cfg)
        self._refresh_violation_table()

        # Start background inference thread
        self.worker = VideoInferenceWorker(
            detector=self.detector,
            anpr=self.anpr,
            db=self.db,
            initial_config=initial_cfg,
            parent=self,
        )
        self.worker.frame_ready.connect(self._on_worker_frame_ready)
        self.worker.stream_error.connect(self._on_worker_stream_error)
        self.worker.status_message.connect(self._on_worker_status_message)
        self.worker.start()

        for k, btn in self.cam_buttons.items():
            btn.setChecked(k == self.current_cam_key)

    def _fit_window_to_screen(self) -> None:
        """Adapts window size to fit the user's screen resolution and Windows DPI scaling."""
        self.setMinimumSize(1000, 600)
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            target_w = min(int(avail.width() * 0.96), 1460)
            target_h = min(int(avail.height() * 0.92), 850)
            self.resize(max(1000, target_w), max(600, target_h))
            # Center window inside available desktop area
            x = avail.x() + (avail.width() - self.width()) // 2
            y = avail.y() + (avail.height() - self.height()) // 2
            self.move(max(avail.x(), x), max(avail.y(), y))
        else:
            self.resize(1280, 720)

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #0b0f17;
                color: #e6edf3;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
            QToolTip {
                background-color: #1b2330;
                color: #e6edf3;
                border: 1px solid #388bfd;
                border-radius: 5px;
                padding: 5px 8px;
                font-size: 11px;
            }
            QGroupBox {
                background-color: #131922;
                border: 1px solid #263042;
                border-radius: 8px;
                margin-top: 8px;
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
                background-color: #1b2330;
                border: 1px solid #2b3648;
                border-radius: 6px;
                padding: 5px 11px;
                min-height: 22px;
                color: #e6edf3;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #243042;
                border-color: #58a6ff;
            }
            QPushButton:pressed {
                background-color: #16202e;
            }
            QPushButton:checked {
                background-color: #1f6feb;
                border-color: #58a6ff;
                color: #ffffff;
            }
            QPushButton:disabled {
                background-color: #131922;
                color: #586270;
                border-color: #1e2633;
            }
            QSplitter::handle {
                background-color: #1e2633;
                width: 4px;
                border-radius: 2px;
                margin: 4px 1px;
            }
            QSplitter::handle:hover {
                background-color: #388bfd;
            }
            QTabWidget::pane {
                border: 1px solid #263042;
                border-radius: 8px;
                background-color: #131922;
                top: -1px;
            }
            QTabBar::tab {
                background-color: #131922;
                color: #8b949e;
                border: 1px solid #263042;
                border-bottom: none;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                padding: 7px 14px;
                margin-right: 4px;
                font-weight: 600;
            }
            QTabBar::tab:hover {
                color: #e6edf3;
                background-color: #1b2330;
            }
            QTabBar::tab:selected {
                background-color: #1f6feb;
                color: #ffffff;
                border-color: #388bfd;
            }
            QTableWidget {
                background-color: #0d121b;
                alternate-background-color: #111824;
                border: 1px solid #263042;
                border-radius: 6px;
                gridline-color: #1e2736;
                selection-background-color: #1b3a66;
                selection-color: #ffffff;
            }
            QTableWidget::item {
                padding: 2px 4px;
            }
            QTableCornerButton::section {
                background-color: #1b2330;
                border: none;
            }
            QHeaderView::section {
                background-color: #1b2330;
                color: #79c0ff;
                font-weight: 700;
                font-size: 11px;
                padding: 6px 4px;
                border: none;
                border-right: 1px solid #263042;
                border-bottom: 1px solid #263042;
            }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                background-color: #0d121b;
                border: 1px solid #2b3648;
                border-radius: 5px;
                padding: 5px 8px;
                color: #e6edf3;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
                border-color: #58a6ff;
            }
            QComboBox QAbstractItemView {
                background-color: #131922;
                border: 1px solid #30363d;
                selection-background-color: #1f6feb;
                color: #e6edf3;
            }
            QCheckBox {
                spacing: 8px;
                color: #d0d7de;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #3b4758;
                background-color: #0d121b;
            }
            QCheckBox::indicator:checked {
                background-color: #1f6feb;
                border-color: #58a6ff;
            }
            QScrollBar:vertical {
                background: #0d121b;
                width: 8px;
                margin: 2px;
                border: none;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #2b3648;
                min-height: 28px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #48566e;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: #0d121b;
            }
            QScrollBar:horizontal {
                background: #0d121b;
                height: 8px;
                margin: 2px;
                border: none;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #2b3648;
                min-width: 28px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #48566e;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: #0d121b;
            }
            """
        )

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(7)

        # 1. HEADER BAR + DEDICATED CAMERA SWITCHER STRIP
        header_card = QFrame()
        header_card.setStyleSheet(
            "QFrame { background-color: #131922; border: 1px solid #263042; border-radius: 8px; }"
        )
        header_v = QVBoxLayout(header_card)
        header_v.setContentsMargins(10, 6, 10, 6)
        header_v.setSpacing(6)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        self.title_lbl = QLabel("🚦 HỆ THỐNG GIÁM SÁT GIAO THÔNG & PHẠT NGUỘI")
        self.title_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.title_lbl.setStyleSheet("color: #60a5fa; border: none; background: transparent;")
        self.title_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.title_lbl.setMinimumWidth(260)
        top_bar.addWidget(self.title_lbl, stretch=1)

        self.lbl_stream_badge = QLabel("● READY")
        self.lbl_stream_badge.setFixedHeight(28)
        self.lbl_stream_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_stream_badge.setStyleSheet(
            "background-color: #123d24; color: #3fb950; border: 1px solid #2ea043; "
            "border-radius: 14px; padding: 2px 12px; font-weight: bold; font-size: 11px;"
        )
        top_bar.addWidget(self.lbl_stream_badge)

        self.btn_voice_ai = QPushButton(
            "🔊 Đọc lỗi AI: BẬT" if self.ai_settings.voice_enabled else "🔇 Đọc lỗi AI: TẮT"
        )
        self.btn_voice_ai.setCheckable(True)
        self.btn_voice_ai.setChecked(self.ai_settings.voice_enabled)
        self.btn_voice_ai.setToolTip("Bật/Tắt tự động đọc lỗi vi phạm qua AI Google Studio")
        self.btn_voice_ai.clicked.connect(self._toggle_voice_ai)
        top_bar.addWidget(self.btn_voice_ai)

        self.btn_settings = QPushButton("⚙️ Cài đặt")
        self.btn_settings.setToolTip("Cài đặt tốc độ đọc, ngôn ngữ hệ thống & giọng đọc")
        self.btn_settings.clicked.connect(self._open_settings_dialog)
        top_bar.addWidget(self.btn_settings)

        self.btn_reset_state = QPushButton("🔁 Reset Bộ Đếm")
        self.btn_reset_state.setToolTip("Reset bộ đếm phương tiện và trạng thái theo dõi ByteTrack")
        self.btn_reset_state.clicked.connect(self._reset_tracking_state)
        top_bar.addWidget(self.btn_reset_state)

        self.btn_pause = QPushButton("⏸ Tạm dừng")
        self.btn_pause.setCheckable(True)
        self.btn_pause.setToolTip("Tạm dừng / Tiếp tục xử lý luồng video hiện tại")
        self.btn_pause.clicked.connect(self._toggle_pause)
        top_bar.addWidget(self.btn_pause)

        header_v.addLayout(top_bar)

        # Subtle horizontal divider between title bar and camera switcher
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #202938; border: none;")
        header_v.addWidget(sep)

        cam_strip_layout = QHBoxLayout()
        cam_strip_layout.setContentsMargins(0, 0, 0, 0)
        cam_strip_layout.setSpacing(6)

        self.cam_buttons: Dict[str, QPushButton] = {}
        self.cam_buttons_layout = QHBoxLayout()
        self.cam_buttons_layout.setSpacing(5)
        cam_strip_layout.addLayout(self.cam_buttons_layout)

        self.lbl_active_cam_name = QLabel("")
        self.lbl_active_cam_name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.lbl_active_cam_name.setStyleSheet(
            "color: #93c5fd; font-size: 11px; font-weight: 600; padding: 0 8px; border: none; background: transparent;"
        )
        cam_strip_layout.addWidget(self.lbl_active_cam_name, stretch=1)

        self.btn_add_cam = QPushButton("➕ Thêm Cam Mới")
        self.btn_add_cam.setStyleSheet("background-color: #238636; border-color: #2ea043; color: white;")
        self.btn_add_cam.setToolTip("Thêm nguồn video MP4 hoặc luồng camera RTSP/Webcam mới")
        self.btn_add_cam.clicked.connect(self._open_add_camera_dialog)
        cam_strip_layout.addWidget(self.btn_add_cam)

        self.btn_del_cam = QPushButton("🗑️ Xóa Cam")
        self.btn_del_cam.setStyleSheet("background-color: #6e2020; border-color: #f85149; color: #ffdfdb;")
        self.btn_del_cam.clicked.connect(self._delete_current_camera)
        cam_strip_layout.addWidget(self.btn_del_cam)

        header_v.addLayout(cam_strip_layout)
        root_layout.addWidget(header_card)

        # 2. MAIN SPLIT CONTENT (User-resizable horizontal splitter)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(5)

        # --- LEFT PANEL: Video Canvas & Interactive Mouse Drawing Toolbar ---
        left_widget = QWidget()
        left_col = QVBoxLayout(left_widget)
        left_col.setContentsMargins(0, 0, 2, 0)
        left_col.setSpacing(6)

        self.video_canvas = InteractiveVideoCanvas(self)
        self.video_canvas.line_updated.connect(self._on_mouse_line_updated)
        self.video_canvas.roi_updated.connect(self._on_mouse_roi_updated)
        self.video_canvas.divider_updated.connect(self._on_mouse_divider_updated)
        self.video_canvas.mode_finished.connect(self._on_draw_mode_finished)
        left_col.addWidget(self.video_canvas, stretch=1)

        draw_bar = QHBoxLayout()
        draw_bar.setSpacing(6)
        self.btn_draw_line = QPushButton("✏️ Vẽ Vạch Đếm")
        self.btn_draw_line.setCheckable(True)
        self.btn_draw_line.setToolTip("Kéo thả chuột trái trên video để vẽ Vạch đếm mới (Chuột phải để hủy)")
        self.btn_draw_line.setStyleSheet("background-color: #163b25; border-color: #2ea043;")
        self.btn_draw_line.clicked.connect(lambda: self._activate_draw_mode("DRAW_LINE"))
        draw_bar.addWidget(self.btn_draw_line)

        self.btn_draw_roi = QPushButton("⬠ Vẽ Vùng ROI")
        self.btn_draw_roi.setCheckable(True)
        self.btn_draw_roi.setToolTip("Click chuột trái chọn các đỉnh vùng ROI, click chuột phải để hoàn tất")
        self.btn_draw_roi.setStyleSheet("background-color: #3a204d; border-color: #8957e5;")
        self.btn_draw_roi.clicked.connect(lambda: self._activate_draw_mode("DRAW_ROI"))
        draw_bar.addWidget(self.btn_draw_roi)

        self.btn_draw_divider = QPushButton("🛣️ Vẽ Phân Làn")
        self.btn_draw_divider.setCheckable(True)
        self.btn_draw_divider.setToolTip("Kéo thả chuột trái dọc theo vạch phân làn giữa 2 chiều đường")
        self.btn_draw_divider.clicked.connect(lambda: self._activate_draw_mode("DRAW_DIVIDER"))
        draw_bar.addWidget(self.btn_draw_divider)

        self.btn_reset_cfg = QPushButton("🔄 Mặc định Cam")
        self.btn_reset_cfg.setToolTip("Khôi phục Vạch đếm, ROI & luật mặc định của Camera đang chọn")
        self.btn_reset_cfg.clicked.connect(self._reset_current_cam_config)
        draw_bar.addWidget(self.btn_reset_cfg)

        left_col.addLayout(draw_bar)

        self.status_banner = QLabel(
            "💡 Hệ thống chạy đa luồng (QThread AI Worker) — Không sinh biển số giả; hỗ trợ kiểm tra & xác nhận thủ công."
        )
        self.status_banner.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.status_banner.setFixedHeight(28)
        self.status_banner.setStyleSheet(
            "background-color: #131922; border: 1px solid #263042; border-radius: 6px; "
            "color: #9da7b3; font-size: 11px; padding: 3px 10px;"
        )
        left_col.addWidget(self.status_banner)
        main_splitter.addWidget(left_widget)

        # --- RIGHT PANEL: Live Telemetry + Tabbed Workspace (Violations vs Rules) ---
        right_widget = QWidget()
        right_col = QVBoxLayout(right_widget)
        right_col.setContentsMargins(2, 0, 0, 0)
        right_col.setSpacing(7)

        # CARD A: Live Telemetry (Clean single-header Card + 6 Accent KPI Cards)
        self.telemetry_group = QFrame()
        self.telemetry_group.setStyleSheet(
            "QFrame#telemetryCard { background-color: #131922; border: 1px solid #263042; border-radius: 8px; }"
        )
        self.telemetry_group.setObjectName("telemetryCard")
        tel_layout = QVBoxLayout(self.telemetry_group)
        tel_layout.setContentsMargins(10, 8, 10, 9)
        tel_layout.setSpacing(7)

        tel_hdr = QHBoxLayout()
        tel_hdr.setSpacing(8)
        self.lbl_telemetry_title = QLabel("📊 THỐNG KÊ LƯU LƯỢNG & TRẠNG THÁI")
        self.lbl_telemetry_title.setStyleSheet(
            "color: #60a5fa; font-weight: bold; font-size: 11px; border: none; background: transparent;"
        )
        tel_hdr.addWidget(self.lbl_telemetry_title)
        tel_hdr.addStretch()

        self.lbl_traffic_status = QLabel("✅ THÔNG THOÁNG (NORMAL)")
        self.lbl_traffic_status.setFixedHeight(24)
        self.lbl_traffic_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_traffic_status.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_traffic_status.setStyleSheet(
            "background-color: #123d24; color: #3fb950; border: 1px solid #2ea043; "
            "border-radius: 12px; padding: 2px 10px;"
        )
        tel_hdr.addWidget(self.lbl_traffic_status)
        tel_layout.addLayout(tel_hdr)

        grid_counts = QGridLayout()
        grid_counts.setSpacing(6)
        self.lbl_counts: Dict[str, QLabel] = {}
        self.lbl_card_titles: Dict[str, QLabel] = {}
        cards_meta = [
            ("Car", "🚗 Ô tô (Car)", "#3fb950", 0, 0),
            ("Motorbike", "🛵 Xe máy (Motor)", "#eab308", 0, 1),
            ("Bus", "🚌 Xe buýt (Bus)", "#bc8cff", 0, 2),
            ("Truck", "🚛 Xe tải (Truck)", "#f0883e", 1, 0),
            ("AvgSpeed", "⚡ Tốc độ TB", "#60a5fa", 1, 1),
            ("InROI", "📍 Xe trong ROI", "#f87171", 1, 2),
        ]
        for key, title, color, r, c in cards_meta:
            frame = QFrame()
            frame.setMinimumHeight(48)
            frame.setStyleSheet(
                f"QFrame {{ background-color: #0d121b; border: 1px solid #222c3c; "
                f"border-left: 3px solid {color}; border-radius: 6px; }}"
            )
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(9, 4, 8, 4)
            fl.setSpacing(1)
            t_lbl = QLabel(title)
            t_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            t_lbl.setStyleSheet("color: #8b949e; font-size: 11px; border: none; background: transparent;")
            v_lbl = QLabel("0")
            v_lbl.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            v_lbl.setStyleSheet(f"color: {color}; border: none; background: transparent;")
            fl.addWidget(t_lbl)
            fl.addWidget(v_lbl)
            self.lbl_card_titles[key] = t_lbl
            self.lbl_counts[key] = v_lbl
            grid_counts.addWidget(frame, r, c)

        tel_layout.addLayout(grid_counts)
        right_col.addWidget(self.telemetry_group)

        # TABBED RIGHT WORKSPACE: Violation Log (Tab 0) & Rules/Calibration (Tab 1)
        self.right_tabs = QTabWidget()

        # TAB 0: Searchable Violation Log & Manual Verification (No redundant nested QGroupBox)
        tab_viol = QWidget()
        tab_viol.setStyleSheet("background-color: transparent;")
        viol_layout = QVBoxLayout(tab_viol)
        viol_layout.setContentsMargins(9, 9, 9, 9)
        viol_layout.setSpacing(7)

        # Search & Filter bar
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(6)
        self.combo_filter_cam = QComboBox()
        self.combo_filter_cam.setMaximumWidth(82)
        self.combo_filter_cam.addItems(["ALL", "CAM 1", "CAM 2", "CAM 3", "CAM 4"])
        self.combo_filter_cam.currentIndexChanged.connect(self._refresh_violation_table)
        filter_bar.addWidget(self.combo_filter_cam)

        self.combo_filter_status = QComboBox()
        self.combo_filter_status.setMaximumWidth(145)
        self.combo_filter_status.addItem("Tất cả trạng thái", "ALL")
        for st in VERIFICATION_STATUSES:
            self.combo_filter_status.addItem(st, st)
        self.combo_filter_status.currentIndexChanged.connect(self._refresh_violation_table)
        filter_bar.addWidget(self.combo_filter_status)

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("🔍 Tìm biển số, loại xe, lỗi vi phạm...")
        self.edit_search.textChanged.connect(self._refresh_violation_table)
        filter_bar.addWidget(self.edit_search, stretch=1)

        self.lbl_viol_count = QLabel("0")
        self.lbl_viol_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_viol_count.setToolTip("Số sự kiện khớp bộ lọc hiện tại")
        self.lbl_viol_count.setStyleSheet(
            "background-color: #1b2330; color: #93c5fd; border: 1px solid #2b3648; "
            "border-radius: 5px; padding: 4px 8px; font-weight: bold; font-size: 11px;"
        )
        filter_bar.addWidget(self.lbl_viol_count)

        viol_layout.addLayout(filter_bar)

        self.table_violations = QTableWidget(0, 7)
        self.table_violations.setHorizontalHeaderLabels(
            ["Giờ", "Cam", "Xe", "Biển số", "Conf", "Tốc độ", "Xác minh / Lỗi"]
        )
        hdr = self.table_violations.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        hdr.setMinimumSectionSize(38)
        self.table_violations.setWordWrap(False)
        self.table_violations.verticalHeader().setVisible(False)
        self.table_violations.verticalHeader().setDefaultSectionSize(29)
        self.table_violations.setAlternatingRowColors(True)
        self.table_violations.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_violations.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_violations.itemSelectionChanged.connect(self._on_violation_row_selected)
        self.table_violations.doubleClicked.connect(self._open_selected_evidence_dialog)
        viol_layout.addWidget(self.table_violations, stretch=1)

        # ANPR Plate Zoom & Action Bar (Dynamic height so details never clip vertically!)
        anpr_box = QFrame()
        anpr_box.setMinimumHeight(88)
        anpr_box.setStyleSheet(
            "QFrame { background-color: #0d121b; border: 1px solid #263042; border-radius: 7px; }"
        )
        anpr_layout = QHBoxLayout(anpr_box)
        anpr_layout.setContentsMargins(8, 7, 8, 7)
        anpr_layout.setSpacing(8)

        self.lbl_plate_thumb = QLabel("Chưa chọn\nsự kiện")
        self.lbl_plate_thumb.setFixedSize(148, 68)
        self.lbl_plate_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_plate_thumb.setStyleSheet(
            "background-color: #131922; border: 1px solid #b4821c; border-radius: 5px; color: #8b949e; font-size: 11px;"
        )
        anpr_layout.addWidget(self.lbl_plate_thumb, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.lbl_anpr_details = QLabel(
            "<b>Biển số ANPR:</b> ---<br>"
            "<b>Xác minh:</b> Chọn 1 dòng để xem bằng chứng & duyệt thủ công"
        )
        self.lbl_anpr_details.setWordWrap(True)
        self.lbl_anpr_details.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_anpr_details.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.MinimumExpanding)
        self.lbl_anpr_details.setStyleSheet("font-size: 11px; border: none; background: transparent;")
        anpr_layout.addWidget(self.lbl_anpr_details, stretch=1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)
        self.btn_view_evidence = QPushButton("🔍 Duyệt / Bằng chứng")
        self.btn_view_evidence.setFixedWidth(154)
        self.btn_view_evidence.setStyleSheet("background-color: #1f6feb; border-color: #388bfd; color: white;")
        self.btn_view_evidence.clicked.connect(self._open_selected_evidence_dialog)
        btn_col.addWidget(self.btn_view_evidence)

        self.btn_speak_selected = QPushButton("🔊 Đọc lỗi (AI)")
        self.btn_speak_selected.setFixedWidth(154)
        self.btn_speak_selected.clicked.connect(self._speak_selected_violation)
        btn_col.addWidget(self.btn_speak_selected)

        self.btn_export_csv = QPushButton("📥 Xuất CSV")
        self.btn_export_csv.setFixedWidth(154)
        self.btn_export_csv.clicked.connect(self._export_csv_report)
        btn_col.addWidget(self.btn_export_csv)

        anpr_layout.addLayout(btn_col)
        viol_layout.addWidget(anpr_box)
        self.right_tabs.addTab(tab_viol, "🚨 Vi phạm && Duyệt ANPR")

        # TAB 1: Traffic Rules & Calibration Config
        tab_rules = QWidget()
        tab_rules.setStyleSheet("background-color: transparent;")
        tab_rules_layout = QVBoxLayout(tab_rules)
        tab_rules_layout.setContentsMargins(10, 10, 10, 10)
        tab_rules_layout.setSpacing(10)

        self.lbl_rules_header = QLabel("⚙️ CẤU HÌNH LUẬT, HIỆU CHUẨN TỐC ĐỘ && NGƯỠNG XÁC NHẬN")
        self.lbl_rules_header.setStyleSheet(
            "color: #60a5fa; font-weight: bold; font-size: 11px; padding-bottom: 4px; border-bottom: 1px solid #263042;"
        )
        tab_rules_layout.addWidget(self.lbl_rules_header)

        rules_layout = QGridLayout()
        rules_layout.setVerticalSpacing(10)
        rules_layout.setHorizontalSpacing(10)

        self.lbl_speed_limit_txt = QLabel("Tốc độ tối đa (km/h):")
        rules_layout.addWidget(self.lbl_speed_limit_txt, 0, 0)
        self.spin_speed_limit = QDoubleSpinBox()
        self.spin_speed_limit.setRange(20.0, 160.0)
        self.spin_speed_limit.setSingleStep(5.0)
        self.spin_speed_limit.valueChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.spin_speed_limit, 0, 1)

        self.lbl_cong_txt = QLabel("Ngưỡng kẹt xe (ROI):")
        rules_layout.addWidget(self.lbl_cong_txt, 0, 2)
        self.spin_congestion = QSpinBox()
        self.spin_congestion.setRange(3, 50)
        self.spin_congestion.valueChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.spin_congestion, 0, 3)

        self.lbl_dir_txt = QLabel("Chiều làn đường:")
        rules_layout.addWidget(self.lbl_dir_txt, 1, 0)
        self.combo_direction = QComboBox()
        for label_txt, mode_val in DIRECTION_MODE_ITEMS:
            self.combo_direction.addItem(label_txt, mode_val)
        self.combo_direction.currentIndexChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.combo_direction, 1, 1, 1, 3)

        self.chk_no_truck = QCheckBox("Bắt cấm xe tải (P.106a)")
        self.chk_no_truck.stateChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.chk_no_truck, 2, 0, 1, 2)

        self.chk_speed = QCheckBox("Bắt quá tốc độ")
        self.chk_speed.stateChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.chk_speed, 2, 2, 1, 2)

        self.chk_wrong_way = QCheckBox("Bắt ngược chiều / Lấn làn")
        self.chk_wrong_way.stateChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.chk_wrong_way, 3, 0, 1, 2)

        self.chk_illegal_parking = QCheckBox("Bắt dừng đỗ trái phép")
        self.chk_illegal_parking.stateChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.chk_illegal_parking, 3, 2, 1, 2)

        self.chk_calibrated = QCheckBox("Đã hiệu chuẩn Homography (m)")
        self.chk_calibrated.stateChanged.connect(self._on_rule_param_changed)
        rules_layout.addWidget(self.chk_calibrated, 4, 0, 1, 4)

        tab_rules_layout.addLayout(rules_layout)

        self.lbl_rule_hint = QLabel(
            "💡 Mẹo vận hành:\n"
            "• Các thay đổi thông số được lưu tự động cho riêng Camera đang chọn.\n"
            "• Dùng các nút 'Vẽ Vạch Đếm', 'Vẽ Vùng ROI', 'Vẽ Phân Làn' bên dưới khung hình để chỉnh nhanh bằng chuột."
        )
        self.lbl_rule_hint.setWordWrap(True)
        self.lbl_rule_hint.setStyleSheet(
            "background-color: #0d121b; border: 1px dashed #2b3648; border-radius: 6px; padding: 10px; color: #8b949e; font-size: 11px;"
        )
        tab_rules_layout.addWidget(self.lbl_rule_hint)
        tab_rules_layout.addStretch()
        self.right_tabs.addTab(tab_rules, "⚙️ Luật && Hiệu chuẩn")

        right_col.addWidget(self.right_tabs, stretch=1)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 53)
        main_splitter.setStretchFactor(1, 47)
        main_splitter.setSizes([680, 600])

        root_layout.addWidget(main_splitter, stretch=1)

        self._rebuild_camera_buttons()

    def _rebuild_camera_buttons(self) -> None:
        """Dynamically builds top-bar Camera buttons and updates the Camera Filter dropdown."""
        while self.cam_buttons_layout.count():
            item = self.cam_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.cam_buttons.clear()

        short_Builtin = {
            "1": "📹 CAM 1",
            "2": "📹 CAM 2",
            "3": "📹 CAM 3",
            "4": "📹 CAM 4",
            "5": "📹 CAM 5",
        }
        sorted_keys = sorted(
            self.config_mgr.configs.keys(),
            key=lambda k: (0, int(k)) if k.isdigit() else (1, k),
        )
        for key in sorted_keys:
            cfg = self.config_mgr.configs[key]
            btn_label = short_Builtin.get(key, f"📹 CAM {key}")
            btn = QPushButton(btn_label)
            btn.setCheckable(True)
            btn.setChecked(key == self.current_cam_key)
            btn.setToolTip(f"{cfg.name}\nNguồn: {cfg.video_path}\nPhím tắt: [{key}]")
            btn.clicked.connect(lambda checked, k=key: self.switch_camera(k))
            self.cam_buttons[key] = btn
            self.cam_buttons_layout.addWidget(btn)

        # Update camera filter dropdown
        if hasattr(self, "combo_filter_cam"):
            prev_filter = self.combo_filter_cam.currentText()
            self.combo_filter_cam.blockSignals(True)
            self.combo_filter_cam.clear()
            self.combo_filter_cam.addItem("ALL")
            for key in sorted_keys:
                self.combo_filter_cam.addItem(f"CAM {key}")
            if prev_filter in [self.combo_filter_cam.itemText(i) for i in range(self.combo_filter_cam.count())]:
                self.combo_filter_cam.setCurrentText(prev_filter)
            self.combo_filter_cam.blockSignals(False)

        if hasattr(self, "btn_del_cam"):
            self.btn_del_cam.setEnabled(self.current_cam_key not in ("1", "2", "3", "4"))

    def switch_camera(self, cam_key: str) -> None:
        self.current_cam_key = cam_key
        for k, btn in self.cam_buttons.items():
            btn.setChecked(k == cam_key)
        if hasattr(self, "btn_del_cam"):
            self.btn_del_cam.setEnabled(cam_key not in ("1", "2", "3", "4"))

        cfg = self.config_mgr.get_config(cam_key)
        self._sync_ui_controls_from_config(cfg)
        if hasattr(self, "worker"):
            self.worker.request_camera_switch(cfg)

    def _open_add_camera_dialog(self) -> None:
        next_key = self.config_mgr.get_next_cam_key()
        dlg = AddCameraDialog(next_key, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.created_config is not None:
            new_cfg = self.config_mgr.add_camera(dlg.created_config)
            self._rebuild_camera_buttons()
            self.switch_camera(new_cfg.cam_key)
            self.status_banner.setText(
                f"✅ Đã thêm thành công [{new_cfg.name}] (Phím tắt [{new_cfg.cam_key}]) từ file: {new_cfg.video_path}"
            )

    def _delete_current_camera(self) -> None:
        if self.current_cam_key in ("1", "2", "3", "4"):
            QMessageBox.information(
                self, "Không thể xóa", "CAM 1..4 là 4 kênh tiêu chuẩn của hệ thống. Chỉ có thể xóa các Camera thêm mới (CAM 5+)."
            )
            return

        cfg = self.config_mgr.get_config(self.current_cam_key)
        ans = QMessageBox.question(
            self,
            "Xác nhận xóa Camera",
            f"Bạn có chắc chắn muốn xóa '{cfg.name}' khỏi danh sách kênh Camera?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            removed_name = cfg.name
            self.config_mgr.remove_camera(self.current_cam_key)
            self.current_cam_key = "1"
            self._rebuild_camera_buttons()
            self.switch_camera("1")
            self.status_banner.setText(f"🗑️ Đã xóa kênh '{removed_name}' và chuyển về CAM 1.")

    def _reset_tracking_state(self) -> None:
        if hasattr(self, "worker"):
            self.worker.request_state_reset()
            self.status_banner.setText("🔁 Đã yêu cầu reset bộ đếm và trạng thái ByteTrack cho Camera hiện tại.")

    def _sync_ui_controls_from_config(self, cfg: CameraConfig) -> None:
        self._updating_ui = True
        if hasattr(self, "lbl_active_cam_name"):
            self.lbl_active_cam_name.setText(f"• {cfg.name}")
            self.lbl_active_cam_name.setToolTip(f"{cfg.name}\n{cfg.video_path}")
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
        self.chk_illegal_parking.setChecked(getattr(cfg, "enable_illegal_parking_rule", False))
        self.chk_calibrated.setChecked(cfg.is_speed_calibrated)
        self._updating_ui = False

    def _on_rule_param_changed(self) -> None:
        if getattr(self, "_updating_ui", False):
            return
        cfg = self.config_mgr.get_config(self.current_cam_key)
        cfg.speed_limit_kmh = float(self.spin_speed_limit.value())
        cfg.congestion_threshold = int(self.spin_congestion.value())
        cfg.direction_mode = str(self.combo_direction.currentData())
        cfg.enable_no_truck_rule = self.chk_no_truck.isChecked()
        cfg.enable_speed_rule = self.chk_speed.isChecked()
        cfg.enable_wrong_way_rule = self.chk_wrong_way.isChecked()
        cfg.enable_illegal_parking_rule = self.chk_illegal_parking.isChecked()
        cfg.is_speed_calibrated = self.chk_calibrated.isChecked()

        ok, msg = cfg.validate()
        if not ok:
            QMessageBox.warning(self, "Cấu hình không hợp lệ", msg)
            return
        self.config_mgr.save()

    def _toggle_pause(self, checked: bool) -> None:
        self.is_paused = checked
        is_en = str(getattr(self.ai_settings, "ui_language", "vi")).lower() == "en"
        self.btn_pause.setText(("▶ Resume" if checked else "⏸ Pause") if is_en else ("▶ Tiếp tục" if checked else "⏸ Tạm dừng"))
        if checked and hasattr(self, "lbl_stream_badge"):
            self.lbl_stream_badge.setText("⏸ PAUSED")
            self.lbl_stream_badge.setStyleSheet(
                "background-color: #3d2e12; color: #d29922; border: 1px solid #d29922; "
                "border-radius: 13px; padding: 2px 12px; font-weight: bold; font-size: 11px;"
            )
        if hasattr(self, "worker"):
            self.worker.set_paused(checked)

    def _activate_draw_mode(self, mode: str) -> None:
        self.btn_draw_line.setChecked(mode == "DRAW_LINE")
        self.btn_draw_roi.setChecked(mode == "DRAW_ROI")
        self.btn_draw_divider.setChecked(mode == "DRAW_DIVIDER")
        self.video_canvas.set_draw_mode(mode)
        if mode == "DRAW_LINE":
            self.status_banner.setText("✏️ Kéo thả chuột trái trên video để vẽ Vạch đếm mới (Chuột phải để hủy).")
        elif mode == "DRAW_ROI":
            self.status_banner.setText("⬠ Click chuột trái chọn các đỉnh đa giác ROI, sau đó click CHUỘT PHẢI để chốt.")
        elif mode == "DRAW_DIVIDER":
            self.status_banner.setText("🛣️ Kéo thả chuột trái dọc theo vạch phân làn giữa 2 chiều đường.")

    def _on_mouse_line_updated(self, points: list) -> None:
        cfg = self.config_mgr.get_config(self.current_cam_key)
        old_line = list(cfg.counting_line)
        cfg.counting_line = points
        ok, msg = cfg.validate()
        if not ok:
            cfg.counting_line = old_line
            QMessageBox.warning(self, "Vạch đếm không hợp lệ", msg)
            return
        self.config_mgr.save()
        self._rerender_paused_canvas()

    def _on_mouse_roi_updated(self, points: list) -> None:
        cfg = self.config_mgr.get_config(self.current_cam_key)
        old_roi = list(cfg.roi_polygon)
        cfg.roi_polygon = points
        ok, msg = cfg.validate()
        if not ok:
            cfg.roi_polygon = old_roi
            QMessageBox.warning(self, "Vùng ROI không hợp lệ", msg)
            return
        self.config_mgr.save()
        self._rerender_paused_canvas()

    def _on_mouse_divider_updated(self, points: list) -> None:
        cfg = self.config_mgr.get_config(self.current_cam_key)
        cfg.divider_line = points
        self.config_mgr.save()
        self._rerender_paused_canvas()

    def _on_draw_mode_finished(self, msg: str) -> None:
        self.btn_draw_line.setChecked(False)
        self.btn_draw_roi.setChecked(False)
        self.btn_draw_divider.setChecked(False)
        self.status_banner.setText(f"✅ {msg} (Đã kiểm tra hợp lệ & lưu vào configs/camera_configs.json)")

    def _reset_current_cam_config(self) -> None:
        cfg = self.config_mgr.reset_camera_to_default(self.current_cam_key)
        if hasattr(self, "worker") and self.worker.engine is not None:
            self.worker.engine.config = cfg
        self._sync_ui_controls_from_config(cfg)
        self._rerender_paused_canvas()
        self.status_banner.setText("🔄 Đã khôi phục cấu hình Vạch đếm & ROI mặc định cho Camera này.")

    def _rerender_paused_canvas(self) -> None:
        if self._last_canvas_frame is not None and hasattr(self, "worker") and self.worker.engine is not None:
            annotated = render_annotated_canvas(
                self._last_canvas_frame, self.worker.engine, self.worker.engine.config
            )
            self.video_canvas.update_frame(annotated)

    def _on_worker_frame_ready(
        self,
        annotated_canvas: np.ndarray,
        raw_canvas: np.ndarray,
        telemetry: dict,
        new_violations: List[ViolationRecord],
    ) -> None:
        self._last_canvas_frame = raw_canvas
        self.video_canvas.update_frame(annotated_canvas)
        self._update_telemetry_labels(telemetry)
        if new_violations:
            self._refresh_violation_table(select_top=True)
            if hasattr(self, "voice_worker") and self.ai_settings.voice_enabled:
                self.voice_worker.enqueue(new_violations[0], force=False)

    def _on_worker_stream_error(self, err_msg: str) -> None:
        self.status_banner.setText(f"❌ LỖI NGUỒN VIDEO: {err_msg}")

    def _on_worker_status_message(self, msg: str) -> None:
        self.status_banner.setText(msg)

    def _update_telemetry_labels(self, telemetry: dict) -> None:
        is_congested = telemetry.get("traffic_status") == "CONGESTED"
        in_roi = telemetry.get("vehicles_in_roi", 0)
        thresh = telemetry.get("congestion_threshold", 9)
        is_en = str(getattr(self.ai_settings, "ui_language", "vi")).lower() == "en"

        if is_congested:
            txt = f"⚠️ CONGESTED ({in_roi}/{thresh} ROI)" if is_en else f"⚠️ KẸT XE ({in_roi}/{thresh} xe ROI)"
            self.lbl_traffic_status.setText(txt)
            self.lbl_traffic_status.setStyleSheet(
                "background-color: #491217; color: #ff7b72; border: 1px solid #f85149; "
                "border-radius: 12px; padding: 2px 10px;"
            )
        else:
            txt = f"✅ NORMAL ({in_roi}/{thresh} ROI)" if is_en else f"✅ THÔNG THOÁNG ({in_roi}/{thresh} xe ROI)"
            self.lbl_traffic_status.setText(txt)
            self.lbl_traffic_status.setStyleSheet(
                "background-color: #123d24; color: #3fb950; border: 1px solid #2ea043; "
                "border-radius: 12px; padding: 2px 10px;"
            )

        counts = telemetry.get("counts", {})
        for v_type in ("Car", "Motorbike", "Bus", "Truck"):
            self.lbl_counts[v_type].setText(str(counts.get(v_type, 0)))

        if hasattr(self, "lbl_stream_badge") and not self.is_paused:
            is_live = bool(telemetry.get("is_live", False))
            health = telemetry.get("stream_health", "ONLINE")
            if health == "RECONNECTING":
                self.lbl_stream_badge.setText("🟡 RECONNECTING")
                self.lbl_stream_badge.setStyleSheet(
                    "background-color: #3d2e12; color: #d29922; border: 1px solid #d29922; "
                    "border-radius: 13px; padding: 2px 12px; font-weight: bold; font-size: 11px;"
                )
            elif is_live:
                self.lbl_stream_badge.setText("🔴 LIVE RTSP")
                self.lbl_stream_badge.setStyleSheet(
                    "background-color: #491217; color: #ff7b72; border: 1px solid #f85149; "
                    "border-radius: 13px; padding: 2px 12px; font-weight: bold; font-size: 11px;"
                )
            else:
                self.lbl_stream_badge.setText("🟢 VIDEO ONLINE")
                self.lbl_stream_badge.setStyleSheet(
                    "background-color: #123d24; color: #3fb950; border: 1px solid #2ea043; "
                    "border-radius: 13px; padding: 2px 12px; font-weight: bold; font-size: 11px;"
                )

        avg_spd = telemetry.get("avg_speed_kmh", 0.0)
        is_cal = telemetry.get("is_speed_calibrated", False)
        prefix = "" if is_cal else "~"
        self.lbl_counts["AvgSpeed"].setText(f"{prefix}{avg_spd:.1f} km/h")
        self.lbl_counts["InROI"].setText(f"{in_roi}/{thresh}")

    def _refresh_violation_table(self, select_top: bool = False) -> None:
        cam_filter = self.combo_filter_cam.currentText() if hasattr(self, "combo_filter_cam") else "ALL"
        status_filter = (
            str(self.combo_filter_status.currentData()) if hasattr(self, "combo_filter_status") else "ALL"
        )
        search_txt = self.edit_search.text() if hasattr(self, "edit_search") else ""

        records = self.db.query_violations(
            camera_filter=cam_filter,
            status_filter=status_filter,
            search_text=search_txt,
            limit=120,
        )

        self.table_violations.setRowCount(0)
        self._violations_by_row.clear()
        if hasattr(self, "lbl_viol_count"):
            is_en = str(getattr(self.ai_settings, "ui_language", "vi")).lower() == "en"
            self.lbl_viol_count.setText(f"{len(records)} events" if is_en else f"{len(records)} sự kiện")

        for row_idx, rec in enumerate(records):
            self.table_violations.insertRow(row_idx)
            self._violations_by_row[row_idx] = rec

            time_short = rec.timestamp.split(" ")[-1] if " " in rec.timestamp else rec.timestamp
            spd_prefix = "" if rec.is_speed_calibrated else "~"
            conf_str = f"{rec.plate_conf*100:.0f}%" if rec.plate_conf > 0 else "N/A"
            status_short = (
                "✅ Tự động"
                if rec.verification_status == "TỰ ĐỘNG XÁC NHẬN"
                else (
                    "☑️ Đã duyệt"
                    if rec.verification_status == "ĐÃ XÁC NHẬN THỦ CÔNG"
                    else ("🚫 Đã hủy" if "HỦY" in rec.verification_status else "⚠️ Cần KT")
                )
            )
            veh_short = "Motor" if rec.vehicle_type == "Motorbike" else rec.vehicle_type
            plate_display = "Chưa đọc" if rec.plate_number == "KHÔNG ĐỌC ĐƯỢC" else rec.plate_number
            cells = [
                time_short,
                rec.camera_id,
                f"{veh_short} #{rec.track_id}",
                plate_display,
                conf_str,
                f"{spd_prefix}{rec.speed_kmh:.1f} km/h",
                f"{status_short} | {rec.violation_type}",
            ]
            row_tooltip = (
                f"{rec.timestamp} | {rec.camera_id} | {rec.vehicle_type} #{rec.track_id}\n"
                f"Biển số: {rec.plate_number} ({conf_str})\n"
                f"Trạng thái: {rec.verification_status}\n"
                f"Lý do: {rec.trigger_reason}"
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(row_tooltip)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    if col == 6
                    else Qt.AlignmentFlag.AlignCenter
                )
                if col == 3:
                    if rec.plate_number == "KHÔNG ĐỌC ĐƯỢC":
                        item.setForeground(QColor("#8b949e"))
                    else:
                        item.setForeground(QColor("#ffea00"))
                        item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                elif col == 6:
                    if "HỦY" in rec.verification_status:
                        item.setForeground(QColor("#8b949e"))
                    elif "CẦN KIỂM TRA" in rec.verification_status:
                        item.setForeground(QColor("#ffa657"))
                    else:
                        item.setForeground(QColor("#3fb950"))
                self.table_violations.setItem(row_idx, col, item)

        if select_top and self.table_violations.rowCount() > 0:
            self.table_violations.selectRow(0)
            self._show_violation_preview(records[0])

    def _on_violation_row_selected(self) -> None:
        row = self.table_violations.currentRow()
        if row in self._violations_by_row:
            self._show_violation_preview(self._violations_by_row[row])

    def _show_violation_preview(self, rec: ViolationRecord) -> None:
        spd_note = "Đã hiệu chuẩn" if rec.is_speed_calibrated else "Ước tính (~)"
        reason_short = rec.trigger_reason if len(rec.trigger_reason) <= 68 else rec.trigger_reason[:65] + "..."
        self.lbl_anpr_details.setText(
            f"<b>Biển số:</b> <span style='color:#ffea00;'>{rec.plate_number}</span> "
            f"<span style='color:#8b949e;'>({rec.plate_conf*100:.0f}% | {rec.plate_status})</span><br>"
            f"<b>Xác minh:</b> <span style='color:#60a5fa;'>{rec.verification_status}</span> "
            f"| <b>Tốc độ:</b> {rec.speed_kmh:.1f} km/h ({spd_note})<br>"
            f"<b>Lý do:</b> <span style='color:#cbd5e1;'>{reason_short}</span>"
        )
        self.lbl_anpr_details.setToolTip(rec.trigger_reason)
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
            dlg = EvidenceAndReviewDialog(self._violations_by_row[row], self.db, self)
            if dlg.exec() == QDialog.DialogCode.Accepted and dlg.updated_record is not None:
                self._refresh_violation_table()
                self._show_violation_preview(dlg.updated_record)
                self.status_banner.setText(
                    f"✅ Đã cập nhật thủ công sự kiện #{dlg.updated_record.id}: Biển số '{dlg.updated_record.plate_number}' ({dlg.updated_record.verification_status})"
                )
        else:
            QMessageBox.information(
                self, "Thông báo", "Hãy chọn một sự kiện trong bảng để xem bằng chứng và duyệt thủ công."
            )

    def _export_csv_report(self) -> None:
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất Báo Cáo Vi Phạm CSV",
            "violations/bao_cao_phat_nguoi.csv",
            "CSV Files (*.csv)",
        )
        if save_path:
            # Export currently filtered records
            filtered_records = [self._violations_by_row[r] for r in sorted(self._violations_by_row.keys())]
            count = self.db.export_to_csv(save_path, records=list(reversed(filtered_records)))
            QMessageBox.information(
                self,
                "Xuất Báo Cáo Thành Công",
                f"Đã xuất {count} bản ghi sự kiện ra file:\n{save_path}",
            )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key_text = event.text().lower()
        if key_text in self.config_mgr.configs:
            self.switch_camera(key_text)
        elif key_text == "q":
            self.close()
        else:
            super().keyPressEvent(event)

    def _open_settings_dialog(self) -> None:
        dlg = SettingsDialog(self.ai_settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.saved_settings is not None:
            self.ai_settings = dlg.saved_settings
            if hasattr(self, "voice_worker"):
                self.voice_worker.update_settings(self.ai_settings)
            self._apply_ui_language()
            v_lang = "EN" if self.ai_settings.voice_language == "en" else "VI"
            if str(self.ai_settings.ui_language).lower() == "en":
                self.status_banner.setText(
                    f"⚙️ Settings saved: UI=EN | Voice={v_lang} | Speed={self.ai_settings.speech_rate:.2f}x"
                )
            else:
                self.status_banner.setText(
                    f"⚙️ Đã lưu cài đặt: Giao diện=VI | Giọng đọc={v_lang} | Tốc độ={self.ai_settings.speech_rate:.2f}x"
                )

    def _apply_ui_language(self) -> None:
        is_en = str(getattr(self.ai_settings, "ui_language", "vi")).lower() == "en"
        rate_tag = f"{float(getattr(self.ai_settings, 'speech_rate', 1.0)):.2f}x"
        self.setWindowTitle(
            "Smart Urban Traffic Monitoring & Reliable ANPR (DACN2)"
            if is_en
            else "Hệ Thống Giám Sát Giao Thông Đô Thị Thông Minh - AI Core & Honest ANPR (DACN2)"
        )
        self.title_lbl.setText(
            "🚦 SMART TRAFFIC MONITORING & ENFORCEMENT"
            if is_en
            else "🚦 HỆ THỐNG GIÁM SÁT GIAO THÔNG & PHẠT NGUỘI"
        )
        self.btn_add_cam.setText("➕ Add Cam" if is_en else "➕ Thêm Cam Mới")
        self.btn_del_cam.setText("🗑️ Delete Cam" if is_en else "🗑️ Xóa Cam")
        self.btn_settings.setText("⚙️ Settings" if is_en else "⚙️ Cài đặt")
        self.btn_reset_state.setText("🔁 Reset Tracker" if is_en else "🔁 Reset Bộ Đếm")
        self.btn_pause.setText(
            ("▶ Resume" if self.is_paused else "⏸ Pause")
            if is_en
            else ("▶ Tiếp tục" if self.is_paused else "⏸ Tạm dừng")
        )
        self.btn_voice_ai.setChecked(self.ai_settings.voice_enabled)
        if is_en:
            self.btn_voice_ai.setText(
                f"🔊 Voice AI ({rate_tag}): ON" if self.ai_settings.voice_enabled else "🔇 Voice AI: OFF"
            )
        else:
            self.btn_voice_ai.setText(
                f"🔊 Đọc lỗi ({rate_tag}): BẬT" if self.ai_settings.voice_enabled else "🔇 Đọc lỗi: TẮT"
            )

        self.btn_draw_line.setText("✏️ Draw Line" if is_en else "✏️ Vẽ Vạch Đếm")
        self.btn_draw_roi.setText("⬠ Draw ROI" if is_en else "⬠ Vẽ Vùng ROI")
        self.btn_draw_divider.setText("🛣️ Draw Divider" if is_en else "🛣️ Vẽ Phân Làn")
        self.btn_reset_cfg.setText("🔄 Default Cam" if is_en else "🔄 Mặc định Cam")

        self.lbl_telemetry_title.setText(
            "📊 LIVE TRAFFIC TELEMETRY & STATUS"
            if is_en
            else "📊 THỐNG KÊ LƯU LƯỢNG & TRẠNG THÁI"
        )
        card_map = {
            "Car": ("🚗 Car", "🚗 Ô tô (Car)"),
            "Motorbike": ("🛵 Motorbike", "🛵 Xe máy (Motor)"),
            "Bus": ("🚌 Bus", "🚌 Xe buýt (Bus)"),
            "Truck": ("🚛 Truck", "🚛 Xe tải (Truck)"),
            "AvgSpeed": ("⚡ Avg Speed", "⚡ Tốc độ TB"),
            "InROI": ("📍 Vehicles in ROI", "📍 Xe trong ROI"),
        }
        for k, (en_t, vi_t) in card_map.items():
            if k in self.lbl_card_titles:
                self.lbl_card_titles[k].setText(en_t if is_en else vi_t)

        self.lbl_rules_header.setText(
            "⚙️ TRAFFIC RULES, SPEED CALIBRATION & THRESHOLDS"
            if is_en
            else "⚙️ CẤU HÌNH LUẬT, HIỆU CHUẨN TỐC ĐỘ && NGƯỠNG XÁC NHẬN"
        )
        self.lbl_speed_limit_txt.setText("Speed limit (km/h):" if is_en else "Tốc độ tối đa (km/h):")
        self.lbl_cong_txt.setText("Jam threshold (ROI):" if is_en else "Ngưỡng kẹt xe (ROI):")
        self.lbl_dir_txt.setText("Road direction:" if is_en else "Chiều làn đường:")
        self.chk_no_truck.setText("No-truck sign (P.106a)" if is_en else "Bắt cấm xe tải (P.106a)")
        self.chk_speed.setText("Overspeed detection" if is_en else "Bắt quá tốc độ")
        self.chk_wrong_way.setText("Wrong-way / Lane violation" if is_en else "Bắt ngược chiều / Lấn làn")
        self.chk_illegal_parking.setText("Illegal parking" if is_en else "Bắt dừng đỗ trái phép")
        self.chk_calibrated.setText("Homography calibrated (m)" if is_en else "Đã hiệu chuẩn Homography (m)")

        self.edit_search.setPlaceholderText(
            "🔍 Search plate, vehicle type, violation..."
            if is_en
            else "🔍 Tìm biển số, loại xe, lỗi vi phạm..."
        )
        self.table_violations.setHorizontalHeaderLabels(
            ["Time", "Cam", "Vehicle", "Plate", "Conf", "Speed", "Status / Violation"]
            if is_en
            else ["Giờ", "Cam", "Xe", "Biển số", "Conf", "Tốc độ", "Xác minh / Lỗi"]
        )
        self.btn_view_evidence.setText("🔍 Review Evidence" if is_en else "🔍 Duyệt / Bằng chứng")
        self.btn_speak_selected.setText("🔊 Read Alert (AI)" if is_en else "🔊 Đọc lỗi (AI)")
        self.btn_export_csv.setText("📥 Export CSV" if is_en else "📥 Xuất CSV")
        if hasattr(self, "right_tabs"):
            self.right_tabs.setTabText(0, "🚨 Violations && ANPR" if is_en else "🚨 Vi phạm && Duyệt ANPR")
            self.right_tabs.setTabText(1, "⚙️ Rules && Calibration" if is_en else "⚙️ Luật && Hiệu chuẩn")
        if hasattr(self, "lbl_rule_hint"):
            self.lbl_rule_hint.setText(
                "💡 Tips:\n"
                "• Changes are automatically saved per selected Camera.\n"
                "• Use 'Draw Line', 'Draw ROI', 'Draw Divider' below the video to adjust zones visually."
                if is_en
                else "💡 Mẹo vận hành:\n"
                "• Các thay đổi thông số được lưu tự động cho riêng Camera đang chọn.\n"
                "• Dùng các nút 'Vẽ Vạch Đếm', 'Vẽ Vùng ROI', 'Vẽ Phân Làn' bên dưới khung hình để chỉnh nhanh bằng chuột."
            )

    def _toggle_voice_ai(self, checked: bool) -> None:
        is_en = str(getattr(self.ai_settings, "ui_language", "vi")).lower() == "en"
        rate_tag = f"{float(getattr(self.ai_settings, 'speech_rate', 1.0)):.2f}x"
        if is_en:
            self.btn_voice_ai.setText(f"🔊 Voice AI ({rate_tag}): ON" if checked else "🔇 Voice AI: OFF")
        else:
            self.btn_voice_ai.setText(f"🔊 Đọc lỗi ({rate_tag}): BẬT" if checked else "🔇 Đọc lỗi: TẮT")
        if hasattr(self, "voice_worker"):
            self.voice_worker.set_enabled(checked)
        self.status_banner.setText(
            "🔊 Đã bật đọc cảnh báo lỗi bằng AI Google Studio."
            if checked
            else "🔇 Đã tắt tự động đọc lỗi AI (vẫn có thể bấm 'Đọc lỗi (AI)')."
        )

    def _speak_selected_violation(self) -> None:
        row = self.table_violations.currentRow()
        rec = self._violations_by_row.get(row)
        if rec is None and self._violations_by_row:
            rec = self._violations_by_row.get(0)
        if rec is None:
            QMessageBox.information(
                self, "Thông báo", "Chưa có sự kiện vi phạm nào để đọc."
            )
            return
        if hasattr(self, "voice_worker"):
            self.status_banner.setText("⏳ AI Google Studio đang tạo câu đọc lỗi...")
            self.voice_worker.enqueue(rec, force=True)

    def _on_voice_narration_ready(self, sentence: str, source: str) -> None:
        tag = "AI Gemini" if source == "gemini" else "Offline"
        self.status_banner.setText(f"🔊 [{tag}] {sentence}")

    def closeEvent(self, event) -> None:
        if hasattr(self, "voice_worker"):
            self.voice_worker.stop()
        if hasattr(self, "worker"):
            self.worker.stop_worker()
        super().closeEvent(event)
