import csv
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from core.logger import get_app_logger

logger = get_app_logger("Database")


@dataclass
class ViolationRecord:
    """Comprehensive audit model for a traffic violation event."""
    id: Optional[int]
    timestamp: str
    camera_id: str
    track_id: int
    vehicle_type: str
    plate_number: str
    plate_conf: float
    plate_status: str
    vehicle_conf: float
    sign_conf: float
    speed_kmh: float
    is_speed_calibrated: bool
    violation_type: str
    verification_status: str
    trigger_reason: str
    model_version: str
    config_snapshot: str
    image_path: str
    plate_crop_path: str = ""
    timeline_path: str = ""
    clip_path: str = ""
    evidence_hash: str = ""


CSV_HEADERS = [
    "ID",
    "Thời gian",
    "Camera",
    "Mã Track",
    "Loại xe",
    "Biển số (ANPR)",
    "Độ tin cậy Biển (%)",
    "Trạng thái Biển",
    "Độ tin cậy Xe (%)",
    "Độ tin cậy Biển báo (%)",
    "Tốc độ (km/h)",
    "Hiệu chuẩn tốc độ",
    "Lỗi vi phạm",
    "Trạng thái xác minh",
    "Lý do cảnh báo",
    "Phiên bản Model",
    "Ảnh biên bản",
    "Ảnh Before-During-After",
    "Video Clip Bằng Chứng",
    "Mã Toàn Vẹn (SHA-256)",
]


class ViolationDatabase:
    """Manages SQLite persistence, schema migration, manual review updates, search/filter, and CSV export."""

    MAX_RECORDS = 1000

    def __init__(
        self,
        db_path: str = "violations/traffic_violations.db",
        csv_path: str = "violations/violations_log.csv",
    ):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.csv_path = csv_path
        self._init_and_migrate_db()
        self.sync_csv()

    def _init_and_migrate_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    plate_number TEXT NOT NULL,
                    plate_conf REAL DEFAULT 0.0,
                    plate_status TEXT DEFAULT 'UNREADABLE',
                    vehicle_conf REAL DEFAULT 0.0,
                    sign_conf REAL DEFAULT 0.0,
                    speed_kmh REAL NOT NULL,
                    is_speed_calibrated INTEGER DEFAULT 0,
                    violation_type TEXT NOT NULL,
                    verification_status TEXT DEFAULT 'CẦN KIỂM TRA THỦ CÔNG',
                    trigger_reason TEXT DEFAULT '',
                    model_version TEXT DEFAULT 'YOLOv8s+VN_Veh+VN_Sign+RapidOCR',
                    config_snapshot TEXT DEFAULT '',
                    image_path TEXT NOT NULL,
                    plate_crop_path TEXT DEFAULT '',
                    timeline_path TEXT DEFAULT '',
                    clip_path TEXT DEFAULT '',
                    evidence_hash TEXT DEFAULT ''
                )
                """
            )
            # Migrate existing tables if older schema is present
            cursor = conn.execute("PRAGMA table_info(violations)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            new_columns = [
                ("plate_conf", "REAL DEFAULT 0.0"),
                ("plate_status", "TEXT DEFAULT 'UNREADABLE'"),
                ("vehicle_conf", "REAL DEFAULT 0.0"),
                ("sign_conf", "REAL DEFAULT 0.0"),
                ("is_speed_calibrated", "INTEGER DEFAULT 0"),
                ("verification_status", "TEXT DEFAULT 'CẦN KIỂM TRA THỦ CÔNG'"),
                ("trigger_reason", "TEXT DEFAULT ''"),
                ("model_version", "TEXT DEFAULT 'YOLOv8s+VN_Veh+VN_Sign+RapidOCR'"),
                ("config_snapshot", "TEXT DEFAULT ''"),
                ("timeline_path", "TEXT DEFAULT ''"),
                ("clip_path", "TEXT DEFAULT ''"),
                ("evidence_hash", "TEXT DEFAULT ''"),
            ]
            for col_name, col_def in new_columns:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE violations ADD COLUMN {col_name} {col_def}")
            conn.commit()

    def insert_violation(
        self,
        camera_id: str,
        track_id: int,
        vehicle_type: str,
        plate_number: str,
        plate_conf: float,
        plate_status: str,
        vehicle_conf: float,
        sign_conf: float,
        speed_kmh: float,
        is_speed_calibrated: bool,
        violation_type: str,
        verification_status: str,
        trigger_reason: str,
        model_version: str,
        config_snapshot: str,
        image_path: str,
        plate_crop_path: str = "",
        timeline_path: str = "",
        clip_path: str = "",
        evidence_hash: str = "",
    ) -> ViolationRecord:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO violations (
                    timestamp, camera_id, track_id, vehicle_type,
                    plate_number, plate_conf, plate_status, vehicle_conf, sign_conf,
                    speed_kmh, is_speed_calibrated, violation_type, verification_status,
                    trigger_reason, model_version, config_snapshot,
                    image_path, plate_crop_path, timeline_path, clip_path, evidence_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    camera_id,
                    track_id,
                    vehicle_type,
                    plate_number,
                    round(plate_conf, 3),
                    plate_status,
                    round(vehicle_conf, 3),
                    round(sign_conf, 3),
                    round(speed_kmh, 1),
                    1 if is_speed_calibrated else 0,
                    violation_type,
                    verification_status,
                    trigger_reason,
                    model_version,
                    config_snapshot,
                    image_path,
                    plate_crop_path,
                    timeline_path,
                    clip_path,
                    evidence_hash,
                ),
            )
            record_id = cursor.lastrowid
            # Enforce storage cap
            cursor.execute(
                """
                DELETE FROM violations WHERE id NOT IN (
                    SELECT id FROM violations ORDER BY id DESC LIMIT ?
                )
                """,
                (self.MAX_RECORDS,),
            )
            conn.commit()

        record = ViolationRecord(
            id=record_id,
            timestamp=timestamp,
            camera_id=camera_id,
            track_id=track_id,
            vehicle_type=vehicle_type,
            plate_number=plate_number,
            plate_conf=round(plate_conf, 3),
            plate_status=plate_status,
            vehicle_conf=round(vehicle_conf, 3),
            sign_conf=round(sign_conf, 3),
            speed_kmh=round(speed_kmh, 1),
            is_speed_calibrated=is_speed_calibrated,
            violation_type=violation_type,
            verification_status=verification_status,
            trigger_reason=trigger_reason,
            model_version=model_version,
            config_snapshot=config_snapshot,
            image_path=image_path,
            plate_crop_path=plate_crop_path,
            timeline_path=timeline_path,
            clip_path=clip_path,
            evidence_hash=evidence_hash,
        )
        self._append_to_csv(record)
        return record

    def update_violation_manual(
        self,
        record_id: int,
        new_plate: str,
        new_violation_type: str,
        new_verification_status: str,
    ) -> Optional[ViolationRecord]:
        """Updates a violation record after manual operator inspection and syncs CSV."""
        new_plate_status = "MANUAL" if new_plate and new_plate != "KHÔNG ĐỌC ĐƯỢC" else "UNREADABLE"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE violations
                SET plate_number = ?,
                    plate_status = ?,
                    violation_type = ?,
                    verification_status = ?
                WHERE id = ?
                """,
                (new_plate, new_plate_status, new_violation_type, new_verification_status, record_id),
            )
            conn.commit()
        self.sync_csv()
        records = self.query_violations(record_id=record_id)
        return records[0] if records else None

    def _row_to_record(self, r: tuple) -> ViolationRecord:
        return ViolationRecord(
            id=r[0],
            timestamp=r[1],
            camera_id=r[2],
            track_id=r[3],
            vehicle_type=r[4],
            plate_number=r[5],
            plate_conf=float(r[6] or 0.0),
            plate_status=str(r[7] or "UNREADABLE"),
            vehicle_conf=float(r[8] or 0.0),
            sign_conf=float(r[9] or 0.0),
            speed_kmh=float(r[10] or 0.0),
            is_speed_calibrated=bool(r[11]),
            violation_type=str(r[12] or ""),
            verification_status=str(r[13] or "CẦN KIỂM TRA THỦ CÔNG"),
            trigger_reason=str(r[14] or ""),
            model_version=str(r[15] or ""),
            config_snapshot=str(r[16] or ""),
            image_path=str(r[17] or ""),
            plate_crop_path=str(r[18] or ""),
            timeline_path=str(r[19] or ""),
            clip_path=str(r[20] or "") if len(r) > 20 else "",
            evidence_hash=str(r[21] or "") if len(r) > 21 else "",
        )

    def query_violations(
        self,
        camera_filter: str = "ALL",
        status_filter: str = "ALL",
        search_text: str = "",
        record_id: Optional[int] = None,
        limit: int = 200,
    ) -> List[ViolationRecord]:
        clauses = []
        params: list = []

        if record_id is not None:
            clauses.append("id = ?")
            params.append(record_id)
        if camera_filter and camera_filter != "ALL":
            clauses.append("camera_id = ?")
            params.append(camera_filter)
        if status_filter and status_filter != "ALL":
            clauses.append("verification_status = ?")
            params.append(status_filter)
        if search_text.strip():
            q = f"%{search_text.strip()}%"
            clauses.append("(plate_number LIKE ? OR violation_type LIKE ? OR vehicle_type LIKE ? OR trigger_reason LIKE ?)")
            params.extend([q, q, q, q])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT id, timestamp, camera_id, track_id, vehicle_type,
                   plate_number, plate_conf, plate_status, vehicle_conf, sign_conf,
                   speed_kmh, is_speed_calibrated, violation_type, verification_status,
                   trigger_reason, model_version, config_snapshot,
                   image_path, plate_crop_path, timeline_path, clip_path, evidence_hash
            FROM violations
            {where_sql}
            ORDER BY id DESC
            LIMIT ?
        """
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_recent_violations(self, limit: int = 100) -> List[ViolationRecord]:
        return self.query_violations(limit=limit)

    def _record_to_csv_row(self, rec: ViolationRecord) -> list:
        return [
            rec.id,
            rec.timestamp,
            rec.camera_id,
            f"#{rec.track_id}",
            rec.vehicle_type,
            rec.plate_number,
            f"{rec.plate_conf * 100:.1f}",
            rec.plate_status,
            f"{rec.vehicle_conf * 100:.1f}",
            f"{rec.sign_conf * 100:.1f}",
            f"{rec.speed_kmh:.1f}",
            "Đã hiệu chuẩn (Homography)" if rec.is_speed_calibrated else "Ước tính (~)",
            rec.violation_type,
            rec.verification_status,
            rec.trigger_reason,
            rec.model_version,
            rec.image_path,
            rec.timeline_path,
            rec.clip_path,
            rec.evidence_hash,
        ]

    def _append_to_csv(self, record: ViolationRecord) -> None:
        try:
            file_exists = os.path.exists(self.csv_path)
            with open(self.csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(CSV_HEADERS)
                writer.writerow(self._record_to_csv_row(record))
        except OSError as e:
            logger.error(f"Lỗi ghi CSV: {e}")

    def sync_csv(self) -> None:
        try:
            records = self.query_violations(limit=self.MAX_RECORDS)
            with open(self.csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)
                for rec in reversed(records):
                    writer.writerow(self._record_to_csv_row(rec))
        except OSError as e:
            logger.error(f"Lỗi đồng bộ CSV: {e}")

    def export_to_csv(self, target_path: str, records: Optional[List[ViolationRecord]] = None) -> int:
        if records is None:
            records = list(reversed(self.query_violations(limit=self.MAX_RECORDS)))
        with open(target_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            for rec in records:
                writer.writerow(self._record_to_csv_row(rec))
        return len(records)

    def clear_all(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM violations")
            conn.commit()
        self.sync_csv()
