import csv
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class ViolationRecord:
    """Data model representing a single traffic violation event."""
    id: Optional[int]
    timestamp: str
    camera_id: str
    track_id: int
    vehicle_type: str
    plate_number: str
    speed_kmh: float
    violation_type: str
    image_path: str
    plate_crop_path: str = ""


class ViolationDatabase:
    """Manages SQLite persistence and CSV reporting for traffic violations."""

    def __init__(self, db_path: str = "violations/traffic_violations.db", csv_path: str = "violations/violations_log.csv"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.csv_path = csv_path
        self._init_db()
        self._ensure_csv_header()

    def _init_db(self) -> None:
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
                    speed_kmh REAL NOT NULL,
                    violation_type TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    plate_crop_path TEXT DEFAULT ''
                )
                """
            )
            conn.commit()

    def _ensure_csv_header(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ID", "Thời gian", "Camera", "Mã Track", "Loại xe",
                    "Biển số (ANPR)", "Tốc độ (km/h)", "Lỗi vi phạm", "Ảnh bằng chứng"
                ])

    def insert_violation(
        self,
        camera_id: str,
        track_id: int,
        vehicle_type: str,
        plate_number: str,
        speed_kmh: float,
        violation_type: str,
        image_path: str,
        plate_crop_path: str = ""
    ) -> ViolationRecord:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO violations (
                    timestamp, camera_id, track_id, vehicle_type,
                    plate_number, speed_kmh, violation_type, image_path, plate_crop_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp, camera_id, track_id, vehicle_type,
                    plate_number, round(speed_kmh, 1), violation_type, image_path, plate_crop_path
                )
            )
            record_id = cursor.lastrowid
            conn.commit()

        record = ViolationRecord(
            id=record_id,
            timestamp=timestamp,
            camera_id=camera_id,
            track_id=track_id,
            vehicle_type=vehicle_type,
            plate_number=plate_number,
            speed_kmh=round(speed_kmh, 1),
            violation_type=violation_type,
            image_path=image_path,
            plate_crop_path=plate_crop_path,
        )
        self._append_to_csv(record)
        return record

    def _append_to_csv(self, record: ViolationRecord) -> None:
        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    record.id,
                    record.timestamp,
                    record.camera_id,
                    f"#{record.track_id}",
                    record.vehicle_type,
                    record.plate_number,
                    f"{record.speed_kmh:.1f}",
                    record.violation_type,
                    record.image_path,
                ])
        except OSError:
            pass

    def get_recent_violations(self, limit: int = 100) -> List[ViolationRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, camera_id, track_id, vehicle_type,
                       plate_number, speed_kmh, violation_type, image_path, plate_crop_path
                FROM violations
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            )
            rows = cursor.fetchall()
        return [
            ViolationRecord(
                id=r[0],
                timestamp=r[1],
                camera_id=r[2],
                track_id=r[3],
                vehicle_type=r[4],
                plate_number=r[5],
                speed_kmh=r[6],
                violation_type=r[7],
                image_path=r[8],
                plate_crop_path=r[9] or "",
            )
            for r in rows
        ]

    def export_to_csv(self, target_path: str) -> int:
        records = self.get_recent_violations(limit=10000)
        with open(target_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ID", "Thời gian", "Camera", "Mã Track", "Loại xe",
                "Biển số (ANPR)", "Tốc độ (km/h)", "Lỗi vi phạm", "Ảnh bằng chứng"
            ])
            for rec in reversed(records):
                writer.writerow([
                    rec.id,
                    rec.timestamp,
                    rec.camera_id,
                    f"#{rec.track_id}",
                    rec.vehicle_type,
                    rec.plate_number,
                    f"{rec.speed_kmh:.1f}",
                    rec.violation_type,
                    rec.image_path,
                ])
        return len(records)

    def clear_all(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM violations")
            conn.commit()
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        self._ensure_csv_header()
