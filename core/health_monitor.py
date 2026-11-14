"""
Giám sát sức khỏe hệ thống & chính sách xoay vòng dung lượng lưu trữ (Retention Policy)
phục vụ vận hành liên tục 24/7.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import shutil
import sqlite3
from typing import List

from core.logger import get_app_logger

logger = get_app_logger("HealthMonitor")


@dataclass
class SystemHealthStatus:
    disk_free_gb: float
    evidence_dir_mb: float
    evidence_files: int
    low_disk_warning: bool


def inspect_system_health(
    output_dir: str = "violations",
    min_free_gb: float = 2.0,
) -> SystemHealthStatus:
    """Kiểm tra dung lượng đĩa trống và mức tiêu thụ của thư mục bằng chứng."""
    os.makedirs(output_dir, exist_ok=True)
    usage = shutil.disk_usage(os.path.abspath(output_dir))
    free_gb = usage.free / (1024.0 ** 3)

    total_bytes = 0
    file_count = 0
    for root, _, files in os.walk(output_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                total_bytes += os.path.getsize(fpath)
                file_count += 1
            except OSError:
                pass

    evidence_mb = total_bytes / (1024.0 ** 2)
    low_warn = free_gb < min_free_gb
    if low_warn:
        logger.warning(
            "Cảnh báo dung lượng ổ đĩa thấp: chỉ còn %.2f GB trống (ngưỡng %.1f GB).",
            free_gb,
            min_free_gb,
        )
    return SystemHealthStatus(
        disk_free_gb=round(free_gb, 2),
        evidence_dir_mb=round(evidence_mb, 2),
        evidence_files=file_count,
        low_disk_warning=low_warn,
    )


def cleanup_expired_evidence(
    db_path: str = "violations/traffic_violations.db",
    output_dir: str = "violations",
    retention_days: int = 30,
    false_positive_days: int = 7,
    max_dir_mb: float = 2048.0,
) -> int:
    """
    Dọn dẹp ảnh/clip cũ theo chính sách lưu trữ:
      1. Xóa tệp bằng chứng của các bản ghi 'BÁO NHẦM (ĐÃ HỦY)' quá `false_positive_days` ngày.
      2. Xóa tệp bằng chứng quá `retention_days` ngày (trừ hồ sơ đã xác nhận giữ lại gần nhất).
      3. Nếu thư mục vẫn vượt `max_dir_mb`, tỉa bớt file cũ nhất (.jpg/.mp4) không thuộc top 200 bản ghi mới nhất.
    Trả về số file đã dọn.
    """
    if not os.path.exists(db_path):
        return 0

    removed_files = 0
    now = datetime.now()
    fp_cutoff = (now - timedelta(days=false_positive_days)).strftime("%Y-%m-%d %H:%M:%S")
    general_cutoff = (now - timedelta(days=retention_days)).strftime("%Y-%m-%d %H:%M:%S")

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, verification_status, image_path, plate_crop_path, timeline_path, clip_path
                FROM violations
                ORDER BY id DESC
                """
            ).fetchall()
    except sqlite3.Error as exc:
        logger.debug("Bỏ qua retention cleanup do chưa đọc được DB (%s).", exc)
        return 0

    protected_paths = set()
    expired_paths: List[str] = []

    for idx, row in enumerate(rows):
        _, ts, ver_status, img_p, plate_p, tl_p, clip_p = row
        media_paths = [p for p in (img_p, plate_p, tl_p, clip_p) if p]
        is_fp = "BÁO NHẦM" in str(ver_status or "")
        if (is_fp and str(ts) < fp_cutoff) or (str(ts) < general_cutoff and idx >= 200):
            expired_paths.extend(media_paths)
        else:
            for p in media_paths:
                protected_paths.add(os.path.abspath(p))

    for p in expired_paths:
        if os.path.exists(p) and os.path.abspath(p) not in protected_paths:
            try:
                os.remove(p)
                removed_files += 1
            except OSError:
                pass

    # Cap total directory size if needed
    health = inspect_system_health(output_dir)
    if health.evidence_dir_mb > max_dir_mb:
        candidates = []
        for fname in os.listdir(output_dir):
            if not fname.lower().endswith((".jpg", ".mp4")):
                continue
            fpath = os.path.join(output_dir, fname)
            if os.path.abspath(fpath) in protected_paths:
                continue
            try:
                candidates.append((os.path.getmtime(fpath), os.path.getsize(fpath), fpath))
            except OSError:
                pass
        candidates.sort(key=lambda item: item[0])
        current_mb = health.evidence_dir_mb
        for _, fsize, fpath in candidates:
            if current_mb <= max_dir_mb:
                break
            try:
                os.remove(fpath)
                current_mb -= fsize / (1024.0 ** 2)
                removed_files += 1
            except OSError:
                pass

    if removed_files > 0:
        logger.info("Đã dọn dẹp %d file bằng chứng hết hạn trong '%s'.", removed_files, output_dir)
    return removed_files
