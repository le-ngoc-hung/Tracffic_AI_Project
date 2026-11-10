"""
Hệ Thống Giám Sát Giao Thông Đô Thị Thông Minh & Phạt Nguội ANPR (DACN2)
------------------------------------------------------------------------
Kiến trúc OOP Clean Code đã được nâng cấp toàn diện:
- core/detector.py          : Multi-model AI (YOLOv8s + VN Vehicle + VN Traffic Sign) & ByteTrack
- core/tracker_analytics.py : Đo tốc độ (km/h), Phát hiện đi ngược chiều, Đếm xe qua vạch, Cảnh báo kẹt xe ROI
- core/plate_anpr.py        : Nhận diện & cắt biển số độ phân giải cao (Full HD ANPR) + Xuất biên bản phạt nguội
- core/database.py          : Lưu trữ lịch sử vi phạm vào SQLite & Xuất báo cáo CSV
- core/config_manager.py    : Quản lý cấu hình Vạch đếm, ROI, Tốc độ tối đa riêng cho từng Camera
- ui/main_window.py         : Giao diện điều khiển PyQt6 Dark Mode + Vẽ Vạch đếm & ROI trực tiếp bằng chuột
"""

from main import main

if __name__ == "__main__":
    main()
