# 🚦 Hệ Thống Giám Sát Giao Thông Đô Thị Thông Minh & Phạt Nguội ANPR (DACN2)

Hệ thống AI giám sát giao thông đa kịch bản sử dụng **Multi-model YOLOv8 + ByteTrack**, tích hợp nhận diện biển số (**ANPR**), đo tốc độ phương tiện (**km/h**), phát hiện đi ngược chiều/lấn làn và giao diện điều khiển **PyQt6 Dark Dashboard** cho phép vẽ vạch đếm & vùng ROI trực tiếp bằng chuột.

---

## 🏗️ Kiến trúc thư mục (OOP Clean Architecture)

```text
Tracffic_AI_Project/
├── configs/
│   └── camera_configs.json      # Lưu tọa độ Vạch đếm, ROI, Phân làn, Tốc độ tối đa riêng cho từng Camera
├── core/
│   ├── config_manager.py        # Quản lý nạp/lưu cấu hình từng Camera (JSON)
│   ├── detector.py              # Multi-model AI Engine (YOLOv8s + VN Vehicle Model + VN Traffic Sign Model)
│   ├── tracker_analytics.py     # Đo tốc độ (km/h), bắt đi ngược chiều, đếm xe qua vạch, đo mật độ ROI
│   ├── plate_anpr.py            # Nhận diện & cắt biển số độ phân giải cao (Full HD ANPR) + tạo ảnh biên bản
│   ├── renderer.py              # Vẽ HUD, quỹ đạo di chuyển (motion trail), tốc độ và cảnh báo lên khung hình
│   └── database.py              # Lưu trữ lịch sử phạt nguội vào SQLite & xuất báo cáo CSV
├── ui/
│   ├── video_canvas.py          # Widget hiển thị video & hỗ trợ vẽ Vạch đếm / Vùng ROI bằng chuột
│   └── main_window.py           # Giao diện trung tâm điều khiển PyQt6
├── models/
│   ├── sign_model.pt            # Mô hình nhận diện 58 loại biển báo giao thông Việt Nam (P.106a, P.102, P.127...)
│   └── vehicle_model.pt         # Mô hình nhận diện phương tiện giao thông Việt Nam (tối ưu xe máy)
├── videos/                      # 4 kịch bản video giao thông tiêu chuẩn (Full HD 1920x1080)
├── violations/                  # Tự động lưu ảnh biên bản phạt nguội, ảnh cắt biển số, SQLite DB & file CSV
├── main.py                      # Điểm khởi chạy chính của ứng dụng
└── step2_business_logic.py      # Script khởi chạy tương thích ngược
```

---

## 🚀 Cài đặt và Chạy (Windows PowerShell)

Nếu thư mục `venv` đã tồn tại và terminal đang hiện `(venv)`, cài đặt bổ sung thư viện và khởi chạy:

```powershell
python -m pip install -r requirements.txt
python main.py
```
*(Hoặc chạy `python step2_business_logic.py` đều mở giao diện trung tâm điều khiển).*

Nếu tạo môi trường lần đầu (Python 3.14):

```powershell
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

---

## ✨ Các tính năng chính

1. **Nhận diện Biển số Phạt nguội (Full HD ANPR):**
   - Tự động ánh xạ tọa độ phương tiện vi phạm lên khung hình gốc `1920x1080` để cắt phương tiện và vùng biển số sắc nét.
   - Tạo **Biên bản phạt nguội tổng hợp** (Ảnh hiện trường + Ảnh xe phóng to + Ảnh cắt biển số ANPR + Thông tin tốc độ, thời gian, lỗi vi phạm).
2. **Đo tốc độ (km/h) & Phát hiện Đi ngược chiều / Lấn làn:**
   - Ước tính vận tốc thực tế (`km/h`) có bù phối cảnh chiều sâu camera (Perspective Compensation) và làm mượt EMA.
   - Kiểm tra vector quỹ đạo di chuyển để bắt lỗi **Đi ngược chiều / Lấn làn** và **Chạy quá tốc độ** theo cấu hình từng làn đường.
3. **Giao diện điều khiển PyQt6 & Vẽ Vạch đếm / ROI bằng chuột:**
   - **Vẽ Vạch Đếm:** Bấm nút `✏️ Vẽ Vạch Đếm` rồi kéo thả chuột trái trực tiếp trên màn hình video.
   - **Vẽ Vùng ROI:** Bấm nút `⬠ Vẽ Vùng ROI`, click chuột trái chọn các đỉnh đa giác trên video, click chuột phải để chốt vùng.
   - Xem trực tiếp danh sách vi phạm, ảnh cắt biển số, xem biên bản phóng to và xuất báo cáo ra file Excel/CSV.
