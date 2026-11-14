# 🚀 HƯỚNG DẪN CÀI ĐẶT, ĐÁNH GIÁ CHẤT LƯỢNG VÀ VẬN HÀNH HỆ THỐNG

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường, chạy bộ kiểm thử đánh giá chất lượng (Benchmark) và vận hành **Hệ Thống Giám Sát Giao Thông Đô Thị Thông Minh & Phạt Nguội Đáng Tin Cậy (DACN2)**.

---

## 1. YÊU CẦU HỆ THỐNG

- **Hệ điều hành:** Windows 10 / Windows 11 (64-bit).
- **Phiên bản Python:** Python 3.10 đến Python 3.14 (Khuyến nghị dùng môi trường `venv` đã cài sẵn trong dự án).
- **Các thư viện chính (`requirements.txt`):**
  - `ultralytics` (YOLOv8 + ByteTrack)
  - `rapidocr-onnxruntime` (PaddleOCR PP-OCRv3 ONNX cho nhận diện biển số thật)
  - `opencv-python`, `numpy`, `lap>=0.5.12`
  - `PyQt6` (Giao diện điều khiển đa luồng)

---

## 2. KHỞI CHẠY ỨNG DỤNG & BỘ ĐO LƯỜNG CHẤT LƯỢNG

### 2.1. Chạy ứng dụng giám sát chính
Mở **PowerShell** tại thư mục dự án và chạy:

```powershell
.\venv\Scripts\python.exe main.py
```
*(Hoặc click đúp vào file `run_app.bat`, hoặc chạy `.\venv\Scripts\python.exe step2_business_logic.py`).*

### 2.2. Chạy bộ kiểm thử & đánh giá chất lượng độc lập (Benchmark Suite)
Để kiểm tra khách quan độ chính xác của hệ thống (ANPR đọc toàn biển & từng ký tự, chống sinh biển giả, đếm xe, kẹt xe, biển báo và sai số tốc độ Homography):

```powershell
.\venv\Scripts\python.exe benchmarks/evaluate_system.py
```
Sau khi chạy xong, mở file báo cáo kết quả tại:
- [`benchmarks/BAO_CAO_DANH_GIA_CHAT_LUONG.md`](file:///d:/Hoc_Tap/DoAnChuyenNganh/DACN2/code_CN2/Tracffic_AI_Project/benchmarks/BAO_CAO_DANH_GIA_CHAT_LUONG.md)
- [`benchmarks/evaluation_results.json`](file:///d:/Hoc_Tap/DoAnChuyenNganh/DACN2/code_CN2/Tracffic_AI_Project/benchmarks/evaluation_results.json)

---

## 3. CÁC TÍNH NĂNG ĐẢM BẢO ĐỘ TIN CẬY TRÊN GIAO DIỆN

### 3.1. Nhận diện biển số trung thực (Honest ANPR) & Duyệt thủ công
- Hệ thống **tuyệt đối không tự tạo biển số giả**. Khi biển số ở quá xa hoặc bị nhòe không đủ độ tin cậy, hệ thống ghi nhận rõ **`KHÔNG ĐỌC ĐƯỢC` (`UNREADABLE`)** và đặt trạng thái sự kiện là **`CẦN KIỂM TRA THỦ CÔNG`**.
- **Cách duyệt/sửa thủ công:**
  1. Nhấp đúp vào một sự kiện trong bảng (hoặc bấm nút **`🔍 Duyệt / Xem Bằng Chứng`**).
  2. Xem tab **Ảnh Biên Bản Tổng Hợp** và tab **Chuỗi Khung Hình Trước - Trong - Sau Sự Kiện (`Before / During / After`)** cùng hộp thông tin **Truy vết hệ thống (Audit Trail)**.
  3. Nhập hoặc chỉnh sửa biển số xe, loại vi phạm và chọn **`💾 Lưu Xác Nhận`** hoặc **`🚫 Đánh Dấu Báo Nhầm`**. Mọi thay đổi được lưu ngay vào SQLite và đồng bộ ra `violations/violations_log.csv`.

### 3.2. Tìm kiếm, Lọc sự kiện và Xuất CSV
- Phía trên bảng sự kiện có thanh công cụ cho phép:
  - Lọc theo Camera (`ALL`, `CAM 1`, `CAM 2`, `CAM 3`, `CAM 4`).
  - Lọc theo Trạng thái xác minh (`CẦN KIỂM TRA THỦ CÔNG`, `TỰ ĐỘNG XÁC NHẬN`, `ĐÃ XÁC NHẬN THỦ CÔNG`, `BÁO NHẦM (ĐÃ HỦY)`).
  - Ô tìm kiếm theo từ khóa biển số, loại xe hoặc lý do cảnh báo.
- Bấm **`📥 Xuất CSV`** để kết xuất danh sách sự kiện đang lọc ra file CSV (bao gồm đầy đủ độ tin cậy Xe, Biển báo, Biển số, trạng thái hiệu chuẩn tốc độ và lý do cảnh báo).

### 3.3. Kiểm tra hợp lệ khi vẽ Vạch đếm & Vùng ROI bằng chuột
- Khi vẽ Vạch đếm (`✏️ Vẽ Vạch Đếm`) hoặc Vùng ROI (`⬠ Vẽ Vùng ROI`), hệ thống tự động kiểm tra độ dài vạch ($\ge 30\text{ px}$) và diện tích đa giác ROI ($\ge 2500\text{ px}^2$). Nếu tọa độ không hợp lệ, hệ thống sẽ hiển thị thông báo cảnh báo và giữ nguyên cấu hình an toàn trước đó.
- **Đọc lỗi vi phạm bằng AI Google Studio (`configs/ai_settings.json`):** Hệ thống tự động chuẩn hóa và đọc thành tiếng các vi phạm mới theo đúng mẫu `"xe '<loại xe>' biển số <biển số> <lỗi của xe đó>"` thông qua Google AI Studio (Gemini text + Gemini TTS trên `QThread` nền). Có thể bật/tắt nhanh trên thanh công cụ (`🔊 Đọc lỗi AI`) hoặc bấm `🔊 Đọc lại lỗi` cho từng dòng trong bảng.
