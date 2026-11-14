# 🚦 Hệ Thống Giám Sát Giao Thông Đô Thị Thông Minh & Phạt Nguội Đáng Tin Cậy (DACN2)

Hệ thống AI giám sát giao thông đa kịch bản sử dụng **Multi-model YOLOv8 + ByteTrack**, tích hợp nhận diện biển số thực tế bằng **RapidOCR (PaddleOCR ONNX)**, hiệu chuẩn tốc độ **Homography (km/h)**, cơ chế **xác nhận vi phạm đa khung hình** tách rời khỏi phát hiện đơn lẻ, và giao diện điều khiển đa luồng **PyQt6 Dark Dashboard**.

---

## 🏗️ Kiến trúc thư mục (Verifiable Clean Architecture)

```text
Tracffic_AI_Project/
├── benchmarks/
│   ├── evaluate_system.py                # Bộ kiểm thử & đo lường chất lượng 7 tiêu chí độc lập
│   ├── evaluation_results.json           # Kết quả đo lường định lượng (JSON)
│   └── BAO_CAO_DANH_GIA_CHAT_LUONG.md    # Báo cáo đối chiếu KPI mục tiêu vs Kết quả thực tế
├── configs/
│   ├── ai_settings.json                  # Cấu hình API key Google AI Studio & đọc lỗi bằng giọng nói
│   └── camera_configs.json               # Cấu hình Vạch đếm, ROI, Homography Quad, Ngưỡng xác nhận
├── core/
│   ├── ai_voice_narrator.py              # Đọc lỗi vi phạm qua Google AI Studio (Gemini + TTS) theo mẫu chuẩn
│   ├── config_manager.py                 # Quản lý & kiểm tra hợp lệ (validate) cấu hình từng Camera
│   ├── detector.py                       # Multi-model AI Engine (YOLOv8s + VN Vehicle + VN Sign)
│   ├── tracker_analytics.py              # Hiệu chuẩn tốc độ Homography, xác nhận đa khung hình, Ring Buffer
│   ├── plate_anpr.py                     # Nhận diện biển số thật (RapidOCR ONNX), multi-frame voting, MP4 clip & SHA-256
│   ├── renderer.py                       # Hiển thị HUD, phân biệt tốc độ đã hiệu chuẩn vs ước tính (~)
│   ├── stream_reader.py                  # Đọc luồng RTSP/HTTP/Webcam thời gian thực & tự kết nối lại khi rớt mạng
│   ├── health_monitor.py                 # Watchdog kiểm tra dung lượng ổ đĩa & tự dọn bằng chứng hết hạn (Retention)
│   ├── database.py                       # SQLite & CSV lưu đầy đủ Audit Trail, clip MP4, SHA-256, hỗ trợ duyệt thủ công
│   └── logger.py                         # Hệ thống ghi log xoay vòng (RotatingFileHandler) tại logs/
├── scripts/
│   └── export_models_onnx.py             # Tiện ích export các model YOLOv8 (.pt -> .onnx) để tăng tốc suy luận
├── ui/
│   ├── inference_worker.py               # Luồng xử lý AI chạy ngầm (QThread) chống đơ giao diện
│   ├── video_canvas.py                   # Widget hiển thị video & vẽ Vạch đếm / Vùng ROI bằng chuột
│   └── main_window.py                    # Giao diện điều khiển, tìm kiếm/lọc sự kiện & xác nhận thủ công
├── docs/                                 # Thư mục tài liệu hướng dẫn cài đặt, nguyên lý & kịch bản bảo vệ
├── main.py                               # Điểm khởi chạy chính của ứng dụng
└── step2_business_logic.py               # Script khởi chạy tương thích ngược
```

---

## 🚀 Cài đặt và Chạy (Windows PowerShell)

### 1. Chạy ứng dụng chính
Nếu thư mục `venv` đã tồn tại:

```powershell
python -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py
```
*(Hoặc click đúp vào file `run_app.bat` hoặc chạy `.\venv\Scripts\python.exe step2_business_logic.py`).*

### 2. Chạy bộ kiểm thử & đo lường chất lượng (Benchmark Suite)
Để đánh giá độc lập độ chính xác ANPR (toàn biển & từng ký tự), đếm xe, cảnh báo kẹt xe, biển báo và sai số tốc độ:

```powershell
.\venv\Scripts\python.exe benchmarks/evaluate_system.py
```
Kết quả chi tiết được xuất ra tại [`benchmarks/BAO_CAO_DANH_GIA_CHAT_LUONG.md`](file:///d:/Hoc_Tap/DoAnChuyenNganh/DACN2/code_CN2/Tracffic_AI_Project/benchmarks/BAO_CAO_DANH_GIA_CHAT_LUONG.md).

---

## ✨ 5 Cải tiến cốt lõi đảm bảo độ tin cậy thực tế

1. **ANPR trung thực (Không sinh biển số giả) & Hiệu chuẩn tốc độ Homography:**
   - Sử dụng **RapidOCR (ONNX Runtime)** đọc ký tự thực tế, lưu độ tin cậy (`plate_conf`) và đánh dấu rõ **`KHÔNG ĐỌC ĐƯỢC` (`UNREADABLE`)** khi biển số ở quá xa hoặc bị nhòe.
   - Hỗ trợ hiệu chuẩn tốc độ bằng 4 điểm chuẩn mặt đường (**Homography Matrix**). Khi chưa hiệu chuẩn, tốc độ được gắn ký hiệu ước tính `~XX.X km/h`.
2. **Tách phát hiện khỏi kết luận vi phạm (Multi-frame Confirmation & Audit Trail):**
   - Yêu cầu phương tiện được theo dõi tối thiểu `min_observation_frames` chu kỳ và duy trì vi phạm liên tiếp `min_violation_frames` chu kỳ trước khi tạo cảnh báo.
   - Tự động phân loại trạng thái **`TỰ ĐỘNG XÁC NHẬN`** hoặc **`CẦN KIỂM TRA THỦ CÔNG`**.
   - Lưu chuỗi 3 ảnh **Trước - Trong - Sau sự kiện (`Before / During / After`)**, độ tin cậy của từng model (Xe, Biển báo, Biển số), phiên bản model và lý do chi tiết tạo cảnh báo.
3. **Đo lường chất lượng bằng bộ kiểm thử có nhãn (`benchmarks/`):**
   - Đo tỷ lệ đọc đúng toàn biển, độ chính xác từng ký tự ($1 - \text{CER}$), tỷ lệ từ chối ảnh mờ, sai số tốc độ MAE và tỷ lệ báo nhầm/bỏ sót.
4. **Kiến trúc đa luồng (`QThread`) & Độ ổn định cao:**
   - Tách hoàn toàn luồng suy luận AI (`VideoInferenceWorker`) khỏi luồng giao diện PyQt6.
   - Ghi log xoay vòng (`logs/traffic_system.log`), giới hạn dung lượng DB và tự động dọn bộ nhớ các track cũ.
5. **Tìm kiếm, Lọc sự kiện & Duyệt/Sửa thủ công trên Giao diện:**
   - Tìm kiếm theo biển số, lọc theo Camera và trạng thái xác minh.
   - Cho phép người vận hành mở hồ sơ bằng chứng, xem ảnh `Before / During / After`, chỉnh sửa biển số/loại lỗi và bấm **Lưu Xác Nhận** hoặc **Đánh Dấu Báo Nhầm**, đồng bộ nhất quán ra file CSV.
