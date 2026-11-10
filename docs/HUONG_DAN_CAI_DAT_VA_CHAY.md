# 🚀 HƯỚNG DẪN CÀI ĐẶT VÀ KHỞI CHẠY HỆ THỐNG

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường và vận hành **Hệ Thống Giám Sát Giao Thông Đô Thị Thông Minh & Phạt Nguội ANPR (DACN2)** trên hệ điều hành Windows.

---

## 1. YÊU CẦU HỆ THỐNG

- **Hệ điều hành:** Windows 10 / Windows 11 (64-bit).
- **Phiên bản Python:** Python 3.10 đến Python 3.14 (Khuyến nghị dùng Python trong thư mục `venv` đã có sẵn của dự án).
- **Phần cứng đề xuất:**
  - CPU: Intel Core i5 / AMD Ryzen 5 trở lên.
  - RAM: Tối thiểu 8 GB (Khuyến nghị 16 GB).
  - GPU (Tùy chọn): Card đồ họa NVIDIA (nếu có CUDA sẽ tăng tốc độ xử lý YOLO lên đáng kể).

---

## 2. KHỞI CHẠY NHANH (MÔI TRƯỜNG ĐÃ CÓ SẴN VENV)

Thư mục `venv` trong dự án đã được cài đặt đầy đủ các thư viện cần thiết (`PyQt6`, `ultralytics`, `opencv-python`, `lap`, `torch`...).

### Cách 1: Chạy trực tiếp qua PowerShell (Khuyến nghị)
Mở cửa sổ **PowerShell** tại thư mục dự án và gõ lệnh:

```powershell
.\venv\Scripts\python.exe main.py
```

Hoặc kích hoạt môi trường ảo trước:
```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

*(Lưu ý: Bạn cũng có thể chạy lệnh `python step2_business_logic.py`, hệ thống sẽ tự động chuyển tiếp vào giao diện chính).*

### Cách 2: Chạy bằng file Click đúp chuột (`run_app.bat`)
Trong thư mục gốc dự án có sẵn file [run_app.bat](file:///d:/Hoc_Tap/DoAnChuyenNganh/DACN2/code_CN2/Tracffic_AI_Project/run_app.bat). Bạn chỉ cần **Click đúp chuột** vào file này, ứng dụng sẽ tự động bật lên mà không cần mở terminal.

---

## 3. CÀI ĐẶT TỪ ĐẦU (NẾU CẦN TẠO LẠI MÔI TRƯỜNG MỚI)

Nếu bạn sao chép mã nguồn sang máy tính khác chưa có môi trường `venv`:

```powershell
# 1. Tạo môi trường ảo mới
py -3.14 -m venv venv

# 2. Kích hoạt môi trường ảo
.\venv\Scripts\Activate.ps1

# 3. Cài đặt toàn bộ thư viện phụ thuộc
python -m pip install -r requirements.txt

# 4. Chạy chương trình
python main.py
```

---

## 4. HƯỚNG DẪN THAO TÁC TRÊN GIAO DIỆN (UI USER GUIDE)

Khi giao diện PyQt6 hiện lên, bạn có các tính năng tương tác sau:

### 4.1. Chuyển đổi giữa 4 kênh Camera tiêu chuẩn
Bấm vào các nút bấm ở thanh công cụ trên cùng:
- **📹 CAM 1: Đô thị VN:** Kịch bản giao thông xe máy đông đúc, theo dõi vận tốc và dòng xe.
- **📹 CAM 2: Kẹt xe:** Kịch bản ùn tắc; khi số lượng xe trong ROI $\ge 9$, hệ thống chuyển sang thẻ đỏ `CONGESTED`.
- **📹 CAM 3: Biển cấm P.106a:** Kịch bản phát hiện xe tải vi phạm biển cấm và tự động kích hoạt chụp phạt nguội ANPR.
- **📹 CAM 4: Cao tốc 2 chiều:** Kịch bản đường cao tốc chia đôi; kiểm tra xe chạy quá tốc độ (> 80 km/h) và xe đi ngược làn.
- **📂 Mở Video...:** Cho phép bạn chọn bất kỳ file video `.mp4`, `.avi` nào khác từ máy tính để phân tích.

### 4.2. Thao tác vẽ trực tiếp bằng chuột trên màn hình Video
- **✏️ Vẽ Vạch Đếm (Kéo chuột):**
  1. Bấm nút `✏️ Vẽ Vạch Đếm`.
  2. Đưa chuột lên màn hình video, **nhấn giữ chuột trái** tại điểm bắt đầu $\rightarrow$ kéo đến điểm kết thúc $\rightarrow$ **thả chuột**.
  3. Vạch đếm màu vàng sẽ được cập nhật ngay lập tức và lưu vào file cấu hình.
- **⬠ Vẽ Vùng ROI (Click chuột trái / Phải để chốt):**
  1. Bấm nút `⬠ Vẽ Vùng ROI`.
  2. Click chuột trái lần lượt vào các điểm để tạo các đỉnh đa giác bao quanh khu vực cần quan sát.
  3. Click **chuột phải** để khép kín đa giác. Vùng ROI bán trong suốt sẽ xuất hiện.
- **🛣️ Vẽ Vạch Phân Làn:**
  1. Bấm nút `🛣️ Vẽ Vạch Phân Làn`.
  2. Kéo thả chuột trái dọc theo dải phân cách giữa đường để hệ thống phân định luồng đi xuôi/ngược.
- **🔄 Khôi phục Mặc định Cam:** Bấm nút này để đưa tọa độ ROI và vạch đếm của camera hiện tại về giá trị chuẩn ban đầu.

### 4.3. Quản lý danh sách phạt nguội & Xuất báo cáo
- **Xem ảnh cận cảnh biển số:** Click chuột trái vào bất kỳ dòng nào trong bảng `DANH SÁCH VI PHẠM PHẠT NGUỘI`, ô góc dưới sẽ phóng to ảnh chụp biển số cùng thông tin chi tiết.
- **Xem Biên bản phạt nguội tổng hợp:** Double-click vào dòng vi phạm (hoặc bấm nút `🔍 Xem Biên Bản`) để mở cửa sổ phóng to toàn bộ biên bản (gồm ảnh toàn cảnh + ảnh xe Full HD + tem kiểm định OCR + thông số tốc độ/thời gian).
- **Xuất file Excel/CSV:** Bấm nút `📥 Xuất CSV` để lưu toàn bộ danh sách vi phạm ra file Excel (`.csv`) dùng làm tài liệu báo cáo.

---

## 5. XỬ LÝ SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING)

1. **Lỗi `Execution of scripts is disabled on this system` khi chạy `Activate.ps1`:**
   - Mở PowerShell dưới quyền Administrator và gõ:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```
   - Sau đó chạy lại lệnh kích hoạt.
2. **Video chạy giật/lag:**
   - Hệ thống đã tích hợp sẵn cơ chế **Frame Skipping (chạy AI mỗi 3 khung hình)** và chuẩn hóa độ phân giải canvas về `854x480` để đạt tốc độ mượt mà. Đảm bảo máy tính không đang chạy các tác vụ nặng khác chiếm CPU.
3. **Cơ sở dữ liệu vi phạm được lưu ở đâu?**
   - SQLite DB: `violations/traffic_violations.db`
   - File log CSV: `violations/violations_log.csv`
   - Ảnh bằng chứng: Thư mục `violations/*.jpg`
