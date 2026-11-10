# 🎓 KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN CHUYÊN NGÀNH 2

Tài liệu này cung cấp kịch bản thực tế từng bước để bạn tự tin báo cáo và demo trước Giảng viên hướng dẫn hoặc Hội đồng chấm phản biện.

---

## 1. DÀN Ý THUYẾT TRÌNH TỔNG QUAN (2 - 3 PHÚT ĐẦU)

* **Lời mở đầu:**  
  *"Kính thưa quý thầy cô trong hội đồng, em xin phép được báo cáo đề tài Đồ án Chuyên ngành 2: **'Xây dựng Hệ Thống Giám Sát Giao Thông Đô Thị Thông Minh, Đo Tốc Độ & Phạt Nguội Tự Động (AI ANPR)'**.*  
  *Mục tiêu của đề tài là xây dựng một nền tảng giám sát giao thông toàn diện, có khả năng xử lý thời gian thực trên nhiều kịch bản thực tế, tự động phát hiện các hành vi vi phạm giao thông, đo vận tốc phương tiện và lập biên bản phạt nguội kèm ảnh trích xuất biển số xe."*

---

## 2. KỊCH BẢN THỰC HIỆN DEMO TỪNG BƯỚC (LIVE DEMO)

Khởi động hệ thống bằng lệnh:
```powershell
.\venv\Scripts\python.exe main.py
```

### Bước 1: Trình diễn Bắt lỗi Biển cấm xe tải (P.106a) & Phạt nguội ANPR (CAM 3)
- Mặc định ứng dụng sẽ mở ở **CAM 3**.
- **Thao tác & Lời thuyết minh:**
  1. Chỉ vào biển báo đỏ góc phải:  
     *"Hệ thống đang tích hợp mô hình `sign_model.pt` quét biển báo theo chu kỳ. Khi phát hiện biển cấm xe tải `P.106a`, quy tắc cấm xe tải được tự động kích hoạt."*
  2. Khi xe tải đi qua vạch đếm:  
     *"Ngay khi xe tải vượt qua vạch đếm, hệ thống lập tức bắt lỗi vi phạm, hộp bao chuyển sang màu đỏ và xuất hiện dòng trạng thái `[VIOLATION]`."*
  3. Chỉ vào bảng phạt nguội bên phải:  
     *"Đồng thời, danh sách vi phạm bên phải cập nhật ngay tức thì: Mã xe, Thời gian, Tốc độ, Lỗi `Cam xe tai (P.106a)` và Biển số xe màu vàng được trích xuất tự động qua module ANPR."*
  4. Double-click vào dòng vi phạm:  
     *"Em xin phép mở Biên bản phạt nguội tổng hợp: Cửa sổ hiển thị toàn cảnh hiện trường, ảnh xe phóng to từ khung hình gốc Full HD 1080p và ảnh cận cảnh biển số kèm tem kiểm định OCR."*

### Bước 2: Trình diễn Khả năng Tương tác Vẽ Vạch đếm & Vùng ROI bằng chuột
- **Thao tác & Lời thuyết minh:**  
  *"Một ưu điểm lớn của hệ thống là không hardcode tọa độ trong mã nguồn, người vận hành có thể vẽ lại trực tiếp bằng chuột cho mọi góc quay camera mới:"*
  1. Bấm nút `✏️ Vẽ Vạch Đếm (Kéo chuột)`: Kéo một vạch chéo bất kỳ qua mặt đường $\rightarrow$ Thả chuột $\rightarrow$ Vạch đếm mới được áp dụng và lưu tự động vào `camera_configs.json`.
  2. Bấm nút `⬠ Vẽ Vùng ROI`: Click chuột trái chọn 4 đỉnh bao quanh một khu vực nút giao $\rightarrow$ Click **chuột phải** để chốt $\rightarrow$ Vùng ROI đổi màu bán trong suốt ngay tức thì.
  3. Bấm `🔄 Khôi phục Mặc định Cam` để đưa về tọa độ tiêu chuẩn của camera.

### Bước 3: Trình diễn Cảnh báo Ùn tắc Kẹt xe (CAM 2)
- Bấm nút **`📹 CAM 2: Kẹt xe`** trên thanh công cụ.
- **Lời thuyết minh:**  
  *"Tại kênh CAM 2, hệ thống theo dõi mật độ phương tiện bên trong vùng ROI. Khi mật độ xe đạt từ 9 xe trở lên, thẻ trạng thái giao thông lập tức đổi sang màu đỏ nhấp nháy `⚠️ TRẠNG THÁI: KẸT XE - CONGESTED`, hỗ trợ các trung tâm điều hành giao thông kích hoạt kịch bản phân luồng."*

### Bước 4: Trình diễn Đo Tốc độ & Phát hiện Đi ngược chiều (CAM 4)
- Bấm nút **`📹 CAM 4: Cao tốc 2 chiều`** trên thanh công cụ.
- **Lời thuyết minh:**  
  *"Ở kịch bản cao tốc 2 chiều, hệ thống chia làn bằng dải phân cách giữa đường:*
  - *Mỗi phương tiện được đo vận tốc thực tế (`km/h`) có bù trừ hệ số phối cảnh chiều sâu camera.*
  - *Phương tiện vượt quá tốc độ tối đa (80 km/h) sẽ bị lập biên bản `Qua toc do`.*
  - *Phương tiện di chuyển sai chiều quy định của làn đường sẽ bị bắt lỗi `Di nguoc chieu / Lan lan`."*

### Bước 5: Trình diễn Xuất Báo Cáo Phạt Nguội (Export CSV)
- Bấm nút `📥 Xuất CSV` $\rightarrow$ Lưu file `bao_cao_phat_nguoi.csv`.
- Mở file CSV cho hội đồng xem:  
  *"Toàn bộ lịch sử vi phạm được lưu trữ song song trong cơ sở dữ liệu SQLite `traffic_violations.db` và có thể xuất ra bảng tính Excel để cán bộ CSGT xử lý phạt nguội."*

---

## 3. CÂU HỎI THƯỜNG GẶP TỪ HỘI ĐỒNG & CÁCH TRẢ LỜI

**Câu 1: Làm sao hệ thống phân biệt được xe máy giữa đám đông dày đặc ở Việt Nam?**
> *Trả lời:* "Hệ thống áp dụng mô hình kép (Multi-model Ensemble): Mô hình chuẩn `yolov8s` phát hiện ô tô, buýt, tải; song song với đó là mô hình `vehicle_model.pt` được train riêng cho giao thông Việt Nam để quét bù xe máy ở xa. Hai mô hình được kết hợp và lọc trùng lặp tự động bằng thuật toán IoU."

**Câu 2: Cơ chế đo tốc độ dựa trên camera có bị sai lệch do góc nhìn xiên không?**
> *Trả lời:* "Dạ có, góc nhìn xiên làm vật thể ở xa di chuyển ít pixel hơn ở gần. Để khắc phục, em đã xây dựng hàm bù trừ phối cảnh (Perspective Compensation Factor) phụ thuộc vào trục tung $Y$, giúp nhân tỉ lệ quãng đường thực tế cho các vật thể ở xa, kết hợp bộ lọc hàm mũ EMA để loại bỏ dao động tức thời."

**Câu 3: Độ phân giải khi nhận diện biển số có bị mờ do resize màn hình không?**
> *Trả lời:* "Dạ không, vì hệ thống áp dụng cơ chế Resolution Mapping: Video hiển thị trên màn hình được resize về 854x480 để chạy mượt, nhưng khi có xe vi phạm, tọa độ được ánh xạ ngược về khung hình gốc Full HD 1920x1080 để crop ảnh xe và biển số, đảm bảo độ nét tối đa cho module ANPR."
