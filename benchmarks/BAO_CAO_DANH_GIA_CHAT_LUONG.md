# 📊 BÁO CÁO ĐO LƯỜNG & ĐÁNH GIÁ CHẤT LƯỢNG HỆ THỐNG (BENCHMARK REPORT)

Báo cáo này được sinh tự động từ bộ kiểm thử độc lập [`benchmarks/evaluate_system.py`](file:///d:/Hoc_Tap/DoAnChuyenNganh/DACN2/code_CN2/Tracffic_AI_Project/benchmarks/evaluate_system.py) nhằm đánh giá khách quan từng thành phần của hệ thống theo đúng tiêu chuẩn kỹ thuật.

---

## 1. BẢNG TỔNG HỢP CHỈ SỐ CHẤT LƯỢNG (KPIs VS TARGETS)

| Hạng mục đánh giá | Chỉ số đo lường (Metric) | Mục tiêu (Target) | Kết quả thực tế | Đánh giá |
| :--- | :--- | :---: | :---: | :---: |
| **1. Nhận diện biển số (ANPR)** | Tỷ lệ đọc đúng toàn biển ( ảnh rõ ) | $\ge 85.0\%$ | **91.7%** | ✅ ĐẠT |
| **2. Nhận diện ký tự ANPR** | Độ chính xác cấp độ ký tự ($1 - \text{CER}$) | $\ge 92.0\%$ | **99.0%** | ✅ ĐẠT |
| **3. Chống sinh biển số giả** | Tỷ lệ từ chối trung thực ảnh mờ (`UNREADABLE`) | $100.0\%$ | **100.0%** | ✅ ĐẠT (0% Bịa) |
| **4. Phát hiện & Theo dõi xe** | Độ ổn định duy trì ID (ByteTrack $\ge 4$ chu kỳ) | $\ge 70.0\%$ | **81.2%** | ✅ ĐẠT |
| **5. Đếm xe qua vạch** | Độ chính xác đếm so với Ground Truth (Cam 3) | $\ge 80.0\%$ | **100.0%** (5/5) | ✅ ĐẠT |
| **6. Cảnh báo kẹt xe ROI** | Tỷ lệ nhận diện đúng trạng thái `CONGESTED` (Cam 2) | $\ge 90.0\%$ | **100.0%** | ✅ ĐẠT |
| **7. Nhận diện biển báo** | Tỷ lệ phát hiện biển cấm `P.106a` (Recall / Conf) | $\ge 85.0\%$ | **100.0%** (Conf: 66.6%) | ✅ ĐẠT |
| **8. Hiệu chuẩn tốc độ BEV** | Sai số tuyệt đối trung bình (Homography MAE) | $\le 3.0\text{ km/h}$ | **0.0 km/h** | ✅ ĐẠT |
| **9. Độ tin cậy vi phạm** | Cảnh báo nhầm (Cam 2 FP) / Bỏ sót P.106a (Cam 3 FN) | $\text{FP} = 0, \text{FN} = 0$ | **FP = 0, FN = 0** | ✅ ĐẠT |

---

## 2. CHI TIẾT KIỂM THỬ MÔ-ĐUN ANPR (RAPIDOCR ONNX)

Bộ kiểm thử bao gồm **12 mẫu biển số có nhãn thực tế** và **4 mẫu cắt trực tiếp từ đuôi/đầu xe ở khoảng cách rất xa trong video (`cam3_violation.mp4`)** để kiểm chứng nguyên tắc **không tự tạo biển số giả**:

| File ảnh mẫu | Nhãn thực tế (Ground Truth) | Kết quả ANPR đọc | Độ tin cậy | Trạng thái | Độ chính xác ký tự | Kết luận |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `plate_01.jpg` | `51C-482.19` | `51C-482.19` | 76.2% | `READ` | 100.0% | ✅ Đạt |
| `plate_02.jpg` | `30H-719.05` | `30H-719.05` | 80.1% | `READ` | 100.0% | ✅ Đạt |
| `plate_03.jpg` | `29A-864.20` | `29-A8 864.20` | 81.5% | `READ` | 87.5% | ❌ Lệch |
| `plate_04.jpg` | `43A-195.88` | `43A-195.88` | 80.3% | `READ` | 100.0% | ✅ Đạt |
| `plate_05.jpg` | `60C-512.34` | `60C-512.34` | 79.4% | `READ` | 100.0% | ✅ Đạt |
| `plate_06.jpg` | `51G-903.41` | `51G-903.41` | 76.6% | `READ` | 100.0% | ✅ Đạt |
| `plate_07.jpg` | `29C-338.12` | `29C-338.12` | 79.6% | `READ` | 100.0% | ✅ Đạt |
| `plate_08.jpg` | `30K-641.77` | `30K-641.77` | 87.8% | `READ` | 100.0% | ✅ Đạt |
| `plate_09.jpg` | `50F-218.65` | `50F-218.65` | 79.6% | `READ` | 100.0% | ✅ Đạt |
| `plate_10.jpg` | `61C-704.92` | `61C-704.92` | 80.9% | `READ` | 100.0% | ✅ Đạt |
| `plate_11.jpg` | `51D-825.10` | `51D-825.10` | 78.7% | `READ` | 100.0% | ✅ Đạt |
| `plate_12.jpg` | `29H-450.33` | `29H-450.33` | 90.6% | `READ` | 100.0% | ✅ Đạt |
| `unreadable_distant_01.jpg` | `KHÔNG ĐỌC ĐƯỢC` | `KHÔNG ĐỌC ĐƯỢC` | 0.0% | `UNREADABLE` | 100.0% | ✅ Đạt |
| `unreadable_distant_02.jpg` | `KHÔNG ĐỌC ĐƯỢC` | `KHÔNG ĐỌC ĐƯỢC` | 0.0% | `UNREADABLE` | 100.0% | ✅ Đạt |
| `unreadable_distant_03.jpg` | `KHÔNG ĐỌC ĐƯỢC` | `KHÔNG ĐỌC ĐƯỢC` | 0.0% | `UNREADABLE` | 100.0% | ✅ Đạt |
| `unreadable_distant_04.jpg` | `KHÔNG ĐỌC ĐƯỢC` | `KHÔNG ĐỌC ĐƯỢC` | 0.0% | `UNREADABLE` | 100.0% | ✅ Đạt |

---

## 3. NHẬN XÉT KỸ THUẬT & HƯỚNG XỬ LÝ TRONG HỆ THỐNG

1. **Về ANPR trên các video quay góc cao/xa (`cam3_violation.mp4`, `cam4_highway.mp4`):**
   - Khi phương tiện ở khoảng cách 40–70m so với camera góc rộng, vùng biển số trên khung hình Full HD 1920x1080 chỉ có kích thước khoảng `28x9 pixels` (mỗi ký tự cao `< 6 pixels`), vượt quá giới hạn phân giải quang học.
   - Thay vì tự sinh biển số giả như phiên bản PoC cũ, hệ thống hiện tại **trả về trung thực `KHÔNG ĐỌC ĐƯỢC` (`UNREADABLE`, conf = `0.0%`)** và tự động gắn cờ bản ghi ở trạng thái **`CẦN KIỂM TRA THỦ CÔNG` (`NEEDS_REVIEW`)** để người vận hành kiểm tra lại hoặc nhập bổ sung.
2. **Về xác nhận vi phạm đa khung hình:**
   - Hệ thống yêu cầu phương tiện phải được theo dõi liên tục ít nhất `min_observation_frames = 4` chu kỳ AI và duy trì hành vi vi phạm tối thiểu `min_violation_frames = 3` chu kỳ trước khi kết luận, giúp triệt tiêu hiện tượng báo nhầm do nhiễu nhận diện đơn lẻ.
