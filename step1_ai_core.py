import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================================
# 1. KHỞI TẠO MÔ HÌNH & CẤU HÌNH
# ==========================================================
# Lần đầu chạy, code sẽ tự động tải file yolov8n.pt (6MB) về máy cho bạn
vehicle_model = YOLO("yolov8n.pt")  

# Nếu đã có sign_model.pt trong thư mục models/ thì mở comment dòng dưới:
# sign_model = YOLO("models/sign_model.pt")

# Trong tập dữ liệu chuẩn của YOLO: 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck"}
COLORS = {
    2: (0, 255, 0),    # Car: Xanh lá
    3: (255, 255, 0),  # Motorbike: Vàng xanh
    5: (255, 0, 255),  # Bus: Tím
    7: (0, 165, 255)   # Truck: Cam
}

# Đổi tên video ở đây để test từng kịch bản (cam1_normal, cam2_jam, cam3_violation, cam4_highway)
VIDEO_PATH = "videos/cam1_normal.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"LỖI: Không tìm thấy video tại '{VIDEO_PATH}'. Hãy kiểm tra lại tên file trong thư mục videos/!")
    exit()

# Cấu hình Frame Skipping giúp máy chạy mượt, không tốn CPU/RAM
frame_count = 0
SKIP_FRAMES = 2  # Cứ 3 khung hình mới chạy AI 1 lần
last_tracked_objects = []

print("Đang chạy hệ thống AI... Nhấn phím 'q' trên cửa sổ video để thoát.")

# ==========================================================
# 2. VÒNG LẶP ĐỌC VIDEO VÀ BÁM VẾT (BYTETRACK)
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        # Hết video thì tự động phát lặp lại từ đầu
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1
    
    # Chuẩn hóa kích thước về 854x480 (16:9) để xử lý nhanh nhất
    frame = cv2.resize(frame, (854, 480))

    if frame_count % (SKIP_FRAMES + 1) == 0 or frame_count == 1:
        # Gọi YOLO + ByteTrack
        results = vehicle_model.track(
            frame,
            persist=True,             # Giữ nguyên ID của xe qua các frame
            tracker="bytetrack.yaml", # Thuật toán ByteTrack
            classes=VEHICLE_CLASSES,  # Chỉ lấy xe máy, ô tô, buýt, tải
            conf=0.30,                # Độ tự tin > 30%
            verbose=False
        )

        last_tracked_objects = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                last_tracked_objects.append((x1, y1, x2, y2, cx, cy, track_id, cls_id))

    # ==========================================================
    # 3. HIỂN THỊ BOUNDING BOX, ID & VẠCH ĐẾM LÊN MÀN HÌNH
    # ==========================================================
    for (x1, y1, x2, y2, cx, cy, track_id, cls_id) in last_tracked_objects:
        color = COLORS.get(cls_id, (0, 255, 0))
        label = f"ID:#{track_id} {CLASS_NAMES.get(cls_id, 'Vehicle')}"

        # Vẽ khung chữ nhật quanh xe
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Vẽ chấm đỏ ở tâm xe (Centroid)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        # Vẽ tên loại xe và mã ID
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Vẽ vạch đếm ảo (Line Crossing) màu vàng ngang màn hình
    cv2.line(frame, (50, 340), (800, 340), (0, 255, 255), 2)
    cv2.putText(frame, "Counting Line", (50, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("AI Core - YOLO + ByteTrack", frame)

    # Bấm phím 'q' để tắt cửa sổ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()