import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# ==========================================================
# 1. KHỞI TẠO HỆ THỐNG AI ĐA NĂNG (UNIFIED PIPELINE)
# ==========================================================
os.makedirs("violations", exist_ok=True)

print("Đang nạp hệ thống AI đa năng...")
# 1. Model quốc tế bản Small (Bắt cực chuẩn ô tô, xe tải, xe buýt trên mọi góc quay)
base_model = YOLO("yolov8s.pt")

# 2. Model xe Việt Nam (Chuyên trị xe máy)
vn_model = YOLO("models/vehicle_model.pt")

# 3. Model biển báo giao thông Việt Nam
sign_model = YOLO("models/sign_model.pt")

COLORS = {
    "Car": (0, 255, 0),         # Xanh lá
    "Motorbike": (255, 255, 0), # Vàng xanh
    "Bus": (255, 0, 255),       # Tím
    "Truck": (0, 165, 255)      # Cam
}

# 4 Kênh Camera tiêu chuẩn của đồ án (Bấm phím 1, 2, 3, 4 để chuyển trực tiếp)
CAMERAS = {
    "1": ("CAM 1: Giao Thong VN (Normal)", "videos/cam1_normal.mp4"),
    "2": ("CAM 2: Canh Bao Ket Xe (Jam)", "videos/cam2_jam.mp4"),
    "3": ("CAM 3: Bat Loi Bien Bao (Violation)", "videos/cam3_violation.mp4"),
    "4": ("CAM 4: Cao Toc 2 Chieu (Highway)", "videos/cam4_highway.mp4")
}

# ==========================================================
# 2. HÀM QUẢN LÝ VÀ CHUYỂN ĐỔI CAMERA
# ==========================================================
current_cam_key = "3"  # Mặc định mở CAM 3

def init_camera(cam_key):
    cam_name, video_path = CAMERAS[cam_key]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở file: {video_path}")
    else:
        print(f"--> Đã chuyển sang: {cam_name} ({video_path})")
    
    state = {
        "counts": {"Car": 0, "Motorbike": 0, "Bus": 0, "Truck": 0},
        "counted_ids": set(),
        "violation_ids": set(),
        "prev_positions": {},
        "traffic_status": "NORMAL",
        "rule_no_truck": False,
        "frame_idx": 0,
        "last_tracked": [],
        "vehicles_in_roi": 0
    }
    return cap, cam_name, state

cap, cam_name, state = init_camera(current_cam_key)

# Cấu hình vạch đếm & vùng ROI (Tối ưu chung cho mọi video 854x480)
LINE_Y = 350
ROI_POLYGON = np.array([[40, 160], [810, 160], [840, 470], [10, 470]], np.int32)
SKIP_FRAMES = 2

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

print("\n========================================================")
print(" HỆ THỐNG GIÁM SÁT GIAO THÔNG ĐÔ THỊ THÔNG MINH (MULTI-SCALE)")
print(" - Phím [1] [2] [3] [4]: Chuyển đổi trực tiếp giữa 4 Camera")
print(" - Phím [q]: Thoát chương trình")
print("========================================================\n")

# ==========================================================
# 3. VÒNG LẶP XỬ LÝ CHÍNH
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    state["frame_idx"] += 1
    frame = cv2.resize(frame, (854, 480))
    canvas = frame.copy()

    # ----------------------------------------------------------
    # MODULE A: QUÉT BIỂN BÁO GIAO THÔNG
    # ----------------------------------------------------------
    if state["frame_idx"] % 20 == 0 or state["frame_idx"] == 1:
        sign_results = sign_model.predict(frame, conf=0.18, verbose=False)
        for box in sign_results[0].boxes:
            sign_name = sign_results[0].names[int(box.cls[0])]
            sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
            if "106" in sign_name or "P.106a" in sign_name:
                state["rule_no_truck"] = True
                state["sign_box"] = (sx1, sy1, sx2, sy2, sign_name)

    if state["rule_no_truck"] and "sign_box" in state:
        sx1, sy1, sx2, sy2, sname = state["sign_box"]
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
        cv2.putText(canvas, f"Sign: {sname}", (max(sx1 - 20, 10), max(sy1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ----------------------------------------------------------
    # MODULE B: NHẬN DIỆN & TRACKING PHƯƠNG TIỆN (THÔNG MINH ĐA TỈ LỆ)
    # ----------------------------------------------------------
    if state["frame_idx"] % (SKIP_FRAMES + 1) == 0 or state["frame_idx"] == 1:
        # 1. Chạy tracking chính với độ phân giải cao 960 để bắt trọn ô tô, xe tải, xe buýt
        results = base_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[2, 3, 5, 7],
            conf=0.22,
            imgsz=960,
            iou=0.45,
            verbose=False
        )

        current_objects = []
        base_boxes = []
        COCO_MAP = {2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck"}

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = box
                v_type = COCO_MAP.get(cls_id, "Car")
                base_boxes.append((x1, y1, x2, y2))
                current_objects.append((x1, y1, x2, y2, track_id, v_type))

        # 2. Quét bù chuyên sâu bằng vn_model với ngưỡng nhạy thấp (conf=0.15) để bắt xe máy ở xa / góc cao
        vn_res = vn_model.predict(frame, conf=0.15, imgsz=640, verbose=False)
        extra_id_counter = 7000
        
        for vbox in vn_res[0].boxes:
            v_cls = int(vbox.cls[0])
            v_raw_name = str(vn_model.names[v_cls]).lower()
            vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])

            # Chấp nhận mọi nhãn xe máy do model tự train nhận diện được
            if "motor" in v_raw_name or "xe" in v_raw_name or v_cls == 6:
                is_duplicate = any(compute_iou((vx1, vy1, vx2, vy2), b) > 0.2 for b in base_boxes)
                if not is_duplicate:
                    extra_id_counter += 1
                    current_objects.append((vx1, vy1, vx2, vy2, extra_id_counter, "Motorbike"))

        # ----------------------------------------------------------
        # MODULE C: XỬ LÝ NGHIỆP VỤ (ĐẾM XE + KỆT XE + VI PHẠM)
        # ----------------------------------------------------------
        state["last_tracked"] = []
        vehicles_in_roi = 0

        for (x1, y1, x2, y2, track_id, v_type) in current_objects:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Đếm xe qua vạch (Ưu tiên ID cố định từ ByteTrack < 7000)
            if track_id < 7000:
                if track_id in state["prev_positions"]:
                    prev_y = state["prev_positions"][track_id]
                    crossed_down = (prev_y < LINE_Y and cy >= LINE_Y)
                    crossed_up = (prev_y > LINE_Y and cy <= LINE_Y)

                    if crossed_down or crossed_up:
                        if track_id not in state["counted_ids"]:
                            state["counts"][v_type] += 1
                            state["counted_ids"].add(track_id)

                            # Bắt lỗi xe tải đi vào đường cấm P.106a
                            if state["rule_no_truck"] and v_type == "Truck":
                                if track_id not in state["violation_ids"]:
                                    state["violation_ids"].add(track_id)
                                    time_str = datetime.now().strftime("%H%M%S")
                                    save_path = f"violations/cam{current_cam_key}_truck_id{track_id}_{time_str}.jpg"
                                    
                                    evidence = frame.copy()
                                    cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.putText(evidence, f"VIOLATION: NO TRUCK (ID #{track_id})",
                                                (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                                    cv2.imwrite(save_path, evidence)
                                    print(f"--> [PHẠT NGUỘI] Phát hiện Xe tải ID #{track_id} vi phạm! Đã lưu ảnh: {save_path}")

                state["prev_positions"][track_id] = cy

            # Đo mật độ xe trong vùng ROI
            if cv2.pointPolygonTest(ROI_POLYGON, (cx, cy), False) >= 0:
                vehicles_in_roi += 1

            state["last_tracked"].append((x1, y1, x2, y2, cx, cy, track_id, v_type))

        state["vehicles_in_roi"] = vehicles_in_roi
        state["traffic_status"] = "CONGESTED" if vehicles_in_roi >= 9 else "NORMAL"

    # ==========================================================
    # 4. HIỂN THỊ GIAO DIỆN LÊN MÀN HÌNH
    # ==========================================================
    roi_color = (0, 0, 255) if state["traffic_status"] == "CONGESTED" else (255, 120, 0)
    cv2.polylines(canvas, [ROI_POLYGON], True, roi_color, 2)
    cv2.line(canvas, (25, LINE_Y), (830, LINE_Y), (0, 255, 255), 2)
    cv2.putText(canvas, f"Counting Line (Y={LINE_Y})", (35, LINE_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for (x1, y1, x2, y2, cx, cy, track_id, v_type) in state["last_tracked"]:
        is_violated = track_id in state["violation_ids"]
        box_color = (0, 0, 255) if is_violated else COLORS.get(v_type, (0, 255, 0))
        label = f"#{track_id} {v_type}" if track_id < 7000 else f"{v_type}"
        if is_violated:
            label += " [VIOLATION]"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(canvas, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

    # Bảng Dashboard góc trên bên trái
    cv2.rectangle(canvas, (10, 10), (310, 195), (25, 25, 25), -1)
    cv2.rectangle(canvas, (10, 10), (310, 195), (180, 180, 180), 1)

    cv2.putText(canvas, cam_name, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    status_color = (0, 0, 255) if state["traffic_status"] == "CONGESTED" else (0, 255, 0)
    cv2.putText(canvas, f"STATUS: {state['traffic_status']} (ROI: {state['vehicles_in_roi']})", (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2)

    rule_text = "BIEN CAM XE TAI (P.106a): CO" if state["rule_no_truck"] else "BIEN CAM XE TAI: KHONG"
    rule_color = (0, 0, 255) if state["rule_no_truck"] else (180, 180, 180)
    cv2.putText(canvas, rule_text, (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rule_color, 1)

    y_offset = 108
    for v_name, count in state["counts"].items():
        cv2.putText(canvas, f"- {v_name}: {count}", (25, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

    # Hướng dẫn điều hướng ở góc dưới
    cv2.rectangle(canvas, (10, 445), (520, 475), (20, 20, 20), -1)
    cv2.putText(canvas, "Bam phim [1] [2] [3] [4] de chuyen Camera | [Q] de thoat", (20, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    cv2.imshow("He Thong Giam Sat Giao Thong Do Thi - Unified AI", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif chr(key) in CAMERAS and chr(key) != current_cam_key:
        current_cam_key = chr(key)
        cap.release()
        cap, cam_name, state = init_camera(current_cam_key)

cap.release()
cv2.destroyAllWindows()