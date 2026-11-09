import cv2
from ultralytics import YOLO

# Nạp model biển báo bạn vừa train xong
sign_model = YOLO("models/sign_model.pt")

# Đọc video xe tải đã ghép biển báo
cap = cv2.VideoCapture("videos/cam3_violation.mp4")

ret, frame = cap.read()
if ret:
    frame = cv2.resize(frame, (854, 480))
    # Nhận diện biển báo trên khung hình
    results = sign_model.predict(frame, conf=0.25)
    
    # Vẽ kết quả nhận diện lên ảnh
    annotated_frame = results[0].plot()
    
    cv2.imshow("Kiem tra bien bao", annotated_frame)
    print("Nhan phim bat ky tren cua so anh de dong...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Khong doc duoc video, kiem tra lai duong dan!")
cap.release()