import cv2
import numpy as np

# Đường dẫn tới các file YOLO
cfg_path = "yolo-fish-2.cfg"
weights_path = "merge_yolo-fish-2.weights"
classes_path = "obj.names"

# Load các class
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load mạng YOLO với CUDA
net = cv2.dnn.readNet(weights_path, cfg_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Hàm để phát hiện đối tượng
def detect_objects(image, conf_threshold=0.4, nms_threshold=0.5):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Lấy các layer đầu ra
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
    # Chạy dự đoán
    detections = net.forward(output_layers)
    
    boxes, confidences, class_ids = [], [], []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Áp dụng Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    result_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            result_boxes.append((box, confidences[i], classes[class_ids[i]]))
    
    return result_boxes

# Đọc video hoặc webcam
video_path = "input.mp4"  # Đổi thành 0 nếu dùng webcam
cap = cv2.VideoCapture(video_path)

# Cấu hình VideoWriter để lưu video
output_path = "output.avi"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not cap.isOpened():
    print("Không thể mở video hoặc webcam.")
    exit()

# Lặp qua từng frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện đối tượng trong frame
    results = detect_objects(frame)

    # Vẽ bounding box và hiển thị kết quả
    for (box, confidence, label) in results:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Hiển thị frame
    cv2.imshow("YOLO Detection", frame)

    # Ghi frame vào video đầu ra
    out.write(frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
