import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# Paths to YOLO configuration files
cfg_path = "yolo-fish-2.cfg"
weights_path = "merge_yolo-fish-2.weights"
classes_path = "obj.names"

# Load class names
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model with CUDA
net = cv2.dnn.readNet(weights_path, cfg_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Initialize Depth Anything model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(DEVICE).eval()

transform = Compose([
    Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# Constants
REFERENCE_DEPTH = 100.0  # Reference depth for scaling
REAL_MEDIAN_LENGTH_CM = 30.0  # Assumed real-world length of the median fish

# Object detection function
def detect_objects(image, conf_threshold=0.4, nms_threshold=0.5):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Perform inference
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

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    result_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            result_boxes.append((box, confidences[i], classes[class_ids[i]]))

    return result_boxes

# Calculate scaled length based on depth
def scale_and_estimate_length(box, center_depth, reference_depth):
    x, y, w, h = box
    if center_depth <= 0:
        return w  # Return original width if depth is invalid
    scaling_factor = reference_depth / center_depth
    scaled_width = w * scaling_factor
    return scaled_width

# Read video
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)
frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
output_path = "output_depth_yolo.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width * 2 + 50, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = detect_objects(frame)

    # Estimate depth
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    frame_transformed = transform({'image': frame_rgb})['image']
    frame_tensor = torch.from_numpy(frame_transformed).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth_map = depth_anything(frame_tensor)

    depth_map = F.interpolate(depth_map[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_map = depth_map.cpu().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

    # Calculate scaled lengths and determine median
    scaled_lengths = []
    fish_data = []

    for (box, confidence, label) in results:
        x, y, w, h = box
        center_x = x + w // 2
        center_y = y + h // 2
        center_x = max(0, min(center_x, frame_width - 1))
        center_y = max(0, min(center_y, frame_height - 1))
        center_depth = depth_map[center_y, center_x]
        scaled_length = scale_and_estimate_length(box, center_depth, REFERENCE_DEPTH)
        scaled_lengths.append(scaled_length)
        fish_data.append((box, confidence, label, scaled_length, center_depth))

    if scaled_lengths:
        median_scaled_length = np.median(scaled_lengths)
        scaling_ratio = REAL_MEDIAN_LENGTH_CM / median_scaled_length if median_scaled_length > 0 else 1.0
    else:
        scaling_ratio = 1.0

    # Annotate video
    for (box, confidence, label, scaled_length, center_depth) in fish_data:
        real_length_cm = scaled_length * scaling_ratio
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"{real_length_cm:.2f} cm", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Combine original frame and depth map
    split_region = np.ones((frame_height, 50, 3), dtype=np.uint8) * 255
    combined_frame = cv2.hconcat([frame, split_region, depth_color])

    # Save and display
    out.write(combined_frame)
    cv2.imshow("YOLO + Depth Map + Real Length", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
