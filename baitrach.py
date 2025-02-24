import cv2
import numpy as np
from cjm_byte_track.core import BYTETracker
from absl import app, flags, logging
from absl.flags import FLAGS
import time
from collections import deque

# Define flags
flags.DEFINE_string('cfg', 'yolo-fish-2.cfg', 'path to cfg file')
flags.DEFINE_string('weights', 'merge_yolo-fish-2.weights', 'path to weights file')
flags.DEFINE_string('video', '0217.mp4', 'path to video file or camera index')
flags.DEFINE_string('output', 'output_bytetrack.mp4', 'path to output video file')
flags.DEFINE_float('conf_threshold', 0.05, 'confidence threshold for YOLO')
flags.DEFINE_float('nms_threshold', 0.45, 'nms threshold for YOLO')
flags.DEFINE_integer('track_buffer', 30, 'buffer size for tracking')
flags.DEFINE_float('track_thresh', 0.5, 'tracking confidence threshold')
flags.DEFINE_float('match_thresh', 0.8, 'matching threshold for tracking')

def main(_argv):
    # Khởi tạo YOLO
    net = cv2.dnn.readNet(FLAGS.weights, FLAGS.cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Khởi tạo BYTETracker với tham số tối ưu cho tracking cá
    tracker_args = {
        'track_thresh': FLAGS.track_thresh,
        'track_buffer': FLAGS.track_buffer,
        'match_thresh': FLAGS.match_thresh,
    }
    tracker = BYTETracker(**tracker_args)

    # Xử lý đầu vào video
    is_camera = FLAGS.video.isdigit()
    cap = cv2.VideoCapture(int(FLAGS.video) if is_camera else FLAGS.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {FLAGS.video}")

    # Cấu hình đầu ra video
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(FLAGS.output, fourcc, fps, frame_size)

    # Quản lý lịch sử tracking
    track_history = {}
    color_palette = np.random.randint(0, 255, (1000, 3))
    processing_times = deque(maxlen=20)

    while cap.isOpened():
        start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện vật thể với YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        # Xử lý kết quả YOLO
        boxes, scores = [], []
        for output in detections:
            for detection in output:
                confidence = detection[5:].max()
                if confidence > FLAGS.conf_threshold:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                    x1 = center_x - width//2
                    y1 = center_y - height//2
                    x2 = x1 + width
                    y2 = y1 + height
                    boxes.append([x1, y1, x2, y2])
                    scores.append(confidence)

        # --- APPLY NMS HERE ---
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=FLAGS.conf_threshold,
            nms_threshold=FLAGS.nms_threshold
        )

        # Filter out only the NMS-selected boxes
        nms_boxes = []
        nms_scores = []
        if len(indices) > 0:
            for i in indices.flatten():
                nms_boxes.append(boxes[i])
                nms_scores.append(scores[i])
        else:
            nms_boxes = []
            nms_scores = []

        # Now nms_boxes, nms_scores contain only the boxes that survived NMS
        if len(nms_boxes) > 0:
            dets = np.hstack([np.array(nms_boxes), np.array(nms_scores).reshape(-1, 1)])
        else:
            dets = np.empty((0, 5))

        # Cập nhật tracker
        online_targets = tracker.update(dets, [frame.shape[0], frame.shape[1]], (frame.shape[0], frame.shape[1]))

        # Visualize kết quả
        for target in online_targets:
            track_id = target.track_id
            bbox = target.tlbr
            x1, y1, x2, y2 = map(int, bbox)
            
            # Lấy màu cố định cho track ID
            color = color_palette[track_id % 1000].tolist()
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label
            label = f"Fish {track_id}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Cập nhật lịch sử di chuyển
            center = ((x1 + x2)//2, (y1 + y2)//2)
            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=30)
            track_history[track_id].append(center)

            # Vẽ đường di chuyển
            points = list(track_history[track_id])
            for i in range(1, len(points)):
                thickness = int(np.sqrt(30 / (i + 1)) * 1.5)
                cv2.line(frame, points[i - 1], points[i], color, thickness)

        # Tính toán FPS
        processing_time = time.perf_counter() - start_time
        processing_times.append(processing_time)
        avg_fps = 1.0 / (sum(processing_times)/len(processing_times))
        
        # Hiển thị thông tin
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Active Tracks: {len(track_history)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Ghi và hiển thị frame
        out.write(frame)
        cv2.imshow("BYTETrack Fish Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass