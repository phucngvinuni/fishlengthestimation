# Fish Detection and Real-World Length Estimation with Depth Mapping

This project combines **YOLO-based object detection** with **depth estimation** to detect fish, scale bounding boxes to a reference depth, and calculate the real-world length of fish in a video. The program overlays the calculated lengths on the video frames and saves the results as a new video.

## Features

- **YOLO Object Detection**:
  - Detects fish in video frames with bounding boxes and confidence scores.
  
- **Depth Estimation**:
  - Uses the `Depth Anything` model to estimate depth for each frame.

- **Real-World Length Estimation**:
  - Scales detected bounding boxes to a reference depth.
  - Calculates the real-world length of fish based on the median scaled length.

- **Overlay Results**:
  - Annotates each bounding box with the detected label, confidence score, and real-world length (in centimeters).

- **Output**:
  - Combines the original video frames and depth map into a side-by-side video for visualization.

## Prerequisites

1. **Python 3.7+**
2. **Libraries**:
   - OpenCV
   - PyTorch
   - torchvision
   - numpy
3. **YOLO Configuration Files**:
   - `yolo-fish-2.cfg`: YOLO configuration file.
   - `merge_yolo-fish-2.weights`: YOLO pre-trained weights.
   - `obj.names`: File containing class names (e.g., "fish").
   - Weight is trained on deepfish and ozfish data: https://github.com/tamim662/YOLO-Fish
4. **Pre-trained Depth Model**:
   - `LiheYoung/depth_anything_vitl14`.

5. **Input Video**:
   - A video file (`input.mp4`) containing fish.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fish-detection-depth-estimation.git
   cd fish-detection-depth-estimation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the YOLO configuration files (`.cfg`, `.weights`, `.names`) in the project directory.

4. Download the pre-trained Depth Anything model:
   ```python
   from depth_anything.dpt import DepthAnything
   depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14')
   ```

## Usage

1. **Run the Program**:
   ```bash
   python main.py
   ```

2. **Input**:
   - Place your video file as `input.mp4` in the project directory.

3. **Output**:
   - The processed video will be saved as `output_depth_yolo.mp4`.

## Key Functions

### `detect_objects(image, conf_threshold=0.4, nms_threshold=0.5)`
- Detects objects in an image using YOLO.
- Parameters:
  - `conf_threshold`: Confidence threshold for filtering detections.
  - `nms_threshold`: Non-Max Suppression threshold.
- Returns:
  - List of detected objects with bounding boxes, confidence scores, and labels.

### `scale_and_estimate_length(box, center_depth, reference_depth)`
- Scales bounding box dimensions to a common depth plane.
- Parameters:
  - `box`: Bounding box coordinates.
  - `center_depth`: Depth at the center of the bounding box.
  - `reference_depth`: Reference depth for scaling.
- Returns:
  - Scaled width (used for length estimation).

## Output Video

The output video combines:
1. **Original Frame**:
   - Displays bounding boxes, labels, confidence scores, and real-world lengths.
2. **Depth Map**:
   - Visualizes depth estimation using a heatmap.

## Example Results

### Original Frame with Annotations
![Original Frame Example](images/original_frame_example.png)

### Depth Map
![Depth Map Example](images/depth_map_example.png)

### Combined Output
[![Combined Output Example](images/combined_output_example.png)](https://github.com/phucngvinuni/fishlengthestimation/blob/main/output_depth_yolo.mp4)

## Future Improvements

- Add unit calibration to convert pixel-based lengths to real-world measurements directly.
- Improve real-world length estimation with multiple reference objects.
- Extend functionality to support other object classes.
## Ready to run link
https://drive.google.com/drive/folders/1yhtynq21lE0aShG1LwarLl-z-r3fVcT_?usp=sharing
Run srcdepth.py for video processing
## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

---


