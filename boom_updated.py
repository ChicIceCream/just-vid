import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from lane_detector import ENet  # Assuming you have a custom ENet class

# Parameters for smoothing
window_length = 5  # Must be odd
polyorder = 2      # Should be less than window_length

# Function to apply smoothing and outlier detection
def smooth_and_filter_detections(left_lane_positions, right_lane_positions):
    if len(left_lane_positions) >= window_length:
        left_lane_smoothed = savgol_filter(left_lane_positions, window_length, polyorder)
        right_lane_smoothed = savgol_filter(right_lane_positions, window_length, polyorder)
    else:
        left_lane_smoothed = np.array(left_lane_positions)
        right_lane_smoothed = np.array(right_lane_positions)
    return left_lane_smoothed[-1], right_lane_smoothed[-1]

# Function to process video and display lane-detected output
def process_and_display_video(input_video_path, model):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file!")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define ROI vertices
    roi_vertices = np.array([[(50, frame_height),  # bottom left
                              (frame_width + 50, frame_height),  # bottom right
                              (frame_width // 2 + 200, frame_height // 2),  # upper right
                              (frame_width // 2 - 200, frame_height // 2)]],  # upper left
                            dtype=np.int32)
    vehicle_center_x = frame_width // 2 + 10
    target_y = 400  # Y-coordinate for lane point detection
    left_lane_positions = []
    right_lane_positions = []

    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot fetch frame.")
            break

        # Apply ROI mask
        mask = np.zeros_like(frame[:, :, 0])
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Preprocess frame for the model
        input_image = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        input_image = cv2.resize(input_image, (512, 256))
        input_image = input_image[..., None] / 255.0  # Normalize to [0, 1]
        input_tensor = torch.from_numpy(input_image).float().permute(2, 0, 1).unsqueeze(0).to('cuda')

        # Model inference
        with torch.no_grad():
            binary_logits, _ = model(input_tensor)
        binary_seg = torch.argmax(binary_logits, dim=1).squeeze().cpu().numpy()
        binary_seg = cv2.resize(binary_seg.astype(np.uint8), (frame_width, frame_height))

        # Apply the ROI mask to binary segmentation
        binary_seg = cv2.bitwise_and(binary_seg, binary_seg, mask=mask)

        # Extract lane points at target_y
        left_lane_points = np.where(binary_seg[target_y, :vehicle_center_x] == 1)[0]
        right_lane_points = np.where(binary_seg[target_y, vehicle_center_x:] == 1)[0]

        if left_lane_points.size > 0:
            left_lane_center = np.mean(left_lane_points)
            left_lane_positions.append(left_lane_center)
        if right_lane_points.size > 0:
            right_lane_center = np.mean(right_lane_points) + vehicle_center_x
            right_lane_positions.append(right_lane_center)

        if len(left_lane_positions) >= window_length and len(right_lane_positions) >= window_length:
            left_lane_center, right_lane_center = smooth_and_filter_detections(left_lane_positions, right_lane_positions)
            lanes_center = (left_lane_center + right_lane_center) / 2
            lateral_offset = (lanes_center - vehicle_center_x) * 0.007

            # Visualize results
            output_image = frame.copy()
            cv2.polylines(output_image, [roi_vertices], isClosed=True, color=(0, 255, 0), thickness=2)

            lane_overlay = np.zeros_like(frame)
            lane_overlay[binary_seg == 1] = [0, 255, 255]
            output_image = cv2.addWeighted(output_image, 0.7, lane_overlay, 0.3, 0)

            # Draw lane centers and vehicle center
            lane_center_y = target_y
            car_center_y = target_y + 100

            cv2.circle(output_image, (int(left_lane_center), target_y), 5, (255, 0, 0), -1)
            cv2.circle(output_image, (int(right_lane_center), target_y), 5, (0, 0, 255), -1)
            cv2.circle(output_image, (int(lanes_center), lane_center_y), 8, (0, 255, 0), -1)
            cv2.circle(output_image, (int(vehicle_center_x), car_center_y), 8, (255, 0, 0), -1)

            # Draw connecting line between lane center and vehicle center
            cv2.line(output_image,
                     (int(lanes_center), lane_center_y),
                     (int(vehicle_center_x), car_center_y),
                     (255, 0, 0), 2)

            # Add labels
            cv2.putText(output_image, f"Lateral Offset: {lateral_offset:.2f} meters",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow('Lane Detection with Lateral Offset', output_image)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Processing completed.")

# Load the pre-trained ENet model
model_path = 'ENET.pth'
enet_model = ENet(2, 4).to('cuda')  # Adjust based on your ENet initialization
enet_model.load_state_dict(torch.load(model_path, map_location='cuda'))
enet_model.eval()

# Run the video processing
input_video_path = "output2.mp4"
process_and_display_video(input_video_path, enet_model)
