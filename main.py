import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model (ensure the file is in the same folder)
model = YOLO("best.pt")

# Check if CUDA is available (GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Running on: {device}")

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam

# Define class IDs for helmet and vest (adjust according to your dataset)
HELMET_CLASS_ID = 0  # Modify as needed
VEST_CLASS_ID = 1    # Modify as needed

# Load door images. (Check the filenames and paths)
open_door_img = cv2.imread("open_door.png")
if open_door_img is None:
    # Create a simulated open door image (green background)
    open_door_img = np.zeros((200, 400, 3), dtype=np.uint8)
    open_door_img[:] = (0, 255, 0)  # Green

closed_door_img = cv2.imread("closed_door.png")
if closed_door_img is None:
    # Create a simulated closed door image (red background)
    closed_door_img = np.zeros((200, 400, 3), dtype=np.uint8)
    closed_door_img[:] = (0, 0, 255)  # Red

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Run YOLOv8 prediction on the frame
    results = model(frame, device=device)

    # Count the number of helmets and vests detected
    helmet_count = 0
    vest_count = 0

    # Loop through results and draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Draw bounding box and label on the frame
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Count detections based on class ID
            if cls == HELMET_CLASS_ID:
                helmet_count += 1
            elif cls == VEST_CLASS_ID:
                vest_count += 1

    # Determine door status and message based on detections
    if helmet_count == 1 and vest_count == 1:
        door_status = cv2.resize(open_door_img, (frame_width, frame_height))
        message = ""
        text_color = (255, 255, 255)  # White text
    else:
        door_status = cv2.resize(closed_door_img, (frame_width, frame_height))
        if helmet_count > 1 or vest_count > 1:
            message = "multiple persons detected"
            text_color = (0, 0, 0)  # Black text for multiple persons detected
        else:
            message = ""
            text_color = (255, 255, 255)  # White text

    # Center the message on the door image
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (door_status.shape[1] - text_size[0]) // 2
    text_y = (door_status.shape[0] + text_size[1]) // 2
    cv2.putText(door_status, message, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Display the images
    cv2.imshow("YOLOv8 Webcam Detection", frame)
    cv2.imshow("Door Status", door_status)

    # Exit when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()