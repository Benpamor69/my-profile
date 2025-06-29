import cv2
import numpy as np
import sys

# Usage example:
# python3 overlay_face.py input_video.mp4 face.png output_video.mp4

if len(sys.argv) != 4:
    print("Usage: python3 overlay_face.py <input_video> <face_image> <output_video>")
    sys.exit(1)

video_path = sys.argv[1]
face_path = sys.argv[2]
output_path = sys.argv[3]

# Load the face image with alpha channel (for transparency)
face_img = cv2.imread(face_path, cv2.IMREAD_UNCHANGED)
if face_img is None:
    raise FileNotFoundError(f"Could not load face image: {face_path}")

# If the face image doesn't have an alpha channel, create one
if face_img.shape[2] == 3:
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2BGRA)

# Resize the face image to a reasonable size (adjust as needed)
face_img = cv2.resize(face_img, (150, 150), interpolation=cv2.INTER_AREA)

# Separate the color and alpha channel
face_color = face_img[:, :, :3]
face_alpha = face_img[:, :, 3] / 255.0

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Could not open video file: {video_path}")

# Prepare video writer with the same properties as input
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Overlay position (top-left corner). Adjust as needed.
face_x, face_y = 50, 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure face image fits in the frame
    y1, y2 = face_y, face_y + face_img.shape[0]
    x1, x2 = face_x, face_x + face_img.shape[1]

    if y2 > frame.shape[0] or x2 > frame.shape[1]:
        break

    # Get the region of interest from the frame
    roi = frame[y1:y2, x1:x2]

    # Blend the face image with the ROI using the alpha mask
    for c in range(3):
        roi[:, :, c] = (face_alpha * face_color[:, :, c] +
                        (1 - face_alpha) * roi[:, :, c])

    frame[y1:y2, x1:x2] = roi

    out.write(frame)

cap.release()
out.release()
print(f"Saved output to {output_path}")
