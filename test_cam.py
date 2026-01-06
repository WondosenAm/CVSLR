import cv2
import time

print("Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
else:
    print("Camera opened successfully")
    ret, frame = cap.read()
    if ret:
        print(f"Read frame of shape {frame.shape}")
    else:
        print("Failed to read frame")
    cap.release()
print("Done")
