print("Importing torch...")
import torch
print("Torch imported.")

print("Checking device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Importing mediapipe...")
import mediapipe as mp
print("MediaPipe imported.")

print("Initializing MediaPipe tasks...")
try:
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    print("MediaPipe classes accessed.")
except Exception as e:
    print(f"MediaPipe error: {e}")

print("Done")
