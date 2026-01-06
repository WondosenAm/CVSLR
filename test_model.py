from video_processor import GestureRecognizer
print("Initializing GestureRecognizer...")
try:
    recognizer = GestureRecognizer()
    print("GestureRecognizer initialized successfully.")
except Exception as e:
    print(f"Failed to initialize GestureRecognizer: {e}")
print("Done")
