import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import atexit
from video_processor import GestureRecognizer, GESTURE_NAMES, GESTURE_TRANSLATIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global state
outputFrame = None
lock = threading.Lock()

class CameraStream:
    def __init__(self):
        self.source = 0  # 0 for webcam, string for file path
        self.queue = []  # Playlist queue
        self.video = None
        self.recognizer = GestureRecognizer()
        self.running = False
        self.lock = threading.Lock()
        self.last_prediction_update = 0
        
        # Start background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def start_source(self, source=0, playlist=None):
        with self.lock:
            if self.video is not None:
                self.video.release()
            
            self.queue = playlist if playlist else []
            self.source = source
            
            # If source is None but we have a playlist, start the first one
            if self.source is None and self.queue:
                self.source = self.queue.pop(0)

            if self.source is not None:
                self.video = cv2.VideoCapture(self.source)
                self.running = True
                if not self.video.isOpened():
                    print(f"Error: Could not open source {self.source}")
                    self.running = False
            else:
                 self.running = False

    def stop_source(self):
        with self.lock:
            # Only stop if queue is empty or forced
            self.running = False
            self.queue = [] # Clear queue on manual stop
            if self.video is not None:
                self.video.release()
                self.video = None

    def update(self):
        global outputFrame
        global latest_prediction
        
        # Track predictions for continuous gloss-level prediction
        frame_buffer = []  # Buffer to collect 30 frames
        gloss_predictions = []  # Store all gloss predictions with confidence
        
        while True:
            if self.running and self.video is not None and self.video.isOpened():
                success, frame = self.video.read()
                if success:
                    # Mirror frame only if webcam
                    if self.source == 0:
                        frame = cv2.flip(frame, 1)
                    
                    # Add frame to buffer
                    frame_buffer.append(frame.copy())
                    
                    # Process frame for visualization
                    try:
                        result = self.recognizer.predict(frame)
                        
                        # Draw results
                        annotated_frame = self.recognizer.draw_landmarks(frame, result['pose_result'], result['hand_result'])
                        
                        with lock:
                            outputFrame = annotated_frame.copy()
                        
                        # CONTINUOUS GLOSS-LEVEL PREDICTION
                        # When we have 30 frames, predict the gloss
                        if len(frame_buffer) >= 30:
                            # Process the 30-frame window for gloss prediction
                            window_result = self.recognizer.predict(frame_buffer[-1])  # Use last frame's result
                            
                            probs = window_result['probabilities']
                            top_3 = []
                            if probs is not None:
                                top_indices = probs.argsort()[-3:][::-1]
                                top_3 = [
                                    {"name": GESTURE_NAMES.get(i, f"G{i}"), "prob": float(probs[i])}
                                    for i in top_indices
                                ]
                            
                            gesture_name = window_result['gesture_name']
                            confidence = float(window_result['confidence'])
                            
                            if gesture_name is None or confidence < 0.6:
                                gesture_name = "Unknown"
                            
                            # For webcam: update immediately
                            # For video: accumulate glosses and show highest confidence
                            if self.source == 0:
                                # Webcam: real-time updates
                                latest_prediction = {
                                    "gesture": gesture_name,
                                    "confidence": confidence,
                                    "top_3": top_3
                                }
                            else:
                                # Video: accumulate gloss predictions
                                if gesture_name != "Unknown":
                                    gloss_predictions.append({
                                        "gesture": gesture_name,
                                        "confidence": confidence,
                                        "top_3": top_3
                                    })
                                    
                                    # Update display with highest confidence gloss so far
                                    best_gloss = max(gloss_predictions, key=lambda x: x['confidence'])
                                    latest_prediction = best_gloss
                            
                            # Slide the window (remove oldest 10 frames, keep overlap)
                            frame_buffer = frame_buffer[10:]
                            
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        
                else:
                    # Video finished
                    if isinstance(self.source, str):
                        print(f"Finished processing video: {self.source}")
                        
                        # Final gloss prediction: highest confidence from all windows
                        if gloss_predictions:
                            best_prediction = max(gloss_predictions, key=lambda x: x['confidence'])
                            latest_prediction = best_prediction
                            print(f"FINAL GLOSS: {best_prediction['gesture']} (confidence: {best_prediction['confidence']:.2%})")
                            print(f"Total gloss predictions: {len(gloss_predictions)}")
                        else:
                            print("No valid gloss predictions found")
                        
                        # Clear buffers for next video
                        frame_buffer = []
                        gloss_predictions = []
                        
                        # Check queue
                        if self.queue:
                            print(f"Starting next video. Remaining: {len(self.queue)}")
                            self.video.release()
                            self.source = self.queue.pop(0)
                            self.video = cv2.VideoCapture(self.source)
                        else:
                            print("All videos processed.")
                            self.running = False
                            if self.video:
                                self.video.release()
                                self.video = None
                    else:
                        time.sleep(0.1)
            else:
                 time.sleep(0.1)

    def stop(self):
        self.stop_source()

# Initialize global vars
latest_prediction = {}
camera_stream = None

def start_camera_stream():
    global camera_stream
    if camera_stream is None:
        camera_stream = CameraStream()
        camera_stream.start_source(0) # Default to webcam

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                # If no frame (e.g. camera stopped), send a blank or loading frame if desired
                # For now just continue or sleep
                time.sleep(0.1)
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03) # Cap at ~30 FPS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify(latest_prediction)

@app.route('/api/camera/control', methods=['POST'])
def camera_control():
    global camera_stream
    data = request.json
    action = data.get('action')
    source_type = data.get('source', 'webcam')
    filename = data.get('filename')

    if camera_stream is None:
        start_camera_stream()

    if action == 'stop':
        # Don't strictly stop, just clear queue and stop current source
        camera_stream.stop_source()
        return jsonify({"status": "stopped"})
    
    elif action == 'start':
        if source_type == 'webcam':
            camera_stream.start_source(0)
        elif source_type == 'video' and filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
            if os.path.exists(filepath):
                camera_stream.start_source(filepath)
            else:
                return jsonify({"error": "File not found"}), 404
        elif source_type == 'playlist':
             filenames = data.get('filenames', [])
             playlist = []
             for fname in filenames:
                 fpath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fname))
                 if os.path.exists(fpath):
                     playlist.append(fpath)
             
             if playlist:
                 # Start first, queue rest
                 camera_stream.start_source(playlist.pop(0), playlist=playlist)
             else:
                 return jsonify({"error": "No valid files in playlist"}), 400

        return jsonify({"status": "started", "source": source_type})

    return jsonify({"error": "Invalid action"}), 400

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Handle multiple files
    files = request.files.getlist('files[]')
    is_batch = True
    if not files:
        # Fallback for single file input
        if 'file' in request.files:
            files = [request.files['file']]
            is_batch = False
        else:
            return jsonify({"error": "No files provided"}), 400
    
    uploaded_filenames = []
    
    for file in files:
        if file.filename == '':
            continue
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_filenames.append(filename)
            print(f"Saved: {filepath}")
            
    if not uploaded_filenames:
         return jsonify({"error": "No valid files uploaded"}), 400

    response = {
        "message": "Files uploaded successfully",
        "filenames": uploaded_filenames
    }
    
    # If single file, add 'filename' key for frontend convenience
    if not is_batch and len(uploaded_filenames) == 1:
        response['filename'] = uploaded_filenames[0]

    return jsonify(response)

# Ensure camera starts strictly once
if __name__ == '__main__':
    start_camera_stream()
    app.run(debug=False, port=8181, use_reloader=False)

