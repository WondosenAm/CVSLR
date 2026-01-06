# Malaysian Sign Language Recognition (CVSLR)

A real-time Malaysian Sign Language recognition system using Computer Vision and Deep Learning.

## Features

- **Real-time Recognition**: Live camera feed processing with gesture detection
- **Video Upload**: Support for single video and batch playlist processing
- **High Accuracy**: CNN-Transformer hybrid model with 50 gesture classes
- **Modern UI**: Elegant glassmorphism design with gradient accents
- **Session Logging**: Track detected gestures with confidence scores

## Technology Stack

- **Backend**: Flask, Python 3.11
- **ML/AI**: PyTorch, MediaPipe, OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **Model**: CNN-Transformer Hybrid Architecture

## Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/WondosenAm/CVSLR.git
   cd CVSLR
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the app**
   - Open browser: `http://127.0.0.1:8181`

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for instructions on deploying to Render or other platforms.

## Model Information

- **Architecture**: CNN-Transformer Hybrid
- **Input**: 30-frame sequences of pose + hand landmarks (258 features)
- **Output**: 50 Malaysian Sign Language gestures
- **Confidence Threshold**: 60%

## Credits

**Developed by**: Group 4 (Vision)  
**Institution**: Faculty of Computer Science and Information Technology  
**Department**: Artificial Intelligence  
**University**: University Malaya

## License

This project is for educational purposes.
