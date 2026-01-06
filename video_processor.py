import cv2
import torch
import numpy as np
import mediapipe as mp
import warnings
import os
import time
from collections import deque
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

class CNNTransformerHybrid(nn.Module):
    """EXACT SAME MODEL AS TRAINING"""
    
    def __init__(self, input_size=258, num_classes=51, dropout=0.35):
        super().__init__()
        
        # CNN FOR LOCAL FEATURES
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.AdaptiveAvgPool1d(16)
        )
        
        # TRANSFORMER FOR TEMPORAL DEPENDENCIES
        self.pos_embed = nn.Parameter(torch.randn(1, 16, 256))
        self.pos_dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # ATTENTION POOLING
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, 256))
        
        # CLASSIFICATION HEAD
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN processing
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        x, _ = self.attention_pool(query, x, x)
        x = x.squeeze(1)
        
        # Classification
        return self.classifier(x)

GESTURE_NAMES = {
    0: 'abang',
    1: 'anak_lelaki', 
    2: 'anak_perempuan',
    3: 'apa',
    4: 'apa_khabar',
    5: 'assalamualaikum',
    6: 'ayah',
    7: 'bagaimana',
    8: 'bahasa_isyarat',
    9: 'baik',
    10: 'bapa_saudara',
    11: 'beli',
    12: 'beli_2',
    13: 'berapa',
    14: 'bila',
    15: 'bomba',
    16: 'buat',
    17: 'emak',
    18: 'emak_saudara',
    19: 'hari',
    20: 'hi',
    21: 'hujan',
    22: 'jahat',
    23: 'jangan',
    24: 'kakak',
    25: 'keluarga',
    26: 'kereta',
    27: 'lelaki',
    28: 'lemak',
    29: 'main',
    30: 'mana',
    31: 'masalah',
    32: 'nasi',
    33: 'nasi_lemak',
    34: 'panas',
    35: 'panas_2',
    36: 'pandai',
    37: 'pandai_2',
    38: 'payung',
    39: 'perempuan',
    40: 'perlahan',
    41: 'perlahan_2',
    42: 'pinjam',
    43: 'polis',
    44: 'pukul',
    45: 'ribut',
    46: 'saudara',
    47: 'sejuk',
    48: 'siapa',
    49: 'tandas'
}

# English translations for Malaysian Sign Language gestures
GESTURE_TRANSLATIONS = {
    'abang': 'Brother',
    'anak_lelaki': 'Son',
    'anak_perempuan': 'Daughter',
    'apa': 'What',
    'apa_khabar': 'How are you',
    'assalamualaikum': 'Peace be upon you',
    'ayah': 'Father',
    'bagaimana': 'How',
    'bahasa_isyarat': 'Sign Language',
    'baik': 'Good',
    'bapa_saudara': 'Uncle',
    'beli': 'Buy',
    'beli_2': 'Buy (variant)',
    'berapa': 'How much',
    'bila': 'When',
    'bomba': 'Firefighter',
    'buat': 'Do/Make',
    'emak': 'Mother',
    'emak_saudara': 'Aunt',
    'hari': 'Day',
    'hi': 'Hi',
    'hujan': 'Rain',
    'jahat': 'Bad',
    'jangan': 'Don\'t',
    'kakak': 'Sister',
    'keluarga': 'Family',
    'kereta': 'Car',
    'lelaki': 'Male',
    'lemak': 'Fat',
    'main': 'Play',
    'mana': 'Where',
    'masalah': 'Problem',
    'nasi': 'Rice',
    'nasi_lemak': 'Nasi Lemak',
    'panas': 'Hot',
    'panas_2': 'Hot (variant)',
    'pandai': 'Smart',
    'pandai_2': 'Smart (variant)',
    'payung': 'Umbrella',
    'perempuan': 'Female',
    'perlahan': 'Slow',
    'perlahan_2': 'Slow (variant)',
    'pinjam': 'Borrow',
    'polis': 'Police',
    'pukul': 'Hit/Time',
    'ribut': 'Storm',
    'saudara': 'Sibling',
    'sejuk': 'Cold',
    'siapa': 'Who',
    'tandas': 'Toilet',
    'Unknown': 'Unknown'
}

class GestureRecognizer:
    def __init__(self, model_path='best_cnn_transformer_hybrid.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_buffer = deque(maxlen=30)
        self.model = self._load_model(model_path)
        self._init_mediapipe()
        
    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = CNNTransformerHybrid(
            input_size=258,
            num_classes=len(GESTURE_NAMES),
            dropout=0.35
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _init_mediapipe(self):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Check model files
        pose_model_path = 'pose_landmarker_lite.task'
        hand_model_path = 'hand_landmarker.task'
        
        if not os.path.exists(pose_model_path) or not os.path.exists(hand_model_path):
            raise FileNotFoundError("MediaPipe task files not found.")

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )

        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=VisionRunningMode.IMAGE
        )

        self.pose_landmarker = PoseLandmarker.create_from_options(pose_options)
        self.hand_landmarker = HandLandmarker.create_from_options(hand_options)

    def extract_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        pose_result = self.pose_landmarker.detect(mp_image)
        hand_result = self.hand_landmarker.detect(mp_image)
        
        features = np.zeros(258, dtype=np.float32)
        
        # Pose
        if pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]
            for i in range(min(33, len(pose_landmarks))):
                idx = i * 4
                features[idx] = pose_landmarks[i].x
                features[idx + 1] = pose_landmarks[i].y
                features[idx + 2] = pose_landmarks[i].z
                features[idx + 3] = pose_landmarks[i].visibility
        
        # Hands
        left_hand_idx = 132
        right_hand_idx = 195
        
        if hand_result.hand_landmarks:
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                if i < len(hand_result.handedness):
                    handedness = hand_result.handedness[i][0].category_name
                    start_idx = left_hand_idx if handedness == 'Left' else right_hand_idx
                    
                    for j in range(min(21, len(hand_landmarks))):
                        idx = start_idx + (j * 3)
                        features[idx] = hand_landmarks[j].x
                        features[idx + 1] = hand_landmarks[j].y
                        features[idx + 2] = hand_landmarks[j].z
                        
        return features, pose_result, hand_result

    def predict(self, frame):
        features, pose_res, hand_res = self.extract_landmarks(frame)
        self.sequence_buffer.append(features)
        
        prediction_idx = None
        confidence = 0.0
        probabilities = None
        
        if len(self.sequence_buffer) == 30:
            sequence_array = np.array(list(self.sequence_buffer), dtype=np.float32)
            sequence_array = np.expand_dims(sequence_array, axis=0)
            input_tensor = torch.tensor(sequence_array, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, dim=1)
                
                prediction_idx = pred.item()
                confidence = conf.item()
                probabilities = probs[0].cpu().numpy()
                
        return {
            'prediction_idx': prediction_idx,
            'confidence': confidence,
            'gesture_name': GESTURE_NAMES.get(prediction_idx) if prediction_idx is not None else None,
            'pose_result': pose_res,
            'hand_result': hand_res,
            'probabilities': probabilities
        }

    def draw_landmarks(self, frame, pose_result, hand_result):
        annotated_frame = frame.copy()
        
        if pose_result.pose_landmarks:
            for landmark in pose_result.pose_landmarks[0]:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
        
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(annotated_frame, (x, y), 3, (255, 0, 0), -1)
                    
        return annotated_frame
