# backend/gesture_recognizer.py
import numpy as np
import os

# ⬇️ ADDED: Import TensorFlow for loading ML model
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print("✅ TensorFlow available for ML-based gesture recognition")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available. Using rule-based recognition only.")

class GestureRecognizer:
    def __init__(self, model_path='models/final_mobilenet_model.h5'):
        self.gestures = [
            'Fist', 'Open_Hand', 'Pointing', 'Peace_Sign', 'Thumbs_Up',
            'OK_Sign', 'Rock', 'Call_Me', 'Thumbs_Down', 'Victory',
            'ILoveYou', 'One', 'Two', 'Three', 'Four', 'Five',
            'Stop', 'Wave'
        ]
        
        # Using rule-based recognition (no ML model needed)
        self.use_ml_model = False
        self.model_path = model_path
        # 🔽 THIS IS WHERE MODEL LOADING HAPPENS
        if TF_AVAILABLE and os.path.exists(model_path):
            try:
                print(f"📥 Loading ML model from: {model_path}")
                self.model = load_model(model_path)
                self.use_ml_model = True
                print(f"✅ ML model loaded successfully!")
                print(f"   Model type: {self.model.__class__.__name__}")
                print(f"   Input shape: {self.model.input_shape}")
                print(f"   Output classes: {self.model.output_shape[-1]}")
            except Exception as e:
                print(f"⚠️  Failed to load ML model: {e}")
                print("   Falling back to rule-based recognition")
                self.use_ml_model = False
        else:
            if not TF_AVAILABLE:
                print("ℹ️  TensorFlow not installed. Using rule-based recognition")
            elif not os.path.exists(model_path):
                print(f"ℹ️  Model file not found: {model_path}")
                print("   Using rule-based gesture recognition")
            else:
                print("ℹ️  Using rule-based gesture recognition")
        print("ℹ Using rule-based gesture recognition")
    
    def recognize(self, landmarks):
        """Recognize gesture from landmarks"""
        # 🔽 THIS DECIDES WHICH METHOD TO USE
        if self.use_ml_model:
            return self._ml_recognize(landmarks)
        else:
            return self._rule_based_recognize(landmarks)
        return self._rule_based_recognize(landmarks)
    
    def _rule_based_recognize(self, landmarks):
        """Rule-based gesture recognition"""
        # Extract key points
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        # Determine which fingers are extended
        fingers_up = self._count_fingers(landmarks)
        
        # Gesture classification
        total = sum(fingers_up)
        
        gesture = 'Unknown'
        confidence = 0.0
        
        if total == 0:
            gesture = 'Fist'
            confidence = 0.95
        elif total == 5:
            gesture = 'Open_Hand'
            confidence = 0.92
        elif fingers_up == [0, 1, 0, 0, 0]:
            gesture = 'Pointing'
            confidence = 0.90
        elif fingers_up == [0, 1, 1, 0, 0]:
            gesture = 'Peace_Sign'
            confidence = 0.88
        elif fingers_up == [1, 0, 0, 0, 0]:
            gesture = 'Thumbs_Up'
            confidence = 0.91
        elif fingers_up == [1, 1, 0, 0, 0]:
            gesture = 'OK_Sign'
            confidence = 0.87
        elif fingers_up == [0, 1, 1, 1, 1]:
            gesture = 'Four'
            confidence = 0.85
        elif fingers_up == [1, 1, 1, 0, 0]:
            gesture = 'Three'
            confidence = 0.84
        elif fingers_up == [0, 1, 1, 0, 0]:
            gesture = 'Two'
            confidence = 0.86
        elif fingers_up == [0, 1, 0, 0, 0]:
            gesture = 'One'
            confidence = 0.88
        elif fingers_up == [0, 0, 0, 0, 1]:
            gesture = 'Call_Me'
            confidence = 0.80
        else:
            gesture = f'{total}_Fingers'
            confidence = 0.70
        
        return {
            'gesture': gesture,
            'confidence': confidence,
            'fingers_extended': fingers_up,
            'total_fingers': total
        }
    
    # 🔽 THIS IS THE ML-BASED RECOGNITION METHOD
    def _ml_recognize(self, landmarks):
        """ML-based gesture recognition using trained model"""
        try:
            # Extract features from landmarks
            features = self._extract_features(landmarks)
            
            # Reshape for model input: (1, 21, 3) or (1, 63) depending on model
            if len(self.model.input_shape) == 3:
                # Model expects (batch, landmarks, coordinates)
                features = features.reshape(1, 21, 3)
            else:
                # Model expects flattened features
                features = features.reshape(1, -1)
            
            # Predict
            predictions = self.model.predict(features, verbose=0)[0]
            print("Check prediction ==",predictions)
            gesture_idx = np.argmax(predictions)
            confidence = predictions[gesture_idx]
            
            # Get gesture name
            if gesture_idx < len(self.gestures):
                gesture_name = self.gestures[gesture_idx]
            else:
                gesture_name = f'Class_{gesture_idx}'
            
            return {
                'gesture': gesture_name,
                'confidence': float(confidence),
                'method': 'ML',
                'all_predictions': {
                    self.gestures[i]: float(predictions[i])
                    for i in range(min(len(self.gestures), len(predictions)))
                }
            }
        except Exception as e:
            print(f"⚠️  ML recognition failed: {e}")
            # Fallback to rule-based
            return self._rule_based_recognize(landmarks)
    
    def _extract_features(self, landmarks):
        """Extract feature vector from landmarks"""
        features = []
        for landmark in landmarks:
            features.extend([landmark['x'], landmark['y'], landmark['z']])
        return np.array(features, dtype=np.float32)
    
    def _count_fingers(self, landmarks):
        """Count extended fingers"""
        fingers = []
        
        # Thumb (special case - check x-axis)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        if abs(thumb_tip['x'] - thumb_mcp['x']) > 0.04:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (check y-axis)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            if landmarks[tip_id]['y'] < landmarks[pip_id]['y']:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def get_supported_gestures(self):
        """Get list of supported gestures"""
        return self.gestures
    
    def get_model_info(self):
        """Get model information"""
        return {
            'type': 'Rule-based',
            'supported_gestures': self.gestures,
            'num_gestures': len(self.gestures)
        }