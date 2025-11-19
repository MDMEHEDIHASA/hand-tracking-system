# backend/gesture_recognizer.py
import numpy as np
import os

class GestureRecognizer:
    def __init__(self, model_path='models/gesture_model.h5'):
        self.gestures = [
            'Fist', 'Open_Hand', 'Pointing', 'Peace_Sign', 'Thumbs_Up',
            'OK_Sign', 'Rock', 'Call_Me', 'Thumbs_Down', 'Victory',
            'ILoveYou', 'One', 'Two', 'Three', 'Four', 'Five',
            'Stop', 'Wave'
        ]
        
        # Using rule-based recognition (no ML model needed)
        self.use_ml_model = False
        print("ℹ Using rule-based gesture recognition")
    
    def recognize(self, landmarks):
        """Recognize gesture from landmarks"""
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