# backend/hand_tracker.py
import cv2
import mediapipe as mp
import numpy as np
import math

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def process_frame(self, frame):
        """Process frame and detect hands"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        output = {
            'hands_detected': False,
            'num_hands': 0,
            'landmarks': [],
            'handedness': []
        }
        
        if results.multi_hand_landmarks:
            output['hands_detected'] = True
            output['num_hands'] = len(results.multi_hand_landmarks)
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Convert to pixel coordinates
                h, w, _ = frame.shape
                landmarks_list = []
                
                for landmark in hand_landmarks.landmark:
                    landmarks_list.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'pixel_x': int(landmark.x * w),
                        'pixel_y': int(landmark.y * h)
                    })
                
                output['landmarks'].append(landmarks_list)
                
                # Get handedness (Left/Right)
                if results.multi_handedness:
                    handedness = results.multi_handedness[hand_idx].classification[0]
                    output['handedness'].append({
                        'label': handedness.label,
                        'score': handedness.score
                    })
        
        return output
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        x1 = point1['pixel_x'] if isinstance(point1, dict) else point1[0]
        y1 = point1['pixel_y'] if isinstance(point1, dict) else point1[1]
        x2 = point2['pixel_x'] if isinstance(point2, dict) else point2[0]
        y2 = point2['pixel_y'] if isinstance(point2, dict) else point2[1]
        
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def draw_landmarks(self, frame, landmarks_data, mode='audio'):
        """Draw hand landmarks on frame"""
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        
        for hand_landmarks in landmarks_data:
            # Draw all landmarks
            for idx, landmark in enumerate(hand_landmarks):
                x, y = landmark['pixel_x'], landmark['pixel_y']
                
                # Different colors for different landmarks
                if idx in [4, 8]:  # Thumb and index tips
                    color = (255, 0, 255)  # Pink
                    radius = 8
                elif idx == 0:  # Wrist
                    color = (0, 255, 255)  # Yellow
                    radius = 10
                else:
                    color = (0, 255, 0)  # Green
                    radius = 5
                
                cv2.circle(annotated_frame, (x, y), radius, color, -1)
                cv2.circle(annotated_frame, (x, y), radius + 2, (255, 255, 255), 2)
            
            # Draw connections
            connections = self.mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                
                cv2.line(annotated_frame,
                        (start['pixel_x'], start['pixel_y']),
                        (end['pixel_x'], end['pixel_y']),
                        (255, 255, 255), 2)
            
            # Mode-specific visualizations
            if mode == 'audio':
                # Draw distance line between thumb and index
                thumb = hand_landmarks[4]
                index = hand_landmarks[8]
                
                cv2.line(annotated_frame,
                        (thumb['pixel_x'], thumb['pixel_y']),
                        (index['pixel_x'], index['pixel_y']),
                        (255, 0, 255), 3)
                
                distance = self.calculate_distance(thumb, index)
                mid_x = (thumb['pixel_x'] + index['pixel_x']) // 2
                mid_y = (thumb['pixel_y'] + index['pixel_y']) // 2
                
                cv2.putText(annotated_frame, f"{distance:.1f}px",
                           (mid_x, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            elif mode == 'draw':
                # Highlight index finger for drawing
                index = hand_landmarks[8]
                cv2.circle(annotated_frame,
                          (index['pixel_x'], index['pixel_y']),
                          15, (0, 255, 255), 3)
            
            elif mode == 'mouse':
                # Show cursor position
                index = hand_landmarks[8]
                cv2.circle(annotated_frame,
                          (index['pixel_x'], index['pixel_y']),
                          20, (0, 0, 255), 2)
                cv2.line(annotated_frame,
                        (index['pixel_x'] - 15, index['pixel_y']),
                        (index['pixel_x'] + 15, index['pixel_y']),
                        (0, 0, 255), 2)
                cv2.line(annotated_frame,
                        (index['pixel_x'], index['pixel_y'] - 15),
                        (index['pixel_x'], index['pixel_y'] + 15),
                        (0, 0, 255), 2)
        
        return annotated_frame