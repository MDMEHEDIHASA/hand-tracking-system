# backend/app.py
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import json
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from audio_controller import AudioController
from mouse_controller import MouseController
import threading
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
hand_tracker = HandTracker()
gesture_recognizer = GestureRecognizer()
audio_controller = AudioController()
mouse_controller = MouseController()

# Global state
current_mode = 'audio'
is_tracking = False
camera = None
recording_data = []
is_recording = False

class VideoCamera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
    def __del__(self):
        if self.camera:
            self.camera.release()
    
    def get_frame(self):
        success, frame = self.camera.read()
        if success:
            frame = cv2.flip(frame, 1)  # Mirror effect
            return frame
        return None

def process_frame_worker():
    """Background worker for processing frames"""
    global is_tracking, current_mode, camera, recording_data, is_recording
    
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    while True:
        if is_tracking and camera:
            frame = camera.get_frame()
            
            if frame is not None:
                # Detect hands
                results = hand_tracker.process_frame(frame)
                
                if results['hands_detected']:
                    landmarks = results['landmarks']
                    
                    # Mode-specific processing
                    if current_mode == 'audio':
                        distance = hand_tracker.calculate_distance(
                            landmarks[0][4], landmarks[0][8]
                        )
                        volume = audio_controller.control_volume(distance)
                        results['volume'] = volume
                        results['distance'] = distance
                        
                    elif current_mode == 'gesture':
                        gesture_data = gesture_recognizer.recognize(landmarks[0])
                        results['gesture'] = gesture_data['gesture']
                        results['confidence'] = gesture_data['confidence']
                        
                    elif current_mode == 'draw':
                        # Air drawing - track index finger tip
                        index_tip = landmarks[0][8]
                        results['drawing_point'] = index_tip
                        
                    elif current_mode == 'mouse':
                        # Virtual mouse control
                        index_tip = landmarks[0][8]
                        thumb_tip = landmarks[0][4]
                        mouse_controller.move_cursor(index_tip)
                        
                        # Pinch to click
                        distance = hand_tracker.calculate_distance(thumb_tip, index_tip)
                        if distance < 30:
                            mouse_controller.click()
                        
                        results['cursor_position'] = index_tip
                    
                    # Visualize
                    annotated_frame = hand_tracker.draw_landmarks(
                        frame, landmarks, current_mode
                    )
                    
                    # Calculate FPS
                    fps_counter += 1
                    if time.time() - fps_start_time >= 1.0:
                        current_fps = fps_counter
                        fps_counter = 0
                        fps_start_time = time.time()
                    
                    results['fps'] = current_fps
                    
                    # Recording
                    if is_recording:
                        recording_data.append({
                            'timestamp': time.time(),
                            'landmarks': landmarks,
                            'mode': current_mode,
                            'results': results
                        })
                    
                    # Encode frame for streaming
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame_bytes = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit to frontend via WebSocket
                    socketio.emit('frame_data', {
                        'frame': frame_bytes,
                        'metrics': results
                    })
                    
        time.sleep(0.01)  # ~100 FPS processing rate

# Start background worker
threading.Thread(target=process_frame_worker, daemon=True).start()

# REST API Endpoints

@app.route('/api/start', methods=['POST'])
def start_tracking():
    """Start hand tracking"""
    global is_tracking, camera
    
    if not is_tracking:
        camera = VideoCamera()
        is_tracking = True
        return jsonify({'status': 'success', 'message': 'Tracking started'})
    
    return jsonify({'status': 'info', 'message': 'Already tracking'})

@app.route('/api/stop', methods=['POST'])
def stop_tracking():
    """Stop hand tracking"""
    global is_tracking, camera
    
    if is_tracking:
        is_tracking = False
        if camera:
            del camera
            camera = None
        return jsonify({'status': 'success', 'message': 'Tracking stopped'})
    
    return jsonify({'status': 'info', 'message': 'Not tracking'})

@app.route('/api/mode', methods=['POST'])
def set_mode():
    """Change tracking mode"""
    global current_mode
    
    data = request.json
    mode = data.get('mode', 'audio')
    
    if mode in ['audio', 'gesture', 'draw', 'mouse']:
        current_mode = mode
        return jsonify({'status': 'success', 'mode': current_mode})
    
    return jsonify({'status': 'error', 'message': 'Invalid mode'}), 400

@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    """Start recording gesture data"""
    global is_recording, recording_data
    
    recording_data = []
    is_recording = True
    return jsonify({'status': 'success', 'message': 'Recording started'})

@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    """Stop recording and return data"""
    global is_recording, recording_data
    
    is_recording = False
    data = recording_data.copy()
    recording_data = []
    
    return jsonify({
        'status': 'success',
        'frames': len(data),
        'data': data
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration"""
    return jsonify({
        'modes': ['audio', 'gesture', 'draw', 'mouse'],
        'current_mode': current_mode,
        'is_tracking': is_tracking,
        'is_recording': is_recording,
        'supported_gestures': gesture_recognizer.get_supported_gestures()
    })

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    return jsonify(gesture_recognizer.get_model_info())

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time()
    })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    print("🚀 Hand Tracking System Backend")
    print("="*50)
    print("Server starting on http://localhost:5000")
    print("WebSocket on ws://localhost:5000")
    print("="*50)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)