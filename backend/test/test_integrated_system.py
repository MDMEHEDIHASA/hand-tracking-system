# backend/test_integrated_system.py
import cv2
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from audio_controller import AudioController
from mouse_controller import MouseController
import time

def test_integrated_system():
    """Test all components working together"""
    
    print("="*70)
    print("TEST 5: INTEGRATED SYSTEM TEST")
    print("="*70)
    
    # Initialize all components
    print("\n1. Initializing all components...")
    tracker = HandTracker()
    recognizer = GestureRecognizer()
    audio = AudioController()
    mouse = MouseController()
    print("✅ All components initialized")
    
    # Open camera
    print("\n2. Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ FAIL - Cannot open camera")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✅ Camera opened")
    
    print("\n" + "="*70)
    print("INTEGRATED SYSTEM TEST - ALL MODES")
    print("="*70)
    print("\nThis test cycles through all 4 modes:")
    print("  1. AUDIO MODE    - Control volume with finger distance")
    print("  2. GESTURE MODE  - Recognize hand gestures")
    print("  3. DRAW MODE     - Track index finger for drawing")
    print("  4. MOUSE MODE    - Control cursor with hand")
    print("\nEach mode runs for 10 seconds")
    print("Total test time: 40 seconds")
    print("\nPress 'q' at any time to quit")
    print("="*70)
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    modes = ['audio', 'gesture', 'draw', 'mouse']
    mode_names = ['AUDIO CONTROL', 'GESTURE RECOGNITION', 'AIR DRAWING', 'VIRTUAL MOUSE']
    mode_duration = 10  # seconds per mode
    
    mode_stats = {mode: {'frames': 0, 'detections': 0} for mode in modes}
    
    for mode_idx, (mode, mode_name) in enumerate(zip(modes, mode_names)):
        print(f"\n{'='*70}")
        print(f"MODE {mode_idx + 1}/4: {mode_name}")
        print(f"{'='*70}")
        
        mode_start = time.time()
        drawing_points = []
        
        while (time.time() - mode_start) < mode_duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            mode_stats[mode]['frames'] += 1
            
            # Detect hands
            results = tracker.process_frame(frame)
            
            if results['hands_detected']:
                mode_stats[mode]['detections'] += 1
                landmarks = results['landmarks']
                
                # Mode-specific processing
                if mode == 'audio':
                    # Audio control
                    thumb = landmarks[0][4]
                    index = landmarks[0][8]
                    distance = tracker.calculate_distance(thumb, index)
                    volume = audio.control_volume(distance)
                    
                    annotated_frame = tracker.draw_landmarks(frame, landmarks, mode)
                    
                    # Display audio info
                    cv2.rectangle(annotated_frame, (10, 10), (400, 150), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, "AUDIO CONTROL MODE", 
                               (20, 40), cv2.