# backend/test_gesture_recognizer.py
import cv2
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
import time

def test_gesture_recognizer():
    """Test gesture recognition"""
    
    print("="*70)
    print("TEST 4: GESTURE RECOGNIZER MODULE")
    print("="*70)
    
    # Initialize
    print("\n1. Initializing components...")
    tracker = HandTracker()
    recognizer = GestureRecognizer()
    print("✅ Components initialized")
    
    # Show supported gestures
    gestures = recognizer.get_supported_gestures()
    print(f"\n2. Supported gestures ({len(gestures)}):")
    for i, gesture in enumerate(gestures, 1):
        print(f"   {i:2d}. {gesture}")
    
    # Open camera
    print("\n3. Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ FAIL - Cannot open camera")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✅ Camera opened")
    
    print("\n" + "="*70)
    print("GESTURE RECOGNITION TEST")
    print("="*70)
    print("\nINSTRUCTIONS:")
    print("Try these gestures:")
    print("  ✊ FIST          - Close all fingers")
    print("  🖐️ OPEN HAND    - Open all fingers")
    print("  ✌️  PEACE SIGN   - Index + middle finger up")
    print("  👍 THUMBS UP    - Only thumb up")
    print("  👆 POINTING     - Only index finger up")
    print("  🤙 CALL ME      - Only pinky up")
    print("  🤘 ROCK         - Index + pinky up")
    print("\nTest runs for 30 seconds. Press 'q' to quit early.")
    print("="*70)
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    start_time = time.time()
    frame_count = 0
    gesture_counts = {}
    last_gesture = None
    gesture_history = []
    
    while (time.time() - start_time) < 30:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Detect hands
        results = tracker.process_frame(frame)
        
        if results['hands_detected']:
            landmarks = results['landmarks']
            
            # Recognize gesture
            gesture_data = recognizer.recognize(landmarks[0])
            gesture_name = gesture_data['gesture']
            confidence = gesture_data['confidence']
            fingers_up = gesture_data.get('fingers_extended', [])
            
            # Track gesture counts
            if gesture_name not in gesture_counts:
                gesture_counts[gesture_name] = 0
            gesture_counts[gesture_name] += 1
            
            # Record gesture changes
            if gesture_name != last_gesture:
                gesture_history.append({
                    'time': time.time() - start_time,
                    'gesture': gesture_name,
                    'confidence': confidence
                })
                last_gesture = gesture_name
            
            # Draw landmarks
            annotated_frame = tracker.draw_landmarks(frame, landmarks, 'gesture')
            
            # Display gesture info
            cv2.rectangle(annotated_frame, (5, 5), (600, 200), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (5, 5), (600, 200), (0, 255, 0), 2)
            
            cv2.putText(annotated_frame, f"GESTURE: {gesture_name}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(annotated_frame, f"Confidence: {confidence*100:.1f}%", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Show finger states
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            fingers_text = " | ".join([f"{name}: {'UP' if fingers_up[i] else 'DOWN'}" 
                                       for i, name in enumerate(finger_names)])
            cv2.putText(annotated_frame, fingers_text, 
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Confidence bar
            bar_width = int(confidence * 550)
            cv2.rectangle(annotated_frame, (20, 150), (20 + bar_width, 180), 
                         (0, 255, 0), -1)
            cv2.rectangle(annotated_frame, (20, 150), (570, 180), (255, 255, 255), 2)
            
        else:
            annotated_frame = frame
            cv2.putText(annotated_frame, "NO HAND DETECTED", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(annotated_frame, "Show your hand to the camera", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Time remaining
        time_left = 30 - int(time.time() - start_time)
        cv2.putText(annotated_frame, f"Time: {time_left}s", 
                   (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Gesture Recognition Test', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Total frames processed: {frame_count}")
    print(f"Total gestures detected: {sum(gesture_counts.values())}")
    print(f"Unique gestures recognized: {len(gesture_counts)}")
    
    print("\nGesture frequency:")
    sorted_gestures = sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True)
    for gesture, count in sorted_gestures:
        percentage = (count / sum(gesture_counts.values())) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {gesture:20s}: {count:4d} frames ({percentage:5.1f}%) {bar}")
    
    print(f"\nGesture transitions detected: {len(gesture_history)}")
    if gesture_history:
        print("\nFirst 10 transitions:")
        for i, trans in enumerate(gesture_history[:10], 1):
            print(f"  {i:2d}. {trans['time']:6.2f}s → {trans['gesture']:20s} "
                  f"({trans['confidence']*100:.1f}%)")
    
    # Pass/Fail
    print("\n" + "="*70)
    if len(gesture_counts) >= 3 and sum(gesture_counts.values()) > 100:
        print("✅ PASS - Gesture recognition working perfectly!")
        print(f"   Recognized {len(gesture_counts)} different gestures")
        print(f"   Processed {sum(gesture_counts.values())} gesture detections")
    elif len(gesture_counts) >= 2:
        print("⚠️  PARTIAL - Gesture recognition works but limited variety")
        print(f"   Try making more different gestures")
    else:
        print("❌ FAIL - Insufficient gesture detection")
        print("   Make sure to show various hand gestures")
    print("="*70)
    
    return len(gesture_counts) >= 3

if __name__ == "__main__":
    test_gesture_recognizer()