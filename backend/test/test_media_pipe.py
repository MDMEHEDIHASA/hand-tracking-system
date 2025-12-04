# backend/test_mediapipe.py
import cv2
import mediapipe as mp
import time

def test_mediapipe():
    """Test MediaPipe hand detection"""
    
    print("="*60)
    print("TESTING MEDIAPIPE HAND DETECTION")
    print("="*60)
    
    # Initialize MediaPipe
    print("\nInitializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    print("✅ MediaPipe initialized")
    
    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ FAIL - Cannot open camera")
        return False
    
    print("✅ Camera opened")
    
    print("\n" + "="*60)
    print("TESTING HAND DETECTION")
    print("="*60)
    print("\nInstructions:")
    print("1. Show your hand to the camera")
    print("2. You should see green dots on your hand")
    print("3. Test will run for 10 seconds")
    print("4. Press 'q' to quit early")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    start_time = time.time()
    hands_detected_count = 0
    frames_processed = 0
    
    while (time.time() - start_time) < 10:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frames_processed += 1
        
        # Flip and convert
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = hands.process(frame_rgb)
        
        # Draw
        if results.multi_hand_landmarks:
            hands_detected_count += 1
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
            
            cv2.putText(frame, "HAND DETECTED!", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show FPS
        fps = frames_processed / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow('MediaPipe Hand Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Frames processed: {frames_processed}")
    print(f"Hands detected in: {hands_detected_count} frames")
    detection_rate = (hands_detected_count / frames_processed * 100) if frames_processed > 0 else 0
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"Average FPS: {fps:.1f}")
    
    if detection_rate > 0:
        print("\n✅ PASS - MediaPipe hand detection working!")
    else:
        print("\n⚠️  WARNING - No hands detected")
        print("   Make sure your hand is visible to the camera")
    
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_mediapipe()