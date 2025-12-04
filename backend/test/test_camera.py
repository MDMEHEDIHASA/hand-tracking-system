# backend/test_camera.py
import cv2

def test_camera():
    """Test if camera is accessible"""
    
    print("="*60)
    print("TESTING CAMERA ACCESS")
    print("="*60)
    
    print("\nAttempting to open camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ FAIL - Cannot open camera")
        print("   Check if camera is connected")
        print("   Check if another app is using the camera")
        return False
    
    print("✅ PASS - Camera opened successfully")
    
    # Try to read a frame
    print("\nAttempting to read frame...")
    ret, frame = cap.read()
    
    if not ret:
        print("❌ FAIL - Cannot read frame from camera")
        cap.release()
        return False
    
    print("✅ PASS - Frame read successfully")
    print(f"   Frame shape: {frame.shape}")
    print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    # Display frame for 3 seconds
    print("\nDisplaying camera feed for 3 seconds...")
    print("   (Press 'q' to quit early)")
    
    import time
    start_time = time.time()
    
    while (time.time() - start_time) < 3:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("✅ PASS - Camera test completed")
    print("\n" + "="*60)
    print("CAMERA TEST COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_camera()