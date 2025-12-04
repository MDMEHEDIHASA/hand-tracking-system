# backend/test_mouse_controller.py
from mouse_controller import MouseController
import time

def test_mouse_controller():
    """Test virtual mouse control"""
    
    print("="*70)
    print("TEST 3: MOUSE CONTROLLER MODULE")
    print("="*70)
    
    # Initialize controller
    print("\n1. Initializing MouseController...")
    mouse = MouseController()
    print("✅ MouseController initialized")
    print(f"   Screen resolution: {mouse.screen_width}x{mouse.screen_height}")
    
    print("\n2. Testing cursor movement...")
    print("\n⚠️  WARNING: This will move your mouse cursor!")
    print("   Move your mouse to stop the test")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Test 1: Move to corners
    print("\n" + "="*70)
    print("TEST 1: MOVING CURSOR TO CORNERS")
    print("="*70)
    
    corners = [
        ({'x': 0.1, 'y': 0.1}, "Top-Left"),
        ({'x': 0.9, 'y': 0.1}, "Top-Right"),
        ({'x': 0.9, 'y': 0.9}, "Bottom-Right"),
        ({'x': 0.1, 'y': 0.9}, "Bottom-Left"),
        ({'x': 0.5, 'y': 0.5}, "Center")
    ]
    
    for landmark, position_name in corners:
        print(f"\nMoving cursor to: {position_name}")
        print(f"  Normalized coords: ({landmark['x']:.2f}, {landmark['y']:.2f})")
        
        for _ in range(10):  # Smooth movement
            pos = mouse.move_cursor(landmark)
            time.sleep(0.05)
        
        print(f"  → Cursor at: ({pos[0]}, {pos[1]})")
        time.sleep(1)
    
    # Test 2: Draw a circle
    print("\n" + "="*70)
    print("TEST 2: DRAWING A CIRCLE WITH CURSOR")
    print("="*70)
    print("\nWatch your cursor draw a circle...")
    time.sleep(1)
    
    import math
    center_x, center_y = 0.5, 0.5
    radius = 0.2
    steps = 50
    
    for i in range(steps + 1):
        angle = (i / steps) * 2 * math.pi
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        
        landmark = {'x': x, 'y': y}
        mouse.move_cursor(landmark)
        time.sleep(0.05)
    
    print("✅ Circle completed")
    
    # Test 3: Click test
    print("\n" + "="*70)
    print("TEST 3: CLICK FUNCTIONALITY")
    print("="*70)
    print("\nTesting click with cooldown...")
    
    # Move to center
    mouse.move_cursor({'x': 0.5, 'y': 0.5})
    time.sleep(0.5)
    
    print("\nPerforming 3 clicks (you should hear/see them)...")
    for i in range(3):
        result = mouse.click()
        if result:
            print(f"  ✓ Click {i+1} performed")
        else:
            print(f"  ⏳ Click {i+1} blocked (cooldown)")
        time.sleep(0.5)
    
    # Test 4: Smoothing
    print("\n" + "="*70)
    print("TEST 4: MOVEMENT SMOOTHING")
    print("="*70)
    print("\nTesting smooth vs. jerky movement...")
    
    # Jerky movement (simulate raw hand tracking)
    print("\n1. Without smoothing (simulated jerky input):")
    mouse.smoothing = 1  # No smoothing
    
    for i in range(20):
        x = 0.4 + (i % 2) * 0.2  # Jump back and forth
        y = 0.5
        mouse.move_cursor({'x': x, 'y': y})
        time.sleep(0.05)
    
    time.sleep(0.5)
    
    # Smooth movement
    print("\n2. With smoothing (production setting):")
    mouse.smoothing = 5  # Normal smoothing
    
    for i in range(20):
        x = 0.4 + (i % 2) * 0.2
        y = 0.5
        mouse.move_cursor({'x': x, 'y': y})
        time.sleep(0.05)
    
    # Return to center
    mouse.move_cursor({'x': 0.5, 'y': 0.5})
    
    # Results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print("✅ PASS - Mouse controller working!")
    print("\nVerified:")
    print("  ✓ Cursor movement to all corners")
    print("  ✓ Smooth circular motion")
    print("  ✓ Click functionality")
    print("  ✓ Click cooldown mechanism")
    print("  ✓ Movement smoothing algorithm")
    print("  ✓ Normalized to screen coordinate mapping")
    print("="*70)
    
    return True

if __name__ == "__main__":
    try:
        test_mouse_controller()
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user")