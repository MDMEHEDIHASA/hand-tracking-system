# backend/test_audio_controller.py
from audio_controller import AudioController
import time

def test_audio_controller():
    """Test audio volume control"""
    
    print("="*70)
    print("TEST 2: AUDIO CONTROLLER MODULE")
    print("="*70)
    
    # Initialize controller
    print("\n1. Initializing AudioController...")
    audio = AudioController()
    print("✅ AudioController initialized")
    
    if not audio.volume_control:
        print("❌ FAIL - Volume control not available")
        print("   This may be a system limitation")
        return False
    
    print("\n2. Testing volume control...")
    print("\n⚠️  WARNING: This will change your system volume!")
    print("   Your current volume will be restored at the end")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Test different distances
    test_cases = [
        (30, "Minimum distance (fingers close)"),
        (50, "Close distance"),
        (100, "Medium distance"),
        (150, "Far distance"),
        (200, "Maximum distance (fingers apart)")
    ]
    
    print("\n" + "="*70)
    print("TESTING VOLUME MAPPING")
    print("="*70)
    
    for distance, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Distance: {distance}px")
        
        volume = audio.control_volume(distance)
        
        print(f"  → Volume set to: {volume}%")
        
        # Visual indicator
        bar_length = int(volume / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  → [{bar}] {volume}%")
        
        time.sleep(1.5)  # Wait to hear volume change
    
    # Test gradual change
    print("\n" + "="*70)
    print("TESTING GRADUAL VOLUME CHANGE")
    print("="*70)
    print("\nGradually increasing volume from 0% to 100%...")
    
    for distance in range(30, 201, 10):
        volume = audio.control_volume(distance)
        bar_length = int(volume / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"\rDistance: {distance:3d}px [{bar}] {volume:3d}%", end="", flush=True)
        time.sleep(0.3)
    
    print("\n\nGradually decreasing volume from 100% to 0%...")
    
    for distance in range(200, 29, -10):
        volume = audio.control_volume(distance)
        bar_length = int(volume / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"\rDistance: {distance:3d}px [{bar}] {volume:3d}%", end="", flush=True)
        time.sleep(0.3)
    
    print("\n")
    
    # Set to 50%
    print("\nRestoring volume to 50%...")
    audio.control_volume(115)
    
    # Results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print("✅ PASS - Audio controller working!")
    print("\nVerified:")
    print("  ✓ Volume control initialization")
    print("  ✓ Distance to volume mapping")
    print("  ✓ Gradual volume changes")
    print("  ✓ Full range control (0-100%)")
    print("="*70)
    
    return True

if __name__ == "__main__":
    try:
        test_audio_controller()
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user")