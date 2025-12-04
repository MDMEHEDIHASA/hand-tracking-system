# backend/test_backend.py
import requests
import time

API_URL = "http://localhost:5000"

def test_endpoints():
    """Test all backend endpoints"""
    
    print("="*60)
    print("TESTING HAND TRACKING BACKEND")
    print("="*60)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{API_URL}/api/health")
        if response.status_code == 200:
            print("✅ PASS - Health check successful")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ FAIL - Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ FAIL - Error: {e}")
    
    # Test 2: Get Config
    print("\n2. Testing Get Config...")
    try:
        response = requests.get(f"{API_URL}/api/config")
        if response.status_code == 200:
            data = response.json()
            print("✅ PASS - Config retrieved")
            print(f"   Modes: {data['modes']}")
            print(f"   Current mode: {data['current_mode']}")
            print(f"   Is tracking: {data['is_tracking']}")
        else:
            print(f"❌ FAIL - Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ FAIL - Error: {e}")
    
    # Test 3: Get Model Info
    print("\n3. Testing Get Model Info...")
    try:
        response = requests.get(f"{API_URL}/api/model/info")
        if response.status_code == 200:
            data = response.json()
            print("✅ PASS - Model info retrieved")
            print(f"   Type: {data['type']}")
            print(f"   Gestures: {data['num_gestures']}")
        else:
            print(f"❌ FAIL - Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ FAIL - Error: {e}")
    
    # Test 4: Start Tracking
    print("\n4. Testing Start Tracking...")
    try:
        response = requests.post(f"{API_URL}/api/start")
        if response.status_code == 200:
            print("✅ PASS - Tracking started")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ FAIL - Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ FAIL - Error: {e}")
    
    # Wait a bit
    print("\n   Waiting 2 seconds...")
    time.sleep(2)
    
    # Test 5: Check if tracking is active
    print("\n5. Testing if tracking is active...")
    try:
        response = requests.get(f"{API_URL}/api/config")
        if response.status_code == 200:
            data = response.json()
            if data['is_tracking']:
                print("✅ PASS - Tracking is active")
            else:
                print("⚠️  WARNING - Tracking should be active but isn't")
        else:
            print(f"❌ FAIL - Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ FAIL - Error: {e}")
    
    # Test 6: Change Mode
    print("\n6. Testing Change Mode...")
    try:
        response = requests.post(f"{API_URL}/api/mode", 
                                json={"mode": "gesture"})
        if response.status_code == 200:
            data = response.json()
            print("✅ PASS - Mode changed")
            print(f"   New mode: {data['mode']}")
        else:
            print(f"❌ FAIL - Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ FAIL - Error: {e}")
    
    # Test 7: Start Recording
    print("\n7. Testing Start Recording...")
    try:
        response = requests.post(f"{API_URL}/api/recording/start")
        if response.status_code == 200:
            print("✅ PASS - Recording started")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ FAIL - Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ FAIL - Error: {e}")
    
    time.sleep(1)
    
    # Test 8: Stop Recording
    print("\n8. Testing Stop Recording...")
    try:
        response = requests.post(f"{API_URL}/api/recording/stop")
        if response.status_code == 200:
            data = response.json()
            print("✅ PASS - Recording stopped")
            print(f"   Frames recorded: {data['frames']}")
        else:
            print(f"❌ FAIL - Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ FAIL - Error: {e}")
    
    # Test 9: Stop Tracking
    print("\n9. Testing Stop Tracking...")
    try:
        response = requests.post(f"{API_URL}/api/stop")
        if response.status_code == 200:
            print("✅ PASS - Tracking stopped")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ FAIL - Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ FAIL - Error: {e}")
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)

if __name__ == "__main__":
    print("\n⚠️  Make sure backend is running on http://localhost:5000")
    print("Press Enter to start tests...")
    input()
    
    test_endpoints()