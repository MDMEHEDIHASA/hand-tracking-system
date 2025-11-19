# backend/mouse_controller.py
import pyautogui
import time

class MouseController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoothing = 5
        self.prev_x = 0
        self.prev_y = 0
        self.click_cooldown = 0.3
        self.last_click_time = 0
        
        pyautogui.FAILSAFE = False
        
        print(f"✓ Mouse control initialized (Screen: {self.screen_width}x{self.screen_height})")
    
    def move_cursor(self, landmark):
        """Move cursor based on hand landmark position"""
        target_x = int(landmark['x'] * self.screen_width)
        target_y = int(landmark['y'] * self.screen_height)
        
        smooth_x = int(self.prev_x + (target_x - self.prev_x) / self.smoothing)
        smooth_y = int(self.prev_y + (target_y - self.prev_y) / self.smoothing)
        
        pyautogui.moveTo(smooth_x, smooth_y, duration=0)
        
        self.prev_x = smooth_x
        self.prev_y = smooth_y
        
        return (smooth_x, smooth_y)
    
    def click(self):
        """Perform mouse click with cooldown"""
        current_time = time.time()
        if current_time - self.last_click_time > self.click_cooldown:
            pyautogui.click()
            self.last_click_time = current_time
            return True
        return False