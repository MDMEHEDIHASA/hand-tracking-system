# backend/audio_controller.py
import platform
import numpy as np

class AudioController:
    def __init__(self):
        self.system = platform.system()
        self.volume_control = None
        
        if self.system == 'Windows':
            self._setup_windows_audio()
        elif self.system == 'Darwin':  # macOS
            self._setup_macos_audio()
        elif self.system == 'Linux':
            self._setup_linux_audio()
        
        self.min_distance = 30
        self.max_distance = 200
    
    def _setup_windows_audio(self):
        """Setup Windows audio control"""
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_control = cast(interface, POINTER(IAudioEndpointVolume))
            self.vol_range = self.volume_control.GetVolumeRange()
            self.min_vol = self.vol_range[0]
            self.max_vol = self.vol_range[1]
            print("✓ Windows audio control initialized")
        except Exception as e:
            print(f"⚠ Windows audio setup failed: {e}")
    
    def _setup_macos_audio(self):
        """Setup macOS audio control"""
        try:
            import subprocess
            subprocess.run(['osascript', '-e', 'get volume settings'],
                         capture_output=True, check=True)
            self.volume_control = 'osascript'
            print("✓ macOS audio control initialized")
        except Exception as e:
            print(f"⚠ macOS audio setup failed: {e}")
    
    def _setup_linux_audio(self):
        """Setup Linux audio control"""
        try:
            import subprocess
            subprocess.run(['amixer', 'get', 'Master'],
                         capture_output=True, check=True)
            self.volume_control = 'amixer'
            print("✓ Linux audio control initialized")
        except Exception as e:
            print(f"⚠ Linux audio setup failed: {e}")
    
    def control_volume(self, distance):
        """Map distance to volume and set system volume"""
        volume_pct = np.interp(
            distance,
            [self.min_distance, self.max_distance],
            [0, 100]
        )
        volume_pct = np.clip(volume_pct, 0, 100)
        
        if self.volume_control:
            try:
                if self.system == 'Windows':
                    vol = np.interp(volume_pct, [0, 100],
                                   [self.min_vol, self.max_vol])
                    self.volume_control.SetMasterVolumeLevel(vol, None)
                
                elif self.system == 'Darwin':
                    import subprocess
                    subprocess.run(['osascript', '-e',
                                  f'set volume output volume {int(volume_pct)}'],
                                 capture_output=True)
                
                elif self.system == 'Linux':
                    import subprocess
                    subprocess.run(['amixer', 'set', 'Master',
                                  f'{int(volume_pct)}%'],
                                 capture_output=True)
            except Exception as e:
                print(f"⚠ Volume control error: {e}")
        
        return int(volume_pct)