from threading import Timer
from datetime import datetime
from pynput import keyboard, mouse
from pynput.mouse import Button
from mss import mss
from typing import List, Tuple, Dict, Optional


def _milli() -> int:
    return int(datetime.time() * 1000)

def _map_mouse_to_window(position: Tuple[int, int], mouse_space: Tuple[int, int], wind_space: Tuple[int, int]) -> Tuple[int, int]:
    x, y = position
    mw, mh = mouse_space
    ww, wh = wind_space
    
    return float(x * ww/mw), float(y * wh/mh)


class Controller:
    def __init__(self, keys: List[str], mouse_space: Tuple[int, int], window_space: Optional[Tuple[int, int]] = None) -> None:
        self.active_keys: Dict[str, bool] = {key: False for key in keys}
        
        self.mouse_space = mouse_space
        
        if window_space is None:
            monitor = mss().monitors[1]
            self.window_space = (monitor['left'], monitor['height'])
        else:
            self.window_space = window_space
        
        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()
        
    def trigger_event(self, key: str, mouse_pos: Tuple[int, int]) -> None:
        """
        Updates active keys and triggers key press/release
        """
        self.mouse_controller.position = mouse_pos
        # check if key is a special char, i.e. scroll or mouse button press
        if key is None:
            return
        elif key[:6] == "SCROLL":
            _, dy = 0, int(key[6:])
            self.mouse_controller.scroll(_, dy)
        elif key == "RMB":
            self.mouse_controller.click(Button.right)
        elif key == "LMB":
            self.mouse_controller.click(Button.left)
        else:
            if self.active_keys[key]:
                self.keyboard_controller.release(key)
            else:
                self.keyboard_controller.press(key)
            
            self.active_keys[key] = not self.active_keys[key]
        
    def schedule_events(self, events: List[Tuple[str, int, Tuple[int, int]]]) -> None:
        """
        Schedules events
        """
        for event in events:
            key, delay, pos = event
            pos = _map_mouse_to_window(pos, self.mouse_space, self.window_space)
            
            Timer(delay/1000, self.trigger_event, [key, pos], {}).start()  # might change this later to have a dedicated thread pool instead of thread spawning like this!
        
    def release(self) -> None:
        """
        Releases all currently active keys. This is useful to 'reset' the bot.
        """
        for key, active in self.active_keys.items():
            if active:
                self.keyboard_controller.release(key)
        
        
    