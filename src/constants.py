from pynput import keyboard

EVENT_TYPE = "event"

KEY_DOWN = "key_down"
KEY_UP = "key_up"

KEY_VALUE = "key"

MOUSE_MOVE = "mouse_move"
MOUSE_SCROLL = "mouse_scroll"
MOUSE_CLICK = "mouse_click"

MOUSE_X = "mouse_x"
MOUSE_Y = "mouse_y"
MOUSE_DX = "mouse_dx"
MOUSE_DY = "mouse_dy"
MOUSE_PRESS = "pressed"

TIMESTAMP = "timestamp"

END_KEY = keyboard.Key.f9.name
PAUSE_KEY = keyboard.Key.f8.name

MAX_SESSION_SIZE = 4 * 60 * 15  # a session size should only be 15 minutes.

SCREENSHOT_SIZE = (400, 400)
FPS = 250
MAX_DATA = int(10E5)