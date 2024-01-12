import mss
import os
import uuid
import json
import simpleaudio as sa

import threading
from threading import Lock

from time import sleep

from PIL import Image

from functools import partial
from datetime import datetime
from hashlib import sha256

import pynput
from pynput import keyboard, mouse
from pynput.keyboard import Key

from typing import List, Optional, Dict, Any, Tuple

# constants

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

# gloabl

running = True
paused = False

play_sound = sa.WaveObject.from_wave_file("assets/audio/play-pause.wav")

# utils

def milli() -> int:
    """
    Gets posix time in milliseconds
    """
    return int(datetime.now().timestamp() * 1000)

def get_session_name() -> str:
    """
    Generates a session ID that is unique to both this machine and the current time
    """
    return sha256(
        string=(hex(milli())+hex(uuid.getnode())).encode(),
        usedforsecurity=False,
    ).hexdigest()

def can_json_encode(obj: Any) -> bool:
    try:
        json.dumps(obj)
    except:
        return False
    return True

def generate_session_directories(session: str) -> None:
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists(f"data/{session}"):
        os.mkdir(f"data/{session}")
    if not os.path.exists(f"data/{session}/screenshots"):
        os.mkdir(f"data/{session}/screenshots")
    if not os.path.exists(f"data/{session}/events"):
        os.mkdir(f"data/{session}/events")

def log_metadata(session: str) -> None:
    with open(f"data/{session}/metadata.json", "w") as f:
        metadata = {
            "fps": FPS,
            "monitor size": mss.mss().monitors[1],
        }

        json.dump(
            obj=metadata,
            fp=f,
        )

def update_sessions(old_session: str) -> str:
    """
    Creates new session with directory and updates info.json in old and new sessions
    Returns new session id
    """
    new_session = get_session_name()
    generate_session_directories(new_session)
    
    info = {}
    if os.path.exists(f"data/{old_session}/info.json"):
        with open(f"data/{old_session}/info.json", "r") as f:
            info = json.load(f)
    else:
        info["last_session"] = None
        
    info["next_session"] = new_session

    with open(f"data/{old_session}/info.json", "w") as f:
        json.dump(info, f)
        
    with open(f"data/{new_session}/info.json", "w") as f:
            info = {
                "last_session": old_session,
                "next_session": None,
            }

            json.dump(info, f)

    log_metadata(new_session)

    print(f"session updated at {datetime.now()}. {old_session} -> {new_session}")
    play_sound.play()
    
    return new_session


# pause, end funcs

def end() -> None:
    global running
    running = False

    print(f"session ended at {datetime.now()}")
    play_sound.play()

def pause_play() -> None:
    global paused
    paused = not paused

    print(f"session {'paused' if paused else 'played'} at {datetime.now()}")
    play_sound.play()

# screen capture code - mss handles the brunt force of this

def take_screenshot(filename: str, monitor: int = 1, size: Tuple[int, int] = None) -> None:
    image = None
    with mss.mss() as sct:
        image = sct.grab(sct.monitors[monitor])
    
    image = Image.frombytes("RGB", image.size, image.bgra, "raw", "BGRX")
    if size is not None:
        image = image.resize(size)
    
    image.save(filename, format="png")

# global event capture code. pynput already provides a non-blocking API, so we use rather than asyncio. Might be prettier with asyncio but alas! Don't code a solution to a problem you don't have (YET!)
        
def create_key_log(type: str, key: int) -> Dict[str, str]:
    """
    Creates a dictionary representing a keyboard event
    """
    return {
        EVENT_TYPE: type,
        KEY_VALUE: key,
        TIMESTAMP: milli(),
    }
        
def log_key_press(key: Key, data: List[Dict[str, str]], mutex: Lock) -> bool:
    """
    Handler for keyboard presses
    """
    if isinstance(key, keyboard.KeyCode):
        key = key.char
    if isinstance(key, keyboard.Key):
        key = key.name

    # returning false cancels the listener
    if key == END_KEY:
        end()
    elif key == PAUSE_KEY:
        pause_play()

        return True

    with mutex:
        data.append(create_key_log(
            type=KEY_DOWN,
            key=key,
        ))
    return True

def log_key_release(key: Key, data: List[Dict[str, Any]], mutex: Lock) -> bool:
    """
    Handler for keyboard releases
    """
    if isinstance(key, keyboard.KeyCode):
        key = key.char
    if isinstance(key, keyboard.Key):
        key = key.name

    with mutex:
        data.append(create_key_log(
            type=KEY_UP,
            key=key,
        ))
    return True

def start_key_listener(data: List[Dict[str, str]], mutex: Lock) -> keyboard.Listener:
    listener = keyboard.Listener(
        on_press=partial(log_key_press, data=data, mutex=mutex),
        on_release=partial(log_key_release, data=data, mutex=mutex)
    )

    listener.start()

    return listener

def create_mouse_log(type: str, x: float, y: float, dx: float = None, dy: float = None, button: int = None, pressed: bool = None) -> Dict[str, Any]:
    """
    Creates a dictionary representing a mouse event
    """
    res = {
        EVENT_TYPE: type,
        MOUSE_X: x,
        MOUSE_Y: y,
        TIMESTAMP: milli(),
    }

    if dx is not None:
        res[MOUSE_DX] = dx
    if dy is not None:
        res[MOUSE_DY] = dy
    if button is not None:
        res[KEY_VALUE] = button
    if pressed is not None:
        res[MOUSE_PRESS] = pressed

    return res

def log_mouse_move(x: int, y: int, data: List[Dict[str, Any]], mutex: Lock) -> bool:
    """
    Handler for mouse move events
    """
    with mutex:
        data.append(
            create_mouse_log(
                type=MOUSE_MOVE,
                x=x,
                y=y,
            )
        )

    return True

def log_mouse_click(x: int, y: int, button: mouse.Button, pressed: bool, data: List[Dict[str, Any]], mutex: Lock) -> bool:
    """
    Handler for mouse click events
    """
    with mutex:
        data.append(
            create_mouse_log(
                type=MOUSE_CLICK,
                x=x,
                y=y,
                button=button.value,
                pressed=pressed,
            )
        )

    return True

def log_mouse_scroll(x: int, y: int, dx: int, dy: int, data: List[Dict[str, Any]], mutex: Lock) -> bool:
    """
    Handler for mouse click events
    """
    with mutex:
        data.append(
            create_mouse_log(
                type=MOUSE_SCROLL,
                x=x,
                y=y,
                dx=dx,
                dy=dy
            )
        )

    return True

def start_mouse_listener(data: List[Dict[str, str]], mutex: Lock) -> mouse.Listener:
    listener = mouse.Listener(
        on_move=partial(log_mouse_move, data=data, mutex=mutex),
        on_click=partial(log_mouse_click, data=data, mutex=mutex),
        on_scroll=partial(log_mouse_scroll, data=data, mutex=mutex)
    )

    listener.start()

    return listener


if __name__ == "__main__":
    session, session_counter = get_session_name(), 0
    generate_session_directories(session)
    log_metadata(session)
    
    data, last_entry, mutex = [], 0, Lock()
    last_capture = milli()

    sleep(5)

    print(f"starting new capture session: {session}")
    play_sound.play()
    
    keyboard_listener, mouse_listener = start_key_listener(data, mutex), start_mouse_listener(data, mutex)

    while running:
        new_capture = milli()

        if paused:
            last_capture = new_capture
            data.clear()
            last_entry = 0

        if new_capture - last_capture >= FPS:
            new_entry = None
            with mutex:
                new_entry = len(data)
                if new_entry >= MAX_DATA:
                    important_data = data[last_entry:new_entry]

                    data.clear()
                    data.extend(important_data)

                    last_entry, new_entry = 0, len(data)
            
            if new_entry - last_entry > 0:
                with open(f"data/{session}/events/{new_capture}.json", "w") as f:
                    json.dump(
                        obj=data[last_entry:new_entry],
                        fp=f,
                        default=(lambda _: None),
                    )

                last_entry = new_entry

            take_screenshot(filename=f"data/{session}/screenshots/{new_capture}.png", size=SCREENSHOT_SIZE)

            last_capture = new_capture
            session_counter += 1

            if session_counter >= MAX_SESSION_SIZE:
                session, session_counter = update_sessions(session), 0

    mouse_listener.stop()
    keyboard_listener.stop()

    sleep(2)

        
