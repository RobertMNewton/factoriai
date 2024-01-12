import os
import json

import torch
from torch import Tensor
from torchvision.io import read_image

from typing import List, Optional, Dict, Iterable, Tuple, Union, Set

from scripts import capture

SCROLL = 'SCROLL'
LMB = 'LMB'
RMB = 'RMB'


def get_sessions(dir: str = "data") -> List[List[str]]:
    """
    Returns a list containing sessions
    """
    session_ids = os.listdir(dir)

    # session index maps session IDs to their session sequences in res
    res: List[List[str]] = []
    session_index: Dict[str, int] = {}
    for session_id in session_ids:
        if len(session_id) != 64:
            continue

        info: Optional[Dict[str, str]] = None
        if os.path.exists(f"{dir}/{session_id}/info.json"):
            with open(f"{dir}/{session_id}/info.json", "r") as f:
                info = json.load(f)

        if info is None:
            session_index[session_id] = len(res)
            res.append([session_id])
        else:
            last_session_id, next_session_id = info.get("last_session", None), info.get("next_session", None)

            if last_session_id in session_index:
                last_session_index = session_index[last_session_id]

                j = res[last_session_index].index(last_session_id)
                res[last_session_index].insert(j + 1, session_id)

                session_index[session_id] = last_session_index

                # need to check if next_session id has been covered yet and correct it if its in a different list
                next_session_index = session_index.get(next_session_id, None)
                if next_session_index is not None:
                    res[last_session_index].extend(res[next_session_index])

                    for s in res[next_session_index]:
                        session_index[s] = last_session_id

                    res.pop(next_session_index)
            elif next_session_id in session_index:
                next_session_index = session_index[next_session_id] #  this is the index of the list that contains last session in res

                j = res[next_session_index].index(next_session_id) #  this is the index of last session in that list
                res[next_session_index].insert(j, session_id)

                session_index[session_id] = next_session_index

                last_session_index = session_index.get(last_session_id, None)
                if last_session_index is not None:
                    res[last_session_index].extend(res[next_session_index])

                    for s in res[next_session_index]:
                        session_index[s] = last_session_id

                    res.pop(next_session_index)
            else:
                session_index[session_id] = len(res)
                res.append([session_id])
    return res

def load_metadata(session_id: str, dir: str = "data") -> dict:
    res = None
    with open(f"{dir}/{session_id}/metadata.json", "r") as f:
        res = json.load(f)

    return res

def load_screenshot(session_id: str, timestamp: Union[int, str], device: torch.device, dtype: torch.dtype = torch.float64, dir: str = "data") -> Tensor:
    """
    Loads screenshot from session id and timestamp
    """
    return read_image(f"{dir}/{session_id}/screenshots/{timestamp}.png").to(dtype).to(device)

def map_delay(delay: int, delay_space: List[int]) -> int:
    """
    Maps delay to the closest available delay in delay space
    """
    delta = [abs(delay - d) for d in delay_space]
    return delay_space[delta.index(min(delta))]

def map_scroll(scroll: int, scroll_space: List[int]) -> int:
    """
    Maps scroll to the closest available scroll in scroll space
    """
    delta = [abs(scroll - d) for d in scroll_space]
    return scroll_space[delta.index(min(delta))]

def map_mouse_pos(pos: Tuple[int, int], mouse_space: [int, int], window_space: [int, int]) -> Tuple[int, int]:
    return int(pos[0] * (mouse_space[0] / window_space[0])), int(pos[1] * (mouse_space[1] / window_space[1]))

def load_events(
    session_id: str,
    timestamp: Union[int, str],
    last_timestamp: Union[int, str],
    keyset: Set[str] | List[str],
    delays: List[str],
    scroll_space: List[int],
    window_space: Tuple[int, int],
    mouse_space: Tuple[int, int],
    dir: str = "data"
    ) -> List[Tuple(Optional[str], int, Tuple[int, int])]:
    """
    Loads events from session id and timestamp and cleans them into a list of tuple actions Tuple(keys: List[(key, pressed)], delay: int, mouse_pos: Tuple[int, int])
    
    NOTE: SCROLL SPACE should be inside keyset or network will throw error
    """
    for scroll in scroll_space:
        assert f"SCROLL_{scroll}" in keyset, \
            f"did not find scroll {scroll} in action space {keyset}" 
    
    events = None
    with open(f"{dir}/{session_id}/{timestamp}.json", "r") as f:
        events = json.load(f)
    
    
    res, last_mouse_pos = [], (0, 0)
    for event in events:
        if event[capture.EVENT_TYPE] == capture.MOUSE_MOVE:
            last_mouse_pos = map_mouse_pos((event[capture.MOUSE_X], event[capture.MOUSE_Y]), mouse_space, window_space)
            res.append(
                (
                    None,
                    map_delay(event[capture.TIMESTAMP] - last_timestamp, delays),
                    last_mouse_pos,
                )
            )
        elif event[capture.EVENT_TYPE] == capture.MOUSE_SCROLL:
            last_mouse_pos = map_mouse_pos((event[capture.MOUSE_X], event[capture.MOUSE_Y]), mouse_space, window_space)
            res.append(
                (
                    f"{SCROLL}{map_scroll(event[capture.MOUSE_DY], scroll_space)}",
                    map_delay(event[capture.TIMESTAMP] - last_timestamp, delays),
                    last_mouse_pos,
                )
            )
        elif event[capture.EVENT_TYPE] == capture.MOUSE_CLICK:
            last_mouse_pos = map_mouse_pos((event[capture.MOUSE_X], event[capture.MOUSE_Y]), mouse_space, window_space)
            res.append(
                (
                    LMB if capture.KEY_VALUE not in event else RMB,
                    map_delay(event[capture.TIMESTAMP] - last_timestamp, delays),
                    last_mouse_pos,
                )
            )
        elif event[capture.EVENT_TYPE] in [capture.KEY_DOWN, capture.KEY_UP] and event[capture.KEY_VALUE] in keyset:
            res.append(
                (
                    event[capture.KEY_VALUE],
                    map_delay(event[capture.TIMESTAMP] - last_timestamp, delays),
                    last_mouse_pos,
                )
            )
    
    return res

def embed_event(event: Tuple[Optional[str], int, Tuple[int, int]], key_space: Dict[str, int], delay_space: Dict[int, int], mouse_space: Tuple[int, int], device: torch.device, dtype: torch.dtype = torch.float64) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Embeds event int three tensors (ready for training!)
    """
    key, delay, mouse_pos = event
    
    keystroke_embedding = torch.zeros((len(key_space),), device=device, dtype=dtype)
    keystroke_embedding[key_space[key]] = 1.0
    
    delay_embedding = torch.zeros((len(delay_space,)), device=device, dtype=dtype)
    delay_embedding[delay_space[delay]] = 1.0
    
    mx, my = mouse_pos
    mouse_embedding = torch.zeros(mouse_space, device=device, dtype=dtype)
    mouse_embedding[mx, my] = 1.0
    
    return key, delay, mouse_pos

def load_data(session: List[str], keys: List[str], delays: List[int], scrolls: list[int], mouse_space: Tuple[int, int], device: torch.device, dtype: torch.dtype = torch.float64, dir: str = "data") -> Iterable[Tensor, List[Tensor]]:
    """
    Loads data in tensor form
    """
    key_map = {key: i for i, key in enumerate(keys)}  # keen observers see this is a reverse key map compared to model
    keys = set(keys)
    
    delay_map = {key: i for i, key in enumerate(delays)}
    
    for session_id in session:
        meta_data = load_metadata(session_id)

        screenshot_ts_l = sorted([int(s.strip(".png")) for s in os.listdir(f"{dir}/{session_id}/screenshots")])
        event_ts_l = sorted([int(s.strip(".png")) for s in os.listdir(f"{dir}/{session_id}/screenshots")])
        
        last_ts = min(screenshot_ts_l + event_ts_l)
        
        event_ts_iter = iter(event_ts_l)
        event_ts = next(event_ts_iter)
        for screenshot_ts in screenshot_ts_l:
            if int(event_ts) - meta_data["fps"]/2 <= int(screenshot_ts) <= int(event_ts) + meta_data["fps"]/2:
                events = load_events(
                    session_id,
                    event_ts,
                    last_ts,
                    keys,
                    delays,
                    scrolls,
                    meta_data["monitor_size"],
                    mouse_space,
                    dir=dir
                    )
                
                yield load_screenshot(session_id, event_ts, device=device, dtype=dtype, dir=dir), [embed_event(event, key_map, delay_map, mouse_space, device=device, dtype=dtype) for event in events]
                last_ts, event_ts = screenshot_ts, next(event_ts_iter)
            else:
                yield load_screenshot(session_id, event_ts, dir=dir), []
