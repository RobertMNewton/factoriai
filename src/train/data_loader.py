import os
import json

import torch
from torch import Tensor
from torchvision import read_image

from typing import List, Optional, Dict, Iterable, Tuple, Union

from ..scripts import capture


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

def load_screenshot(session_id: str, timestamp: Union[int, str], dir: str = "data") -> Tensor:
    return read_image(f"{dir}/{session_id}/screenshots/{timestamp}.png")

def load_events(session_id: str, timestamp: Union[int, str], keymap: Dict[str, int], win_size: Tuple[int, int], keystroke_encoding: Optional[Tensor] = None, dir: str = "data") -> Tuple[Tensor, Optional[List[str]]]:
    if keystroke_encoding is None:
        size = max(keymap.values())
        keystroke_encoding = torch.cat(
            tensors=(
                torch.zeros((size, 1)),
                torch.ones((size, 1)),
            ),
            dim=-1
        )

    mouse_pos = (0, 0)

    events = None
    with open(f"{dir}/{session_id}/events/{timestamp}.json", "r") as f:
        events = json.load(f)
    

    repeats = {}
    for event in events:
        if event[capture.EVENT_TYPE] in [capture.KEY_DOWN, capture.KEY_UP]:
            keystroke_ix = keymap[capture.KEY_VALUE]
            keystroke_encoding[keystroke_ix] = Tensor([1, 0]) if event[capture.EVENT_TYPE] == capture.KEY_DOWN else Tensor([0, 1])

            if keymap[capture.KEY_VALUE] in repeats:
                keymap[capture.KEY_VALUE] += 1
            else:
                keymap[capture.KEY_VALUE] = 1
        else:
            


def load_data(session: List[str], keymap: Dict[str, int], dir: str = "data") -> Iterable[Tensor, Tensor]:
    for session_id in session:
        meta_data = load_metadata(session_id)

        screenshot_ts_l = sorted([int(s.strip(".png")) for s in os.listdir(f"{dir}/{session_id}/screenshots")])
        event_ts_l = iter(sorted([int(s.strip(".png")) for s in os.listdir(f"{dir}/{session_id}/screenshots")]))

        event_ts = next(event_ts_l)
        for screenshot_ts in screenshot_ts_l:
            # this is still generic. Events loading needs to be implemented
            if screenshot_ts - (meta_data["fps"] / 2) <= event_ts <= screenshot_ts + (meta_data["fps"] / 2):
                yield load_screenshot(session_id, screenshot_ts), load_events(session_id, screenshot_ts)
                screenshot_ts = next(screenshot_ts_l)
            else:
                yield load_screenshot(session_id, screenshot_ts), load_events(session_id, screenshot_ts)
