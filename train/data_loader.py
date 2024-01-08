import os
import json
from typing import List, Optional, Dict


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


