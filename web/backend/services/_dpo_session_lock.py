"""Serialize DPO session admission across tools that share dataset manifests."""

import threading
from functools import wraps

_admission_lock = threading.RLock()


def dpo_session_admission(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with _admission_lock:
            return function(*args, **kwargs)

    return guarded
