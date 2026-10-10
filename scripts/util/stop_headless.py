"""Send Ctrl+C to a detached headless training run (Windows).

scripts/train.py turns a KeyboardInterrupt into an orderly stop: with
backup_before_save on, the trainer backs up at the current step and saves the
final model before exiting. A detached run owns a hidden console, so the signal
has to be delivered through that console rather than taskkill (which would just
terminate the process and lose everything since the last backup).

Usage: python scripts/util/stop_headless.py <pid>
"""

import ctypes
import sys
import time

CTRL_C_EVENT = 0


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print(__doc__)
        return 2
    pid = int(sys.argv[1])
    kernel32 = ctypes.windll.kernel32

    # Detach from our own console, attach to the target's, and ignore the
    # Ctrl+C we are about to raise so only the target reacts to it. Once
    # detached, stdout may be gone, so failures are reported by exit code.
    kernel32.FreeConsole()
    if not kernel32.AttachConsole(pid):
        return 1
    kernel32.SetConsoleCtrlHandler(None, True)
    if not kernel32.GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0):
        return 1
    # Give the event time to reach the target before we detach again.
    time.sleep(1.0)
    kernel32.FreeConsole()
    return 0


if __name__ == "__main__":
    sys.exit(main())
