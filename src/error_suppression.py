import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_stderr():
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)
