import contextlib
import time


@contextlib.contextmanager
def timer(msg=None, log_func=print):
    begin_time = time.perf_counter()
    yield
    time_elapsed = time.perf_counter() - begin_time
    log_func(f"{msg or 'timer'} | {time_elapsed:.2f} sec elapsed ")
