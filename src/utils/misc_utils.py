from time import time


def stopwatch():
    started_at = time()
    return lambda: time() - started_at


def dt_str(dt: float) -> str:
    """
    :param dt: time in seconds
    """
    if dt >= 0.1:
        return f"{dt:.1f}s"
    ms = dt * 1000
    if ms >= 0.1:
        return f"{ms:.1f}"
    ns = ms * 1000000
    return f"{int(ns)}"


def stopwatch_str():
    stopwatch_ = stopwatch()
    return lambda: dt_str(stopwatch_())
