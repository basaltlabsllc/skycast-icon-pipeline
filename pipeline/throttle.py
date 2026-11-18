# pipeline/throttle.py — retry a ranged GET that the server throttled (HTTP 429
# or 503), so one throttle costs seconds, not a whole forecast step. Used on the
# wind fetch path (globalfeed/build.py build_wind), where a step is ~20-60
# ranged requests and one failed step drops the whole run's wind.
#
# Only 429 and 503 are retried; every other HTTP error (404, 403, 500) and every
# network error raises at once, as before. The wait honours a numeric
# Retry-After (seconds), else backs off 2, 4, 8, 16 s; each wait is capped at
# CAP_S, at most ATTEMPTS tries in all, and never past `deadline` (the build's
# budget clock): a request still throttled then raises its error.
import urllib.error

THROTTLED = (429, 503)
ATTEMPTS = 5        # tries in all: four retries
BASE_S = 2.0
CAP_S = 30.0


def _wait_for(err, attempt):
    ra = err.headers.get("Retry-After") if getattr(err, "headers", None) is not None else None
    try:
        s = float(ra) if ra is not None else None
    except ValueError:
        s = None                              # an HTTP-date Retry-After: fall back to the backoff
    if s is None or s < 0:
        s = BASE_S * 2 ** attempt
    return min(s, CAP_S)


def retry_throttled(get, sleep, clock=None, deadline=None, attempts=ATTEMPTS):
    """get(url, rng=None) -> bytes, wrapped: an HTTP 429 / 503 is retried after a
    wait (see the header). sleep(s) is injected; with clock and deadline, a retry
    whose wait would end past the deadline is not made (the error raises)."""
    def wrapped(url, rng=None):
        for attempt in range(attempts):
            try:
                return get(url, rng)
            except urllib.error.HTTPError as e:
                if e.code not in THROTTLED or attempt == attempts - 1:
                    raise
                s = _wait_for(e, attempt)
                if clock is not None and deadline is not None and clock() + s > deadline:
                    raise
                sleep(s)
    return wrapped
