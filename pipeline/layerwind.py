# pipeline/layerwind.py — each cloud deck's wind over the model's own cloud
# layer, the rule the app's ICON texture wind uses (src/cloudclues.js deckOf),
# here on pressure levels.
#
# Decks (hPa; deck i holds hi > p >= lo, and the low deck also p == ps):
#   ncep  (GFS, HRRR, RRFS)  low ps..642      mid 642..350      high 350..150
#   icon                     low ps..800      mid 800..400      high 400..150
#   ecmwf (IFS, AIFS)        low ps..0.8 ps   mid 0.8..0.45 ps  high 0.45 ps..150
# 150 hPa tops every high deck. A level below the ground (p > ps) never counts.
#
# A level's weight is the pressure span it stands for inside its deck: from the
# midpoint to the next level below (or the deck's bottom, min(hi, ps)) to the
# midpoint to the next level above (or the deck's top, lo). The deck's wind is
# the weighted vector mean over its thickest unbroken run of cloudy levels
# (thickness = summed weights; ties go to the lower run), else over every
# level in the deck; NaN where neither has a level with a wind.
import numpy as np

DECK_KEYS = (850, 500, 250)          # ICW1 keys: low, mid, high deck
CONDENSATE_CLOUDY = 1e-6             # kg/kg cloud water + ice (HRRR, RRFS)
TCDC_CLOUDY = 5.0                    # % (GFS)
MOIST_RH = 90.0                      # % effective RH (ICON, IFS, AIFS)
DECK_TOP = 150.0                     # hPa

_NCEP = ((642.0, None), (350.0, 642.0), (DECK_TOP, 350.0))
_ICON = ((800.0, None), (400.0, 800.0), (DECK_TOP, 400.0))


def deck_bounds(kind, ps):
    """[(lo, hi)] per deck (low, mid, high), each an array of ps's shape (hPa)."""
    ps = np.asarray(ps, np.float64)
    if kind in ("ncep", "icon"):
        table = _NCEP if kind == "ncep" else _ICON
        return [(np.full(ps.shape, lo), ps.copy() if hi is None else np.full(ps.shape, hi)) for lo, hi in table]
    if kind == "ecmwf":
        a = np.maximum(0.8 * ps, DECK_TOP)
        b = np.maximum(0.45 * ps, DECK_TOP)
        return [(a, ps.copy()), (b, a.copy()), (np.full(ps.shape, DECK_TOP), b.copy())]
    raise ValueError(f"unknown deck kind {kind!r}")


def _esw(t):
    return 611.2 * np.exp(17.62 * (t - 273.15) / (t - 30.03))


def _esi(t):
    return np.exp(9.550426 - 5723.265 / t + 3.53068 * np.log(t) - 0.00728332 * t)


def _effective_rh(rh_water, t):
    """RH in %, over ice below 273.15 K (ICON's rhWaterToIce)."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return np.where(t < 273.15, rh_water * _esw(t) / _esi(t), rh_water)


def moist_from_rh(rh_water_pct, t_k):
    rh = np.asarray(rh_water_pct, np.float64)
    t = np.asarray(t_k, np.float64)
    with np.errstate(invalid="ignore"):
        # missing T or RH is never moist (the app skips such points)
        return np.isfinite(rh) & np.isfinite(t) & (_effective_rh(rh, t) >= MOIST_RH)


def moist_from_q(q, t_k, p_hpa):
    """Specific humidity (kg/kg), T (K), p (hPa) -> RH >= 90 % (over ice below 0 C)."""
    q = np.asarray(q, np.float64)
    t = np.asarray(t_k, np.float64)
    p = np.asarray(p_hpa, np.float64) * 100.0
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        e = q * p / (0.622 + 0.378 * q)
        rh_water = 100.0 * e / _esw(t)
        return np.isfinite(rh_water) & np.isfinite(t) & (_effective_rh(rh_water, t) >= MOIST_RH)


def deck_winds(levels_hpa, u, v, cloudy, ps, bounds):
    """{(deck_key, 'u'|'v'): float32 [npts]} — see the header for the rule."""
    levels = np.asarray(levels_hpa, np.float64)
    order = np.argsort(-levels, kind="stable")          # ground up
    p = levels[order]
    ps = np.asarray(ps, np.float64)
    npts = ps.shape[0]
    out = {}
    for d, (key, (lo, hi)) in enumerate(zip(DECK_KEYS, bounds)):
        lo = np.broadcast_to(np.asarray(lo, np.float64), ps.shape)
        hi = np.broadcast_to(np.asarray(hi, np.float64), ps.shape)
        bottom = np.minimum(hi, ps)

        # bounds[0] must be the low deck: only it includes a level exactly at its hi (= ps).
        def member(k):
            if k < 0 or k >= p.size:
                return np.zeros(npts, bool)
            inside = (p[k] < hi) | (d == 0) & (p[k] <= hi)
            return inside & (p[k] >= lo) & (p[k] <= ps)

        all_u = np.zeros(npts); all_v = np.zeros(npts); all_w = np.zeros(npts)
        cur_on = np.zeros(npts, bool)
        cur_t = np.zeros(npts); cur_u = np.zeros(npts); cur_v = np.zeros(npts); cur_w = np.zeros(npts)
        best_t = np.full(npts, -1.0)
        best_u = np.zeros(npts); best_v = np.zeros(npts); best_w = np.zeros(npts)

        def close_runs(ending):
            take = ending & cur_on & (cur_t > best_t)
            best_t[take] = cur_t[take]; best_u[take] = cur_u[take]
            best_v[take] = cur_v[take]; best_w[take] = cur_w[take]
            for a in (cur_t, cur_u, cur_v, cur_w):
                a[ending] = 0.0
            cur_on[ending] = False

        m_prev, m_cur = member(-1), member(0)
        for k in range(p.size):
            m_next = member(k + 1)
            below = np.where(m_prev, 0.5 * (p[k - 1] + p[k]) if k > 0 else 0.0, bottom)
            above = np.where(m_next, 0.5 * (p[k] + p[k + 1]) if k + 1 < p.size else 0.0, lo)
            w = np.where(m_cur, below - above, 0.0)
            uk = np.asarray(u[order[k]], np.float64)
            vk = np.asarray(v[order[k]], np.float64)
            ok = m_cur & np.isfinite(uk) & np.isfinite(vk)
            wu = np.where(ok, w * np.where(ok, uk, 0.0), 0.0)
            wv = np.where(ok, w * np.where(ok, vk, 0.0), 0.0)
            ww = np.where(ok, w, 0.0)
            all_u += wu; all_v += wv; all_w += ww
            c = m_cur & np.asarray(cloudy[order[k]], bool)
            close_runs(~c)
            cur_on |= c
            cur_t += np.where(c, w, 0.0)
            cur_u += np.where(c, wu, 0.0); cur_v += np.where(c, wv, 0.0); cur_w += np.where(c, ww, 0.0)
            m_prev, m_cur = m_cur, m_next
        close_runs(np.ones(npts, bool))

        # A single cloudy pressure level is a valid run (unlike deckOf's fine grid,
        # where seg.length < 2 falls back to the band): pressure levels are coarse.
        use_run = best_w > 0
        su = np.where(use_run, best_u, all_u)
        sv = np.where(use_run, best_v, all_v)
        sw = np.where(use_run, best_w, all_w)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[(key, "u")] = np.where(sw > 0, su / np.where(sw > 0, sw, 1.0), np.nan).astype(np.float32)
            out[(key, "v")] = np.where(sw > 0, sv / np.where(sw > 0, sw, 1.0), np.nan).astype(np.float32)
    return out
