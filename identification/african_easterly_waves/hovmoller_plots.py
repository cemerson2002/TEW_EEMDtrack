from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr

from feature.util.data_utils import pb_str_to_datetime
from feature.tracking.african_easterly_waves.tracking import AEWTracking
    
def _lon_in_band(props, lat_band):
    lons = [p.lon for p in props.line_pts if lat_band[0] <= p.lat <= lat_band[1]]
    if not lons:
        return None
    pts = [(p.lon, p.lat) for p in props.line_pts]
    lo = max(lat_band[0], min(p[1] for p in pts))
    hi = min(lat_band[1], max(p[1] for p in pts))
    mean_lon = float(np.mean(lons))
    if len(pts) > 1 and hi > lo:
        interp = AEWTracking._lon_at_lats(pts, np.linspace(lo, hi, max(3, int((hi - lo) / 0.5) + 1)))
        if not np.isnan(interp).all():
            mean_lon = float(np.nanmean(interp))
    return mean_lon, min(lons), max(lons)


def _finish(ax, title, fname):
    ax.invert_yaxis()                                    # time runs downward
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.set_xlabel('longitude [deg]')
    ax.set_ylabel('time')
    ax.set_title(title)
    ax.grid(alpha=.3)
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


def plot_hovmoller_detections(ds, data_desc, lat_band=(8, 12), fname='hov_detections.png'):
    """All identified troughs (data_desc), available in AEWIdentification.postprocess().
    Dot = mean lon of trough inside lat band, bar = its lon extent."""
    fig, ax = plt.subplots(figsize=(8, 10))
    n = 0
    for s in data_desc.sets:
        for ts in s.timesteps:
            t = pb_str_to_datetime(ts.time)
            for f in ts.features:
                r = _lon_in_band(f.properties, lat_band)
                if r is None:
                    continue
                c, lo, hi = r
                ax.plot([lo, hi], [t, t], color='tab:red', lw=2, alpha=.6)
                ax.plot(c, t, 'o', ms=3, color='tab:red')
                n += 1
    _finish(ax, f'detections in {lat_band[0]}-{lat_band[1]}N (n={n})', fname)


def _node_lon(props, lat_band):
    """Mean lon of the trough inside the lat band. Falls back to the whole line (flag False)
    so out-of-band nodes of a track are still visible instead of silently dropped."""
    r = _lon_in_band(props, lat_band)
    if r is not None:
        return r[0], True
    return float(np.mean([p.lon for p in props.line_pts])), False


def _draw_track(ax, tr, lat_band, color, lw=1.5, ms=10, zorder=3):
    """Draw one DataGraph track via its edges (parent -> children).
    Filled dot = node inside lat band, hollow dot / dotted line = node outside the band."""
    centers = {}                                          # feature id -> (lon, time, in_band)
    for n in tr.nodes:
        lon, inb = _node_lon(n.feature.properties, lat_band)
        centers[n.feature.id] = (lon, pb_str_to_datetime(n.time), inb)
    for e in tr.edges:
        p = centers.get(e.parent.feature.id)
        for ch in e.children:
            c = centers.get(ch.feature.id)
            if p and c:
                ls = '-' if (p[2] and c[2]) else ':'
                ax.plot([p[0], c[0]], [p[1], c[1]], ls, color=color, lw=lw, zorder=zorder - 1)
    for inb in (True, False):
        pts = [(x, t) for x, t, b in centers.values() if b == inb]
        if pts:
            xs, ts = zip(*pts)
            ax.scatter(xs, ts, s=ms, zorder=zorder, edgecolors=color,
                       facecolors=color if inb else 'none')
    return [t for _, t, _ in centers.values()]


def plot_hovmoller_tracks(tracks, ds=None, lat_band=(10, 20), fname='hov_tracks.png',
                          cfg=None, rejected_tracks=None, xlim=(-100, 30), tlim=None):
    """Tracked waves, one color per track. Call from DataGraph.generate_tracks() (depth 0).

    ds              optional xr.Dataset with 'cv' for the grey background.
    rejected_tracks optional tracks drawn in light grey underneath (what adjust_track threw away).
    cfg             optional config; draws the assumed min/max AEW speed as dashed reference lines.
    xlim, tlim      fixed axis ranges so the plot always shows the whole domain / month.
                    tlim=(start, end) as datetimes; default: cfg.start_date/end_date, else the data.
    """
    fig, ax = plt.subplots(figsize=(8, 10))

    all_t = []
    for tr in rejected_tracks or []:
        all_t += _draw_track(ax, tr, lat_band, color='0.6', lw=0.8, ms=4, zorder=2)

    cmap = plt.get_cmap('tab20')
    for i, tr in enumerate(tracks):
        all_t += _draw_track(ax, tr, lat_band, color=cmap(i % 20))

    if tlim is None:
        if cfg is not None and getattr(cfg, 'start_date', None) and getattr(cfg, 'end_date', None):
            tlim = (np.datetime64(cfg.start_date).astype(datetime),
                    np.datetime64(cfg.end_date).astype(datetime))
        elif all_t:
            tlim = (min(all_t), max(all_t))
    ax.set_xlim(*xlim)
    if tlim is not None:
        ax.set_ylim(tlim[0], tlim[1])                     # _finish() inverts: start at top

        if cfg is not None:                               # speed reference, top right, 3 days
            lon0 = xlim[1] - 5
            t0 = tlim[0] + timedelta(hours=6)
            for v in cfg.speed_deg_per_h:
                ax.plot([lon0, lon0 + v * 72], [t0, t0 + timedelta(hours=72)], 'k--', lw=1)
                ax.axvline(x=-16, color='black', lw=2)
                ax.axvline(x=-60, color='black', lw=2)

    _finish(ax, f'tracks in {lat_band[0]}-{lat_band[1]}N (n={len(tracks)})', fname)
