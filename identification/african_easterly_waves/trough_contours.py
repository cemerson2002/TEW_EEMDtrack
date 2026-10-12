import numpy as np
import contourpy
from scipy.ndimage import binary_dilation


def split_valid_runs(seg, flags, max_gap=4, min_pts=3):
    """Cut a contour polyline to the stretches where `flags` is True.

    seg      : (N, 2) array of (lon, lat) vertices
    flags    : (N,) bool, True where the trough conditions hold at that vertex
    max_gap  : False stretches of <= max_gap vertices *between* valid vertices are bridged
               (vertices are ~1 per grid cell, so 4 ~= 1 deg on a 0.25 deg grid)
    min_pts  : drop runs shorter than this many vertices
    """
    flags = np.asarray(flags, dtype=bool).copy()
    n = len(flags)

    # bridge short interior gaps
    i = 0
    while i < n:
        if flags[i]:
            i += 1
            continue
        j = i
        while j < n and not flags[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            flags[i:j] = True
        i = j

    # collect runs of True
    runs, i = [], 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        if j - i >= min_pts:
            runs.append(seg[i:j])
        i = j
    return runs


def extract_trough_paths(adv1, ok, lon, lat, max_gap=4, min_pts=3, counter=None):
    """Zero-contours of the FULL adv1 field, trimmed to where the trough conditions hold.

    adv1 : (lat, lon) first advection term, NOT masked by the conditions
    ok   : (lat, lon) bool, conditions (CV-percentile & adv2 > thr & u < thr) fulfilled
    lon, lat : 1-D ascending, uniformly spaced coordinates
    counter  : optional Counter to record debug statistics

    Returns a list of paths (list of (lon, lat) arrays), same format as before.
    """
    adv1 = np.squeeze(adv1)
    ok = np.squeeze(ok).astype(bool)
    if not ok.any():
        return []

    ok_d = binary_dilation(ok, iterations=1)       # vertices sit on cell edges: be tolerant by 1 cell
    dlon, dlat = lon[1] - lon[0], lat[1] - lat[0]

    gen = contourpy.contour_generator(lon, lat, np.ma.masked_invalid(adv1))
    paths = []
    for seg in gen.lines(0.0):
        iy = np.clip(np.rint((seg[:, 1] - lat[0]) / dlat).astype(int), 0, len(lat) - 1)
        ix = np.clip(np.rint((seg[:, 0] - lon[0]) / dlon).astype(int), 0, len(lon) - 1)
        runs = split_valid_runs(seg, ok_d[iy, ix], max_gap=max_gap, min_pts=min_pts)
        if counter is not None:
            counter['contour_lines'] += 1
            counter['contour_runs'] += len(runs)
        paths.extend(list(r) for r in runs)
    return paths
