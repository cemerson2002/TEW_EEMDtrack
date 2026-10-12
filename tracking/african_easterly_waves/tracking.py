from ..feature_compare_tracking import FeatureComparisonTracking

import feature.identification.african_easterly_waves.configuration as cfg
from feature.identification.african_easterly_waves.processing import interpolate_wts, add_wts_to_ds
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
import numpy as np

from shapely.geometry import Polygon, LineString, MultiPoint, Point
from tqdm import tqdm
from feature.util.data_utils import pb_str_to_datetime, clip
from feature.util.graph import DataGraph
from shapely.ops import unary_union

class AEWTracking(FeatureComparisonTracking):
    """
    Tracking of AEWs.
    Based on comparisons on wavetroughs
    """

    def __init__(self, **kwargs):
        self.config = cfg
        self.set_max_delta_compare(cfg.max_cmp_delta)
        self._match_score = {}             # ((time1, id1), (time2, id2)) -> score of each match, see
                                           # correspond() and _select_best_matches()

                # ---- DEBUG counters (remove when done) ----
        self.dbg_corr = Counter()          # correspond() outcomes
        self.dbg_corr_dt = Counter()       # pairs tested per time difference [h]
        self.dbg_corr_miss_dist = []       # distance [deg] between predicted polygon and actual line on misses
        self.dbg_best_reject = {}          # (time, feature id) -> (polygon distance [deg], reason) of
                                           # its closest rejected later candidate
        self.dbg_all_nodes = set()
        self.dbg_has_succ = set()
        self.dbg_has_pred = set()
        self.dbg_adj = Counter()           # adjust_track() outcomes
        self.dbg_adj_speeds = []           # deg/h values that reached the speed check
        self.dbg_adj_track_len = []        # number of nodes per track seen
        self.dbg_adj_track_dur_h = []      # duration [h] per track seen
        self._speed_cache = {}             # (time, feature id) -> speed verdict, see
                                              # _centered_window_keep()
        pass
 
    # ------------------------------------------------------------------ DEBUG
    @staticmethod
    def _pct(vals, qs=(0, 10, 25, 50, 75, 90, 100)):
        if not vals:
            return "n/a"
        v = sorted(vals)
        return ", ".join(f"p{q}={v[min(len(v) - 1, int(q / 100 * (len(v) - 1) + 0.5))]:.3g}" for q in qs)
 
    def _dbg_note_reject(self, k1, dist, reason):
        best = self.dbg_best_reject.get(k1)
        if best is None or dist < best[0]:
            self.dbg_best_reject[k1] = (dist, reason)

    def _dbg_chain_end_reasons(self):
        """For detections with a predecessor but no successor (chain ends), classify the closest
        rejected candidate at t+6h/t+12h."""
        last_time = max(t for t, _ in self.dbg_all_nodes)
        cnt = Counter()
        for k in self.dbg_has_pred - self.dbg_has_succ:
            if last_time - k[0] <= self.max_compare_delta:
                continue
            d, reason = self.dbg_best_reject.get(k, (None, None))
            if d is None:
                cnt['none'] += 1
            elif reason == 'zz' and d == 0.0:
                cnt['zz'] += 1
            else:
                cnt['<0.5' if d < 0.5 else '0.5-1' if d < 1 else '1-2' if d < 2 else '>=2'] += 1
        return sum(cnt.values()), ", ".join(f"{c}: {cnt[c]}" for c in ['zz', '<0.5', '0.5-1', '1-2', '>=2', 'none'])

    def dbg_report_correspond(self):
        """Print + reset the correspond() counters. Called by DataGraph.generate_tracks."""
        c = self.dbg_corr
        n_all = len(self.dbg_all_nodes)
        linked = self.dbg_has_succ | self.dbg_has_pred
        both = self.dbg_has_succ & self.dbg_has_pred
        
        def pct(n):
            return f"{n} ({100 * n / max(n_all, 1):.1f}%)"
        
        print("\n[DEBUG correspond] ----------------------------------------")
        print(f"  detections seen:                       {n_all}")
        print(f"  matched to >=1 other detection:        {pct(len(linked))}")
        print(f"    with a successor (later match):      {pct(len(self.dbg_has_succ))}")
        print(f"    with a predecessor (earlier match):  {pct(len(self.dbg_has_pred))}")
        print(f"    both (mid-track):                    {pct(len(both))}")
        print(f"  isolated (no match at all):            {pct(n_all - len(linked))}")
        print(f"  pairs tested / matched / poly miss:    {c['pairs_tested']} / {c['match']} / {c['no_match']}")
        print(f"  failed zig-zag test:                   {c['zig_zag_fail']} "
              f"({c['zz_fail_poly_hit']} would pass the polygon)")
        print(f"  invalid polygons:                      {c['polygon_invalid']}")
        print(f"  pairs by dt [h]:                       {dict(sorted(self.dbg_corr_dt.items()))}")
        print(f"  miss distance [deg]:                   {self._pct(self.dbg_corr_miss_dist)}")
        n_ends, reasons = self._dbg_chain_end_reasons()
        print(f"  chain ends ({n_ends}), closest rejected candidate:")
        print(f"    {reasons}")
        print(f"    (zz = in polygon but zig-zag rejected; others = polygon miss distance [deg])")
        print("-----------------------------------------------------------\n")
#        self.dbg_corr.clear()
#        self.dbg_corr_dt.clear()
#        self.dbg_corr_miss_dist.clear()
 
    def dbg_report_adjust(self):
        """Print + reset the adjust_track() counters. Called by DataGraph.build_filter_tracks."""
        c = self.dbg_adj
        print("\n[DEBUG adjust_track] --------------------------------------")
        print(f"  tracks seen:                    {c['tracks_seen']}")
        print(f"    rejected: duration < {self.config.duration_threshold}: {c['rej_duration']}")
        print(f"    passed duration gate:         {c['tracks_seen'] - c['rej_duration']}")
        print(f"    rejected: no node kept:       {c['rej_no_node_kept']}")
        print(f"    accepted (>=1 node kept):     {c['accepted']}")
        print(f"  track length [nodes]:           {self._pct(self.dbg_adj_track_len)}")
        print(f"  track duration [h]:             {self._pct(self.dbg_adj_track_dur_h)}")
        print(f"  --- per-node window checks (only tracks that passed duration gate) ---")
        print(f"  windows examined:               {c['windows']}")
        print(f"    window too short (< {self.config.avg_speed_t}): {c['window_short']}")
        print(f"    too horizontal (mean lon/lat>1): {c['horizontal']}")
        print(f"    reached speed check:          {c['speed_checked']}")
        print(f"      speed OK (kept):            {c['speed_ok']}")
        print(f"      too slow / eastward:        {c['speed_fail']}")
        print(f"  speed [deg/h] at check:         {self._pct(self.dbg_adj_speeds)}")
        print(f"  speed threshold [deg/h]:        {self.config.avg_speed_min_deg_per_h:.4f}  (keep if speed < this)")
        print(f"  nodes kept / total:             {c['nodes_kept']} / {c['nodes_total']}")
        print("-----------------------------------------------------------\n")
        self.dbg_adj.clear()
        self.dbg_adj_speeds.clear()
        self.dbg_adj_track_len.clear()
        self.dbg_adj_track_dur_h.clear()
    # ---------------------------------------------------------------- /DEBUG
    
    @staticmethod
    def _lon_at_lats(pts, lats):
        """Longitude of polyline `pts` [(lon, lat), ...] at each latitude in `lats` (mean over all
        crossings if the line doubles back in latitude, NaN where the line does not reach)."""
        p = np.asarray(pts, dtype=float)
        x0, y0, x1, y1 = p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1]
        lat = lats[:, None]
        crosses = (np.minimum(y0, y1) <= lat) & (lat <= np.maximum(y0, y1))
        dy = np.where(y1 == y0, 1.0, y1 - y0)
        frac = np.where(y1 == y0, 0.5, (lat - y0) / dy)
        lon = np.where(crosses, x0 + frac * (x1 - x0), np.nan)
        with np.errstate(invalid='ignore'):
            return np.nanmean(lon, axis=1) if lon.shape[1] else np.full(len(lats), np.nan)
 
    def _zonal_shift_at_common_lats(self, pts1, pts2):
        """Median zonal shift [deg] from trough 1 to trough 2 over their common latitudes, or None
        if they share less than config.zigzag_min_lat_overlap_deg of latitude."""
        lat1 = [p[1] for p in pts1]
        lat2 = [p[1] for p in pts2]
        lo, hi = max(min(lat1), min(lat2)), min(max(lat1), max(lat2))
        if len(pts1) < 2 or len(pts2) < 2 or hi - lo < self.config.zigzag_min_lat_overlap_deg:
            return None
        lats = np.linspace(lo, hi, max(3, int((hi - lo) / 0.5) + 1))
        with np.errstate(invalid='ignore'):
            shift = self._lon_at_lats(pts2, lats) - self._lon_at_lats(pts1, lats)
        shift = shift[~np.isnan(shift)]
        return float(np.median(shift)) if len(shift) else None

    def correspond(self, time1, obj1, time2, obj2):
        # returns True if features are same wave

        # access the properties set earlier
        prop1 = obj1.properties
        prop2 = obj2.properties
        lineseg1 = prop1.line_pts
        lineseg2 = prop2.line_pts

        bb1_lon_center = (prop1.bb.max.lon + prop1.bb.min.lon) / 2.0
        bb2_lon_center = (prop2.bb.max.lon + prop2.bb.min.lon) / 2.0

        time_diff = time2 - time1
        #time_diff_h = time_diff.seconds // 3600
        time_diff_h = int(time_diff.total_seconds() // 3600)
        
        # DEBUG
        self.dbg_corr['pairs_tested'] += 1
        self.dbg_corr_dt[time_diff_h] += 1
        
        k1, k2 = (time1, obj1.id), (time2, obj2.id)
        self.dbg_all_nodes.update((k1, k2))

        # predicted polygonal area
        # area swept by the trough line shifted between min and max speed: union of one
        # parallelogram per segment. Joining the two shifted lines into a single ring instead
        # self-intersects (invalid polygon) whenever the trough doubles back in latitude.
        shift_min = self.config.speed_deg_per_h[0] * time_diff_h
        shift_max = self.config.speed_deg_per_h[1] * time_diff_h
        pts1 = [(p.lon, p.lat) for p in lineseg1]
        if len(pts1) == 1:
            pts1 = pts1 * 2
        predicted_polygon = unary_union([
            MultiPoint([(x0 + shift_min, y0), (x1 + shift_min, y1),
                        (x1 + shift_max, y1), (x0 + shift_max, y0)]).convex_hull
            for (x0, y0), (x1, y1) in zip(pts1, pts1[1:])])

        # The polygon above is built at fixed latitude (trough shifted only in longitude). Troughs
        # drift/tilt and change shape between steps, so the next detection can fall just outside
        # it and the chain snaps. Buffer it to add tolerance in all directions.
        # Set match_buffer_deg = 0 to restore the old behaviour.
        if self.config.match_buffer_deg > 0:
            predicted_polygon = predicted_polygon.buffer(self.config.match_buffer_deg)

        # actual line string
        p2_pt_list = []
        for i in range(len(lineseg2)):
            p2_pt_list.append((lineseg2[i].lon, lineseg2[i].lat))
        actual_linestring = LineString(p2_pt_list)

        c1 = sum(p.lon for p in lineseg1) / len(lineseg1)
        c2 = sum(p.lon for p in lineseg2) / len(lineseg2)
        dx, buf = c2 - c1, self.config.zigzag_buffer_deg
        
        dx_min = self.config.speed_deg_per_h[1] * time_diff_h - buf
        dx_max = self.config.speed_deg_per_h[0] * time_diff_h + buf
 
        # cheap gate on the centroid shift: pairs this far off are different waves
        if not (dx_min - 5.0 <= dx <= dx_max + 5.0):
            self.dbg_corr['zig_zag_fail'] += 1
            self._dbg_note_reject(k1, float('inf'), 'poly')  # DEBUG
            return False
 
        # The centroid shifts whenever the trough grows, shrinks or gains/loses a merged piece,
        # even if the wave moved normally. Measure the shift at the latitudes both troughs cover
        # instead; fall back to the centroid if they barely overlap in latitude.
        dx_lat = self._zonal_shift_at_common_lats(pts1, p2_pt_list)
        if dx_lat is not None:
            dx = dx_lat
        
        if not (dx_min <= dx <= dx_max):
          self.dbg_corr['zig_zag_fail'] += 1
          # DEBUG: would the polygon test have accepted this pair?
          if predicted_polygon.intersects(actual_linestring):
              self.dbg_corr['zz_fail_poly_hit'] += 1
              self._dbg_note_reject(k1, 0.0, 'zz')
          else:
              self._dbg_note_reject(k1, predicted_polygon.distance(actual_linestring), 'poly')
          return False

        if not predicted_polygon.is_valid:  # DEBUG
            self.dbg_corr['polygon_invalid'] += 1

        if not predicted_polygon.intersects(actual_linestring):  
        # linesting poly not where expected?
    
            # DEBUG
            self.dbg_corr['no_match'] += 1
            miss_dist = predicted_polygon.distance(actual_linestring)
            self._dbg_note_reject(k1, miss_dist, 'poly')
            if len(self.dbg_corr_miss_dist) < 20000:
                self.dbg_corr_miss_dist.append(miss_dist)

            return False
            
        # score the match for _select_best_matches(): mean distance [deg] of the trough points at
        # time2 from where the trough at time1 is expected after moving at the typical AEW speed
        expected_shift = self.config.match_expected_speed_deg_per_h * time_diff_h
        expected_pts = [(p.lon + expected_shift, p.lat) for p in lineseg1]
        expected_geom = LineString(expected_pts) if len(expected_pts) > 1 else Point(expected_pts[0])
        self._match_score[(k1, k2)] = statistics.mean(expected_geom.distance(Point(pt)) for pt in p2_pt_list)

        self.dbg_corr['match'] += 1  # DEBUG
        self.dbg_has_succ.add(k1)    # DEBUG
        self.dbg_has_pred.add(k2)    # DEBUG

        return True
    
    def adjust_track(self, track):

        # keep track if persists longer than duration_threshold
        
        # create return feature
        # reduced_tg = tracking_graph.tr_tech.pb_reference.TrackableSet()
        # reduced_tg.CopyFrom(tracking_graph.set_desc)
        # del reduced_tg.graph.edges[:]
        # del reduced_tg.tracks[:]
        
        nodes = track.nodes

        if len(nodes) == 0:
            print("Size zero track called in adjust_track?")
            exit(1)
        
        min_time = pb_str_to_datetime(nodes[0].time)
        max_time = pb_str_to_datetime(nodes[-1].time)
        duration = max_time - min_time
        
        # DEBUG
        self.dbg_adj['tracks_seen'] += 1
        self.dbg_adj_track_len.append(len(nodes))
        self.dbg_adj_track_dur_h.append(duration.total_seconds() / 3600)

        if duration < self.config.duration_threshold:
            self.dbg_adj['rej_duration'] += 1  # DEBUG
            return None

        times = [pb_str_to_datetime(node.time) for node in nodes]
        center_lons = [(node.feature.properties.bb.max.lon + node.feature.properties.bb.min.lon) / 2.0 for node in nodes]
        center_lats = [(node.feature.properties.bb.max.lat + node.feature.properties.bb.min.lat) / 2.0 for node in nodes]
        extent_lons = [(node.feature.properties.bb.max.lon - node.feature.properties.bb.min.lon) for node in nodes]
        extent_lats = [(node.feature.properties.bb.max.lat - node.feature.properties.bb.min.lat) for node in nodes]

        # init nodes as False = should be discarded
        keep_node = [False] * len(nodes)

        # iterate over all nodes: if fast enough from this to this+dt, keep
        for node_idx, node in enumerate(nodes):

            ds_nodes = track.get_nodes_after(self.config.avg_speed_t, node_idx)

            for end_node_idx, way_node_idxs in ds_nodes.items():
                self.dbg_adj['windows'] += 1  # DEBUG
                way_node_idxs = list(set(way_node_idxs)) # make them unique. might be dup if multiple paths
                end_node = nodes[end_node_idx]
                
                dt = pb_str_to_datetime(end_node.time) - pb_str_to_datetime(node.time)
                dt_h = dt.total_seconds() / 3600.0
                
                is_terminal = (end_node_idx == len(nodes) - 1) or (len(track.get_children(end_node)) == 0)

                # Skip if interval is under threshold AND hasn't reached track termination
                if dt < self.config.avg_speed_t and not is_terminal:
                    self.dbg_adj['window_short'] += 1  # DEBUG
                    continue

                if dt_h <= 0:
                    continue
                    # could not track wave that long, dont set to True
                    print('Could not track wave that long')
                    continue

                # if nodes along track are predominantly horizontal orientated discard them.
                lon_lat_ratio = [extent_lons[i] / extent_lats[i] for i in way_node_idxs]
                if statistics.mean(lon_lat_ratio) > 1:
                    self.dbg_adj['horizontal'] += 1  # DEBUG
                    continue

                # check avg speed in that 48h
                end_node_lon_center = (end_node.feature.properties.bb.max.lon + end_node.feature.properties.bb.min.lon) / 2.0

#                avg_speed_window_h = self.config.avg_speed_t.total_seconds() / 3600
#                #deg_per_hr_avg = (end_node_lon_center - center_lons[node_idx]) / uavg_speed_window_h
                deg_per_hr_avg = (end_node_lon_center - center_lons[node_idx]) / dt_h
                
                # DEBUG
                self.dbg_adj['speed_checked'] += 1
                self.dbg_adj_speeds.append(deg_per_hr_avg)
                
                # speed check. fast enough -> set to True
                if self.config.avg_speed_max_deg_per_h < deg_per_hr_avg < self.config.avg_speed_min_deg_per_h:
                    self.dbg_adj['speed_ok'] += 1  # DEBUG
                    for way_node_idx in way_node_idxs:
                        keep_node[way_node_idx] = True
                else:
                    self.dbg_adj['speed_fail'] += 1  # DEBUG

                # us_nodes = get_upstream_nodes_at_time(t)
                # nodes_in_path_ab = cut(ds_nodes, us_nodes)
                
        # DEBUG
        self.dbg_adj['nodes_total'] += len(nodes)
        self.dbg_adj['nodes_kept'] += sum(keep_node)
        if not any(keep_node):
            self.dbg_adj['rej_no_node_kept'] += 1
        else:
            self.dbg_adj['accepted'] += 1

        # then generate_tracks()
        node_id_to_keep = {node.feature.id: keep for node, keep in zip(nodes, keep_node)}
        kept_edges = [edge for edge in track.edges if node_id_to_keep.get(edge.parent.feature.id, False)]
        # add back empty parent nodes:


        # heuristic fix: if x1->n and x2->n and times t(x1) < t(x2) < t(n) in same track
        # reconnect to x1->x2, x2->n
        # reintroduces transitive edges, since x1->n still there.
        for edge1 in kept_edges:
            edge1_to_add = []
            x1 = edge1.parent
            x1t = pb_str_to_datetime(x1.time)
            for edge2 in kept_edges:
                if edge1 == edge2:
                    continue

                x2 = edge2.parent
                x2t = pb_str_to_datetime(x2.time)
                n = [ch for ch in edge1.children if ch in edge2.children] # common children

                if len(n) == 0:
                    continue
                
                # just take first
                nt = pb_str_to_datetime(n[0].time)

                if x1t < x2t < nt:
                    edge1_to_add.append(x2)

            edge1.children.extend(edge1_to_add)
        
        # return list of edges
        return kept_edges
    
