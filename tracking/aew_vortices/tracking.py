from ..feature_compare_tracking import FeatureComparisonTracking

import feature.identification.aew_vortices.configuration as cfg
from datetime import datetime, timedelta
from collections import defaultdict
import statistics, math

from shapely.geometry import Polygon, LineString
from feature.util.data_utils import pb_str_to_datetime, clip
from feature.util.graph import DataGraph

zero_dt = timedelta(seconds=0)

class AEWVortexTracking(FeatureComparisonTracking):
    """
    Tracking of AEWs.
    Based on comparisons on wavetroughs
    """

    def __init__(self, **kwargs):
        self.config = cfg
        pass

    def correspond(self, time1, obj1, time2, obj2):
        # returns True if features are same wave
        time_diff = time2 - time1
        time_diff_h = time_diff.seconds // 3600

        # access the properties set earlier
        prop1 = obj1.properties
        prop2 = obj2.properties
        
        lat1 = prop1.pos.lat
        lat2 = prop2.pos.lat
        lon1 = prop1.pos.lon
        lon2 = prop2.pos.lon
        o1 = prop1.is_ccw
        o2 = prop2.is_ccw
        
        if o1 != o2:
            return False # dont track ccw that suddently become cw
        
        dlat = abs(lat1 - lat2)
        dlon = abs(lon1 - lon2)
        
        # dist_deg = math.sqrt(dlat * dlat + dlon * dlon)
        max_dist_deg_lon = time_diff_h * self.config.max_dist_deg_per_hr_lon
        max_dist_deg_lat = time_diff_h * self.config.max_dist_deg_per_hr_lat
        
        if dlon > max_dist_deg_lon:
            return False
        
        if dlat > max_dist_deg_lat:
            return False
        
        return True

    # get dict of key=endnode, values=nodes on way) for endnodes at end if time_delta
    @staticmethod
    def get_nodes_after(time_delta, edges, start_index):

        this_node = edges[start_index].parent
        if time_delta <= zero_dt or len(edges[start_index].children) == 0:
            dd = defaultdict(list)
            dd[start_index] = [start_index]
            return dd

        obj_node_list = [e.parent for e in edges]
        children_ = defaultdict(list)

        for child in edges[start_index].children:

            dt = pb_str_to_datetime(child.time) - pb_str_to_datetime(this_node.time)
            remaining_time_delta = time_delta - dt

            # get index of connected node in graph
            c_node_idx = obj_node_list.index(child)
            after_nodes = AEWVortexTracking.get_nodes_after(remaining_time_delta, edges, c_node_idx)
            # add own node to it (this_node)
            for endn_, pathn_ in after_nodes.items():
                children_[endn_].extend(pathn_ + [start_index])

        return children_

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

        if duration < self.config.duration_threshold:
            return None
        
        return track
        """
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
                way_node_idxs = list(set(way_node_idxs)) # make them unique. might be dup if multiple paths
                end_node = nodes[end_node_idx]

                # if dt < 48h -> didnt reach 2 days.
                if pb_str_to_datetime(end_node.time) - pb_str_to_datetime(node.time) < self.config.avg_speed_t:
                    # could not track wave that long, dont set to True
                    continue

                # if nodes along track are predominantly horizontal orientated discard them.
                lon_lat_ratio = [extent_lons[i] / extent_lats[i] for i in way_node_idxs]
                if statistics.mean(lon_lat_ratio) > 1:
                    continue

                # check avg speed in that 48h
                end_node_lon_center = (end_node.feature.properties.bb.max.lon + end_node.feature.properties.bb.min.lon) / 2.0

                avg_speed_window_h = self.config.avg_speed_t.total_seconds() / 3600
                deg_per_hr_avg = (end_node_lon_center - center_lons[node_idx]) / avg_speed_window_h
                # speed check. fast enough -> set to True
                if deg_per_hr_avg < self.config.avg_speed_min_deg_per_h:
                    for way_node_idx in way_node_idxs:
                        keep_node[way_node_idx] = True

                # us_nodes = get_upstream_nodes_at_time(t)
                # nodes_in_path_ab = cut(ds_nodes, us_nodes)


        # then generate_tracks()
        kept_edges = [edge for x, edge in enumerate(track.edges) if keep_node[x]]
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
        """