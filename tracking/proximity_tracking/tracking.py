from ..feature_compare_tracking import FeatureComparisonTracking
from datetime import datetime, timedelta
from feature.util.data_utils import pb_str_to_datetime, rgetattr
from feature.util.enstools_utils import distance
import numpy as np
deg_to_rad = lambda deg: deg * np.pi / 180.0

class ProximityTracking(FeatureComparisonTracking):
    """
    Proximity tracking tracking: Features are tracked if their speed is within a user-set speed window.
    """

    def __init__(self, lon_name, lat_name, speed_range_ms=[0,20], min_duration=timedelta(hours=24), **kwargs):
        """
        Creates a new instance of the tracking strategy.
        
        Parameters
        ----------
        lon_name: str
            Where in the feature properties the longitude is, e.g. position.longitude
        lat_name: str
            Where in the feature properties the latitude is, e.g. position.latitude
        speed_range_ms: list
            Minimum and maximum expected speed thresholds for the features.
        """
        self.lon_name = lon_name
        self.lat_name = lat_name
        
        self.min_speed_ms = speed_range_ms[0]
        self.max_speed_ms = speed_range_ms[1]
        self.min_duration = min_duration
        

    def correspond(self, time1, obj1, time2, obj2):
        # Compare obj1 and obj2, based on their .proto definition
        
        properties1 = obj1.properties
        properties2 = obj2.properties
        
        lon1 = rgetattr(properties1, self.lon_name)
        lon2 = rgetattr(properties2, self.lon_name)
        lat1 = rgetattr(properties1, self.lat_name)
        lat2 = rgetattr(properties2, self.lat_name)
        
        print(lon1, lon2, lat1, lat2)
        
        dist_m = distance(deg_to_rad(lat1), deg_to_rad(lat2), deg_to_rad(lon1), deg_to_rad(lon2))
        time_s = (time2 - time1).total_seconds()
        speed_ms = dist_m / time_s
        
        print(dist_m, time_s, speed_ms)
        
        return self.min_speed_ms < speed_ms < self.max_speed_ms


    # can be overwritten to filter and adjust tracks after the generation process
    # for example, only keep tracks that sustain longer than a certain time range
    def adjust_track(self, track_graph):
        nodes = [edge.parent for edge in track_graph.edges]
        times = [node.time for node in nodes]
        # print(times)
        first_node = nodes[0]
        last_node = nodes[-1] # get time of first and last node in track

        end_time = pb_str_to_datetime(last_node.time)
        start_time = pb_str_to_datetime(first_node.time)

        if end_time - start_time > self.min_duration:  # only keep if track exists for longer than X
            # return the original track
            return track_graph
        # return no track
        return None
