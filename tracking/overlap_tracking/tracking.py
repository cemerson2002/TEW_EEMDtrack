from ..tracking import TrackingStrategy
from feature.util.data_utils import get_subset_by_description, get_split_dimensions, squeeze_nodes, \
    SplitDimension, pb_str_to_datetime
from enstools.misc import get_time_dim
import xarray as xr
import numpy as np


class OverlapTracking(TrackingStrategy):
    """
    Implementation of simple overlap tracking: Features of consecutive timestamps are considered as same if
    they spatially overlap. This requires the given field which to check for overlaps.

    This field should be of data type "int", where the field is
        zero, if at the location is no feature
        i, if feature with the ID i (from identification) is at the location.

    Optionally, min_duration can be set as a datetime.timedelta, indicating the minimum time an feature has
    to be alive in filtering.
    """

    def __init__(self, field_name=None, min_duration=None):
        self.field_name = field_name
        self.min_duration = min_duration
        pass

    def track(self, trackable_set, subset: xr.Dataset):
        """
        Overlap tracking implementation.
        Always fetch field_name of two consecutive timesteps, and search for overlaps in fields
        """

        time_dim = get_time_dim(subset)
        subset_field = subset[self.field_name]
        connections = []

        timesteps = trackable_set.timesteps
        for t in range(0, len(timesteps) - 1):
            id_connections = set()
            t1 = timesteps[t]
            t2 = timesteps[t + 1]

            # data blocks 2d or 3d to check for overlap
            data_t1 = subset_field.sel({time_dim: t1.time})
            data_t2 = subset_field.sel({time_dim: t2.time})

            # search overlaps
            and_areas = np.logical_and(data_t1.data, data_t2.data)
            where_both = list(zip(*np.where(and_areas)))

            # for each overlap pixel, add start/end ID to set
            for idx in where_both:
                obj1_id = data_t1.data[idx]
                obj2_id = data_t2.data[idx]
                id_connections.add(tuple([obj1_id, obj2_id]))

            # to list so there are no duplicates
            id_connections = list(id_connections)
            # id connections to GraphConnections
            for id_con in id_connections:
                con = self.get_new_connection_from_id(t1, id_con[0], t2, id_con[1])
                connections.append(con)

        return connections

    def adjust_track(self, tracking_graph):
        if self.min_duration is None:
            return tracking_graph

        # keep track if persists longer than min_duration
        nodes = [edge.parent for edge in tracking_graph.edges]
        min_time = pb_str_to_datetime(nodes[0].time)
        max_time = pb_str_to_datetime(nodes[-1].time)
        duration = max_time - min_time

        if duration < self.min_duration:
            return None
        return tracking_graph
