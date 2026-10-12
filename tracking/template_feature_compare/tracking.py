from ..feature_compare_tracking import FeatureComparisonTracking
from datetime import datetime, timedelta
from feature.util.data_utils import pb_str_to_datetime


class TrackingCompareTemplate(FeatureComparisonTracking):
    """
    Template for simple feature-comparision tracking: Features of consecutive timestamps are considered the same
    if correspond() returns True.
    This can be used if the tracking can simply be done by this pairwise comparision. More complex algorithms, which
    rely on a broader state or which need information from multiple time steps before, need a more general approach.
    """

    def __init__(self, **kwargs):
        # called from example_template.py, to set parameters.
        pass

    def correspond(self, time1, obj1, time2, obj2):
        # Compare obj1 and obj2, based on their .proto definition
        # For example, if we want to track features from the identificaton template, we can compare their centroid, as shown below.

        # Assume this is used with IdentificationTemplate
        # get ID of this feature, unique for given timestep
        o1_id = obj1.id
        o2_id = obj2.id

        # access the properties set earlier
        properties1 = obj1.properties
        properties2 = obj2.properties

        if abs(properties1.centroid.x - properties2.centroid.x) < 1:
            # some random heuristic as example
            # if this is True, obj1 at timestep time1 and obj2 at timestep obj2 are the same
            return True

        return False

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

        if end_time - start_time > timedelta(hours=4):  # only keep if track exists for longer than 4h
            # return the original track
            return track_graph
        # return no track
        return None
