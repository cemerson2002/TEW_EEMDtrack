from ..tracking import TrackingStrategy
import xarray as xr
from enstools.misc import get_time_dim


class TrackingTemplate(TrackingStrategy):
    """
    Template for a default "fall-back" tracking strategy.
    There is also a more specific template called template_feature_compare, which provides a template for strategies
    which are solely based on comparing the features of the feature descriptions of consecutive time steps.
    """

    def track(self, trackable_set, subset: xr.Dataset):
        """
        Implementation of track() for general tracking strategies.
        In this template, we iterate over the valid_times of this forecast/reanalysis and compare all features of two
        consecutive timestamps. If they correspond according to some metric, their IDs should be added to the list.
        See also the template_feature_compare for a more specific case.

        Parameters
        ----------
        trackable_set : identification_pb2.TrackableSet
                The timesteps, a list of timestamps of this (re)forecast.
        subset : xarray.Dataset
                The subset (i.e., current member) to track.

        Returns
        -------
        List of tuples of feature IDs, representing the pairwise connections between features.
        """

        connections = []

        time_dim = get_time_dim(subset)

        timesteps = trackable_set.timesteps
        for t in range(0, len(timesteps) - 1):
            t1 = timesteps[t]
            data_t1 = subset.sel({time_dim: t1.time})

            t2 = timesteps[t + 1]
            data_t2 = subset.sel({time_dim: t2.time})
            
            # Could be parallelized here, but might be not possible based on the application.
            # Check the FeatureComparisonTracking for a parallelized template.
            for o1 in t1.features: 
                for o2 in t2.features:
                    if True: # put something meaningful here. This just creates a complete graph based on the time constraints.
                        connections.append(self.get_new_connection(t1.time, o1, t2.time, o2))


        return connections

    def adjust_track(self, track):
        # keep all tracks after generating them
        return track
