from .tracking import TrackingStrategy
from abc import ABC, abstractmethod
from datetime import datetime
from dask import delayed, compute
import numpy as np
import xarray as xr
from tqdm import tqdm
from feature.util.data_utils import pb_str_to_datetime, SplitDimension
from multiprocessing.pool import ThreadPool as Pool
from functools import partial

class FeatureComparisonTracking(TrackingStrategy):
    """
    Implementation of a tracking strategy which can track features by a simple pairwise comparison of their feature
    descriptions. This acts as an abstract class for comparison tracking strategies.
    It implements the track() method and provides an abstract correspond() method with gives a binary answer if two
    features of (consecutive) timesteps are the same. Features do not have neccessarily be associated with consecutive
    timesteps. Using set_max_delta_compare() the user can ser a maximum timedelta.
    """
    
    def __init__(self):
        self.threads = None
        
    def set_threads(self, threads):
        self.threads = threads

    def set_max_delta_compare(self, cmp_delta):
        """
        Set the maximum delta time of features to compare. If None, only compare features of consecutive timesteps.
        Parameters
        ----------
        cmp_delta: timedelta
            The maximum timedelta between features to consider.

        Returns
        -------

        """
        self.max_compare_delta = cmp_delta

    @abstractmethod
    def correspond(self, time1, obj1, time2, obj2):
        """
        Abstract method. Implementations should check here if obj1 and obj2 of consecutive timestamps can be regarded
        as same feature, creating a tracking link.

        Parameters
        ----------
        obj1 : identification_pb2.Feature
                Feature to compare from timestamp t
        obj2 : identification_pb2.Feature
                Feature to compare from timestamp t+1

        Returns
        -------
        True if the features can be considered the same. Otherwise False.
        """
        return False

    def track(self, tracking_set, subset: xr.Dataset):
        """
        Implementation of track() for tracking strategies which are based on pairwise comparisons of features. Using
        this tracking strategy, the correspond() method has to implement a boolean function which returns True if
        the given feature pair of consecutive time steps should be considered as the same feature.

        Parameters
        ----------
        tracking_set : pb2.TrackingSet The timesteps, a list of timestamps of this (re)forecast.
        subset: the corresponding xarray subset

        Returns
        -------
        List of pb2.GraphConnection s.
        """

        # get pair of timesteps, check if their delta has been requested
        timesteps = tracking_set.timesteps
        
        if not hasattr(self, 'threads'):
            print("FeatureCompareTracking threads not set. Fall back to 16.")
            self.set_threads(16)
        
        pool = Pool(processes=self.threads)
        
        # Make progress bar if only one trackable set. Then, we can handle the progress bar, not the super-parallel-level.
        if self.num_sets == 1:
            results = list(tqdm(pool.imap(partial(self.track_start_time_idx, tracking_set=tracking_set, subset=subset), range(len(timesteps))), total=len(timesteps), desc="comparison_tracking()"))
        else:
            results = list(pool.imap(partial(self.track_start_time_idx, tracking_set=tracking_set, subset=subset), range(len(timesteps))))  # iterate over timesteps.

        # print(list(results))
        outputs = [len(result) for result in results]
        
        # flatten output
        connections = [i for res in results for i in res]
        return np.asarray(connections, dtype=object)
        

    def track_start_time_idx(self, start_time_idx, tracking_set=None, subset=None):
            
        timesteps = tracking_set.timesteps
        
        t1_ts = timesteps[start_time_idx]
        t1_dt = pb_str_to_datetime(t1_ts.time)
        
#        tqdm.write("Start generating edges for t = " + str(t1_dt))
        t_conns = []
        
        try:
            cmp_delta = self.max_compare_delta
        except AttributeError:
            cmp_delta = None
        
        for t2 in range(start_time_idx + 1, len(timesteps)):

            t2_ts = timesteps[t2]
            t2_dt = pb_str_to_datetime(t2_ts.time)

            if cmp_delta is None:
                consider = (t2 - start_time_idx == 1)
            else:
                consider = t2_dt - t1_dt <= cmp_delta
                
            if consider:

                for o1 in t1_ts.features:
                    for o2 in t2_ts.features:
                        if self.correspond(t1_dt, o1, t2_dt, o2):
                            # new feature
                            t_conns.append(self.get_new_connection(t1_ts.time, o1, t2_ts.time, o2))
        
        # print(t_conns)
        return t_conns
                            
                            