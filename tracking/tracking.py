import numpy as np
from abc import ABC, abstractmethod
import xarray as xr
from tqdm import tqdm
from feature.util.data_utils import get_subset_by_description, get_split_dimensions, squeeze_nodes, \
    SplitDimension
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from feature.util.data_utils import pb_str_to_datetime
from feature.util.graph import DataGraph


class TrackingStrategy(ABC):
    """
    Base abstract class for feature tracking algorithms.
    Implementations need to override the abstract track() method.
    """

    def __init__(self):
        self.workers = None
        self.pb_reference = None
        self.tracks_requested = True
        self.num_sets = None

    @abstractmethod
    def track(self, track_set, subset: xr.Dataset):
        """
        Abstract tracking method. This gets called for each time steps list of the feature descriptions.
        This timeline can be for example a reforecast or a member of an ensemble forecast, in which detected features should
        be tracked over multiple timestamps. This method gets called in parallel for all timelines in the dataset.

        This method should compute links of corresponding features of two time steps.
        See template/ and especially template_feature_compare/ for examples

        Parameters
        ----------
        track_set : iterable of identification_pb2.TrackingSet
                The data to be tracked
        subset : xarray.Dataset
                Subset to be tracked. Forwarded from identification

        Returns
        -------
        connections : list of pb_reference.RefGraphConnections
                The connections forming the path of the features in the time steps.
        """
        connections = []
        return connections

    def preprocess(self, feature_desc):
        """
        Base method for the preprocess of the tracking.

        Parameters
        ----------
        feature_desc: pb2.DatasetDescription
            The whole pb2.DatasetDescription dataset description
        
        Returns
        -------
        feature_desc: 
            Modified pb2.DatasetDescription
        """
        
        return feature_desc
    
    def postprocess(self, feature_desc):
        """
        Base method for the postprocess of the tracking.

        Parameters
        ----------
        feature_desc: pb2.DatasetDescription
            The whole pb2.DatasetDescription dataset description
        
        Returns
        -------
        feature_desc: 
            Modified pb2.DatasetDescription
        """
        
        return feature_desc

    def track_set(self, set_idx, obj_desc=None, dataset=None):
        """
        Tracks a given TrackableSet.

        Parameters
        ----------
        set_idx: index of the set in the feature description
        obj_desc: the feature description
        dataset: the dataset

        Returns
        -------
        """

        # get according subset
        obj_set = obj_desc.sets[set_idx]
        dataset_sel = get_subset_by_description(dataset, obj_set, self.processing_mode)
        split_dims = get_split_dimensions(dataset, self.processing_mode)

        split_string = '; '.join([str(dim.name) + ": " + str(getattr(obj_set, dim.dim)) for dim in split_dims])
        if split_string == '':
            tqdm.write("Start tracking singular trackable set.")
        else:
            tqdm.write("Start tracking set defined by:   " + split_string)

        # track this set
        cons = self.track(obj_set, dataset_sel)
        # print(cons)
        if cons is None:
            cons = []

        # squeeze result
        # build empty graph nodes
        # for each feature add (n1,[]) to list. then add connections.
        empty_cons = []
        for idx_ts, ts in enumerate(obj_set.timesteps):
            cur_time = ts.time
            for obj in ts.features:
                empty_cons.append(self.get_new_connection(cur_time, obj))

        cons = squeeze_nodes(list(empty_cons) + list(cons))
        # sort them by time (string key)
        cons = sorted(cons, key=lambda c: c.parent.time)
        
        # put them into a graph
        tqdm.write("Put tracking result into a graph.")
        g = DataGraph(cons, self, remove_transitive=True)
        
        # if tracks requested, generate tracks from connections.
        if self.tracks_requested:
            g.generate_tracks(apply_filter=True)
            
        g.fill_protobuf(obj_set)
        return None

    
        
    def execute(self, feature_desc, dataset_ref: xr.Dataset):
        """
        Execute the tracking procedure.
        The description is split into the different timelines which can be executed in parallel.

        Parameters
        ----------
        feature_desc : pb2.DatasetDescription
                The description of the detected features from the identification strategy.
        dataset_ref : xarray.Dataset
                Reference to the dataset used in the pipeline.
        workers : int
                Number of workers.
        """
        self.num_sets = len(feature_desc.sets)
        
        feature_desc = self.preprocess(feature_desc)
        # parallel for all tracking sets
        pool = Pool(self.workers)
        
        # Make progress bar if there are multiple trackable sets.
        
        if self.num_sets > 1:
            list(tqdm(pool.imap(partial(self.track_set, obj_desc=feature_desc, dataset=dataset_ref),
                 range(len(feature_desc.sets))), total=len(feature_desc.sets), desc="track()"))
        else:
            list(pool.imap(partial(self.track_set, obj_desc=feature_desc, dataset=dataset_ref),
                 range(len(feature_desc.sets))))
        
        feature_desc = self.postprocess(feature_desc)  # only postprocess feature desc, THEN to graph

    def adjust_track(self, track):
        """
        Override this method to filter, change and split certain tracks.
        When generate_tracks() on a Graph is called, each track is checked via this method and can be adjusted.
        For example, you could check here if the track holds together for longer than a certain time range.
        Each track is a subgraph of the total graph, and each node is part of exactly one subgraph.

        Parameters
        ----------
        track: the track as util.graph.DataGraph

        Returns
        -------
        Either 
            None, if discard track
            track: DataGraph: he adjusted track
            list(edges): list of edges to keep. This will trigger a recompute of adjusting tracks.
        """
        return None


    def get_new_connection(self, start_time, start_obj, end_time=None, end_obj=None):
        """
        Create a new connection for the tracking graph and populates it.
        Takes start time and feature, and if existent end time and feature.

        Parameters
        ----------
        start_time : start time (str)
        start_obj : start feature
        end_time : end time (str)
        end_obj : end feature

        Returns
        -------
        the pb2.GraphConnection
        """
        new_connection = self.pb_reference.GraphConnection()

        new_connection.parent.time = start_time
        new_connection.parent.feature.CopyFrom(start_obj)
        if end_time is not None and end_obj is not None:
            n2 = new_connection.children.add()
            n2.time = end_time
            n2.feature.CopyFrom(end_obj)

        return new_connection

    def get_new_connection_from_id(self, timestep_start, obj_id_start, timestep_end=None, obj_id_end=None):
        """
        Create a new connection for the tracking graph and populates it.
        Takes start timestep and feature ID, and if existent end timestep and feature ID.

        Parameters
        ----------
        timestep_start : start time pb2.Timestep
        obj_id_start : start feature id
        timestep_end : start time pb2.Timestep
        obj_id_end : end feature id

        Returns
        -------
        the pb2.GraphConnection
        """
        new_connection = self.pb_reference.GraphConnection()
        new_connection.parent.time = timestep_start.time

        # search given feature ID in features at timestep_start
        ids_start = [obj.id for obj in timestep_start.features]
        try:
            start_obj_index = ids_start.index(obj_id_start)
        except ValueError as err:
            print(err)
            print("Could not find feature ID " + str(
                obj_id_start) + " in given description for timestep at " + timestep_start.time)
            exit(1)
        # add this feature
        new_connection.parent.feature.CopyFrom(timestep_start.features[start_obj_index])

        # if end node specified, do the same
        if timestep_end is not None and obj_id_end is not None:
            ids_end = [obj.id for obj in timestep_end.features]
            try:
                end_obj_index = ids_end.index(obj_id_end)
            except ValueError as err:
                print(err)
                print("Could not find feature ID " + str(
                    obj_id_end) + " in given description for timestep at " + timestep_end.time)
                exit(1)

            # add this feature
            n2 = new_connection.children.add()
            n2.time = timestep_end.time
            n2.feature.CopyFrom(timestep_end.features[end_obj_index])

        return new_connection

    
