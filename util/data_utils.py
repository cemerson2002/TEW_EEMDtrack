import xarray as xr
from enum import Enum
from datetime import datetime
import numpy as np
import json
from enstools.misc import get_ensemble_dim, get_time_dim
from feature.util.enstools_utils import get_vertical_dim, get_init_time_dim, get_longitude_dim, \
    get_latitude_dim, get_possible_time_dims, get_valid_time_dim
from multiprocessing import Lock
import pandas as pd
from types import SimpleNamespace
lock = Lock()
from google.protobuf.json_format import MessageToJson
from google.protobuf import json_format
import functools

class SplitDimension:
    """
    Dimensions the input data set is split by to be parallelized for identification
    """
    
    class SplitDimensionDim(str, Enum):
        """
        Supported split dimensions
        """
        INIT_TIME = "init_time"
        ENSEMBLE_MEMBER = "member"
        LEVEL = "level"
        VALID_TIME = "valid_time" # this is only split dim during indentification

    def __init__(self, dim: SplitDimensionDim, name, size, is_scalar):
        self.dim = dim
        self.name = name
        self.size = size
        self.is_scalar = is_scalar

    # tracking set ID to string
    @staticmethod
    def set_track_desc_to_sd_str(set_desc):
        str_ = ""
        for sd in SplitDimension.SplitDimensionDim:
            if sd == SplitDimension.SplitDimensionDim.VALID_TIME:
                continue
            if set_desc.HasField(sd):
                str_ = str_ + sd + ": " + str(getattr(set_desc, sd)) + "; "
        return str_


    def __str__(self):
        return str(self.dim) + " " + self.name


def get_valid_time_split_dimension(dataset):

    """
    Get the split-dimension object for the valid_time dimension.

    Parameters
    ----------
    dataset: xr.Dataset

    Returns
    -------
    The SplitDimension object of the valid time.
    """
    # search time
    time_dim = get_valid_time_dim(dataset)
    time_dim_in_input = (time_dim is not None)
    if time_dim_in_input and time_dim in dataset.dims:
        time_dim_len = len(dataset.coords[time_dim].values)
        split_dim = SplitDimension(SplitDimension.SplitDimensionDim.VALID_TIME, time_dim, time_dim_len, False)
    time_var_value = None
    if not time_dim_in_input and 'time' in dataset.variables:
        # time dimension as scalar in dataset, also remember it
        split_dim = SplitDimension(SplitDimension.SplitDimensionDim.VALID_TIME, 'time', 1, True)

    return split_dim

def valid_time_to_dt(valid_time_data, init_time_data): # (dataset, valid_time_split_dim, valid):
    """
    Get actual datetime from init time and valid time
    """

    if np.issubdtype(valid_time_data.dtype, np.datetime64):
        # valid time is already datetime64
        return valid_time_data
    elif np.issubdtype(valid_time_data.dtype, np.timedelta64):
        # valid is a timedelta. find init time and add.
        if init_time_data is None:
            print("Timedeltas in valid time, but no init time given.")
            exit(1)
        return init_time_data + valid_time_data

    else:
        print("Cant convert valid time dimension value " + str(valid_time_data) + " to datetime. Make sure it is a datetime or the step dimension is a timedelta, and the init time a datetime.")
        exit(1)

def get_split_dimensions(dataset: xr.Dataset, processing_mode):
    """
    Get the split dimensions in the data set, so which can be parallelized over.

    Parameters
    ----------
    dataset: xr.Dataset
        the dataset
    processing_mode : '2d' or '3d'
        if '2d', the vertical dimension is also a split dimension to be parallelized over

    Returns
    -------
    List of SplitDimension objects from the dataset.
    """
    split_dims = []

    #   get all dimensions besides latitude and longitude, and try to parallelize them.
    lat_dim = get_latitude_dim(dataset)
    lon_dim = get_longitude_dim(dataset)
    lev_dim = get_vertical_dim(dataset)
    possible_split_dims = []
    for dim in dataset.dims:
        if dim != lat_dim and dim != lon_dim:
            if dim == lev_dim and processing_mode == '3d':
                continue
            possible_split_dims.append(dim)

    init_time_dim = get_init_time_dim(dataset)
    valid_time_dim = get_valid_time_dim(dataset)

    # search ensemble
    ensemble_dim = get_ensemble_dim(dataset)
    ensemble_dim_in_input = (ensemble_dim is not None)
    if ensemble_dim_in_input and ensemble_dim in possible_split_dims:
        ensemble_dim_len = len(dataset.coords[ensemble_dim].values)
        split_dims.append(
            SplitDimension(SplitDimension.SplitDimensionDim.ENSEMBLE_MEMBER, ensemble_dim, ensemble_dim_len, False))
    elif not ensemble_dim_in_input and ensemble_dim in dataset.variables:
        ensemble_dim_len = 1
        split_dims.append(SplitDimension(SplitDimension.SplitDimensionDim.ENSEMBLE_MEMBER, ensemble_dim, 1, True))

    # search init_time
    init_time_dim_in_input = (init_time_dim is not None)
    if init_time_dim_in_input and init_time_dim in possible_split_dims:
        init_time_dim_len = len(dataset.coords[init_time_dim].values)
        split_dims.append(
                SplitDimension(SplitDimension.SplitDimensionDim.INIT_TIME, init_time_dim, init_time_dim_len, False))
    elif not init_time_dim_in_input and init_time_dim in dataset.variables:
        init_time_dim_len = 1
        split_dims.append(SplitDimension(SplitDimension.SplitDimensionDim.INIT_TIME, init_time_dim, 1, True))

    if processing_mode == '2d':
        # search level
        level_dim = get_vertical_dim(dataset)
        level_dim_in_input = (level_dim is not None)
        if level_dim_in_input and level_dim in dataset.dims:
            level_dim_len = len(dataset.coords[level_dim].values)
            split_dims.append(SplitDimension(SplitDimension.SplitDimensionDim.LEVEL, level_dim, level_dim_len, False))
        elif not level_dim_in_input and level_dim in dataset.variables:
            level_dim_len = 1
            split_dims.append(SplitDimension(SplitDimension.SplitDimensionDim.LEVEL, level_dim, 1, True))

    lon = get_longitude_dim(dataset)
    lat = get_latitude_dim(dataset)
    tim = get_time_dim(dataset)
    level_dim = get_vertical_dim(dataset)
    # are there dimensions that could not be resolved?
    for dim in dataset.dims:
        if dim != lon and dim != lat and dim != tim and dim not in [sd.name for sd in split_dims]:
            if processing_mode == '3d' and dim == level_dim:
                continue # level in 3d mode, ok
            print("Found unknown dimension: " + str(dim) + ", these are not getting parallelized...")
            print("Consider using CF Naming Conventions.")

    return split_dims


def get_subset_by_description(dataset, set_desc, proc_mode):
    """
    Get the subset of a data set defined by the protobuf set.
    
    Parameters
    ----------
    dataset: xr.Dataset
    set_desc: pb2.TrackableSet

    Returns
    -------
    the subset of the dataset
    """
    split_dims = get_split_dimensions(dataset, proc_mode)

    sub_ds = dataset
    for split_dim in split_dims:

        if split_dim.dim == SplitDimension.SplitDimensionDim.VALID_TIME:
            continue

        if set_desc.HasField(split_dim.dim.value):
            sub_ds = sub_ds.sel(dict({split_dim.name: getattr(set_desc, split_dim.dim.value)}))

    return sub_ds


def squeeze_nodes(nodes):
    """
    Util function to squeeze nodes in graph.
    (parent1,[child1]), (parent1,[child2])
        -> make them (parent1, [child1, child2])
        
    Parameters
    ----------
    nodes: list of pb2.GraphConnection

    Returns
    -------
    Squeezed nodes
    """
    # nodes are list of (n1,[n2]) -> make them (n1, [n2, ...]) for same n1

    i = 0
    while i < len(nodes):
        node = nodes[i]
        # node.n1, node.n2[]
        # search backwards all other nodes with same n1
        for j in range(len(nodes) - 1, i, -1):
            if node.parent == nodes[j].parent:
                node.children.extend(nodes[j].children)
                nodes = np.delete(nodes, j)
        i += 1
    return nodes

def pb_str_to_datetime(time_str):
    """
    Convert string date as present in pb2 (also ISO) to datetime.

    Parameters
    ----------
    time_str: str
        Time string in %Y-%m-%dT%H:%M:%S format

    Returns
    -------
    datetime object
    """
    return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

def datetime_to_pb_str(time_dt):
    """
    Convert datetime to string as expected by pb2.

    Parameters
    ----------
    time_dt: datetime.datetime
        Datetime to convert

    Returns
    -------
    Time string in %Y-%m-%dT%H:%M:%S format
    """
    return time_dt.strftime('%Y-%m-%dT%H:%M:%S')

def datetime64_to_pb_str(time_dt):
    return pd.to_datetime(str(time_dt)).strftime('%Y-%m-%dT%H:%M:%S')

def pb_str_to_datetime64(time_str):
    dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
    return np.datetime64(dt)

def proto_to_json(obj):
    """
    Convert proto to json recursively.
    """
    return MessageToJson(obj)

def proto_to_simplenamespace(obj):
    """
        Convert proto to SimpleNamespace recursively.
        """
    j = MessageToJson(obj)
    sn = json.loads(j, object_hook=lambda d: SimpleNamespace(**d))
    return sn

def clip(tup, mint, maxt):
    return np.clip(tup, mint, maxt)

def simplenamespace_to_json(sn):
    return json.dumps(sn, default=lambda s: vars(s))

def simplenamespace_to_proto(sn, proto_obj):

    json_str = simplenamespace_to_json(sn)
    pb = json_format.Parse(json_str, proto_obj, ignore_unknown_fields=False)
    return pb

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))