import logging
import six
from enstools.misc import is_additional_coordinate_variable
import numpy as np


# this one is already in enstools.misc, but not in the version on the w2w cluster. simple copy.
def distance(lat1, lat2, lon1, lon2, radius=6371229.0):
    """
    distance between two points on a globe.

    Parameters
    ----------
    lat1:
            latitude of first point in radian.

    lat2:
            latitude of second point in radian.

    lon1:
            longitude of first point in radian.

    lon2:
            logitude of second point in radian.

    radius:
            radius of the globe in m.

    Returns
    -------
    distance in m
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c


def get_latitude_dim(ds):
    """
    get the name of the latitude dimension from a dataset or array

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    str or None :
            if no latitude dimension was found, None is returned.
    """
    lat_names = ["lat", "latitude"]
    for lat_name in lat_names:
        if lat_name in ds.dims:
            logging.debug("get_latitude_dim: found name '%s'" % lat_name)
            return lat_name
    return None


def get_longitude_dim(ds):
    """
    get the name of the longitude dimension from a dataset or array

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    str or None :
            if no longitude dimension was found, None is returned.
    """
    lon_names = ["lon", "longitude"]
    for lon_name in lon_names:
        if lon_name in ds.dims:
            logging.debug("get_longitude_dim: found name '%s'" % lon_name)
            return lon_name
    return None

def get_u_var(ds):
    u_names = ["u", "U", "var131"]
    for u_name in u_names:
        if u_name in ds.data_vars:
            logging.debug("get_u_dim: found name '%s'" % u_name)
            return u_name
    return None

def get_v_var(ds):
    v_names = ["v", "V", "var132"]
    for v_name in v_names:
        if v_name in ds.data_vars:
            logging.debug("get_v_dim: found name '%s'" % v_name)
            return v_name
    return None


def get_vertical_dim(ds):
    """
    get the name of the vertical dimension from a dataset or array.
    Needs to be expanded.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    str or None :
            if no vertical dimension was found, None is returned.
    """
    v_names = ["pres", "p", "lev", "level", "isobaric", "layer", "hybrid", "height", "height_2", "th_level", "plev", "isobaricInhPa"]
    for v_name in v_names:
        if v_name in ds.dims:
            logging.debug("get_vertical_dim: found name '%s'" % v_name)
            return v_name
    return None


def get_possible_time_dims(ds):
    """
        Get list of dimensions that can be associated with 'time'.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray

        Returns
        -------
        str or None :
                if no init_time dimension was found, None is returned.
        """
    v_names = ["init_time", "time", "times", "valid_time", "step"]
    found = []
    for v_name in v_names:
        if v_name in ds.dims:
            logging.debug("get_possible_time_dims: found name '%s'" % v_name)
            found.append(v_name)
    return found

def get_init_time_dim(ds):
    """
    get the name of the init_time dimension from a dataset or array.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    str or None :
            if no init_time dimension was found, None is returned.
    """
    pos = get_possible_time_dims(ds)
    if 'init_time' in pos:
        return 'init_time'

    if len(pos) == 2:
        # cant find 'init time'. Do we have 'step'?
        if 'step' in pos: # then use the other dim.
            init_time_dim = pos[0] if pos[1] == 'step' else pos[1]
            return init_time_dim
        else:
            print("Cant resolve init_time: " + str(pos) + " Consider renaming to init_time.")
            exit(1)
    if len(pos) > 2:
        print("Cant resolve that many time dimensions: " + str(pos))
        exit(1)

    return None

def get_valid_time_dim(ds):
    """
    get the name of the valid_time dimension from a dataset or array.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    str or None :
            if no valid time dimension was found, None is returned.
    """
    pos = get_possible_time_dims(ds)

    if 'valid_time' in pos:
        return 'valid_time'
    if len(pos) == 1 and pos[0] == 'time':
        return 'time'

    if len(pos) == 2:
        # cant find 'init time'. Do we have 'step'?
        if 'init_time' in pos: # then use the other dim.
            valid_time_dim = pos[0] if pos[1] == 'init_time' else pos[1]
            return valid_time_dim
        elif 'step' in pos: # if step in times, use step.
            return 'step'
        else:
            print("Cant resolve valid_time: " + str(pos) + " Consider renaming to valid_time.")
            exit(1)
    if len(pos) > 2:
        print("Cant resolve that many time dimensions: " + str(pos))
        exit(1)

    return None


def add_time_dim(ds, time, inplace=True):
    """
    create an ensemble dimension with in the dataset

    Parameters
    ----------
    ds : xarray.Dataset

    time : datetime
            time of the dataset

    inplace : bool
            modify the dataset directly?

    Returns
    -------
    xarray.Dataset:
            A copy of the dataset expanded by the time dimension or the expanded dataset
            if the expansion was done inplace.
    """
    # create a lazy copy of the dataset
    if inplace:
        new_ds = ds
    else:
        new_ds = ds.copy()

    if time is None:
        import datetime
        time = datetime.datetime.now()

    # loop over all data variables.
    # those with time dimension are extended behind the time dimension, others at the front
    for one_name, one_var in six.iteritems(ds.data_vars):
        if is_additional_coordinate_variable(one_var):
            continue
        new_ds[one_name] = one_var.expand_dims("time")
    new_ds.coords["time"] = [time]

    return new_ds


def add_vertical_dim(ds, vertical_value=0, vertical_name='level', inplace=True):
    """
    create a vertical dimension with in the dataset

    Parameters
    ----------
    ds : xarray.Dataset

    vertical_value : float
            value for the added vertical dimension of the dataset, default 0

    vertical_name : str
            name for the added vertical dimension of the dataset, default 'level'

    inplace : bool
            modify the dataset directly?

    Returns
    -------
    xarray.Dataset:
            A copy of the dataset expanded by the vertical dimension or the expanded dataset
            if the expansion was done inplace.
    """
    # create a lazy copy of the dataset
    if inplace:
        new_ds = ds
    else:
        new_ds = ds.copy()

    # loop over all data variables.
    # those with time dimension are extended behind the time dimension, others at the front
    for one_name, one_var in six.iteritems(ds.data_vars):
        if is_additional_coordinate_variable(one_var):
            continue
        if len(one_var.dims) > 0 and one_var.dims[0] == "time":
            new_ds[one_name] = one_var.expand_dims(vertical_name, 1)
        else:
            new_ds[one_name] = one_var.expand_dims(vertical_name)
    new_ds.coords[vertical_name] = [vertical_value]

    return new_ds
