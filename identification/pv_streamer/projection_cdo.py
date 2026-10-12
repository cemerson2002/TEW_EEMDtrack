import configparser
import math
import os

import numpy as np
import xarray as xr
from cdo import *
import itertools

from feature.util.enstools_utils import get_latitude_dim, get_longitude_dim


def read_grid_config():
    """
    read data from grid file
    """
    gridfile_path = str(os.path.dirname(os.path.realpath(__file__))) + "/gridfile"
    with open(gridfile_path) as f:
        file_content = '[dummy_section]\n' + f.read()
    grid_config = configparser.ConfigParser()
    grid_config.read_string(file_content)
    return grid_config


def precompute_distance_map_simple(stereo_dataset):
    """
    Precompute in simple fashion the distance map for stereographic dataset
    Uses information from grid file to compute distortion parameter and from there the distances
    """
    grid_config = read_grid_config()
    xinc, yinc = int(grid_config['dummy_section']['xinc']), int(grid_config['dummy_section']['yinc'])
    pixel_dist = abs(int(xinc))  # distance of one pixel in km

    # cosine(lat/2)^2 is distortion factor in stereographic projection, when center is at pole.
    # need to handle that distortion starts at 90N
    cos_lat_2 = np.cos(np.deg2rad((90.0 - stereo_dataset['lat_field'].values) / 2.0))
    h_factor = np.square(cos_lat_2)

    return h_factor * pixel_dist  # scaled distance by distortion


def precompute_area_map_simple(distance_map):
    """
    Area map approximated as square of distance map
    """
    am = np.square(distance_map)
    np.nan_to_num(am, copy=False)
    print("Use area map with total area of " + str(np.sum(am)))
    return am


#
def project_latlon_to_stereo(dataset: xr.Dataset):
    """
    project dataset data into stereographic projection according to parameters from grid file
    uses CDO as underlying wrapper (CDO installation needed!)
    """
    lat_dim = get_latitude_dim(dataset)
    lon_dim = get_longitude_dim(dataset)

    lon_field = xr.DataArray(coords=[dataset.coords[lat_dim], dataset.coords[lon_dim]])
    lon_field = lon_field.where(False, lon_field[lon_dim])
    lat_field = xr.DataArray(coords=[dataset.coords[lat_dim], dataset.coords[lon_dim]])
    lat_field = lat_field.where(False, lat_field[lat_dim])

    dataset['lon_field'] = lon_field
    dataset['lat_field'] = lat_field

    cdo = Cdo()
    cdo.debug = True

    this_dir = os.path.dirname(os.path.realpath(__file__))  # gridfile in same dir as this script
    # be careful with interpolation scheme: for example interpolating lons might give wrong results at antimeridian
    # remapbil: bilinear, remapnn: nearest neighbour
    stereo_ds = cdo.remapbil("" + str(this_dir) + "/gridfile", input=dataset, returnXDataset=True)

    return stereo_ds


def project_stereo_to_latlon(dataset: xr.Dataset, gridfile_path):
    
    #lat_dim = get_latitude_dim(dataset)
    #lon_dim = get_longitude_dim(dataset)

    #lon_field = xr.DataArray(coords=[dataset.coords[lat_dim], dataset.coords[lon_dim]])
    #lon_field = lon_field.where(False, lon_field[lon_dim])
    #lat_field = xr.DataArray(coords=[dataset.coords[lat_dim], dataset.coords[lon_dim]])
    #lat_field = lat_field.where(False, lat_field[lat_dim])

    #dataset['lon_field'] = lon_field
    #dataset['lat_field'] = lat_field

    cdo = Cdo()
    cdo.debug = True

    this_dir = os.path.dirname(os.path.realpath(__file__))  # gridfile in same dir as this script
    # be careful with interpolation scheme: for example interpolating lons might give wrong results at antimeridian
    # remapbil: bilinear, remapnn: nearest neighbour
    latlon_ds = cdo.remapnn(gridfile_path, input=dataset, returnXDataset=True) # nearest neighbour for streamers!
    
    return latlon_ds
