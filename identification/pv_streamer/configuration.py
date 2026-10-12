import xarray as xr
from feature.util.enstools_utils import get_latitude_dim, get_longitude_dim, get_vertical_dim
import logging
from feature.identification.pv_streamer.data_util import get_pv_field_name
import numpy as np

class PVConfig:
    # algorithm specific config
    def __init__(self):
        pass

    @staticmethod
    def get_default(dataset: xr.Dataset, vdim_name):
        config = PVConfig()

        # size of radius of stereographic projection (radius per hemisphere), therefore img=2rX4r
        config.proj_radius = 200
        # w_bar distance in km (Spr: 1500-2000)
        config.w_bar = 750.0 # 2*wbar=maxwidth
        # min length of 2D streamer in km (only 2d mode)
        config.min_streamer_len = 1000.0
        # volume threshold of 3D streamers
        config.min_streamer_volume = 50000 # km2K
        # 3D bridges Kelvin Threshold
        config.bridge_K = 8 # xK bridges remove.
        config.exclude_isolated_cutoffs = False

        # preprocess remove holes so distance mask works
        config.remove_holes_maxarea = int(config.proj_radius)
        if vdim_name is None:
            # 2D
            config.dH = 1 * 125
            return config

        # 3D: distance between levels
        th_values = dataset.coords[vdim_name].data
        res_theta = abs(th_values[1] - th_values[0])

        # scale analysis. vertical extent in range 1000km x 1000km, typical height 10K -> stretch vertically by 100"km/K"
        # so extent in scale 1000x1000x1000. w_bar = 1000 as well. (max w 2000)
        config.dH = res_theta * 125

        return config


class DetectionConfig:
    # general detection config
    def __init__(self):
        pass

    @staticmethod
    def init_def_atm_table():
        # default atmosphere for ERAI from https://rda.ucar.edu/datasets/ds627.1/docs/Pressure_and_isentropic_levels/
        # translate z to theta and pressure so we have an approx z distance
        z_km = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        th = [288.1, 289.8, 291.5, 293.2, 294.9, 296.7, 298.4, 300.3, 302.1, 304.1, 305.9, 308.0, 309.9, 312.0, 314.1, 316.2, 318.4, 320.5, 322.9, 325.2, 327.5, 332.4, 347.4, 363.4, 380.1, 397.5, 415.7, 434.8, 454.7, 475.6, 497.3]
        p = [1013.25, 954.61, 898.76, 845.59, 795.01, 746.91, 701.21, 657.80, 616.60, 577.52, 540.48, 505.39, 472.17, 440.75, 411.05, 382.99, 356.51, 331.54, 308.0, 285.84, 264.99, 226.99, 193.99, 165.79, 141.7, 121.11, 103.52, 88.497, 75.652, 64.674, 55.293]
        return np.asarray(z_km), np.asarray(th), np.asarray(p)

    @staticmethod
    def config_from_dataset(dataset: xr.Dataset):
        config = DetectionConfig()

        config.latitude_dim = get_latitude_dim(dataset)
        if config.latitude_dim is None:
            raise ValueError("Could not detect latitude dimension in dataset.")
        config.longitude_dim = get_longitude_dim(dataset)
        if config.longitude_dim is None:
            raise ValueError("Could not detect longitude dimension in dataset.")
        config.vertical_dim = get_vertical_dim(dataset)
        
        config.pv_dim = get_pv_field_name(dataset)
        if config.pv_dim is None:
            raise ValueError("Error, can't find PV in dataset")
            
        if config.vertical_dim not in dataset[config.pv_dim].dims:
            print("Executing for 2 dimensions.")
            config.dims = 2
        else:
            config.dims = 3

        if config.vertical_dim == 'th_level':
            config.vtype = 'theta'
        elif config.vertical_dim in ['level', 'lev']:
            config.vtype = 'pressure'
        else:
            config.vtype = None
            print("Unknown vtype, assume 2D")

        config.res_lon = abs(dataset.coords[config.longitude_dim].values[1] - dataset.coords[config.longitude_dim].values[0])
        config.res_lat = abs(dataset.coords[config.latitude_dim].values[1] - dataset.coords[config.latitude_dim].values[0])
        if config.vertical_dim is not None:
            config.res_z = abs(dataset.coords[config.vertical_dim].values[1] - dataset.coords[config.vertical_dim].values[0])
        else:
            config.res_z = -1

        # order of latitudes in dataset
        if (dataset.coords[config.latitude_dim][1].values - dataset.coords[config.latitude_dim][0].values) > 0:
            config.lat_order = 'asc'
        else:
            config.lat_order = 'desc'

        minLon = dataset.coords[config.longitude_dim].values[0]
        maxLon = dataset.coords[config.longitude_dim].values[-1]
        if minLon > maxLon:
            minLon, maxLon = maxLon, minLon

        minLat = dataset.coords[config.latitude_dim].values[0]
        maxLat = dataset.coords[config.latitude_dim].values[-1]
        if minLat > maxLat:
            minLat, maxLat = maxLat, minLat

        config.lon_min = minLon
        config.lon_max = maxLon
        config.lat_min = minLat
        config.lat_max = maxLat

        config.lon_indices_len = len(dataset.coords[config.longitude_dim].values)
        config.lat_indices_len = len(dataset.coords[config.latitude_dim].values)

        if maxLon - minLon < 360 - config.res_lon:
            raise ValueError("Error: Longitude range not 360 degrees, needed in this algorithm for feasible results.")

        if maxLat - minLat == 180:
            config.data_extent = 'global'
        elif minLat == 0 and maxLat == 90:
            config.data_extent = 'nh'
        elif minLat == -90 and maxLat == 0:
            config.data_extent = 'sh'
        else:
            raise ValueError("Unknown data extent: Either NH, SH or global.")

        logging.debug("Detected type: " + config.data_extent)

        if config.data_extent != 'nh':
            raise ValueError("Currently only northern hemispheric datasets usable. Use lat:0..90,lon:0..360 datasets.")

        config.tab_z, config.tab_th, config.tab_p = DetectionConfig.init_def_atm_table()

        ##############
        # from vertical data dim (theta,pressure...) to approx z height
        # for real-world volume
        ##############
        ######
        # scale analysis -> assume distance inbetween theta levels the same, need stretch for distances in algorithm
        # use mixed units for distances km^2*K, but stretch vertically so algorithm makes sense
        ######
        # includes config.alg.dH

        if config.dims == 2:
            # 2D
            config.alg = PVConfig.get_default(dataset, config.vertical_dim)
            return config

        if config.vtype == 'theta':
            v_data = dataset.coords[config.vertical_dim].data

            idx = np.searchsorted(config.tab_th, v_data)
            if 0 in idx:
                print("todo idx == 0 ->> access -1")
            idx_bw = (v_data - config.tab_th[idx - 1]) / (config.tab_th[idx] - config.tab_th[idx - 1])
            config.v_data_z = config.tab_z[idx - 1] + idx_bw * (config.tab_z[idx] - config.tab_z[idx - 1])

        elif config.vtype == 'pressure':
            v_data = dataset.coords[config.vertical_dim].data

            idx = config.tab_p.size - np.searchsorted(config.tab_p[::-1], v_data, side='right') # tab_p is descending -> inverse and inverse indices
            if 0 in idx:
                print("todo idx == 0 ->> access -1")

            # where index == config.tab_p.size
            idx = np.where(idx == config.tab_p.size, config.tab_p.size - 1, idx)

            idx_bw = (v_data - config.tab_p[idx - 1]) / (config.tab_p[idx] - config.tab_p[idx - 1])
            config.v_data_z = config.tab_z[idx - 1] + idx_bw * (config.tab_z[idx] - config.tab_z[idx - 1])

        # distances between levels
        config.v_data_z_dist = np.abs(np.diff(config.v_data_z))

        if config.vtype != 'theta':
            print("only theta currently supported")
            exit()
        config.alg = PVConfig.get_default(dataset, config.vertical_dim)

        return config
