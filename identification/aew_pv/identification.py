from feature.identification import IdentificationStrategy
import xarray as xr
import numpy as np
import os, sys
from .util import split_wt_files_to_timedict, get_wts_for_subset, filter_pv, create_wt_area, gen_feature_descriptions
from .plotting import plot_kw_style

import json
from types import SimpleNamespace
from matplotlib import pyplot as plt
from feature.identification._proto_gen import aew_pv_pb2
import cartopy.crs as ccrs

from skimage.draw import line_aa
from skimage.morphology import binary_dilation, binary_erosion, dilation, erosion
from feature.util.enstools_utils import get_u_var, get_v_var, get_vertical_dim, get_longitude_dim, get_latitude_dim, get_init_time_dim, get_valid_time_dim
import threading
from skimage.draw import line

aew_pv_lock = threading.Lock() # need lock to write to obj reference list within identify

class AEWPVIdentification(IdentificationStrategy):

    def __init__(self, cfg, **kwargs):
        """
        Initialize

        """

        self.config = cfg  # config
        
        json_path = self.config.in_wts
        with open(json_path) as f:
            wt_data = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

        self.json_desc = wt_data
        self.circles = xr.load_dataset(self.config.circles_file).circles
        self.wts_time_dict = dict()

        pass

    def precompute(self, dataset: xr.Dataset, **kwargs):
        print("Precompute for AEW identification...")

        plt.switch_backend('agg')  # this is thread safe matplotlib but cant display.
        
        # dataset = dataset.isel(time=200)

        # dataset = dataset.drop_vars(["env_R", "env_I"])
        u_name = self.config.u_dim if hasattr(self.config, 'u_dim') and (self.config.u_dim is not None) else get_u_var(dataset)
        v_name = self.config.v_dim if hasattr(self.config, 'v_dim') and (self.config.v_dim is not None) else get_v_var(dataset)

        if u_name is None or v_name is None:
            print("Could not locate u and v fields in dataset. Needed to compute advection terms.")
            exit()

        level_str = get_vertical_dim(dataset)
        lat_str = get_latitude_dim(dataset)
        lon_str = get_longitude_dim(dataset)
        init_time_str = get_init_time_dim(dataset)
        valid_time_str = get_valid_time_dim(dataset)

        # --------------- SUBSET DATA ACCORDING TO CFG
        start_date_dt = np.datetime64(self.config.start_date) if self.config.start_date is not None else None
        end_date_dt = np.datetime64(self.config.end_date) if self.config.end_date is not None else None

        # if data is lon=0..360, change it to -180..180
        dataset.coords[lon_str] = (dataset.coords[lon_str] + 180) % 360 - 180
        dataset = dataset.sortby(dataset[lon_str])

        filter_time_str = init_time_str if init_time_str is not None else valid_time_str

        # subset lon,lat,lev,time
        dataset = dataset.sel(
            # **{lat_str: slice(lat_range[0], lat_range[1])},
            # **{lon_str: slice(lon_range[0], lon_range[1])},
            #  **{level_str: sel_levels},
            **{filter_time_str: slice(start_date_dt, end_date_dt)}
        )
        
        dataset['wavetroughs'] = dataset.wavetroughs.fillna(0)
        dataset['tracks'] = dataset.tracks.fillna(0)
        
        # get the data we want to investigate
        dataset = dataset.sortby(lat_str) # in case of descending
        dataset['pv_bin'] = xr.zeros_like(dataset.pv, dtype=bool)
        dataset['pv_filter'] = xr.zeros_like(dataset.pv, dtype=int)
        dataset['pv_morphed'] = xr.zeros_like(dataset.pv, dtype=int)
        dataset['pv_opened'] = xr.zeros_like(dataset.pv, dtype=int)
        dataset['wt_infl'] = xr.zeros_like(dataset[self.config.use_wt_field], dtype=int)
        
        dataset['obj_reference'] = xr.zeros_like(dataset.pv, dtype=int).isel(**{lat_str: 0, lon_str: 0, level_str: 0}).squeeze(drop=True)
        
        # preprocess json input to dict
        self.wts_time_dict = split_wt_files_to_timedict([self.config.in_wts], "tracks")
        self.map_objs = [] # after identify
        
        # pv field to binary via threshold, lat boundaries, phase boundaries
        dataset = filter_pv(dataset, self.config.pvu_thr_upper, self.config.phase_range, self.config.phase_level, self.config.lat_range, self.config.lev_range)
        
        self.selem_plane = np.zeros(shape=(3, 3, 3))
        self.selem_plane[1, 1, 1] = 1
        self.selem_plane[1, 1, 0] = 1
        self.selem_plane[1, 1, 2] = 1
        self.selem_plane[1, 0, 1] = 1
        self.selem_plane[1, 2, 1] = 1
        self.selem_vertical = np.zeros(shape=(3, 3, 3))
        self.selem_vertical[1, 1, 1] = 1
        self.selem_vertical[0, 1, 0] = 1
        self.selem_vertical[2, 1, 2] = 1
        
        return dataset

    def identify(self, data_chunk: xr.Dataset, **kwargs):
        
        u_name = self.config.u_dim
        v_name = self.config.v_dim
        
        current_wts = get_wts_for_subset(data_chunk, self.wts_time_dict)
        
        # already in field
        # data_chunk = make_wt_field_by_id(data_chunk, current_wts)
        
        # for each id in ds -> expand by circle
        data_chunk = create_wt_area(data_chunk, self.config.use_wt_field, self.circles, radius=self.config.wt_radius)
        
        # filter pv_bin by WT area
        # data_chunk['pv_filter'] = data_chunk.pv_filter.transpose(..., "level")
        
        data_chunk['pv_filter'].data = xr.where(data_chunk.pv_bin, data_chunk.wt_infl, 0)
        
        # dilate/erode PV_bin
        assert(data_chunk.pv_filter.dims[0] == 'level') # levels in front for selem
        
        data_chunk['pv_filter'].data = xr.where(data_chunk.pv_bin, data_chunk.wt_infl, 0)
        # need to make negative ids temporarily positive to make morphology work
        
        shift_ = -data_chunk.pv_filter.data.min()+1 if data_chunk.pv_filter.data.min() < 0 else 0 # set minimum id to +1 in case of negative ids, all > 0.
        shifted_ = xr.where(data_chunk.pv_filter != 0, data_chunk.pv_filter + shift_, 0)
        
        # closing
        dilated = dilation(shifted_.data)
        dilated = dilation(dilated)
        dilated = dilation(dilated)
        eroded = erosion(dilated)
        eroded = erosion(eroded)
        eroded = erosion(eroded)
        
        # shift back negative ids to be negative
        data_chunk['pv_morphed'].data = np.where(eroded != 0, eroded - shift_, 0)
        
        # apply lev filter after morph
        data_chunk['pv_morphed'] = xr.where(np.logical_and(data_chunk.level > self.config.lev_range[0], data_chunk.level < self.config.lev_range[1]), data_chunk.pv_morphed, 0)
        
        """
        import copy
        a = copy.deepcopy(shifted_.data)
        a = erosion(a)
        a = dilation(a)
        
        # shift back negative ids to be negative
        data_chunk['pv_opened'].data = np.where(a != 0, a - shift_, 0)
        # apply lev filter after morph
        data_chunk['pv_opened'] = xr.where(np.logical_and(data_chunk.level > self.config.lev_range[0], data_chunk.level < self.config.lev_range[1]), data_chunk.pv_opened, 0)
        """
        
        # dont cut morphed at lower threshold -> we want to combine debris.
        # starting with PV, expand to area using lower thr. nah or? with dilate basically same?!
        
        objs = gen_feature_descriptions(data_chunk, current_wts, self)
        # cant put attributes in netcdf here since variable length. depends on amount of WTs in previous timesteps
        
        objs_f = objs
        
        with aew_pv_lock:
            map_id = len(self.map_objs)
            self.map_objs.append(objs_f)

        data_chunk['obj_reference'] = map_id
        
        return data_chunk, objs_f

    def postprocess(self, dataset: xr.Dataset, data_desc, **kwargs):

        # plot_kw_style(dataset, self.json_desc, self.config, lev=700)
        # plot_kw_style(dataset, self.json_desc, self.config, lev=850)
        # print(dataset)
        # dataset.to_netcdf("/home/ws/he7273/phd_all/data/coll_oper/jjaso2021_test.nc")
        # add vortices to tracks?
        
        for v in dataset.data_vars:
            if v not in ["pv_bin", "pv_morphed", "pv_filter", "pv_opened", "wt_infl", "obj_reference"]:
                dataset = dataset.drop_vars(v)
                
        if "pv_bin" in dataset.data_vars:
            dataset["pv_bin"] = dataset.pv_bin.transpose("time", ...)

        # add geometry to netcdf if present
        
        dataset['geometry_container1'] = xr.DataArray(
            attrs={
                "geometry_type": "ellipsoid", 
                "start_index": "start_index_ellipsoid_pc", 
                "node_count": "node_count_ellipsoid_pc", 
                "node_coordinates": "x_c y_c z_c x_major y_major z_major x_med y_med z_med x_minor y_minor z_minor",
                "long_name": "Principal Component Ellipsoid"
            })
        dataset['geometry_container2'] = xr.DataArray(
            attrs={
                "geometry_type": "ellipsoid", 
                "start_index": "start_index_ellipsoid_minvol", 
                "node_count": "node_count_ellipsoid_minvol", 
                "node_coordinates": "x_c_mv y_c_mv z_c_mv x_major_mv y_major_mv z_major_mv x_med_mv y_med_mv z_med_mv x_minor_mv y_minor_mv z_minor_mv",
                "long_name": "Minimum Volume Ellipsoid"
            })
        dataset['geometry_container3'] = xr.DataArray(
            attrs={
                "geometry_type": "ellipsoid", 
                "start_index": "start_index_ellipsoid_sqdist", 
                "node_count": "node_count_ellipsoid_sqdist", 
                "node_coordinates": "x_c_sq y_c_sq z_c_sq x_major_sq y_major_sq z_major_sq x_med_sq y_med_sq z_med_sq x_minor_sq y_minor_sq z_minor_sq",
                "long_name": "Minimal Sqdist Ellipsoid"
            })
        
        # TODO ellipsoids here currently only writable for reanalysis data, so one time line.
        dataset["node_count_ellipsoid_pc"] = xr.DataArray(np.zeros(shape=(len(dataset.time.values)), dtype=int), dims=("time",))
        dataset["start_index_ellipsoid_pc"] = xr.DataArray(np.zeros(shape=(len(dataset.time.values)), dtype=int), dims=("time",))
        
        dataset["node_count_ellipsoid_minvol"] = xr.DataArray(np.zeros(shape=(len(dataset.time.values)), dtype=int), dims=("time",))
        dataset["start_index_ellipsoid_minvol"] = xr.DataArray(np.zeros(shape=(len(dataset.time.values)), dtype=int), dims=("time",))
        
        dataset["node_count_ellipsoid_sqdist"] = xr.DataArray(np.zeros(shape=(len(dataset.time.values)), dtype=int), dims=("time",))
        dataset["start_index_ellipsoid_sqdist"] = xr.DataArray(np.zeros(shape=(len(dataset.time.values)), dtype=int), dims=("time",))
        
        x_c, y_c, z_c, x_major, y_major, z_major, x_med, y_med, z_med, x_minor, y_minor, z_minor = [],[],[],[],[],[],[],[],[],[],[],[]
        x_c_mv, y_c_mv, z_c_mv, x_major_mv, y_major_mv, z_major_mv, x_med_mv, y_med_mv, z_med_mv, x_minor_mv, y_minor_mv, z_minor_mv = [],[],[],[],[],[],[],[],[],[],[],[]
        x_c_sq, y_c_sq, z_c_sq, x_major_sq, y_major_sq, z_major_sq, x_med_sq, y_med_sq, z_med_sq, x_minor_sq, y_minor_sq, z_minor_sq = [],[],[],[],[],[],[],[],[],[],[],[]
        # x_s, y_s, x_e, y_e = [],[],[],[]

        for time_id in range(len(dataset.time.values)):
            
            subset = dataset.isel(time=time_id)
            objs = self.map_objs[subset['obj_reference'].item()]
            
            subset["start_index_ellipsoid_pc"].values = len(x_c) # start index is current length
            subset["start_index_ellipsoid_minvol"].values = len(x_c) # start index is current length
            subset["start_index_ellipsoid_sqdist"].values = len(x_c) # start index is current length
            # subset["start_index_lines"].values = len(x_s) # start index is current length
            
            count = 0 
            count_l = 0
            for obj_idx, obj in enumerate(objs):
                count += 1
                x_c.append(obj.properties.ell.center.lon)
                y_c.append(obj.properties.ell.center.lat)
                z_c.append(obj.properties.ell.center.lev)
                    
                x_major.append(obj.properties.ell.major_axis.lon)
                y_major.append(obj.properties.ell.major_axis.lat)
                z_major.append(obj.properties.ell.major_axis.lev)
                
                x_med.append(obj.properties.ell.medium_axis.lon)
                y_med.append(obj.properties.ell.medium_axis.lat)
                z_med.append(obj.properties.ell.medium_axis.lev)
                
                x_minor.append(obj.properties.ell.minor_axis.lon)
                y_minor.append(obj.properties.ell.minor_axis.lat)
                z_minor.append(obj.properties.ell.minor_axis.lev)
                
                x_c_mv.append(obj.properties.ell_minvol.center.lon)
                y_c_mv.append(obj.properties.ell_minvol.center.lat)
                z_c_mv.append(obj.properties.ell_minvol.center.lev)
                    
                x_major_mv.append(obj.properties.ell_minvol.major_axis.lon)
                y_major_mv.append(obj.properties.ell_minvol.major_axis.lat)
                z_major_mv.append(obj.properties.ell_minvol.major_axis.lev)
                
                x_med_mv.append(obj.properties.ell_minvol.medium_axis.lon)
                y_med_mv.append(obj.properties.ell_minvol.medium_axis.lat)
                z_med_mv.append(obj.properties.ell_minvol.medium_axis.lev)
                
                x_minor_mv.append(obj.properties.ell_minvol.minor_axis.lon)
                y_minor_mv.append(obj.properties.ell_minvol.minor_axis.lat)
                z_minor_mv.append(obj.properties.ell_minvol.minor_axis.lev)
                
                x_c_sq.append(obj.properties.ell_sqdist.center.lon)
                y_c_sq.append(obj.properties.ell_sqdist.center.lat)
                z_c_sq.append(obj.properties.ell_sqdist.center.lev)
                    
                x_major_sq.append(obj.properties.ell_sqdist.major_axis.lon)
                y_major_sq.append(obj.properties.ell_sqdist.major_axis.lat)
                z_major_sq.append(obj.properties.ell_sqdist.major_axis.lev)
                
                x_med_sq.append(obj.properties.ell_sqdist.medium_axis.lon)
                y_med_sq.append(obj.properties.ell_sqdist.medium_axis.lat)
                z_med_sq.append(obj.properties.ell_sqdist.medium_axis.lev)
                
                x_minor_sq.append(obj.properties.ell_sqdist.minor_axis.lon)
                y_minor_sq.append(obj.properties.ell_sqdist.minor_axis.lat)
                z_minor_sq.append(obj.properties.ell_sqdist.minor_axis.lev)
                """
                for pt_idx in range(len(obj.properties.wt.line_pts) - 1):
                    count_l += 1
                    x_s.append(obj.properties.wt.line_pts[pt_idx].lon)
                    y_s.append(obj.properties.wt.line_pts[pt_idx].lat)
                    x_e.append(obj.properties.wt.line_pts[pt_idx + 1].lon)
                    y_e.append(obj.properties.wt.line_pts[pt_idx + 1].lat)
                """
                
            subset["node_count_ellipsoid_pc"].values = count
            subset["node_count_ellipsoid_minvol"].values = count
            subset["node_count_ellipsoid_sqdist"].values = count
            # subset["node_count_lines"].values = count_l
            
            dataset.start_index_ellipsoid_pc[dict(time=time_id)] = subset["start_index_ellipsoid_pc"]
            dataset.node_count_ellipsoid_pc[dict(time=time_id)] = subset["node_count_ellipsoid_pc"]
            
            dataset.start_index_ellipsoid_minvol[dict(time=time_id)] = subset["start_index_ellipsoid_minvol"]
            dataset.node_count_ellipsoid_minvol[dict(time=time_id)] = subset["node_count_ellipsoid_minvol"]
            
            dataset.start_index_ellipsoid_sqdist[dict(time=time_id)] = subset["start_index_ellipsoid_sqdist"]
            dataset.node_count_ellipsoid_sqdist[dict(time=time_id)] = subset["node_count_ellipsoid_sqdist"]
            # dataset.start_index_lines[dict(time=time_id)] = subset["start_index_lines"]
            # dataset.node_count_lines[dict(time=time_id)] = subset["node_count_lines"]
        
        dataset.coords['x_c'] = xr.DataArray(x_c, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_c'] = xr.DataArray(y_c, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_c'] = xr.DataArray(z_c, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})
        
        dataset.coords['x_major'] = xr.DataArray(x_major, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_major'] = xr.DataArray(y_major, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_major'] = xr.DataArray(z_major, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})
        
        dataset.coords['x_med'] = xr.DataArray(x_med, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_med'] = xr.DataArray(y_med, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_med'] = xr.DataArray(z_med, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})

        dataset.coords['x_minor'] = xr.DataArray(x_minor, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_minor'] = xr.DataArray(y_minor, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_minor'] = xr.DataArray(z_minor, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})
        
        dataset.coords['x_c_mv'] = xr.DataArray(x_c_mv, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_c_mv'] = xr.DataArray(y_c_mv, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_c_mv'] = xr.DataArray(z_c_mv, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})
        
        dataset.coords['x_major_mv'] = xr.DataArray(x_major_mv, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_major_mv'] = xr.DataArray(y_major_mv, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_major_mv'] = xr.DataArray(z_major_mv, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})
        
        dataset.coords['x_med_mv'] = xr.DataArray(x_med_mv, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_med_mv'] = xr.DataArray(y_med_mv, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_med_mv'] = xr.DataArray(z_med_mv, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})

        dataset.coords['x_minor_mv'] = xr.DataArray(x_minor_mv, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_minor_mv'] = xr.DataArray(y_minor_mv, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_minor_mv'] = xr.DataArray(z_minor_mv, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})
        
        dataset.coords['x_c_sq'] = xr.DataArray(x_c_sq, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_c_sq'] = xr.DataArray(y_c_sq, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_c_sq'] = xr.DataArray(z_c_sq, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})
        
        dataset.coords['x_major_sq'] = xr.DataArray(x_major_sq, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_major_sq'] = xr.DataArray(y_major_sq, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_major_sq'] = xr.DataArray(z_major_sq, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})
        
        dataset.coords['x_med_sq'] = xr.DataArray(x_med_sq, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_med_sq'] = xr.DataArray(y_med_sq, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_med_sq'] = xr.DataArray(z_med_sq, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})

        dataset.coords['x_minor_sq'] = xr.DataArray(x_minor_sq, dims=("ellipsoid_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_minor_sq'] = xr.DataArray(y_minor_sq, dims=("ellipsoid_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        dataset.coords['z_minor_sq'] = xr.DataArray(z_minor_sq, dims=("ellipsoid_node",), attrs={"units": "hPa", "standard_name": "level", "axis": "Z"})
        
        """
        dataset.coords['x_s'] = xr.DataArray(x_s, dims=("line_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_s'] = xr.DataArray(y_s, dims=("line_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        
        dataset.coords['x_e'] = xr.DataArray(x_e, dims=("line_node",), attrs={"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        dataset.coords['y_e'] = xr.DataArray(y_e, dims=("line_node",), attrs={"units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
        """
        
        return dataset, data_desc

