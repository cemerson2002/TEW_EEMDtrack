import IMTreatment.vortex_detection.vortex_detection
import skimage.morphology

from feature.identification import IdentificationStrategy
import xarray as xr
import numpy as np
import os, sys
from tqdm import tqdm
from .util import compute_field, interpolate_wavetroughs, create_wt_troughs_and_areas, bpf, compute_rortex, detect_local_maxima

import json
from types import SimpleNamespace
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter, generate_binary_structure
from scipy.ndimage.filters import gaussian_filter
from skimage.draw import line_aa
from feature.util.enstools_utils import get_u_var, get_v_var, get_vertical_dim, get_longitude_dim, get_latitude_dim, get_init_time_dim, get_valid_time_dim
import threading
from skimage.draw import line
from IMTreatment.vortex_criterions import get_lambda2, get_gamma
from IMTreatment.core import VectorField, ScalarField, SpatialScalarFields, SpatialVectorFields


class AEWVortexIdentification(IdentificationStrategy):

    def __init__(self, **kwargs):
        """
        Initialize

        """

        import feature.identification.aew_vortices.configuration as cfg
        self.config = cfg  # config
        self.field = self.config.field
        self.bpf = self.config.bpf
        
        json_path = cfg.in_wts
        with open(json_path) as f:
            wt_data = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

        self.json_desc = wt_data
        self.circles = xr.load_dataset(self.config.circles_file).circles

        pass

    def precompute(self, dataset: xr.Dataset, **kwargs):
        print("Precompute for AEW identification...")

        plt.switch_backend('agg')  # this is thread safe matplotlib but cant display.

        # --------------- CLIMATOLOGY

        self.config.u_dim = self.config.u_dim if hasattr(self.config, 'u_dim') and (self.config.u_dim is not None) else get_u_var(dataset)
        self.config.v_dim = self.config.v_dim if hasattr(self.config, 'v_dim') and (self.config.v_dim is not None) else get_v_var(dataset)
        if 'vo' in dataset.data_vars:
            dataset = dataset.rename(vo='rv')
        print(self.config.u_dim)
        print(self.config.v_dim)

        level_str = get_vertical_dim(dataset)
        lat_str = get_latitude_dim(dataset)
        lon_str = get_longitude_dim(dataset)
        init_time_str = get_init_time_dim(dataset)
        valid_time_str = get_valid_time_dim(dataset)

        # --------------- SUBSET DATA ACCORDING TO CFG
        dataset = dataset.sortby('time')
        start_date_dt = np.datetime64(self.config.start_date) if self.config.start_date is not None else None
        end_date_dt = np.datetime64(self.config.end_date) if self.config.end_date is not None else None

        # if data is lon=0..360, change it to -180..180
        dataset.coords[lon_str] = (dataset.coords[lon_str] + 180) % 360 - 180
        dataset = dataset.sortby(dataset[lon_str])

        sel_levels = self.config.levels
        filter_time_str = init_time_str if init_time_str is not None else valid_time_str

        # subset lon,lat,lev,time
        dataset = dataset.sel(
            # **{lat_str: slice(lat_range[0], lat_range[1])},
            # **{lon_str: slice(lon_range[0], lon_range[1])},
            **{level_str: sel_levels},
            **{filter_time_str: slice(start_date_dt, end_date_dt)})
        
        # get the data we want to investigate
        dataset = dataset.sortby(lat_str) # in case of descending

        if len(dataset.time.values) == 0:
            print("Given start and end time leads to no data to process.")
            exit(1)

        # --------------- DO NUMPY PARALLELIZED STUFF: CREATE TROUGH MASKS

        if (self.config.u_dim is None and self.config.v_dim is None) and (self.config.field not in dataset.data_vars):
            print("Could not locate u and v, and no vorticity fields in dataset. Needed to compute advection terms.")
            exit()

        if self.config.method == 'hodges':
            # input should be low res T42 like 2.8 deg.
            # use spline interpolation to refine grid. cf Hodges95 Sect 3
            
            print("interp")
            lon_range = (dataset[lon_str].values[0], dataset[lon_str].values[-1])
            lat_range = (dataset[lat_str].values[0], dataset[lat_str].values[-1])
            
            lons_05 = np.arange(lon_range[0], lon_range[1]+0.01, 0.5)
            lats_05 = np.arange(lat_range[0], lat_range[1]+0.01, 0.5)
            dataset = dataset.interp({lon_str: lons_05, lat_str: lats_05}, method="cubic")
            
            # coarsen to T42 ~ 2.8deg
            #dataset = dataset.interp(coords={lon_str: lons_t42, lat_str: lats_t42}, method="linear")
            print(dataset)
            
        # BPF (optional)
        if self.bpf and (self.config.u_dim is not None and self.config.v_dim is not None):
            dataset = bpf(dataset, [self.config.u_dim, self.config.v_dim], min_days=2, max_days=6)
            self.config.u_dim = self.config.u_dim + '_bpf'
            self.config.v_dim = self.config.v_dim + '_bpf'
            u_name = self.config.u_dim
            v_name = self.config.v_dim
        elif self.bpf and (self.config.field in dataset.data_vars):
            dataset = bpf(dataset, [self.config.field], min_days=2, max_days=6)
            self.config.field = self.config.field + '_bpf'
        else:
            print("Dont know what to put a BPF on.")
            exit(1)
        
        print(dataset)
        
            
        # self.json_desc = interpolate_wavetroughs(self.json_desc, african_easterly_waves_pb2, self.pb_dataset, start_date_dt, end_date_dt)

        # create field to use
        print(self.config.u_dim)
        print(self.config.v_dim)
        if self.config.u_dim is not None and self.config.v_dim is not None: # uv in data, compute new field
            dataset = compute_field(dataset, self.config.u_dim, self.config.v_dim, self.field)
        dataset = dataset.transpose(..., lat_str, lon_str)

        data_field = dataset[self.field]
        # dataset['vortices'] = xr.zeros_like(dataset[self.field], dtype=int)
        dataset['vortices'] = xr.zeros_like(dataset[self.field], dtype=int)
        dataset.vortices.values = np.zeros(shape=dataset[self.field].data.shape)

        return dataset

    def identify(self, data_chunk: xr.Dataset, index_accessor, **kwargs):

        objs = []
        
        vortices_da = data_chunk.vortices
        
        u_name = self.config.u_dim
        v_name = self.config.v_dim
        lat_str = get_latitude_dim(data_chunk)
        lon_str = get_longitude_dim(data_chunk)
        
        
        if self.config.method == 'critpts':
            sx = ScalarField()
            sx.import_from_arrays(data_chunk[lat_str].values.tolist(), data_chunk[lon_str].values.tolist(), data_chunk[u_name].data)
            sy = ScalarField()
            sy.import_from_arrays(data_chunk[lat_str].values.tolist(), data_chunk[lon_str].values.tolist(), data_chunk[v_name].data)

            v = VectorField()
            try:
                v.import_from_sfs(sx, sy)
            except (ValueError, AttributeError) as e:
                tqdm.write("Error in import_from_sfs(). You might have to edit the IMTreatment package, which computes critical points.")
                tqdm.write("Go to IMTreatment/core/vectorfield.py, and replace the import_from_sfs(self, vx, vy) function (line 575) with:")
                tqdm.write("""    def import_from_sfs(self, vx, vy):
        # overwrite previous
        self.__clean()
        # Be sure axes matches
        if (vx.axe_x != vy.axe_x).any() or (vx.axe_y != vy.axe_y).any():
            raise ValueError('Axes do not match')
        # Be sure units match
        if vx.unit_x != vy.unit_x or vx.unit_y != vy.unit_y:
            raise ValueError('Axes units do not match')
        # Be sure units match
        if vx.unit_values != vy.unit_values:
            raise ValueError('Data units do not match')
        # Combine masks
        mask = np.logical_and(vx.mask, vy.mask)
        # Store data
        self.axe_x = vx.axe_x
        self.axe_y = vx.axe_y
        self.comp_x = vx.values
        self.comp_y = vy.values
        self.mask = mask
        self.unit_x = vx.unit_x
        self.unit_y = vx.unit_y
        self.unit_values = vx.unit_values
        self._values_dtype = vx._values_dtype""")
                raise ValueError('Need package fix.')
            
            cpp = IMTreatment.vortex_detection.vortex_detection.get_critical_points(v, smoothing_size=2) # smoothing_size 
            #vor = IMTreatment.vortex_criterions.vortex_criterions.get_vorticity(v)
            #strength = IMTreatment.vortex_criterions.vortex_criterions.get_swirling_strength(v)

            #rortex,_,_,_ = compute_rortex(strength.values, vor.values, data_chunk[u_name].data, data_chunk[v_name].data, 0.5, 0.5, [1])
            #print(rortex)

            sadd = cpp.sadd[0].xy
            foc = cpp.foc[0].xy
            foc_c = cpp.foc_c[0].xy
            node_i = cpp.node_i[0].xy
            node_o = cpp.node_o[0].xy

            # each vortex
            for i in range(sadd.shape[0]):
                obj = self.get_new_feature(index_accessor)

                properties = obj.properties

                properties.pos.lat = sadd[i][0]
                properties.pos.lon = sadd[i][1]

                interp_rv = data_chunk.rv.interp(coords={lon_str: [sadd[i][1]], lat_str: [sadd[i][0]]}, method="linear")
                interp_rv_value = interp_rv.values[0][0]
                
                properties.rel_vort = interp_rv_value
                
                # get indices of nearest lonlat in data set
                ilon = list(vortices_da[lon_str].values).index(vortices_da.sel({lon_str:sadd[i][1]}, method='nearest')[lon_str])
                ilat = list(vortices_da[lat_str].values).index(vortices_da.sel({lat_str:sadd[i][0]}, method='nearest')[lat_str])

                properties.is_ccw = True if interp_rv_value > 0 else False

                # vortices_da.loc[{lon_str: round_off_rating(sadd[i][1]), lat_str: round_off_rating(sadd[i][0])}] = 1 if properties.is_ccw else -1
                vortices_da[{lon_str: ilon, lat_str: ilat}] = 1 if properties.is_ccw else -1
                objs.append(obj)

            data_chunk['vortices'] = vortices_da
            
        elif self.config.method == 'hodges':
            
            # T42? 2.79@eq also ORDER? uv@t42 or only rv
            
            rv_data = data_chunk[self.config.field].data
            max_data = detect_local_maxima(rv_data)
            
            thr = self.config.base_threshold[self.config.field]
            
            max_data = np.where(rv_data < thr, 0, max_data)
            
            data_chunk['vortices'].values = max_data # no ccw/cw...
            maxima = np.where(data_chunk.vortices == 1)
            
            for i in range(len(maxima[0])):
                
                pt = data_chunk.vortices[maxima[0][i], maxima[1][i]]
                if pt.latitude.item() < 0 or pt.latitude.item() > 24:
                    data_chunk.vortices[maxima[0][i], maxima[1][i]] = 0
                    continue # only keep band
                
                obj = self.get_new_feature(index_accessor)
                obj.id = i + 1
                properties = obj.properties

                properties.pos.lat = pt.latitude.item()
                properties.pos.lon = pt.longitude.item()
                properties.is_ccw = True

                objs.append(obj)
                
            
            print(len(objs))
            
        
        
        

            
        
        # IMTreatment.vortex_criterions.vortex_criterions.get_swirling_strength(vf, raw=False)
        
        
                
        # v = data_chunk.vortices
        #cp = data_chunk.saddle

        #ds = data_chunk.stack(z=["latitude", "longitude"])
        #ds = ds.where(ds.saddle > 0, drop=True)

        # CLUSTER
        # prominence of 2deg -> keep if it is the max peak in a 2deg circle?
        #                    -> or discard if any peak within 2deg is higher? or is this the same?
        # For each vortex...
        """
        for z in range(len(ds.z.data)):
            if ds.saddle[z].item() == 0:
                print("skip?")
                continue
            lon_vortex = ds.longitude[z].item()
            lat_vortex = ds.latitude[z].item()
            data_vertex = ds[self.field][z].item()

            circle = self.circles.sel(longitude_center=lon_vortex, latitude_center=lat_vortex)

            # is this efficient? compare manually with all other vortex pts
            for zc in range(len(ds.z.data)):
                lon_vc = ds.longitude[zc].item()
                lat_vc = ds.latitude[zc].item()
                data_c = ds[self.field][zc].item()

                distance_c = circle.sel(longitude=lon_vc, latitude=lat_vc).item()
                # if distance in promenance radius and other point has higher vorticity -> delete this point.
                if distance_c < self.config.prominence_radius and data_c > data_vertex:
                    # vortex in vicinity is higher. discard this.
                    cp.loc[dict(longitude=lon_vortex, latitude=lat_vortex)] = 0
                    break
        """
        # data_chunk.vortices.values = skimage.morphology.binary_dilation(data_chunk.vortices.values)

        # for each vortex
        #  associate with its corresponding WT
        #  put into json



        return data_chunk, objs

    def postprocess(self, dataset: xr.Dataset, data_desc, **kwargs):
    
        dataset.to_netcdf("test_ip.nc")
        
        return dataset, data_desc

    """
    def update_vortices_field(self, dataset, set_desc, lev=700):
        # given vortices of tracks, overwrite there value from +/-1 to +/-
        from collections import defaultdict
        
        vortices_da = dataset.vortices
        level_str = get_vertical_dim(dataset)
        lat_str = get_latitude_dim(dataset)
        lon_str = get_longitude_dim(dataset)
        track_nodes = [e.parent for track in set_desc.tracks for e in track.edges]
        
        # bucket nodes by time
        bucket_set = defaultdict(list)
        
        for track_node in track_nodes:
            bucket_set[track_node.time].append(track_node)
        
        for time, nodes in bucket_set.items():
            print("Update vortex field", time, lev)
            
            vortices_da_time = vortices_da.sel({level_str: lev, 'time': time})
            
            for node in nodes:
                
                node_lat = node.feature.properties.pos.lat
                node_lon = node.feature.properties.pos.lon
                
                interp_vortex = vortices_da_time.sel({lon_str: round_off_rating(node_lon), lat_str: round_off_rating(node_lat)})
                interp_vortex_value = interp_vortex.values
                assert(interp_vortex_value in [-1.0,1.0])
                vortices_da_time.loc[{lon_str: round_off_rating(node_lon), lat_str: round_off_rating(node_lat)}] = interp_vortex_value * 10 # tracks to -2,+2
                
            vortices_da.loc[dict(time=time)] = vortices_da_time
            
        dataset['vortices'] = vortices_da
    """