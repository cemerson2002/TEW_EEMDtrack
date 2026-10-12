from feature.identification import IdentificationStrategy
import xarray as xr
import numpy as np
import os, sys

from feature.util.data_utils import pb_str_to_datetime

import metpy.calc as mpcalc
from .util import calc_adv
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

from .processing import populate_feature, compute_cv
from skimage.draw import line_aa
from feature.util.enstools_utils import get_u_var, get_v_var, get_vertical_dim, get_longitude_dim, get_latitude_dim, get_init_time_dim, get_valid_time_dim
import threading
from skimage.draw import line
from collections import Counter

from shapely.geometry import LineString, Point

import contourpy

class AEWIdentification(IdentificationStrategy):

    def __init__(self, wt_out_file=False, wt_traj_dir=None, cv='cv', year_summer=None, month=None, **kwargs):
        """
        Initialize the AEW Identification.

        Parameters (experimental)
        ----------
        kwargs
        wt_out_file: output the wavetroughs as new and only out-field in 0.5x0.5
        year_summer: if set, process AEW season (01.06.-31.10.) of given year
        
        """

        import feature.identification.african_easterly_waves.configuration as cfg
        self.config = cfg  # config
        self.config.out_traj_dir = wt_traj_dir
        self.config.cv_name = cv
        
        self.orig_dataset = None
        
        if year_summer is not None:
            if month is not None:
                m_str = str(month).zfill(2)
                self.config.start_date = str(year_summer) + '-' + m_str + '-01T00:00'
                self.config.end_date = str(year_summer) + '-' + m_str + '-30T00:00'
            else:
#                self.config.start_date = str(year_summer) + '-06-01T00:00'
#                self.config.end_date = str(year_summer) + '-10-31T23:00'
                self.config.start_date = str(year_summer) + '-08-01T00:00'
                self.config.end_date = str(year_summer) + '-08-31T23:00'

        self.config.out_wt = wt_out_file
        if wt_out_file:
            self.config.sum_over_all = True
        self.lock_ = threading.Lock()
        
                        # ---- DEBUG counters (remove when done) ----
        self.dbg_wave = Counter()
        self.wave_size = []              
        
        pass
        
    @staticmethod
    def _pct(vals, qs=(80, 85, 90, 95, 100)):
        if not vals:
            return "n/a"
        v = sorted(vals)
        return ", ".join(f"p{q}={v[min(len(v) - 1, int(q / 100 * (len(v) - 1) + 0.5))]:.3g}" for q in qs)
        
    def _dbg_identify_summary(self):
        prints = self.dbg_wave
        print("\n[DEBUG identify] ---------------------------------------------")
        print(f"  troughs checked:         {prints['wave_tested']}")
        print(f"  failed because not in right area:       {prints['not_in_area']}")
        print(f"  failed because too small:        {prints['too_small']}")
        print(f"  wave size [deg]: {self._pct(self.wave_size)}")
        print(f"  failed because bad lats:  {prints['bad_lats']}")
        print(f"  failed because bad props. :{prints['bad_props']}")
        print(f"  passed: {prints['passed']}")
        print(f"  merged into another piece of the same trough: {prints['merged_away']} "
              f"-> {prints['passed'] - prints['merged_away']} troughs")
        print("-----------------------------------------------------------\n")
        self.dbg_wave.clear()
        self.wave_size.clear()

    def precompute(self, dataset: xr.Dataset, **kwargs):
        print("Precompute for AEW identification...")

        plt.switch_backend('agg')  # this is thread safe matplotlib but cant display.

        # --------------- CLIMATOLOGY
        lon_range = self.config.data_lon
        lat_range = self.config.data_lat
        clim_file = self.config.get_clim_file()

        u_name = self.config.u_dim if hasattr(self.config, 'u_dim') and (self.config.u_dim is not None) else get_u_var(dataset)
        v_name = self.config.v_dim if hasattr(self.config, 'v_dim') and (self.config.v_dim is not None) else get_v_var(dataset)

        if u_name is None or v_name is None:
            print("Could not locate u and v fields in dataset. Needed to compute advection terms.")
            exit()

        # restrict dimensions only to the ones present in u/v
        all_dims = [d for d in dataset.dims]
        for d in all_dims:
            if d not in dataset[u_name].dims:
                dataset = dataset.drop_dims(d)

        level_str = get_vertical_dim(dataset)
        lat_str = get_latitude_dim(dataset)
        lon_str = get_longitude_dim(dataset)
        init_time_str = get_init_time_dim(dataset)
        valid_time_str = get_valid_time_dim(dataset)
        
        cv_clim = None

        if os.path.isfile(clim_file):
            cv_clim = xr.open_dataset(clim_file)
        if cv_clim is None:
            from .climatology import compute_climatology
            print("Climatology file not found. Computing climatology...")
            cv_clim = compute_climatology(self.config)
            cv_clim.to_netcdf(clim_file)

        lat_str_clim = get_latitude_dim(cv_clim)
        lon_str_clim = get_longitude_dim(cv_clim)
#        cv_clim = cv_clim.sel(
#            **{lat_str_clim: slice(lat_range[0], lat_range[1])},
#            **{lon_str_clim: slice(lon_range[0], lon_range[1])})

        # --------------- SUBSET DATA ACCORDING TO CFG
        start_date_dt = np.datetime64(self.config.start_date) if self.config.start_date is not None else None
        end_date_dt = np.datetime64(self.config.end_date) if self.config.end_date is not None else None
        
        # if data is lon=0..360, change it to -180..180
        dataset.coords[lon_str] = (dataset.coords[lon_str] + 180) % 360 - 180
        dataset = dataset.sortby(dataset[lon_str])

        filter_time_str = init_time_str if init_time_str is not None else valid_time_str
        
        # get the data we want to investigate
        dataset = dataset.sortby(lat_str) # in case of descending
        dataset = dataset.sortby(filter_time_str) # I am looking at you, ERA5
        
        dataset = dataset.sel(
#            **{lat_str: slice(lat_range[0], lat_range[1])},
#            **{lon_str: slice(lon_range[0], lon_range[1])},
            **{filter_time_str: slice(start_date_dt, end_date_dt)})

        if len(dataset.time.values) == 0:
            print("Given start and end time leads to no data to process.")
            exit(1)

        # dataset = dataset.expand_dims('level')
        # level_str = 'level'
        if level_str in dataset[u_name].dims:
            dataset = dataset.sel(**{level_str: self.config.levels}) # 3-D wind field, select levels
        elif level_str is not None:
            dataset = dataset.sel(**{level_str: self.config.levels}) # 2-D, reduce 3-D field too
            dataset[u_name] = dataset[u_name].expand_dims('level')
            dataset[v_name] = dataset[v_name].expand_dims('level')
        else:
            dataset[u_name] = dataset[u_name].expand_dims(level=self.config.levels)
            dataset[v_name] = dataset[v_name].expand_dims(level=self.config.levels)
            level_str = 'level'
        
        # rename cv_clim dimensions to be same as in data.
        cv_clim = cv_clim.rename({'lat': lat_str, 'lon': lon_str})
        if 'plev' in cv_clim.dims and 'plev' != level_str:
            print("plev from clim to level: div by 100.")
            cv_clim = cv_clim.rename({'plev': level_str})
            cv_clim = cv_clim.assign_coords({level_str: cv_clim[level_str] / 100})

        # also only use levels also present in data
        #cv_clim = cv_clim.sel({level_str: dataset[level_str].values})

#        if self.config.cv_name not in dataset.data_vars:
#            print("Curvature Vorticity not found, trying to compute it out of U and V...")
#            dataset = compute_cv(dataset, u_name, v_name, self.config.cv_name)
        
        self.orig_dataset = dataset
        
        # make dataset to 2.5 (or same as cv_clim)
        dataset = dataset.interp({lat_str: cv_clim.coords[lat_str], lon_str: cv_clim.coords[lon_str]})

        # make sure that lat and lon are last two dimensions
        if lat_str not in dataset[self.config.cv_name].coords.dims[-2:] or lon_str not in dataset[
                                                                                              self.config.cv_name].coords.dims[
                                                                                          -2:]:
            print("Reordering dimensions so lat and lon at back. Required for metpy.calc.")
            dataset = dataset.transpose(..., lat_str, lon_str)

        # --------------- DO NUMPY PARALLELIZED STUFF: CREATE TROUGH MASKS

        u = dataset[u_name]
        v = dataset[v_name]
        cv = dataset[self.config.cv_name]
        # smooth CV with kernel
        
        cv = mpcalc.smooth_gaussian(cv, n=13).metpy.dequantify()

#        # create hourofyear to get anomalies
#        cv = cv.assign_coords(hourofyear=cv.time.dt.strftime("%m-%d %H"))
#        cv_anom = cv.groupby('hourofyear') - cv_clim.cv
#        
#        # compute advection of cv: first and second derivative
#        adv1, adv2 = calc_adv(cv_anom, u, v)
        adv1, adv2 = calc_adv(cv, u, v)

#        # xr.where() anomaly data exceeds the percentile from the hourofyear climatology:
#        # replace data time with hourofyear -> compare with climatology percentile -> back to real time
#        cv_anom_h = cv_anom.swap_dims(dims_dict={'time': 'hourofyear'})
#        perc_mask_h = cv_anom_h.where(
#            cv_anom_h > cv_clim.cva_quantile_hoy.sel(dict(hourofyear=cv_anom.hourofyear.data)))
#        perc_mask = perc_mask_h.swap_dims(dims_dict={'hourofyear': 'time'})

        sigma_map = cv_clim.cv_sigma
        sigma_floor = np.nanpercentile(sigma_map.values, self.config.cv_sigma_floor_pctile)
        thresh_map = self.config.cv_sigma_k * np.maximum(sigma_map, sigma_floor)

#        cv_perc_thresh = np.nanpercentile(cv, self.config.cv_percentile)  # 66th percentile of cv anomalies
#        # print(cv_perc_thresh)

        print('Locating wave troughs...')
#        # filter the advection field given our conditions:
#        # trough_mask = adv1
#        
#        cond = (~np.isnan(perc_mask) & # percentile of anomaly over threshold from climatology
#        (adv2.squeeze().values > self.config.second_advection_min_thr) & # second time derivative > 0: dont detect local minima over the percentile
#        (u.squeeze().values < self.config.max_u_thresh) # threshold for propagation speed -> keep only westward
#        )

        cond = (
            (cv > thresh_map) & # filtered CV exceeds sigma-based threshold
            (adv2 > self.config.second_advection_min_thr) & # dont detect local min over percentile
            (u < self.config.max_u_thresh) # propagation speed threshold
        )
            
#        trough_mask = adv1.squeeze().where(cond)
        trough_mask = adv1.where(cond).squeeze()
        
        dataset['trough_mask'] = trough_mask
        dataset['adv1'] = adv1.squeeze()
        dataset['trough_ok'] = (~trough_mask.isnull()).astype('int8')
        
        """
        # create 0.5x0.5 dataarray for wavetroughs
        min_lat = dataset[lat_str].data.min()
        max_lat = dataset[lat_str].data.max()
        min_lon = dataset[lon_str].data.min()
        max_lon = dataset[lon_str].data.max()

        lat05 = np.linspace(min_lat, max_lat, int((max_lat - min_lat) * 2) + 1)
        lon05 = np.linspace(min_lon, max_lon, int((max_lon - min_lon) * 2) + 1)

        # 0.5x0.5 for wavetroughs
        wt = xr.zeros_like(dataset['trough_mask'], dtype=float)
        wt = wt.isel(**{lat_str: 0}).drop(lat_str).isel(**{lon_str: 0}).drop(lon_str)
        wt = wt.expand_dims(lon05=lon05).expand_dims(lat05=lat05)
        wt = wt.transpose(..., 'lat05', 'lon05')

        dataset['wavetroughs'] = wt
        dataset['wavetroughs'].attrs['units'] = 'prob'
        dataset['wavetroughs'].attrs['standard_name'] = 'wavetroughs'
        dataset['wavetroughs'].attrs['long_name'] = 'position_of_wavetrough'
        dataset['lat05'].attrs['long_name'] = 'latitude'
        dataset['lat05'].attrs['standard_name'] = 'latitude'
        dataset['lon05'].attrs['long_name'] = 'longitude'
        dataset['lon05'].attrs['standard_name'] = 'longitude'
        dataset['lat05'].attrs['units'] = 'degrees_north'
        dataset['lon05'].attrs['units'] = 'degrees_east'
        """
        
        return dataset

    def identify(self, data_chunk: xr.Dataset, index_accessor=None, **kwargs):

        objs = []
        trough_mask_cur = data_chunk.trough_mask
        if np.isnan(trough_mask_cur).all():
            return data_chunk, objs

        def clip(tup, mint, maxt):
            return np.clip(tup, mint, maxt)

        #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})
        # generate zero-contours with matplotlib core
#
#        c = trough_mask_cur.plot.contour(transform=ccrs.PlateCarree(), colors='blue', levels=[0.0],
#                                         subplot_kws={'projection': ccrs.PlateCarree()})
#        
#        path_data = c.get_paths()[0]
#        
#        # disassemble in smaller paths
#        path_list = []
#        current_path = []
#        # 1 = new start, 2 = continue
#        for (vertex, vtype) in path_data.iter_segments():
#            if vtype == 1:
#                if len(current_path):
#                    path_list.append(current_path)
#                current_path = [vertex] # new path
#            elif vtype == 2:
#                current_path.append(vertex)
#            elif vtype == 79:
#                # Contour is closed loop. Happens rarely.
#                current_path.append(current_path[0])
#            else:
#                print("Unknown path type " + str(vtype))

#        if len(current_path):
#            path_list.append(current_path)

        from .trough_contours import extract_trough_paths
        
        lat_str = get_latitude_dim(data_chunk)
        lon_str = get_longitude_dim(data_chunk)
        path_list = extract_trough_paths(
            data_chunk.adv1.values, data_chunk.trough_ok.values,
            data_chunk[lon_str].values, data_chunk[lat_str].values,
            max_gap=self.config.contour_gap_pts, counter=self.dbg_wave)

        
        # id_ = 1
        
        
        """
        # Debug for writing contours to file
        import csv
        def export_contours(fname, cn):
            
            with open(fname, "w") as f:
                # for each contour line
                paths_repr = []
                for path in paths:
                    path_repr = []
                    for v_idx, v in enumerate(path.vertices):
                        path_repr.append([v[0], v[1]])
                    
                    paths_repr.append(path_repr)
                wr = csv.writer(f)
                wr.writerows(paths_repr)
                #np.savetxt(fp, contour, comments='# ',delimiter=',', newline='\n' )
                #fp.write('\n')
        
        export_contours('20220913-contour-raw.csv', paths)
        exit()
        """
        
        for path in path_list:

            # get new feature, set id
            o = self.get_new_feature(index_accessor)
            
            # populate it
            populate_feature(o.properties, path, data_chunk, self.config)

            # add to features if keep
            if not self.keep_wavetrough(o.properties):
                continue

            objs.append(o)
        
        if self.config.trough_merge_dist_deg > 0 and len(objs) > 1:
            objs = self.merge_trough_pieces(objs, data_chunk)
 
        return data_chunk, objs
 
    def merge_trough_pieces(self, objs, data_chunk):
        """
        One wave trough is often detected as several pieces (the contour is cut where the trough
        conditions fail, or the trough consists of separate contour lines). Tracked separately, the
        pieces give parallel duplicate tracks and tracks hopping between pieces. Merge kept troughs
        whose lines come within config.trough_merge_dist_deg of each other into one trough: pieces
        are oriented south -> north, ordered by latitude and joined into one line.
        """
        lines = []
        for o in objs:
            pts = [(p.lon, p.lat) for p in o.properties.line_pts]
            lines.append(LineString(pts) if len(pts) > 1 else Point(pts[0]))
 
        # union-find over pieces closer than the merge distance
        parent = list(range(len(objs)))
 
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
 
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if lines[i].distance(lines[j]) <= self.config.trough_merge_dist_deg:
                    parent[find(i)] = find(j)
 
        groups = {}
        for i in range(len(objs)):
            groups.setdefault(find(i), []).append(i)
 
        merged = []
        for members in groups.values():
            o = objs[members[0]]
            if len(members) > 1:
                pieces = []
                for i in members:
                    pts = [(p.lon, p.lat) for p in objs[i].properties.line_pts]
                    if pts[0][1] > pts[-1][1]:
                        pts = pts[::-1]
                    pieces.append(pts)
                pieces.sort(key=lambda pts: pts[0][1])
                o.properties.Clear()
                populate_feature(o.properties, [pt for pts in pieces for pt in pts], data_chunk, self.config)
                self.dbg_wave['merged_away'] += len(members) - 1
            merged.append(o)
        return merged

    def postprocess(self, dataset: xr.Dataset, data_desc, **kwargs):
    
        self._dbg_identify_summary()

        lat_str = get_latitude_dim(dataset)
        lon_str = get_longitude_dim(dataset)
        
            
        """
        # drop everything, only keep WTs as 0.5x0.5
        if self.config.out_wt:
            for var in dataset.data_vars:
                if var not in ['wavetroughs']:
                    dataset = dataset.drop_vars([var])
            # wavetroughs are 0.5x0.5 in lat05,lon05 field. remove other stuff
            for dim in dataset.dims:
                if dim in [lat_str, lon_str, 'hourofyear', 'quantile']:
                    dataset = dataset.drop_vars([dim])

            dataset = dataset.rename({'lat05': lat_str, 'lon05': lon_str})

            level_str = get_vertical_dim(dataset)
            if level_str is not None:
                dataset = dataset.squeeze(drop=True)

            if self.config.sum_over_all:
                dataset['wavetroughs'] = dataset.wavetroughs.sum(dim='time')

        # create met3d like trajectories, not really working right now...
        if self.config.out_traj_dir:
            if not os.path.exists(self.config.out_traj_dir):
                os.makedirs(self.config.out_traj_dir)

            assert (len(data_desc.sets) == 1)
            desc_set = data_desc.sets[0]
            desc_times = desc_set.timesteps

            for idx, ts in enumerate(desc_times):
                # need to make separate dataset for each init-time
                # because number of trajs (WTs) are different from time to time
                dataset_wt = xr.Dataset()

                lon_list = []
                lat_list = []
                pres_list = []
                max_pts_in_wt = -1
                for o in ts.features:  # get lons and lats
                    pt_list = o.properties.line_pts
                    lon_list.append(np.array([pt.lon for pt in pt_list]))
                    lat_list.append(np.array([pt.lat for pt in pt_list]))
                    pres_list.append(np.array([850.0 for pt in pt_list]))
                    max_pts_in_wt = max(max_pts_in_wt, len(lon_list[-1]))
                # go again and fill with NaNs at end
                for i in range(len(lon_list)):  # get lons and lats
                    lon_list[i] = np.pad(lon_list[i], (0, max_pts_in_wt - len(lon_list[i])), mode='constant',
                                         constant_values=np.nan)
                    lat_list[i] = np.pad(lat_list[i], (0, max_pts_in_wt - len(lat_list[i])), mode='constant',
                                         constant_values=np.nan)
                    pres_list[i] = np.pad(pres_list[i], (0, max_pts_in_wt - len(pres_list[i])), mode='constant',
                                          constant_values=np.nan)

                dataset_wt = dataset_wt.expand_dims(
                    time=np.arange(0, max_pts_in_wt).astype(dtype=float))  # fake traj time
                dataset_wt = dataset_wt.expand_dims(ensemble=[0])
                dataset_wt = dataset_wt.expand_dims(trajectory=np.arange(1, len(ts.features) + 1))

                lons = xr.DataArray(np.zeros((1, len(ts.features), max_pts_in_wt)),
                                    dims=("ensemble", "trajectory", "time"))
                lons.attrs['standard_name'] = "longitude"
                lons.attrs['long_name'] = "longitude"
                lons.attrs['units'] = "degrees_east"

                lats = xr.zeros_like(lons)
                lats.attrs['standard_name'] = "latitude"
                lats.attrs['long_name'] = "latitude"
                lats.attrs['units'] = "degrees_north"

                pres = xr.zeros_like(lons)
                pres.attrs['standard_name'] = "air_pressure"
                pres.attrs['long_name'] = "pressure"
                pres.attrs['units'] = "hPa"
                pres.attrs['positive'] = "down"
                pres.attrs['axis'] = "Z"

                dataset_wt['lon'] = lons
                dataset_wt['lat'] = lats
                dataset_wt['pressure'] = pres
                
                lon_list_np = np.array(lon_list)
                lat_list_np = np.array(lat_list)
                pres_list_np = np.array(pres_list)

                dataset_wt['lon'].data[0] = lon_list_np
                dataset_wt['lat'].data[0] = lat_list_np
                dataset_wt['pressure'].data[0] = pres_list_np

                dataset_wt['time'].attrs['standard_name'] = "time"
                dataset_wt['time'].attrs['long_name'] = "time"
                dataset_wt['time'].attrs['units'] = "hours since " + ts.time.replace('T', ' ')
                dataset_wt['time'].attrs['trajectory_starttime'] = ts.time.replace('T', ' ')
                dataset_wt['time'].attrs['forecast_inittime'] = ts.time.replace('T',
                                                                                      ' ')  # '2006-09-01 12:00:00'

                out_path = self.config.out_traj_dir + ts.time.replace(':', '_') + '.nc'
                dataset_wt.to_netcdf(out_path)
        
        """
        return dataset, data_desc

    # filter: keep current wavetrough if:
    #   troughs are within a certain spatial window
    #   length of the trough < threshold
    def keep_wavetrough(self, properties):
        """
        Called for each wavetrough, check if kept based on filtering heuristics:
        - WT requires any point of wavetrough in config.wave_filter range
        - WT requires minimum length threshold
        - WT lat center not in 5..25

        Parameters
        ----------
        properties

        Returns
        -------
        True if kept
        """
        
        self.dbg_wave['wave_tested'] += 1

        in_area = False
        for line_pt in properties.line_pts:

            # check if any point is outside filtering area
            if (self.config.wave_filter_lon[0] < line_pt.lon < self.config.wave_filter_lon[1]
                    and self.config.wave_filter_lat[0] < line_pt.lat < self.config.wave_filter_lat[1]):
                in_area = True

        if not in_area:  # no point of line segment in our area
            self.dbg_wave['not_in_area'] += 1
            return False

        self.wave_size.append(properties.length_deg)
        if properties.length_deg <= self.config.degree_len_thr:  # too small
            self.dbg_wave['too_small'] += 1
            return False
        
        mid_lat = (properties.bb.max.lat + properties.bb.min.lat) / 2.0
        #mid_lat = properties.bb.max.lat - properties.bb.min.lat
        if mid_lat < 5.0 or mid_lat > 25.0:
            self.dbg_wave['bad_lats'] += 1
            return False
        
        height = properties.bb.max.lat - properties.bb.min.lat
        width = properties.bb.max.lon - properties.bb.min.lon
        
        if height < self.config.proportion_n * width:
            self.dbg_wave['bad_props'] += 1
            return False
        
        self.dbg_wave['passed'] += 1
        return True
