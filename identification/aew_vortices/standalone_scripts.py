import numpy as np
import xarray as xr
import json, sys, os
from types import SimpleNamespace
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
from collections import defaultdict

def get_nodes_per_track(data_desc, level, split='year'):
    # returns
    # node_dict[year] = [ {0: t0}, {1: t1}, ..., {None: leftovers}] with tn nodes of track n
    # and leftovers node not part of any track
    # split checked for start of any track
    
    def get_dict_key(split, node):
        key = None
        if split == 'year':
            key = node.time[:4]
        elif split == 'month':
            key = node.time[:7]
        elif split == 'all':
            key = 'all'
        return key
    
    level_set = None
    for set_ in data_desc.sets:
        if set_.level == level:
            level_set = set_
            
    if level_set is None:
        print("Could not find set for given level")
        exit()
    
    node_dict = dict()
    used_ids = []
    for track_id, track in enumerate(level_set.tracks):
        
        track_nodes = [edge.parent for edge in track.edges]
        track_node_times = [n.time for n in track_nodes]
        
        starting_node = track_nodes[0]
        key = get_dict_key(split, starting_node)
        if key not in node_dict:
                node_dict[key] = []
            
        # if it has a merge/(split, cant just return list of parents. instead plot list of track segments.
        has_split_or_merge = (len(track_node_times) != len(list(set(track_node_times))))
        
        if has_split_or_merge:
            
            track_segments = []
            
            for track_edge in track.edges:
                parent = track_edge.parent
                if hasattr(track_edge, 'children'):
                    for c in track_edge.children:
                        track_segments.append([parent,c])
            node_dict[key].append(track_segments)
            
        else:
            node_dict[key].append(track_nodes)
            for n in track_nodes:
                used_ids.append(n.feature.id)

    
    """
    for k,v in node_dict.items():
        print('key', k)
        for k_,v_ in v.items():
            print('k_', k_)
            print('v_', len(v_))
    """
    return node_dict

def plot_vortex_tracks(data_desc, method, year, bpf_str, level, split='year'):
    ## Plot vortex in given input desc split by month
    
    node_dict = get_nodes_per_track(data_desc, level, split)
    
    for split_item, tracks in node_dict.items():
        print(split_item)
        
        resolution = 2000
        cbar_space_px = 80
        subplotparameters = mpl.figure.SubplotParams(left=0, bottom=0, right=1 - cbar_space_px / resolution, top=1,
                                                     wspace=0, hspace=0)
        fig, ax = plt.subplots(figsize=(resolution / 150, resolution / 450),
                               dpi=100,
                               subplotpars=subplotparameters,
                               subplot_kw=dict(projection=ccrs.PlateCarree()))

        extent = [-75, 45, -10, 40]
        
        for t_key, track in enumerate(tracks):
            
            if type(track[0]) is list:
                
                # plot individual track segments to resolve merge/split
                for track_segment in track:
                    pos_list = [track_vortex.feature.properties.pos for track_vortex in track_segment]
                    line = patches.Path([[p.lon, p.lat] for p in pos_list])
                    if track_segment[0].feature.properties.isCcw:
                        col = 'blue'
                    else:
                        col = 'red'
            
                    patch = patches.PathPatch(line, linewidth=1, facecolor='none', edgecolor=col)  # cmap(time_weight)
                    ax.add_patch(patch)
                
            else:
                pos_list = [track_vortex.feature.properties.pos for track_vortex in track]
                line = patches.Path([[p.lon, p.lat] for p in pos_list])
            
                if track[0].feature.properties.isCcw:
                    col = 'blue'
                else:
                    col = 'red'
            
                patch = patches.PathPatch(line, linewidth=1, facecolor='none', edgecolor=col)  # cmap(time_weight)
                ax.add_patch(patch)



        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=list(np.array([255, 225, 171]) / 255))

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        ax.set_title("")
        fig.tight_layout()
        
        print("Saving...")
        if not os.path.isdir(f'{year}'):
            os.makedirs(f'{year}')
        plt.savefig(f'{year}/{method}tracks_{bpf_str}_{level}_{split_item}.png', format='png', backend='agg')


def plot_vortex_tracks_heat_map(data_descs, circles, method, bpf_str, level, split='year', ip=4, rad=100000): # 100km
    cached = os.path.exists(f"heatmaps/ds_{method}_{bpf_str}_{level}_{split}.nc")
            
    if not cached and (type(data_descs) is list): # multiple years:
        node_dicts = [get_nodes_per_track(data_desc_y, level, split) for data_desc_y in data_descs]
        
        node_dict = defaultdict(list)
        # if split by month reorder.
        if split == 'month':
            
            for node_dict_ in node_dicts:
                for split_item, tracks in node_dict_.items():
                    if split_item is not None:
                        split_month = split_item[-2:]
                        node_dict[split_month].extend(tracks)
                    
        elif split == 'year':
            # just merge
            for node_dict_ in node_dicts:
                for split_item, tracks in node_dict_.items():
                    if split_item is not None:
                        split_year = split_item
                        node_dict[split_year].extend(tracks)
                
        elif split == 'all':
            # put them under 'all'
            for node_dict_ in node_dicts:
                for split_item, tracks in node_dict_.items():
                    if split_item is not None:
                        node_dict['all'].extend(tracks)
            
            
    elif not cached:
        node_dict = get_nodes_per_track(data_descs, level, split)

    field_template = xr.zeros_like(circles.isel(longitude_center=0, latitude_center=0), dtype=bool)
    
    if cached:
        node_dict = {split: None}
    
    for split_item, tracks in node_dict.items():
        print(split_item)
        
        if os.path.exists(f"heatmaps/ds_{method}_{bpf_str}_{level}_{split_item}.nc"):
            # use cached one
            heat_maps = xr.open_dataset(f"heatmaps/ds_{method}_{bpf_str}_{level}_{split_item}.nc")
            heat_map_cyc = heat_maps.cyc
            heat_map_acyc = heat_maps.acyc
            print(heat_maps)
        
        else:
            # generate from data
            heat_map_cyc = xr.zeros_like(field_template, dtype=int)
            heat_map_acyc = xr.zeros_like(heat_map_cyc, dtype=int)

            for t_key, track in enumerate(tracks):
                print(t_key)
                
                if type(track[0]) is list: # if merger/splitter: create intermediate points for all track segments
                    lons = []
                    lats = []
                    is_ccw = track[0][0].feature.properties.isCcw
                    for track_segment in track:
                        pos_list = [track_vortex.feature.properties.pos for track_vortex in track_segment]
                        assert(len(pos_list) == 2)
                        print(pos_list)
                        
                        # create ip intermediate points to have it more smooth
                        lons_seg = [np.nan] * (ip * len(pos_list) - (ip-1))
                        lats_seg = [np.nan] * (ip * len(pos_list) - (ip-1))
                        lons_seg[::ip] = [p.lon for p in pos_list]
                        lats_seg[::ip] = [p.lat for p in pos_list]
                        lons_seg = pd.Series(lons_seg).interpolate().tolist()
                        lats_seg = pd.Series(lats_seg).interpolate().tolist()
                        
                        lons.extend(lons_seg)
                        lats.extend(lats_seg)
                else:

                    is_ccw = track[0].feature.properties.isCcw
                    pos_list = [track_vortex.feature.properties.pos for track_vortex in track]
                    # create ip intermediate points to have it more smooth
                    lons = [np.nan] * (ip * len(pos_list) - (ip-1))
                    lats = [np.nan] * (ip * len(pos_list) - (ip-1))
                    lons[::ip] = [p.lon for p in pos_list]
                    lats[::ip] = [p.lat for p in pos_list]
                    lons = pd.Series(lons).interpolate().tolist()
                    lats = pd.Series(lats).interpolate().tolist()

                # line = patches.Path([[p.lon, p.lat] for p in pos_list])
                # patch = patches.PathPatch(line, linewidth=1, facecolor='none', edgecolor='red')  # cmap(time_weight)
                # ax.add_patch(patch)
                track_mask = xr.zeros_like(field_template, dtype=bool)

                for p_idx in range(len(lons)):

                    p_circle = circles.sel(longitude_center=lons[p_idx], latitude_center=lats[p_idx], method="nearest")
                    p_circle_mask = p_circle < rad 

                    track_mask = np.logical_or(track_mask, p_circle_mask)
                
                if is_ccw:
                    heat_map_cyc = heat_map_cyc + track_mask.astype(dtype=int)
                else:
                    heat_map_acyc = heat_map_acyc + track_mask.astype(dtype=int)
                

            heat_maps = heat_map_cyc.to_dataset(name="cyc")
            heat_maps["acyc"] = heat_map_acyc
            
            heat_maps.to_netcdf(f"heatmaps/ds_{method}_{bpf_str}_{level}_{split_item}.nc")

            
        if method == 'hodges':
            # regrid heatmap to F32 resolution and let matplotlib handle the interpolation
            # just hardcode w/e
            lats_ = [37.6730896290453, 34.8825209937735, 32.091943881744, 
    29.3013596217627, 26.510769325211, 23.7201739335347, 20.9295742544895, 
    18.1389709902394, 15.3483647594915, 12.5577561152307, 9.76714555919557, 
    6.97653355394864, 4.18592053318915, 1.3953069108195, -1.3953069108195, 
    -4.18592053318915, -6.97653355394864, -9.76714555919557]
            lons_ = [-73.125, -70.3125, -67.5, -64.6875, -61.875, -59.0625, -56.25, 
    -53.4375, -50.625, -47.8125, -45, -42.1875, -39.375, -36.5625, -33.75, 
    -30.9375, -28.125, -25.3125, -22.5, -19.6875, -16.875, -14.0625, -11.25, 
    -8.4375, -5.625, -2.8125, 0, 2.8125, 5.625, 8.4375, 11.25, 14.0625, 
    16.875, 19.6875, 22.5, 25.3125, 28.125, 30.9375, 33.75, 36.5625, 39.375, 
    42.1875, 45]
            heat_map_cyc = heat_map_cyc.interp({'longitude': lons_, 'latitude': lats_}, method="linear")
            
        # plot
        
        print(heat_map_cyc.max())
        print(heat_map_acyc.max())
        resolution = 2000
        cbar_space_px = 80
        #subplotparameters = mpl.figure.SubplotParams(left=0, bottom=0, right=1 - cbar_space_px / resolution, top=1,
        #                                             wspace=0, hspace=0)
        fig, ax = plt.subplots(figsize=(16,5),
                               dpi=100,
                               # subplotpars=subplotparameters,
                               subplot_kw=dict(projection=ccrs.PlateCarree()))

        extent = [-75, 45, -10, 40]
        
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=list(np.array([255, 225, 171]) / 255))

        heat_map_cyc.plot.contourf(ax=ax, cmap='Blues', add_colorbar=True)
        if method != 'hodges':
            heat_map_acyc.plot.contour(ax=ax, cmap='Reds', add_colorbar=True)
        
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3)
        
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        ax.set_title("")
        fig.tight_layout()
        
        print("Saving...")
        if not os.path.isdir(f'heatmaps'):
            os.makedirs(f'heatmaps')
        plt.savefig(f'heatmaps/heat_{method}_{bpf_str}_{level}_{split_item}.png', format='png', backend='agg')

    
    

if sys.argv[1] == 'tracks':
    # python standalone_scripts.py tracks hodges/critpts 2018 True/False(bpf) OPT[700/850 month/year]
    method = sys.argv[2]
    year = sys.argv[3]
    bpf_str = sys.argv[4]
    fp = f"/project/meteo/w2w/C3/fischer/data/aew_new/{year}/easterly_vortices_{method}_rv_{bpf_str}.json"
    data_desc = json.load(open(fp), object_hook= lambda x: SimpleNamespace(**x))
    
    if len(sys.argv) > 5:
        levs = [int(sys.argv[5])]
    else:
        levs = [600,700,850]
    if len(sys.argv) > 6:
        splits = [sys.argv[6]]
    else:
        splits = ['year','month']
    
    for lev in levs:
        for split in splits:
            plot_vortex_tracks(data_desc, method, year, bpf_str, lev, split=split)

elif sys.argv[1] == 'track_heatmap':
    # python standalone_scripts.py track_heatmap hodges/critpts T/F 700/850 month/year/all
    years = range(2002,2023)
    method = sys.argv[2]
    bpf_str = sys.argv[3]
    
    fps = [f"/project/meteo/w2w/C3/fischer/data/aew_new/{year}/easterly_vortices_{method}_rv_{bpf_str}.json" for year in years]
    data_descs = [json.load(open(fp), object_hook= lambda x: SimpleNamespace(**x)) for fp in fps]
    
    circles = xr.load_dataset("/project/meteo/w2w/C3/fischer/data/circles.nc").circles
    
    if len(sys.argv) > 4:
        levs = [int(sys.argv[4])]
    else:
        levs = [600,700,850]
    if len(sys.argv) > 5:
        splits = [sys.argv[5]]
    else:
        splits = ['year','month', 'all']
    if method == 'hodges':
        rad = 100000
    else:
        rad = 400000
    for lev in levs:
        for split in splits:
            plot_vortex_tracks_heat_map(data_descs, circles, method, bpf_str, lev, split=split, rad=rad)