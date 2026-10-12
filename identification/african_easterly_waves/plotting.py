import os.path

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import matplotlib
import matplotlib as mpl
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from feature.util.data_utils import pb_str_to_datetime
from pathlib import Path
import feature.identification.african_easterly_waves.configuration as cfg
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def get_kitweather_rain_cm(ens_mode):
    rgb_colors = []
    
    pathtotxtfile = '/project/meteo/w2w/C3/fischer/belanger/feature/feature/identification/african_easterly_waves/' # '/home/iconeps/icon_data/additional_data/colorpalettes/'
    filename_colorpalette = 'colorpalette_dyamond_prec_rate.txt'
    
    if ens_mode:
        filename_colorpalette = 'colormap_WhiteBeigeGreenBlue_16.txt'
    else:
        filename_colorpalette = 'colorpalette_dyamond_prec_rate.txt'

    with open(pathtotxtfile + filename_colorpalette, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        rgb_colors.append([float(line[0:3])/255, float(line[4:7])/255, float(line[8:11])/255, 1])
    rgb_colors = [[1, 1, 1, 0]] + rgb_colors + [[0.35, 0, 0.4, 1]]
    cmap = mpl.colors.ListedColormap(rgb_colors[1:-1]) # , name=colorpalette
    cmap = cmap.with_extremes(bad='white', under=rgb_colors[0], over=rgb_colors[-1])

    if ens_mode:
        levels = [0.01,0.02,0.05,0.1,0.12,0.15,0.2,0.25,0.3,0.4,0.5,0.75]
    else:
        levels = [0.1,0.2,0.3,0.5,1,2,3,5,10,20,30,50]
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    return levels, cmap, norm


def crop_top_bottom_whitespace(path):

    # pixels from image left where a vertical column is scanned from top and bottom for non-white pixels
    x_scan_position = 450
    add_bottom_delta = 20

    im = Image.open(path)
    image_array_y = np.where(np.asarray(im.convert('L')) < 255, 1, 0)[:, x_scan_position]
    vmargins = [np.where(image_array_y[2:] == 1)[0][0] + 2 + 1,
                image_array_y[:-2].shape[0] - np.where(image_array_y[:-2] == 1)[0][-1] + 2]
    im_cropped = Image.new('RGBA',(im.size[0], im.size[1] - vmargins[0] - vmargins[1] + add_bottom_delta), (0, 0, 0, 0))
    im_cropped.paste(im.crop((0, vmargins[0], im.size[0], im.size[1] - vmargins[1] + add_bottom_delta)), (0, 0))
    im.close()
    im_cropped.save(path, 'png')
    im_cropped.close()

    return


## MAIN PLOTTING FUNC FOR KITWEATHER PLOTS - DETERMINISTIC MODE
def plot_ts_filtered_waves(wts_part_of_tracks, fig_name, ds=None, tp=None, ens_mode=False):
    
    from timeit import default_timer as timer
    t1 = timer()

    resolution = 1600
    cbar_space_px = 80
    subplotparameters = mpl.figure.SubplotParams(left=0, bottom=0, right=1 - cbar_space_px / resolution, top=1,
                                             wspace=0, hspace=0)
    fig, ax = plt.subplots(figsize=(resolution / 100, resolution / 100),
                       dpi=100,
                       subplotpars=subplotparameters,
                       subplot_kw=dict(projection = ccrs.PlateCarree()))

    extent = [-100, 35, -10, 35]

    levels_rain, rain_cm, norm = get_kitweather_rain_cm(ens_mode)
    distance_plot_to_cbar = 0.010
    axins = ax.inset_axes([1 + distance_plot_to_cbar, 0.05, 0.015, 0.93],
                      transform=ax.transAxes)
    ticks_list = levels_rain
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=rain_cm, norm=norm),
                    cax=axins, extend='both', extendfrac=0.03,
                    ticks=ticks_list)
    unit_text = '>1 mm/hr\nprobability' if ens_mode else 'mm/hr'
    y_off = -0.075 if ens_mode else -0.06
    axins.text(0.25, y_off, unit_text, transform=axins.transAxes,
           horizontalalignment='left', verticalalignment='center')

    t2 = timer()

    if ds is not None and not ens_mode: # no uv for ens

        # print("Before dec")
        # streamplot_func = _add_transform_first_to_streamplot(ds.plot.streamplot)
        # print("After dec")
        ds.plot.streamplot(x='lon', y='lat', u='u', v='v', linewidth=0.6,
                   arrowsize = 0.5,
                   density=6,
                   color='black') # , transform_first=True not working, or is already implemented. still slow.
        
    t3 = timer()

    if tp is not None:
        # transform to mm
        tp.plot.contourf(levels=levels_rain, extend='max', subplot_kws={'transform_first': True},
            cmap=rain_cm, norm=norm, add_colorbar=False)

    t4 = timer()

    # generate plot per pressure level, per time step

    for obj_idx, node in enumerate(wts_part_of_tracks):
        line_pts = node.feature.properties.line_pts
        line = patches.Path([[p.lon, p.lat] for p in line_pts])
        if ens_mode: # ensemble way thinner
            patch = patches.PathPatch(line, linewidth=1, facecolor='none', edgecolor='crimson')
        else:
            patch = patches.PathPatch(line, linewidth=3, facecolor='none', edgecolor='crimson') # cmap(time_weight)
        ax.add_patch(patch)

    t5 = timer()

    # ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=list(np.array([255, 225, 171])/255))

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax.set_title("")
    fig.tight_layout()

    plt.savefig(fig_name, format='png', backend='agg')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    crop_top_bottom_whitespace(fig_name)

    t6 = timer()
    """
    print("Init: " + str(t2 - t1))
    print("Streamplot: " + str(t3 - t2))
    print("Rain: " + str(t4 - t3))
    print("Wavetroughs: " + str(t5 - t4))
    print("Finalize: " + str(t6 - t5))

    print("Saved to " +  fig_name)
    exit()
    """
    print("Saved to " +  fig_name)
    return


"""
## MAIN PLOTTING FUNC FOR KITWEATHER PLOTS - ENSEMBLE MODE
def plot_ts_filtered_waves(wts_part_of_tracks, fig_name, ds=None, tp=None):
    
    from timeit import default_timer as timer
    t1 = timer()

    resolution = 1600
    cbar_space_px = 80
    subplotparameters = mpl.figure.SubplotParams(left=0, bottom=0, right=1 - cbar_space_px / resolution, top=1,
                                             wspace=0, hspace=0)
    fig, ax = plt.subplots(figsize=(resolution / 100, resolution / 100),
                       dpi=100,
                       subplotpars=subplotparameters,
                       subplot_kw=dict(projection = ccrs.PlateCarree()))

    extent = [-100, 35, -10, 35]


    levels_rain, rain_cm, norm = get_kitweather_rain_cm()
    distance_plot_to_cbar = 0.010
    axins = ax.inset_axes([1 + distance_plot_to_cbar, 0.05, 0.015, 0.93],
                      transform=ax.transAxes)
    ticks_list = levels_rain
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=rain_cm, norm=norm),
                    cax=axins, extend='both', extendfrac=0.03,
                    ticks=ticks_list)
    axins.text(0.5, -0.06, 'mm/hr', transform=axins.transAxes,
           horizontalalignment='left', verticalalignment='center')

    t2 = timer()

    if ds is not None:

        # print("Before dec")
        # streamplot_func = _add_transform_first_to_streamplot(ds.plot.streamplot)
        # print("After dec")
        ds.plot.streamplot(x='lon', y='lat', u='u', v='v', linewidth=0.6,
                   arrowsize = 0.5,
                   density=6,
                   color='black') # , transform_first=True not working, or is already implemented. still slow.
        
    t3 = timer()

    if tp is not None:
        # transform to mm
        tp.plot.contourf(levels=levels_rain, extend='max', subplot_kws={'transform_first': True},
            cmap=rain_cm, norm=norm, add_colorbar=False)

    t4 = timer()

    # generate plot per pressure level, per time step

    for obj_idx, node in enumerate(wts_part_of_tracks):
        line_pts = node.feature.properties.line_pts
        line = patches.Path([[p.lon, p.lat] for p in line_pts])
        patch = patches.PathPatch(line, linewidth=3, facecolor='none', edgecolor='crimson') # cmap(time_weight)
        ax.add_patch(patch)

    t5 = timer()

    # ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=list(np.array([255, 225, 171])/255))

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax.set_title("")
    fig.tight_layout()

    plt.savefig(fig_name, format='png', backend='agg')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    crop_top_bottom_whitespace(fig_name)

    t6 = timer()
    print("Saved to " +  fig_name)
    return
"""

# plots the wave state (all wavetroughs given specific timestep in a set) ts: pb2.Timestep
def plot_wavetroughs(ts, fig_name, cv=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 4), subplot_kw=dict(projection=ccrs.PlateCarree()))

    x_ticks = [-100, -95, -85, -75, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35]
    y_ticks = [0, 10, 20, 30]
    extent = [-100, -45, -10, 35]

    if cv is not None:
        levelfc = np.asarray([0, 0.5, 1, 2, 3]) * 1e-5
        cv.plot.contourf(levels=levelfc, vmin=0, extend='max', cmap='Blues')

    # generate plot per pressure level, per time step

    # colors per time step
    # min_time = wave_thr_list[0].time.astype('float64')
    # max_time = wave_thr_list[-1].time.astype('float64')
    # cmap = matplotlib.cm.get_cmap('rainbow')
    # color_wgts = np.linspace(0.0, 1.0, len(wave_thr_list))
    # colors = ['red', 'yellow', 'green', 'blue', 'purple']

    vt = ts.time
    for obj_idx, obj in enumerate(ts.features):
        # time64 = wave.time.astype('float64')
        # time_weight = (time64 - min_time) / (max_time - min_time) if max_time > min_time else 1.0

        line_pts = obj.properties.line_pts
        line = patches.Path([[p.lon, p.lat] for p in line_pts])
        patch = patches.PathPatch(line, linewidth=2, facecolor='none', edgecolor='red') # cmap(time_weight)
        ax.add_patch(patch)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    yt1 = ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
    xt1 = ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())

    figure_name = fig_name.replace(':', '_') + '_aew_troughs.png'
    plt.savefig(figure_name, format='png')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    return figure_name


def plot_timesteps_from_desc(feature_desc, cv=None):
    # plot for each set for each timestep everything detected.
    from feature.util.data_utils import get_subset_by_description

    for set_idx, od_set in enumerate(feature_desc.sets):
        fn = cfg.plot_dir + "ts_set_" + str(set_idx)

        cv_set = get_subset_by_description(cv, od_set, '2d')

        for ts in od_set.timesteps:
            cv_st = cv_set.sel(time=ts.time).cv
            fnt = fn + "_" + ts.time
            print(fnt)
            # ts.validTime / .features
            fout_name = plot_wavetroughs(ts, fnt, cv=cv_st)
            print("Plot to " + fout_name)

    return None

def plot_track(track, fn):

    nodes = [edge.parent for edge in track.edges]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 4), subplot_kw=dict(projection=ccrs.PlateCarree()))

    x_ticks = [-100, -95, -85, -75, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35]
    y_ticks = [0, 10, 20, 30]
    extent = [-100, -45, -10, 35]

    # generate plot per pressure level, per time step

    # colors per time step

    min_time = pb_str_to_datetime(nodes[0].time).timestamp()
    max_time = pb_str_to_datetime(nodes[-1].time).timestamp()
    cmap = matplotlib.cm.get_cmap('rainbow')
    color_wgts = np.linspace(0.0, 1.0, len(nodes))
    colors = ['red', 'yellow', 'green', 'blue', 'purple']

    for node_idx, node in enumerate(nodes):
        obj = node.feature
        time_d = pb_str_to_datetime(node.time).timestamp()
        time_weight = (time_d - min_time) / (max_time - min_time) if max_time > min_time else 1.0

        line_pts = obj.properties.line_pts
        line = patches.Path([[p.lon, p.lat] for p in line_pts])
        patch = patches.PathPatch(line, linewidth=2, facecolor='none', edgecolor=cmap(time_weight))
        ax.add_patch(patch)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    yt1 = ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
    xt1 = ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())


    figure_name = cfg.plot_dir + fn  + '.png' # .replace(':', '_')
    plt.title(nodes[0].time + " - " + nodes[-1].time)
    print("Plot to " + str(figure_name))
    plt.savefig(figure_name, format='png')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    return figure_name


from collections import defaultdict
def plot_differences(set_graph, tracks, ds=None, tp=None, plot_prefix=None):
    print("plot_differences() deprecated")
    exit() # OLD FUNC.
    # plot the differences of the total graph and the tracks
    # so check which WTs are part of a track and which have been dropped.

    set_nodes = [e.parent for e in set_graph.graph.edges]
    is_in_set_nodes = [False] * len(set_nodes)

    for track in tracks:
        track_nodes = [e.parent for e in track.edges]

        for track_node in track_nodes:
            try:
                # track node in list, set to True
                idx = set_nodes.index(track_node)
                is_in_set_nodes[idx] = True
            except ValueError:
                # not in it
                continue

    # lists which WTs are part of tracks, and which are not part of tracks
    wts_in_tracks_list = [set_nodes[i] for i, b in enumerate(is_in_set_nodes) if b]
    wts_not_in_tracks_list = [set_nodes[i] for i, b in enumerate(is_in_set_nodes) if not b]

    # make these lists to dicts with date as key
    wts_in_tracks = defaultdict(list)
    for wt_in_track in wts_in_tracks_list:
        wts_in_tracks[wt_in_track.time].append(wt_in_track)
    wts_not_in_tracks = defaultdict(list)
    for wt_not_in_track in wts_not_in_tracks_list:
        wts_not_in_tracks[wt_not_in_track.time].append(wt_not_in_track)

    dates = set()
    dates.update(wts_in_tracks.keys())
    dates.update(wts_not_in_tracks.keys())
    dates_list = list(dates)
    dates_list.sort()

    if plot_prefix is None:
        plot_prefix = cfg.plot_dir
    # create subdirs if needed
    plot_dir = '/'.join(plot_prefix.split('/')[:-1]) + '/'
    os.makedirs(plot_dir, exist_ok=True)

    for date in dates_list:
        fig_name = plot_prefix + date[0:4] + date[5:7] + date[8:10] + "T" + date[11:13] + ".png"
        try:
            ds_ss = ds.sel(time=date)
        except (KeyError, AttributeError) as e:
            print("No ds data for " + str(date))
            ds_ss = None
        try:
            tp_ss = tp.sel(time=date).tp
        except (KeyError, AttributeError) as e:
            print("No rain data for " + str(date))
            tp_ss = None
        plot_ts_filtered_waves(wts_in_tracks[date], fig_name, ds=ds_ss, tp=tp_ss) # wts_not_in_tracks[date], 
        # except KeyError:
        #     print("No rain data for " + str(date))
            
        #     tp_ss = None
            # plot_ts_part_of_track(wts_in_tracks[date], wts_not_in_tracks[date], fig_name, ds=ds_ss, tp=tp_ss)


import pandas as pd
def plot_kw(tracks, ds, tp=None, plot_prefix=None, ens_mode=False):

    dates_dt = pd.to_datetime(ds.time.values)

    wavetroughs_in_tracks = dict() # by time
    for date in dates_dt:
        wavetroughs_in_tracks[date] = []

    # put nodes of all tracks in buckets by time
    for track in tracks:
        track_nodes = [e.parent for e in track.edges]
        for track_node in track_nodes:
            
            track_node_time = pb_str_to_datetime(track_node.time)
            wavetroughs_in_tracks[track_node_time].append(track_node)

    if plot_prefix is None:
        plot_prefix = cfg.plot_dir
        # create subdirs if needed
    else:
        plot_dir = '/'.join(plot_prefix.split('/')[:-1]) + '/'
    os.makedirs(plot_dir, exist_ok=True)

    if ens_mode and tp is not None:
        tp_thr = (tp > cfg.ens_rain_threshold).astype(dtype=float)
        tp_prob = tp_thr.mean(dim="member")
        tp = tp_prob

    # call plotting for each date
    for date in dates_dt:
        fig_name = plot_prefix + date.strftime("%Y%m%dT%H") + ".png"

        ds_ss = ds.sel(time=date).squeeze()
        try:
            tp_ss = tp.sel(time=date)

        except (KeyError, AttributeError) as e:
            print("No rain data for " + str(date))
            tp_ss = None
        plot_ts_filtered_waves(wavetroughs_in_tracks[date], fig_name, ds=ds_ss, tp=tp_ss, ens_mode=ens_mode)


def plot_track_from_graph(track_desc, fig_name_prefix, cv=None):

    nodes = [edge.parent for edge in track_desc.edges]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 4), subplot_kw=dict(projection=ccrs.PlateCarree()))

    x_ticks = [-100, -95, -85, -75, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35]
    y_ticks = [0, 10, 20, 30]
    extent = [-100, -45, -10, 35]

    if cv is not None:
        cv = cv.isel(time=0)
        levelfc = np.asarray([0, 0.5, 1, 2, 3]) * 1e-5
        cv.plot.contourf(levels=levelfc, vmin=0, extend='max', cmap='Blues')

    # generate plot per pressure level, per time step

    # colors per time step

    min_time = pb_str_to_datetime(nodes[0].time).timestamp()
    max_time = pb_str_to_datetime(nodes[-1].time).timestamp()
    cmap = matplotlib.cm.get_cmap('rainbow')
    color_wgts = np.linspace(0.0, 1.0, len(nodes))
    colors = ['red', 'yellow', 'green', 'blue', 'purple']

    for node_idx, node in enumerate(nodes):
        obj = node.feature
        time_d = pb_str_to_datetime(node.time).timestamp()
        time_weight = (time_d - min_time) / (max_time - min_time) if max_time > min_time else 1.0

        line_pts = obj.properties.line_pts
        line = patches.Path([[p.lon, p.lat] for p in line_pts])
        patch = patches.PathPatch(line, linewidth=2, facecolor='none', edgecolor=cmap(time_weight))
        ax.add_patch(patch)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    yt1 = ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
    xt1 = ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())

    figure_name = fig_name_prefix  + '_troughs.png' # .replace(':', '_')
    plt.title(nodes[0].time + " - " + nodes[-1].time)
    print("Plot to " + str(figure_name))
    plt.savefig(figure_name, format='png')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    return figure_name

def plot_wt_list(nodes, fn):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 4), subplot_kw=dict(projection=ccrs.PlateCarree()))

    x_ticks = [-100, -95, -85, -75, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35]
    y_ticks = [0, 10, 20, 30]
    extent = [-100, -45, -10, 35]

    # generate plot per pressure level, per time step

    # colors per time step
    # min_time = wave_thr_list[0].time.astype('float64')
    # max_time = wave_thr_list[-1].time.astype('float64')
    # cmap = matplotlib.cm.get_cmap('rainbow')
    # color_wgts = np.linspace(0.0, 1.0, len(wave_thr_list))
    # colors = ['red', 'yellow', 'green', 'blue', 'purple']

    for node in nodes:
        line_pts = node.feature.properties.line_pts
        line = patches.Path([[p.lon, p.lat] for p in line_pts])
        patch = patches.PathPatch(line, linewidth=2, facecolor='none', edgecolor='red')  # cmap(time_weight)
        ax.add_patch(patch)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    yt1 = ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
    xt1 = ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())

    figure_name = fn + '.png'
    plt.savefig(figure_name, format='png')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    return figure_name

def plot_track_in_ts(track):
    
    fn = cfg.plot_dir + 'singletrack_'

    per_ts_wts = dict()
    for edge in track.edges:
        node = edge.parent
        key = node.time.replace(':', '_')
        if key in per_ts_wts:
            per_ts_wts[key].append(node)
        else:
            per_ts_wts[key] = [node]

    for time, nodes in per_ts_wts.items():
        plot_wt_list(nodes, fn + time)


def plot_tracks_from_desc(graph_desc, ds=None):

    from feature.util.data_utils import get_subset_by_description

    for set_idx, od_set in enumerate(graph_desc.sets):

        # cv_set = get_subset_by_description(ds, od_set, '2d')
        for set_tr, track in enumerate(od_set.tracks):
            fn = cfg.plot_dir + 'set_' + str(set_idx) + "_track_" + str(set_tr)
            plot_track_from_graph(track, fn, cv=None)

    pass

