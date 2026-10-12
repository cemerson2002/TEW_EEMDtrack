import xarray as xr
import numpy as np
import json, os
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from PIL import Image
from datetime import datetime
from pathlib import Path
import feature.identification.aew_vortices.configuration as cfg
from feature.util.enstools_utils import get_u_var, get_v_var, get_vertical_dim, get_longitude_dim, get_latitude_dim

def get_kitweather_rain_cm(rain_cm_file):
    rgb_colors = []

    with open(rain_cm_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        rgb_colors.append([float(line[0:3])/255, float(line[4:7])/255, float(line[8:11])/255, 1])
    rgb_colors = [[1, 1, 1, 0]] + rgb_colors + [[0.35, 0, 0.4, 1]]
    cmap = mpl.colors.ListedColormap(rgb_colors[1:-1]) # , name=colorpalette
    cmap = cmap.with_extremes(bad='white', under=rgb_colors[0], over=rgb_colors[-1])

    levels = [0.1,0.2,0.3,0.5,1,2,3,5,10,20,30,50]
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    return levels, cmap, norm

def pb_str_to_datetime(time_str):
    return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

def get_track_wts_of_time(dt, all_nodes):
    nodes = [n for n in all_nodes if pb_str_to_datetime(n.time) == dt]
    return nodes

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


def plot_kw_style(dataset, set_, wt_set_, config, rea_wts_set=None, lev=700):
    dataset_rain = xr.load_dataset(cfg.in_data_rain) if (cfg.in_data_rain is not None and os.path.exists(cfg.in_data_rain)) else None
    dataset_fft = xr.load_dataset(cfg.in_data_fft) if (cfg.in_data_fft is not None and os.path.exists(cfg.in_data_fft)) else None
    
    if dataset_rain is not None:
        dataset_rain = dataset_rain.rename({'lon': 'longitude','lat':'latitude'}).transpose(...,'latitude','longitude')
    
    lv = lev
    level_str = get_vertical_dim(dataset)
    lon_str = get_longitude_dim(dataset)
    lat_str = get_latitude_dim(dataset)

    track_nodes = [e.parent for track in wt_set_.tracks for e in track.edges]
    if rea_wts_set:
        rea_track_nodes = [e.parent for track in rea_wts_set.tracks for e in track.edges]

    for tidx, t in enumerate(dataset.time.data):
        time_dt = datetime.utcfromtimestamp(t.astype(datetime) / 1e9)
        fig_name = time_dt.strftime(str(lv) + "_%Y%m%dT%H.png")

        # get WTs which are part of tracks for current ts
        wts = get_track_wts_of_time(time_dt, track_nodes)
        print(wts)
        if rea_wts_set:
            rea_wts = get_track_wts_of_time(time_dt, rea_track_nodes)
            print(rea_wts)
        
        ds_t = dataset.sel(time=t)
        if dataset_fft is not None:
            ds_fft_t = dataset_fft.sel(time=t)
        ds_rain_t = None
        if dataset_rain is not None:
            try:
                ds_rain_t = dataset_rain.sel(time=t)
            except:
                print("Time not in rain ds.")

        resolution = 1600
        cbar_space_px = 80
        subplotparameters = mpl.figure.SubplotParams(left=0, bottom=0, right=1 - cbar_space_px / resolution, top=1,
                                                 wspace=0, hspace=0)
        fig, ax = plt.subplots(figsize=(resolution / 100, resolution / 100),
                           dpi=100,
                           subplotpars=subplotparameters,
                           subplot_kw=dict(projection=ccrs.PlateCarree()))

        extent = [-75, 45, -10, 40]
        
        levels_rain, rain_cm, norm = get_kitweather_rain_cm(config.kw_rain_cm_file)
        distance_plot_to_cbar = 0.010
        axins = ax.inset_axes([1 + distance_plot_to_cbar, 0.05, 0.015, 0.93],
                          transform=ax.transAxes)
        ticks_list = levels_rain
        cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=rain_cm, norm=norm),
                        cax=axins, extend='both', extendfrac=0.03,
                        ticks=ticks_list)
        unit_text = 'mm/hr'
        y_off = -0.06
        axins.text(0.25, y_off, unit_text, transform=axins.transAxes,
               horizontalalignment='left', verticalalignment='center')

        # field contour
        levels = np.linspace(0, 1e-4, 50)

        # RelVort?
        # ds_t[config.field].sel(level=lv).plot.contourf(levels=levels, cmap='Blues', subplot_kws={'transform_first': True})

        """
        ds_t.sel(level=700).plot.streamplot(x='longitude', y='latitude',
                                            u=config.u_dim, v=config.v_dim,
                           linewidth=0.6,
                           arrowsize=0.5,
                           density=6,
                           color='blue')  # , transform_first=True not working, or is already implemented. still slow.
        """
        
        ds_t.sel({level_str: lv}).rv.plot.contourf(subplot_kws={'alpha': 0.5, 'transform_first': True}) # , add_colorbar=False

        if ds_rain_t is not None:
            ds_rain_t = ds_rain_t.sel(longitude=slice(-75.0,45.01), latitude=slice(-10,40.01))
            ds_rain_t.precipitationCal.plot.contourf(levels=levels_rain, extend='max', subplot_kws={'transform_first': True},
                         cmap=rain_cm, norm=norm, add_colorbar=False)
        else:
            print("No rain")
        
        if config.u_dim is not None:
            ds_t.sel({level_str: lv}).plot.streamplot(x=lon_str, y=lat_str,
                                            u=config.u_dim, v=config.v_dim,
                                            linewidth=0.3,
                                            arrowsize=0.6,
                                            density=6,
                                            color='red')
            

    # generate plot per pressure level, per time step

        for obj_idx, node in enumerate(wts):
            line_pts = node.feature.properties.linePts
            line = patches.Path([[p.lon, p.lat] for p in line_pts])

            if node.feature.id == -1:
                patch = patches.PathPatch(line, linewidth=3, facecolor='none', edgecolor='orange')  # cmap(time_weight)
            else:
                patch = patches.PathPatch(line, linewidth=3, facecolor='none', edgecolor='red')  # cmap(time_weight)

            ax.add_patch(patch)
        
        if rea_wts_set: # compare with rea -m mode
            for obj_idx, node in enumerate(rea_wts):
                line_pts = node.feature.properties.linePts
                line = patches.Path([[p.lon, p.lat] for p in line_pts])

                if node.feature.id == -1:
                    patch = patches.PathPatch(line, linewidth=3, facecolor='none', edgecolor='olive')  # cmap(time_weight)
                else:
                    patch = patches.PathPatch(line, linewidth=3, facecolor='none', edgecolor='darkolivegreen')  # cmap(time_weight)

                ax.add_patch(patch)

        # plot vortices
        # ds_t.sel(level=700).vortices.plot.contourf(levels=[-0.5,0.5,99], colors=('#00000000', 'blue'), subplot_kws={'transform_first': True}, add_colorbar=False)
        ds_t.sel({level_str: lv}).vortices.plot.contourf(levels=[-99,-10.5,-0.5,0.5,10.5,99], colors=('red', 'orange', '#00000000', 'cyan', 'blue'), subplot_kws={'transform_first': True}, add_colorbar=False)
        
        #ds_t.sel(level=lv).foc_c.plot.contourf(levels=[-0.5, 0.5, 99], colors=('#00000000', 'orange'), subplot_kws={'transform_first': True}, add_colorbar=False)
        #ds_t.sel(level=lv).foc.plot.contourf(levels=[-0.5, 0.5, 99], colors=('#00000000', 'yellow'), subplot_kws={'transform_first': True}, add_colorbar=False)
        #ds_t.sel(level=lv).node_i.plot.contourf(levels=[-0.5, 0.5, 99], colors=('#00000000', 'green'), subplot_kws={'transform_first': True}, add_colorbar=False)
        
        # saddles?
        # ds_t.sel(level=lv).node_o.plot.contourf(levels=[-0.5, 0.5, 99], colors=('#00000000', 'blue'), subplot_kws={'transform_first': True}, add_colorbar=False)
        
            
        # ax.coastlines()
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
        
        Path(config.fig_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(config.fig_dir + fig_name, format='png', backend='agg')

        plt.close(fig)
        crop_top_bottom_whitespace(config.fig_dir + fig_name)

        print("Saved to " + fig_name)
        



# for each timestep...