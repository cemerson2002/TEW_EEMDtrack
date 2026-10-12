from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib.path as mpath
import matplotlib.colors
import copy
import matplotlib.colors as mcolors

# some misc plotting routines...

### HEMISPHERE CIRCLE
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

### READ PROJECTION INFOS
from .projection_cdo import read_grid_config

grid_config = read_grid_config()
dx, dy = int(grid_config['dummy_section']['xinc']) * 1e3, int(grid_config['dummy_section']['yinc']) * 1e3  # meters
xstart, ystart = int(grid_config['dummy_section']['xfirst']) * 1e3, int(
    grid_config['dummy_section']['yfirst']) * 1e3  # meters
x = np.arange(xstart, -xstart, dx)
y = np.arange(ystart, -ystart, dy)

### COLORMAPS
binary_grey = matplotlib.colors.ListedColormap(['white', 'grey'])

### PLOTTING
plt_extent = [xstart * 0.7, -xstart * 0.7, ystart * 0.7, -ystart * 0.7]
fs = (10, 8)
d_cmap = plt.cm.cividis


def plot_pvu2_binary(ds: xr.Dataset, config, level=0):
    if config.vertical_dim is not None:
        ds_2d = ds.isel(dict({config.vertical_dim: level, "time": 0}))
    else:
        ds_2d = ds.isel(time=0)
    fig = plt.figure(figsize=fs)

    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=1)
    ax.gridlines()
    ax.set_extent(plt_extent, ccrs.NorthPolarStereo())

    pv = ds_2d.variables['pvu2'][:]
    pvu2intvar = pv.astype(dtype=int)

    plt.contour(x, y, pvu2intvar, [0.5], colors='black', transform=ccrs.NorthPolarStereo(), linewidths=2)
    plt.contourf(x, y, pv, 1, cmap=binary_grey, transform=ccrs.NorthPolarStereo())

    # plt.show()
    plt.savefig("step1.png")


def plot_pvu2_pp(ds: xr.Dataset, pp_field, config, level=0):
    pp_field = pp_field[level]
    if config.vertical_dim is not None:
        ds_2d = ds.isel(dict({config.vertical_dim: level, "time": 0}))
    else:
        ds_2d = ds.isel(time=0)
    fig = plt.figure(figsize=fs)

    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=1)
    ax.gridlines()
    ax.set_extent(plt_extent, ccrs.NorthPolarStereo())

    pv = ds_2d.variables['pvu2'][:]
    pvu2intvar = pp_field.astype(dtype=int)

    plt.contour(x, y, pvu2intvar, [0.5], colors='black', transform=ccrs.NorthPolarStereo(), linewidths=2)
    plt.contourf(x, y, pp_field, 1, cmap=binary_grey, transform=ccrs.NorthPolarStereo())

    # plt.show()
    plt.savefig("step1a.png")


def plot_erosion(ds: xr.Dataset, dist_field, config, level=0):
    dist_field = dist_field[level]
    fig = plt.figure(figsize=fs)

    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=1)
    ax.gridlines()
    ax.set_extent(plt_extent, ccrs.NorthPolarStereo())

    # ax.pcolormesh(x, y, pv, cmap=binary_grey, transform=ccrs.NorthPolarStereo())

    CS = plt.contour(x, y, dist_field, [1000], colors='red', transform=ccrs.NorthPolarStereo(), linewidths=2)

    cmap = copy.copy(d_cmap)
    a = np.ma.masked_where(dist_field < 0.05, dist_field)
    cmap.set_bad(color='white')

    plt.contourf(x, y, a, [0, 500, 1000, 1500, 2000, 2500, 3000, 4000], cmap=cmap, transform=ccrs.NorthPolarStereo(),
                 norm=mcolors.PowerNorm(0.8, vmin=0, vmax=4000))  #

    plt.colorbar()
    plt.show()
    plt.savefig("step2.png")


def plot_erosion_animate(ds: xr.Dataset, dist_field, config, level=0):
    dist_field = dist_field[level]
    fig = plt.figure(figsize=fs)

    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=1)
    ax.gridlines()
    ax.set_extent(plt_extent, ccrs.NorthPolarStereo())

    # ax.pcolormesh(x, y, pv, cmap=binary_grey, transform=ccrs.NorthPolarStereo())

    cmap = copy.copy(d_cmap)
    a = np.ma.masked_where(dist_field < 0.05, dist_field)
    cmap.set_bad(color='white')

    from matplotlib import animation
    def animate(i):
        cur_field = np.ma.copy(a)
        cur_field = np.ma.masked_where(cur_field > 50 * i, cur_field)

        cf = ax.contourf(x, y, cur_field, [0, 500, 1000, 1500, 2000, 2500, 3000, 4000], cmap=cmap,
                         transform=ccrs.NorthPolarStereo(), norm=mcolors.PowerNorm(0.8, vmin=0, vmax=4000))  #
        cblack = ax.contour(x, y, dist_field, [0.5], colors='black', transform=ccrs.NorthPolarStereo(), linewidths=2)
        if 50 * i > 1000:
            cred = plt.contour(x, y, dist_field, [1000], colors='red', transform=ccrs.NorthPolarStereo(), linewidths=2)
            return cf, cblack, cred
        # plt.colorbar()
        else:
            return cf, cblack

    # plt.colorbar()
    # call the animator.  blit=True means only re-draw the parts that have changed.

    anim = animation.FuncAnimation(fig, animate,  # init_func=init,
                                   frames=80, interval=300, blit=False)

    anim.save('basic_animation.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    # plt.show()
    exit()
    plt.savefig("step2.png")


def plot_inner_binary(ds: xr.Dataset, inner_field, config, level=0):
    inner_field = inner_field[level]
    if config.vertical_dim is not None:
        ds_2d = ds.isel(dict({config.vertical_dim: level, "time": 0}))
    else:
        ds_2d = ds.isel(time=0)
    fig = plt.figure(figsize=fs)

    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=1)
    ax.gridlines()
    ax.set_extent(plt_extent, ccrs.NorthPolarStereo())

    pv = ds_2d.variables['pvu2'][:]
    pvu2intvar = inner_field.astype(dtype=int)

    plt.contour(x, y, pvu2intvar, [0.5], colors='black', transform=ccrs.NorthPolarStereo(), linewidths=2)
    plt.contourf(x, y, pv, 1, cmap=binary_grey, transform=ccrs.NorthPolarStereo())

    # plt.show()
    plt.savefig("step3.png")


def plot_dilation(ds: xr.Dataset, dist_field, config, level=0):
    dist_field = dist_field[level]
    fig = plt.figure(figsize=fs)

    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=1)
    ax.gridlines()
    ax.set_extent(plt_extent, ccrs.NorthPolarStereo())

    # ax.pcolormesh(x, y, pv, cmap=binary_grey, transform=ccrs.NorthPolarStereo())

    CS = plt.contour(x, y, dist_field, [0.5, 1000], colors=['black', 'red'], transform=ccrs.NorthPolarStereo(),
                     linewidths=2)
    plt.contourf(x, y, dist_field, [0, 500, 1000, 1500, 2000, 2500, 3000], cmap=d_cmap,
                 transform=ccrs.NorthPolarStereo(), norm=mcolors.PowerNorm(0.8, vmin=0, vmax=4000))  #

    plt.colorbar()
    plt.show()
    plt.savefig("step4.png")


def plot_dilation_animate(ds: xr.Dataset, dist_field, config, level=0):
    if config.vertical_dim is not None:
        ds_2d = ds.isel(dict({config.vertical_dim: level, "time": 0}))
    else:
        ds_2d = ds.isel(time=0)
    pv = ds_2d.variables['pvu2'][:]
    pvu2intvar = pv.astype(dtype=int)

    dist_field = dist_field[level]
    fig = plt.figure(figsize=fs)

    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=1)
    ax.gridlines()
    ax.set_extent(plt_extent, ccrs.NorthPolarStereo())

    cmap = copy.copy(d_cmap)
    a = np.ma.masked_where(dist_field < 0.05, dist_field)
    cmap.set_bad(color='grey')

    from matplotlib import animation
    def animate(i):
        cur_field = np.ma.copy(a)
        cur_field = np.ma.masked_where(cur_field > 50 * i, cur_field)

        cf = ax.contourf(x, y, cur_field, [0, 500, 1000, 1500, 2000, 2500, 3000, 5000], cmap=cmap,
                         transform=ccrs.NorthPolarStereo(), norm=mcolors.PowerNorm(0.8, vmin=0, vmax=5000))  #
        cblack = ax.contour(x, y, dist_field, [0.5], colors='black', transform=ccrs.NorthPolarStereo(), linewidths=2)
        cblack2 = ax.contour(x, y, pvu2intvar, [0.5], colors='black', transform=ccrs.NorthPolarStereo(), linewidths=1)
        if 50 * i > 1000:
            cred = plt.contour(x, y, cur_field, [1000], colors='red', transform=ccrs.NorthPolarStereo(), linewidths=2)
            return cf, cblack, cblack2, cred
        # plt.colorbar()
        else:
            return cf, cblack, cblack2

    anim = animation.FuncAnimation(fig, animate,  # init_func=init,
                                   frames=100, interval=300, blit=False)
    # plt.show()
    anim.save('basic_animation_2.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    exit()
    #
    # plt.show()
    plt.savefig("step4.png")


### this is used for publication.
def plot_streamer_areas(ds: xr.Dataset, dist_field, streamer_areas, config, level=0):
    streamer_areas = streamer_areas[level]
    dist_field = dist_field[level]
    if config.vertical_dim is not None:
        ds_2d = ds.isel(dict({config.vertical_dim: level}))

    if "time" in ds.dims:
        ds_2d = ds_2d.isel(time=0)

    fig = plt.figure(figsize=fs)

    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=1)
    ax.gridlines()
    ax.set_extent(plt_extent, ccrs.NorthPolarStereo())

    pv = ds_2d.variables['pvu2'][:]
    pvu2intvar = pv.astype(dtype=int)

    # CS = plt.contour(x, y, dist_field, [1000], colors=['black'], transform=ccrs.NorthPolarStereo(), linewidths=2)
    plt.contour(x, y, pvu2intvar, [0.5], colors='black', transform=ccrs.NorthPolarStereo(), linewidths=2)
    plt.contourf(x, y, streamer_areas, 1, cmap=binary_grey, transform=ccrs.NorthPolarStereo())

    plt.savefig("step5.png")

def generate_2d_plot(subset, fig_name):

    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=1)
    ax.gridlines()
    ax.set_extent(plt_extent, ccrs.NorthPolarStereo())

    pv = subset.variables['pv'][:]
    pvu2intvar = (pv > 2e-6).astype(dtype=int)

    # CS = plt.contour(x, y, dist_field, [1000], colors=['black'], transform=ccrs.NorthPolarStereo(), linewidths=2)
    plt.contour(x, y, pvu2intvar, [0.5], colors='black', transform=ccrs.NorthPolarStereo(), linewidths=2)
    plt.contourf(x, y, subset.streamer, [0.5, 1000], cmap=binary_grey, transform=ccrs.NorthPolarStereo())

    print("Saving " + fig_name)
    plt.savefig(fig_name + '.png')



def generate_2d_plots(ds: xr.Dataset):

    if 'member' in ds.coords:
        members = ds.coords['member'].values
        if not isinstance(members, np.ndarray):
            members = [members]
    else:
        members = [-1]
    if 'time' in ds.coords:
        times = ds.coords['time'].values
        if not isinstance(times, np.ndarray):
            times = [times]
    else:
        times = [-1]

    for member in members:
        for ts in times:
            figname = ''
            if member != -1:
                subset = ds.sel({'member': member, 'time': ts})
                figname += 'mem_' + str(int(member)) + '_time_' + str(ts)[:13]
            else:
                subset = ds.sel({'time': ts})
                figname += 'time_' + str(ts)[:13]

            ss_sq = subset.squeeze()
            generate_2d_plot(ss_sq, figname)


