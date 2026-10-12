
import numpy as np
import xarray as xr
import math, json
from feature.util.enstools_utils import get_u_var, get_v_var, get_vertical_dim, get_longitude_dim, \
    get_latitude_dim
from skimage.draw import line
from feature.util.data_utils import simplenamespace_to_proto, clip, pb_str_to_datetime64, datetime64_to_pb_str, proto_to_json, proto_to_simplenamespace, pb_str_to_datetime
from types import SimpleNamespace
from google.protobuf import json_format
from scipy.signal import butter, lfilter, sosfiltfilt


# calculates the dx's and dy's for each grid cell
# takes list of latitudes and longitudes as input and returns field with dimensions len(lats) x len(lons)
def calculate_dx_dy(lats, lons):
    lat_res = lats[1] - lats[0]
    lon_res = lons[1] - lons[0]
    nlat = len(lats)
    nlon = len(lons)
    lon_percent = (lon_res * nlon) / 360.0  # percentage of global longitudes. e.g. if 110W to 70E -> 0.5
    lat_percent = (lat_res * nlat) / 180.0  # percentage of global latitudes
    Re = 6378100

    # field filled with the latitude value at each grid point
    lat_f = np.tile(lats, (nlon, 1)).transpose()
    # field filled with the longitude value at each grid point
    lon_f = np.tile(lons, (nlat, 1))
    # dx: circumference at current latitude  *  percentage of longitudes in dataset  /  amount of lon grid points
    dx = np.cos(lat_f * math.pi / 180.0) * 2.0 * math.pi * Re * lon_percent / nlon
    # dy: constant everywhere: half circumference  *  latitude percentage /  amount of lat points
    dy = (lat_f * 0) + lat_percent * math.pi * Re / nlat

    return dx, dy


def compute_vort(dataset, vtype, u_str, v_str, field_str):
    xr.set_options(keep_attrs=True)  # assign coords keeps attributes

    lon_str = get_longitude_dim(dataset)
    lat_str = get_latitude_dim(dataset)

    lats = dataset.coords[lat_str].data
    lons = dataset.coords[lon_str].data

    # be careful if your data is non-continuous by the chosen window (e.g. +120E..-120W)
    # reorder axis for efficient numpy broadcasting
    dataset = dataset.sortby(lat_str)
    dataset = dataset.transpose(..., lat_str, lon_str)

    dataset[field_str] = xr.zeros_like(dataset[u_str], dtype=float)

    # calculate dx and dy for each cell in grid in meters
    dx, dy = calculate_dx_dy(lats, lons)

    # Relative Vorticity = dv/dx - du/dy, use central differences
    u_arr = dataset[u_str].data
    v_arr = dataset[v_str].data
    ndims = u_arr.ndim

    # roll axes so we have a reference to each cell's neighbours
    # ax[-1] = lon, ax[-2] = lat
    # works as expected if longitude band is full 360 degrees. Rolling does exactly that.
    v_xp = np.roll(v_arr, -1, axis=ndims - 1)  # roll -1 to get +1. v_xp = v(x+1)
    v_xm = np.roll(v_arr, 1, axis=ndims - 1)
    v_yp = np.roll(v_arr, -1, axis=ndims - 2)
    v_ym = np.roll(v_arr, 1, axis=ndims - 2)

    u_xp = np.roll(u_arr, -1, axis=ndims - 1)  # roll -1 to get +1
    u_xm = np.roll(u_arr, 1, axis=ndims - 1)
    u_yp = np.roll(u_arr, -1, axis=ndims - 2)
    u_ym = np.roll(u_arr, 1, axis=ndims - 2)

    # central differences:  (V(x+1)-V(x-1)) / 2x - (U(y+1)-U(y-1)) / 2y
    RV = ((v_xp - v_xm) / (2 * dx)) - ((u_yp - u_ym) / (2 * dy))
    # results verified to ERA5 vo-data
    if vtype == 'relative':
        dataset[field_str].values = RV
        dataset[field_str].attrs = {'long_name': "Relative Vorticity", 'units': "s**-1"}
        dataset[field_str] = dataset[field_str].chunk({lat_str: len(lats), lon_str: len(lons)})
        return dataset


    # split into shear + curvature: RV = -dV/dn + V/R
    # shear vorticity:
    # rate of change of wind speed in the direction of flow
    # -dV/dn

    # wind direction of this cell
    ref_angle = np.arctan2(v_arr, u_arr)

    ### METHOD 1 ### from analytical standpoint

    sin_angle = np.sin(ref_angle)
    cos_angle = np.cos(ref_angle)
    # here is where the magic happens...
    # to cartesian: dV/dn = - dV/dx * sin(phi) + dV/dy * cos(phi) ## n is normal to V, split it into x,y respecting the
    # n-direction, where phi is the rotation angle of the natural coordinate system, this also is the angle of the
    # wind vector
    #
    # we get the magnitude of shear as the projection of dV/dn onto the vector itself (direction e_t)
    # with e_t = cos(phi)*e_x + sin(phi)*e_y , so:
    # dV/dn * e_t = du/dx * (-sin*cos) + du/dy cos^2 + dv/dx *(-sin^2) + dv/dy (cos*sin)
    SV = (u_xp - u_xm) / (2 * dx) * (-sin_angle * cos_angle) + \
         (u_yp - u_ym) / (2 * dy) * (cos_angle ** 2) + \
         (v_xp - v_xm) / (2 * dx) * (-(sin_angle ** 2)) + \
         (v_yp - v_ym) / (2 * dy) * (sin_angle * cos_angle)
    SV = -SV  # -dV/dn
    if vtype == 'shear':
        dataset[field_str].values = SV
        dataset[field_str].attrs = {'long_name': "Shear Vorticity", 'units': "s**-1"}
        dataset[field_str] = dataset[field_str].chunk({lat_str: len(lats), lon_str: len(lons)})
        return dataset

    # remainder is CV
    CV = RV - SV
    if vtype == 'curvature':
        dataset[field_str].values = CV
        dataset[field_str].attrs = {'long_name': "Curvature Vorticity", 'units': "s**-1"}
        dataset[field_str] = dataset[field_str].chunk({lat_str: len(lats), lon_str: len(lons)})
        return dataset
    return None


def compute_field(dataset, u_str, v_str, field_name):

    if field_name in ['cv', 'cva', 'stormrel_cv', 'stormrel_cva']:
        dataset = compute_vort(dataset, 'curvature', u_str, v_str, 'cv')
    elif field_name in ['rv', 'rva']:
        dataset = compute_vort(dataset, 'relative', u_str, v_str, 'rv')
    else:
        print("NYI")
        exit(1)

    if field_name in ['rva', 'cva', 'stormrel_cv', 'stormrel_cva']:
        print("NYI")
        exit(1)

    return dataset


def create_fake_wt_edges(edge, child_idx, pb_ref, pb_desc):
    # make multiple edges out of this one


    parent_node = edge.parent
    parent_node_pb = simplenamespace_to_proto(parent_node, pb_ref.GraphNode())
    child_node = edge.children[child_idx]
    child_node_pb = simplenamespace_to_proto(child_node, pb_ref.GraphNode())

    parent_node_time = pb_str_to_datetime64(parent_node.time)
    child_node_time = pb_str_to_datetime64(child_node.time)

    delt = child_node_time - parent_node_time
    if delt != np.timedelta64(12, 'h'):
        print("NO 12h abort")
        print(delt)
        exit(1)

    fake_node_time = parent_node_time + 0.5 * delt # assert only skip 1

    # create fake node as copy of parent
    fake_node = pb_ref.GraphNode() # json_format.Parse(json.dumps(parent_json), pb_ref.GraphNode(), ignore_unknown_fields=False)
    fake_node.time = datetime64_to_pb_str(fake_node_time)

    # id and flag False
    fake_node.feature.id = -1
    fake_node.feature.flag = False

    # set properties: bb as mean of parent and child
    parent_props = parent_node.feature.properties
    child_props = child_node.feature.properties
    fake_props = fake_node.feature.properties

    fake_props.bb.min.lat = (parent_props.bb.min.lat + child_props.bb.min.lat) / 2.0
    fake_props.bb.max.lat = (parent_props.bb.max.lat + child_props.bb.max.lat) / 2.0
    fake_props.bb.min.lon = (parent_props.bb.min.lon + child_props.bb.min.lon) / 2.0
    fake_props.bb.max.lon = (parent_props.bb.max.lon + child_props.bb.max.lon) / 2.0

    avg_lon = (fake_props.bb.min.lon + fake_props.bb.max.lon) / 2.0
    fake_props.line_pts.add()
    fake_props.line_pts[0].lat = fake_props.bb.min.lat
    fake_props.line_pts[0].lon = avg_lon
    fake_props.line_pts.add()
    fake_props.line_pts[1].lat = fake_props.bb.max.lat
    fake_props.line_pts[1].lon = avg_lon

    fake_props.length_deg = math.sqrt((fake_props.bb.max.lat - fake_props.bb.min.lat) ** 2 + (fake_props.bb.max.lon - fake_props.bb.min.lon) ** 2)

    # new edge:
    edge1 = pb_ref.GraphConnection()
    edge1.parent.CopyFrom(parent_node_pb)
    edge1.children.append(fake_node)

    edge2 = pb_ref.GraphConnection()
    edge2.parent.CopyFrom(fake_node)
    edge2.children.append(child_node_pb)

    return [edge1, edge2]

def interpolate_wavetroughs(json_desc, pb_ref, pb_desc, start_dt, end_dt):

    for set_ in json_desc.sets:

        for track in set_.tracks:
            # old_edges = []
            new_edges = []

            for edge in track.edges:
                parent_node = edge.parent
                parent_node_time = pb_str_to_datetime64(parent_node.time)
                if parent_node_time < start_dt or parent_node_time > end_dt:
                    continue
                if not hasattr(edge, 'children'):
                    continue

                for child_idx, child_node in enumerate(edge.children):
                    child_node_time = pb_str_to_datetime64(child_node.time)

                    if child_node_time - parent_node_time > np.timedelta64(6, 'h'):
                        print("Add fake WT at " + parent_node.time)
                        wt_edges = create_fake_wt_edges(edge, child_idx, pb_ref, pb_desc)
                        # old_edges.append((edge, child_idx))
                        new_edges.extend(wt_edges)


            # remove old_edges from this track and from graph

            new_edges_sn = [proto_to_simplenamespace(e) for e in new_edges]
            track.edges.extend(new_edges_sn)
            track.edges.sort(key=lambda item: item.parent.time)

            set_.graph.edges.extend(new_edges_sn)
            set_.graph.edges.sort(key=lambda item: item.parent.time)
            # print("EOT")
            # not to TS!

    return json_desc

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=-1):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data, axis=axis)
    return y


def bpf(dataset, data_vars, min_days=2, max_days=6): # 2-6day bpf as default
    time_res_hrs = (dataset.coords['time'].values[1] - dataset.coords['time'].values[0]).astype(
        'timedelta64[h]') / np.timedelta64(1, 'h')  # 6 if 6hrly
    time_res_days = time_res_hrs / 24.0  # 6/24 = 1/4
    
    for var in data_vars:
        print("Execute bpf for " + str(min_days) + " to " + str(max_days) + " days. Variable: " + var)
        var_filtered = butter_bandpass_filter(dataset[var].data, time_res_days / max_days, time_res_days / min_days, 1, axis=0)  # time is dim0
        var_bpf = xr.zeros_like(dataset[var])
        var_bpf.data = var_filtered
        dataset[var + '_bpf'] = var_bpf

    return dataset

def create_wt_troughs_and_areas(dataset, in_json, circles, use_fc, d=500000): # 500 km?
    print("Create WT lines...")

    if use_fc:
        print("Forecast data not supported")
        exit(1)
    else:
        dataset['wt_rea'] = xr.zeros_like(dataset.u.isel(level=0).squeeze(), dtype=int)
        dataset['infl_rea'] = xr.zeros_like(dataset.u.isel(level=0).squeeze(), dtype=int)

        #  for ts in ana_set.timesteps:

    min_lat = dataset.latitude.data.min()
    max_lat = dataset.latitude.data.max()
    min_lon = dataset.longitude.data.min()
    max_lon = dataset.longitude.data.max()
    lons = len(dataset.longitude.data)
    lats = len(dataset.latitude.data)

    wt_rea = dataset.wt_rea
    infl_rea = dataset.infl_rea

    for wt_set in in_json.sets:

        cur_set = wt_set

        if use_fc:
            initTime = wt_set.initTime
            set_ds = dataset.sel(time=initTime)  # init time
        else:
            set_ds = dataset

        for track in cur_set.tracks:
            print("Next Track")
            track_nodes = [e.parent for e in track.edges]

            for node in track_nodes:

                vt = node.time
                """
                try:
                if use_fc:
                    diff_time = pb_str_to_datetime(vt) - pb_str_to_datetime(initTime)
                    wt_da = set_ds.wt_fc.sel(step=diff_time)
                else:
                """
                try:
                    wt_da = wt_rea.sel(time=vt)
                    infl_da = infl_rea.sel(time=vt)
                except KeyError:
                    print("Skipping timestep (not in dataset) " + str(vt))
                    continue
                #except KeyError:
                #continue

                props = node.feature.properties
                if not hasattr(props, 'linePts'):
                    print("?")

                for v_idx in range(len(props.linePts) - 1):
                    start_lonlat = props.linePts[v_idx].lon, props.linePts[v_idx].lat
                    end_lonlat = props.linePts[v_idx + 1].lon, props.linePts[v_idx + 1].lat

                    start_idx = ((start_lonlat[0] - min_lon) / (max_lon - min_lon) * lons,
                                 (start_lonlat[1] - min_lat) / (max_lat - min_lat) * lats)
                    # start_idx = clip(start_idx, (0, 0), (lons, lats))

                    end_idx = ((end_lonlat[0] - min_lon) / (max_lon - min_lon) * lons,
                               (end_lonlat[1] - min_lat) / (max_lat - min_lat) * lats)
                    # end_idx = clip(end_idx, (0, 0), (lons, lats))

                    rr, cc = line(int(start_idx[0]), int(start_idx[1]), int(end_idx[0]), int(end_idx[1]))
                    rr = clip(rr, 0, lons - 1)
                    cc = clip(cc, 0, lats - 1)

                    wt_da.values[cc, rr] = node.feature.id

                    # make circle
                    for px_idx in range(len(rr)):

                        circle = circles.isel(longitude_center=rr[px_idx], latitude_center=cc[px_idx])
                        influence_area = circle.where(circle < d, -1)

                        # update influence area dataarray
                        infl_da = xr.where(influence_area >= 0, node.feature.id, infl_da)

                wt_rea.loc[dict(time=vt)] = wt_da.values
                infl_rea.loc[dict(time=vt)] = infl_da.values

    return dataset




"""
SWIRL Code
    rortex.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This code contains the routines needed to compute the
rortex criterion.
"""
# Imports
import numpy as np
# ----------------


def create_U(vx, vy, dx, dy, l):
    """
    This routine computes the velocity gradient tensor given
    a velocity field, the size of the grid cells and the stencil
    to be used.

    Parameters
    ----------
    v : 2D vector
        Velocity field
    dl : 2D vector
        Grid cell size
    l : int
        Stencil to use

    Returns
    -------
    U : array
        The array containing the velocity gradient tensors.

    Raises
    ------
    """
    # prepare U array
    nx = vx.shape[0]
    ny = vx.shape[1]
    U = np.zeros((2, 2, nx, ny))

    # Fill the velocity gradient tensor matrix.
    U[:, :, l:-l, l:-l] = np.array([[(vx[2*l:, l:-l] - vx[:-2*l, l:-l])/(2*dx*l),
                                     (vx[l:-l, 2*l:] - vx[l:-l, :-2*l])/(2*dy*l)],

                                    [(vy[2*l:, l:-l] - vy[:-2*l, l:-l])/(2*dx*l),
                                     (vy[l:-l, 2*l:] - vy[l:-l, :-2*l])/(2*dy*l)]
                                    ], dtype=np.float32)

    # Reorder axis to give it to linalg.eig
    U = np.moveaxis(U, [0, 1, 2, 3], [2, 3, 0, 1])

    return U

def compute_rortex(S, W, vx, vy, dx, dy, l):
    """
    Computes the rortex criterion given the swirling strength and
    the vorticity.

    Parameters
    ----------
    S : list of arrays
        List of swirling strength arrays
    W : list of arrays
        List of vorticity arrays

    Returns
    -------
    R : list of arrays
        List of rortex arrays

    Raises
    ------
    """
    """
    # Check if need to compute vorticity:
    if W is None:
        W, _ = compute_vorticity(v, dl, l)

    # Check if need to compute swirling strength:
    if S is None:
        S, _ = compute_swirlingstrength(v, dl, l, param)
    """
    # prepare R,U
    R = []
    U = []
    nx = vx.shape[0]
    ny = vx.shape[1]

    # loop over stencils:
    for i in np.arange(len(l)):

        # Prepare velocity gradient tensor instance
        Ui = np.zeros((nx, ny))

        # fill velocity gradient tensor:
        Ui = create_U(vx, vy, dx, dy, l[i])

        # get sign of rortex
        signR = np.sign(S)

        # compute total shear part
        R1 = W**2 - S**2
        mask = R1 < 0.0
        R1[mask] = 0.0

        # compute rortex
        R2 = np.abs((W*signR) + np.abs(np.sqrt(R1)))
        Ri = R2*signR

        # Clean velocity gradient tensors
        mask = (Ri == 0.0)
        Ui[mask] = np.array([[1., 0.], [0., 1.]])

        # append result
        R.append(Ri)
        U.append(Ui)

    return R, S, U, W


# https://stackoverflow.com/questions/3986345/how-to-find-the-local-minima-of-a-smooth-multidimensional-array-in-numpy-efficien
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
def detect_local_maxima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_max = (filters.maximum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_maxima = local_max ^ eroded_background
    return detected_maxima # np.where(detected_minima)