
import numpy as np
import xarray as xr
import math, json
from feature.util.enstools_utils import get_u_var, get_v_var, get_vertical_dim, get_longitude_dim, \
    get_latitude_dim
from skimage.draw import line
from feature.util.data_utils import simplenamespace_to_proto, clip, pb_str_to_datetime64, datetime64_to_pb_str, proto_to_json, proto_to_simplenamespace, pb_str_to_datetime
from skimage.measure import label
from types import SimpleNamespace
from google.protobuf import json_format
from scipy.signal import butter, lfilter, sosfiltfilt
from collections import defaultdict
from types import SimpleNamespace
from datetime import datetime
from scipy.spatial import ConvexHull
from tqdm import tqdm


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


def interpolate_wavetroughs(json_desc, pb_ref, pb_desc):

    for set_ in json_desc.sets:

        for track in set_.tracks:
            # old_edges = []
            new_edges = []

            for edge in track.edges:
                parent_node = edge.parent
                parent_node_time = pb_str_to_datetime64(parent_node.time)
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


def bpf(dataset, u_str, v_str, min_days=2, max_days=6): # 2-6day bpf as default
    time_res_hrs = (dataset.coords['time'].values[1] - dataset.coords['time'].values[0]).astype(
        'timedelta64[h]') / np.timedelta64(1, 'h')  # 6 if 6hrly
    time_res_days = time_res_hrs / 24.0  # 6/24 = 1/4
    tqdm.write("Execute bpf for " + str(min_days) + " to " + str(max_days) + " days.")

    u_filtered = butter_bandpass_filter(dataset[u_str].data, time_res_days / max_days, time_res_days / min_days, 1, axis=0)  # time is dim0
    v_filtered = butter_bandpass_filter(dataset[v_str].data, time_res_days / max_days, time_res_days / min_days, 1, axis=0)  # time is dim0

    u_bpf = xr.zeros_like(dataset[u_str])
    v_bpf = xr.zeros_like(dataset[v_str])
    u_bpf.data = u_filtered
    v_bpf.data = v_filtered

    # u_bpf = ds.u.filter.bandpass([time_res_days/6, time_res_days/2], dim='time') # 2-6 days
    # v_bpf = ds.v.filter.bandpass([time_res_days/6, time_res_days/2], dim='time')

    dataset['u_bpf'] = u_bpf
    dataset['v_bpf'] = v_bpf
    return dataset

def create_wt_area(data_chunk, field_name, circles, radius=500000): # 500 km?

    wt_px_data = data_chunk[field_name].stack(z=[...])
    wt_px_data = wt_px_data.where(np.logical_and(~np.isnan(wt_px_data), wt_px_data != 0), drop=True)
    for z in range(len(wt_px_data.z.data)): # for each px of WTs

        lon = wt_px_data.longitude[z]
        lat = wt_px_data.latitude[z]

        wt_index = int(wt_px_data.data[z])
        
        # get precomputed radius circle
        circle = circles.sel(longitude_center=lon.item(), latitude_center=lat.item())
        influence_area = circle.where(circle < radius, -1)

        # update influence area dataarray
        data_chunk['wt_infl'].data = xr.where(influence_area >= 0, wt_index, data_chunk['wt_infl'].data)

    # wt_rea.loc[dict(time=vt)] = wt_da.values

    

    return data_chunk

def split_wt_files_to_timedict(wts_json_list, ty):
    # wts_json_list: list of json files of feature descriptions
    # ty: tracks or wavetroughs
    
    nodes = []
    ids = [] # keep track of ids to exclude dups.
    for wts_json in wts_json_list:
        
        # load json
        with open(wts_json) as json_file:
            json_data = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
            
        if ty == 'tracks':
            tracks = json_data.sets[0].tracks
            for t in tracks:
                for e in t.edges: # for each track for each edge, add node if not yet in.
                    if e.parent.feature.id not in ids:
                        nodes.append(e.parent)
                        ids.append(e.parent.feature.id)
                        # print(ids)
        
        elif ty == 'wavetroughs':
            edges = json_data.sets[0].graph.edges
            for e in edges:
                if e.parent.feature.id not in ids:
                    nodes.append(e.parent)
                    ids.append(e.parent.feature.id)
        
    # ids = list(set(ids)) # could contain dups.

    wt_dict = defaultdict(list)
    for node in nodes:
        wt_dict[pb_str_to_datetime64(node.time)].append(node)
    
    return wt_dict
    

def get_wts_for_subset(subset, wts_time_dict):
    subset_dt = datetime.utcfromtimestamp(subset.time.item() / 1e9)

    # print(wts_time_dict)
    time_wts = wts_time_dict[np.datetime64(subset_dt)]
    return time_wts


def filter_pv(data_chunk, pvu_thr, phase_range, phase_level, lat_range, lev_range):
    tqdm.write("Filter PV based on lat, lev, phase, pvu")
    pv = data_chunk.pv
    phase_information = data_chunk.env_P.sel(level=phase_level)
    
    # threshold mask
    pv_bin_pvu_thr = xr.where(pv > pvu_thr * 1e-6, True, False)
    
    # lat mask
    mask_lat = xr.where(np.logical_and(pv.latitude >= lat_range[0], pv.latitude <= lat_range[1]), True, False)
    
    if phase_range[1] > phase_range[0]:
        mask_phase_lev = xr.where(np.logical_and(phase_information > phase_range[0], phase_information < phase_range[1]), True, False)
    else:
        mask_phase_lev = xr.where(np.logical_or(phase_information > phase_range[0], phase_information < phase_range[1]), True, False)
    # phase mask
    
    data_chunk['pv_bin'].data = np.logical_and(np.logical_and(pv_bin_pvu_thr, mask_lat), mask_phase_lev)
    # filter by level
    data_chunk['pv_bin'] = xr.where(np.logical_and(data_chunk.level > lev_range[0], data_chunk.level < lev_range[1]), data_chunk.pv_bin, False)
    
    return data_chunk
    
    
def copy_wt_props(wt_props, obj_props):
    
    for line_pt in wt_props.linePts:    
        obj_props.wt.line_pts.add()
        obj_props.wt.line_pts[-1].lat = line_pt.lat
        obj_props.wt.line_pts[-1].lon = line_pt.lon
    
    obj_props.wt.bb.min.lat = wt_props.bb.min.lat
    obj_props.wt.bb.max.lat = wt_props.bb.max.lat
    obj_props.wt.bb.min.lon = wt_props.bb.min.lon
    obj_props.wt.bb.max.lon = wt_props.bb.max.lon
    
    obj_props.wt.length_deg = wt_props.lengthDeg
    

# https://github.com/aleksandrbazhin/ellipsoid_fit_python/blob/master/ellipsoid_fit.py
# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
# for arbitrary axes
def ellipsoid_fit(levs, lats, lons):
    
    if levs is None:
        # couldnt create hull
        tqdm.write("SQERR failed")
        center = [np.nan, np.nan, np.nan]
        evecs = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
        radii = [np.nan, np.nan, np.nan]
        return center, evecs, radii, None
    
    
    x = levs
    y = lats
    z = lons
    D = np.array([x * x + y * y - 2 * z * z,
                 x * x + z * z - 2 * y * y,
                 2 * x * y,
                 2 * x * z,
                 2 * y * z,
                 2 * x,
                 2 * y,
                 2 * z,
                 1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii, v
    
# Option 2: Not best fit by least squares of convex hull, but smallest that encloses all points    
def getMinVolEllipse(P=None, tolerance=0.01):
        """ Find the minimum volume ellipsoid which holds all the points
        
        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!
        
        Here, P is a numpy array of N dimensional points like this:
        P = [[x,y,z,...], <-- one point per line
             [x,y,z,...],
             [x,y,z,...]]
        
        Returns:
        (center, radii, rotation)
        
        """
        if len(P) == 0:
            tqdm.write("MinVol failed len(P)")
            center = [np.nan, np.nan, np.nan]
            evecs = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
            radii = [np.nan, np.nan, np.nan]
            return center, radii, evecs
        
        
        (N, d) = np.shape(P)
        d = float(d)
    
        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)]) 
        QT = Q.T
        
        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            
            try:
                M = np.diag(np.dot(QT , np.dot(np.linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
            except np.linalg.LinAlgError:
                tqdm.write("MinVol failed singular inv.")
                center = [np.nan, np.nan, np.nan]
                evecs = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
                radii = [np.nan, np.nan, np.nan]
                return center, radii, evecs
                
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse 
        center = np.dot(P.T, u)
        
        
        try:
            # the A matrix for the ellipse
            A = np.linalg.inv(
                       np.dot(P.T, np.dot(np.diag(u), P)) - 
                       np.array([[a * b for b in center] for a in center])
                       ) / d
        except np.linalg.LinAlgError:
            tqdm.write("MinVol failed singular inv (2).")
            center = [np.nan, np.nan, np.nan]
            evecs = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
            radii = [np.nan, np.nan, np.nan]
            return center, radii, evecs
        
        # Get the values we'd like to return
        U, s, rotation = np.linalg.svd(A)
        radii = 1.0/np.sqrt(s)
        
        return (center, radii, rotation)


def getPrincipalComponentAxes(xs, ys, zs):
    
    # https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
    xm = np.mean(xs)
    ym = np.mean(ys)
    zm = np.mean(zs)
    center = [xm, ym, zm]
    xs_ = xs - xm
    ys_ = ys - ym
    zs_ = zs - zm
    coords = np.vstack([xs_, ys_, zs_])
    
    cov = np.cov(coords)
    # evals, evecs = np.linalg.eig(cov) # eigenvectors of obj
    
    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        tqdm.write("PC Failed")
        center = [np.nan, np.nan, np.nan]
        evecs = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
        radii = [np.nan, np.nan, np.nan]
        return center, radii, evecs 
        
    
    elens = 2 * np.sqrt(evals)
    
    # sort by length
    # sort_indices = np.argsort(elens)[::-1]
    # x_v1, y_v1, z_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    # evecs_sorted = [evecs[:, sort_indices[0]], evecs[:, sort_indices[1]], evecs[:, sort_indices[2]]]
    # elens_sorted = elens[sort_indices]
    
    return center, elens, evecs.T # we want it transposed
    
    
    
from scipy.spatial.qhull import QhullError
def generate_pv_desc(data_chunk, o, cfg):
    
    resolution = cfg.data_resolution
    
    lev_deltas = np.gradient(data_chunk.level.values)
    
    one_deg_in_km = 111.1
    one_deg_in_km_at_10N = 0.985 * one_deg_in_km
    
    id_ = o.id
    props = o.properties
    
    morph_da = data_chunk.pv_morphed
    
    if cfg.vsplit_mode is not None:
        if cfg.vsplit_mode == 'lower':
            print("lower mode, override data")
            morph_da = xr.where(morph_da.level >= 600, morph_da, 0)
        elif cfg.vsplit_mode == 'upper':
            print("upper mode, override data")
            morph_da = xr.where(morph_da.level <= 600, morph_da, 0)
        else:
            print("Err")
            exit(1)
    
    bin_obj = (morph_da == o.id)
    
    # take biggest chunk
    labels = label(bin_obj)
    if labels.max() != 0: # assume at least 1 region
        # set it to zero and restore subset later.
        data_chunk.pv_morphed.data[np.where(bin_obj)] = 0
        
        sizes = np.bincount(labels.flat)[1:]
        
        for i in range(len(sizes)):
            if sizes[i] < 6:
                # if obj very small after morphing, remove it from morphed 
                bin_obj[np.where(labels == (i+1))] = 0
        
        # largestCC = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
        # overwrite bin_obj with only sizeable ones
        # restore
        data_chunk.pv_morphed.data[np.where(bin_obj)] = o.id
        
        # bin_obj = largestCC
    
    morph_obj_ids = np.where(bin_obj)

    pv_obj_vals = data_chunk.pv.values[morph_obj_ids]
    pv_obj_vals_sum = pv_obj_vals.sum()
    
    levs_idx = morph_obj_ids[0]
    lats_idx = morph_obj_ids[1]
    lons_idx = morph_obj_ids[2]
    lats = data_chunk.latitude.values[lats_idx]
    lons = data_chunk.longitude.values[lons_idx]
    levs = data_chunk.level.values[levs_idx]
    
    # volume
    volume_km2_hPa = 0
    for lat_idx, lon_idx, lev_idx in zip(lats_idx, lons_idx, levs_idx):
        
        # POINTS - VERY VERBOSE
        props.volume_rep.points.add()
        props.volume_rep.points[-1].lev = data_chunk.level.values[lev_idx]
        props.volume_rep.points[-1].lat = data_chunk.latitude.values[lat_idx]
        props.volume_rep.points[-1].lon = data_chunk.longitude.values[lon_idx]
        
        # 1° at 10N
        lat_dx = resolution * one_deg_in_km
        lon_dx = resolution * one_deg_in_km_at_10N
        lev_dx = lev_deltas[lev_idx]
        
        volume_km2_hPa += lat_dx * lon_dx * lev_dx
        
    props.volume = volume_km2_hPa
    
    # BBOX volume
    props.bb.min.lat = lats.min() if len(lats) > 0 else np.nan
    props.bb.min.lon = lons.min() if len(lons) > 0 else np.nan
    props.bb.min.lev = levs.min() if len(levs) > 0 else np.nan
    props.bb.max.lat = lats.max() if len(lats) > 0 else np.nan
    props.bb.max.lon = lons.max() if len(lons) > 0 else np.nan
    props.bb.max.lev = levs.max() if len(levs) > 0 else np.nan
    
    # max pv, avg pv (intensity)
    props.max_pv = pv_obj_vals.max() if len(pv_obj_vals) > 0 else np.nan
    props.avg_pv = pv_obj_vals.mean() if len(pv_obj_vals) > 0 else np.nan
    
    # center of mass
    center_lat = np.dot(lats, pv_obj_vals) / pv_obj_vals_sum
    center_lon = np.dot(lons, pv_obj_vals) / pv_obj_vals_sum
    center_lev = np.dot(levs, pv_obj_vals) / pv_obj_vals_sum
    props.com.lat = center_lat
    props.com.lon = center_lon
    props.com.lev = center_lev
    
    # maybe histogram?
    
    # main axes - moments
    
    px_list = np.asarray([levs, lats, lons]) # np.asarray(morph_obj_ids)
    px_list_T = np.transpose(px_list)
    
    has_hull = False
    try:
        hull = ConvexHull(px_list_T, incremental=False, qhull_options=None)
        has_hull = True
        
        px_list_hull = np.transpose(px_list_T[hull.vertices])
        
        
    except (QhullError, ValueError) as e:
        px_list_hull = [None, None, None]
    
    # https://stackoverflow.com/questions/22146383/covariance-matrix-of-an-ellipse/47539536#47539536
    # To expand on Rody's answer, the covariance matrix for a solid ellipse has eigenvalues given by lambda_i = r_i^2/4. This leads to OP's equation of 
    #  !!!  r_i = 2*sqrt(lambda_i).
    # For a (non-solid) ellipse, as in the OP's case, the eigenvalues are double those of the solid case: lambda_i = r_i^2/2, leading to r_i = sqrt(2*lambda_i) (which is equal to Rody's 2*sqrt(lambda_i/2)).
            
    
    # OPTION 1. LOWEST SQ DIST ERROR
    center_sqdist, evecs_sqdist, radii_sqdist, v_sqdist = ellipsoid_fit(px_list_hull[0], px_list_hull[1], px_list_hull[2])
    
    # OPTION 2. SMALLEST VOLUME ENCLOSING SHAPE
    center_minvol, radii_minvol, evecs_minvol = getMinVolEllipse(P=px_list_T)
    
    # OPTION 3. Principal component axes (image moments)
    center_pc, radii_pc, evecs_pc = getPrincipalComponentAxes(levs, lats, lons)
    
    # set PC
    radii_pc_sorted_i = np.argsort(radii_pc)
        
    props.ell.center.lev = center_pc[0]
    props.ell.center.lat = center_pc[1]
    props.ell.center.lon = center_pc[2]
        
    props.ell.major_axis.lev = evecs_pc[radii_pc_sorted_i[2]][0] * radii_pc[radii_pc_sorted_i[2]]
    props.ell.major_axis.lat = evecs_pc[radii_pc_sorted_i[2]][1] * radii_pc[radii_pc_sorted_i[2]]
    props.ell.major_axis.lon = evecs_pc[radii_pc_sorted_i[2]][2] * radii_pc[radii_pc_sorted_i[2]]

    props.ell.medium_axis.lev = evecs_pc[radii_pc_sorted_i[1]][0] * radii_pc[radii_pc_sorted_i[1]]
    props.ell.medium_axis.lat = evecs_pc[radii_pc_sorted_i[1]][1] * radii_pc[radii_pc_sorted_i[1]]
    props.ell.medium_axis.lon = evecs_pc[radii_pc_sorted_i[1]][2] * radii_pc[radii_pc_sorted_i[1]]

    props.ell.minor_axis.lev = evecs_pc[radii_pc_sorted_i[0]][0] * radii_pc[radii_pc_sorted_i[0]]
    props.ell.minor_axis.lat = evecs_pc[radii_pc_sorted_i[0]][1] * radii_pc[radii_pc_sorted_i[0]]
    props.ell.minor_axis.lon = evecs_pc[radii_pc_sorted_i[0]][2] * radii_pc[radii_pc_sorted_i[0]]
    
    # set MINVOL
    radii_minvol_sorted_i = np.argsort(radii_minvol)
        
    props.ell_minvol.center.lev = center_minvol[0]
    props.ell_minvol.center.lat = center_minvol[1]
    props.ell_minvol.center.lon = center_minvol[2]
        
    props.ell_minvol.major_axis.lev = evecs_minvol[radii_minvol_sorted_i[2]][0] * radii_minvol[radii_minvol_sorted_i[2]]
    props.ell_minvol.major_axis.lat = evecs_minvol[radii_minvol_sorted_i[2]][1] * radii_minvol[radii_minvol_sorted_i[2]]
    props.ell_minvol.major_axis.lon = evecs_minvol[radii_minvol_sorted_i[2]][2] * radii_minvol[radii_minvol_sorted_i[2]]

    props.ell_minvol.medium_axis.lev = evecs_minvol[radii_minvol_sorted_i[1]][0] * radii_minvol[radii_minvol_sorted_i[1]]
    props.ell_minvol.medium_axis.lat = evecs_minvol[radii_minvol_sorted_i[1]][1] * radii_minvol[radii_minvol_sorted_i[1]]
    props.ell_minvol.medium_axis.lon = evecs_minvol[radii_minvol_sorted_i[1]][2] * radii_minvol[radii_minvol_sorted_i[1]]

    props.ell_minvol.minor_axis.lev = evecs_minvol[radii_minvol_sorted_i[0]][0] * radii_minvol[radii_minvol_sorted_i[0]]
    props.ell_minvol.minor_axis.lat = evecs_minvol[radii_minvol_sorted_i[0]][1] * radii_minvol[radii_minvol_sorted_i[0]]
    props.ell_minvol.minor_axis.lon = evecs_minvol[radii_minvol_sorted_i[0]][2] * radii_minvol[radii_minvol_sorted_i[0]]
    
    # set SQDIST
    radii_sqdist_sorted_i = np.argsort(radii_sqdist)
        
    props.ell_sqdist.center.lev = center_sqdist[0]
    props.ell_sqdist.center.lat = center_sqdist[1]
    props.ell_sqdist.center.lon = center_sqdist[2]
        
    props.ell_sqdist.major_axis.lev = evecs_sqdist[radii_sqdist_sorted_i[2]][0] * radii_sqdist[radii_sqdist_sorted_i[2]]
    props.ell_sqdist.major_axis.lat = evecs_sqdist[radii_sqdist_sorted_i[2]][1] * radii_sqdist[radii_sqdist_sorted_i[2]]
    props.ell_sqdist.major_axis.lon = evecs_sqdist[radii_sqdist_sorted_i[2]][2] * radii_sqdist[radii_sqdist_sorted_i[2]]

    props.ell_sqdist.medium_axis.lev = evecs_sqdist[radii_sqdist_sorted_i[1]][0] * radii_sqdist[radii_sqdist_sorted_i[1]]
    props.ell_sqdist.medium_axis.lat = evecs_sqdist[radii_sqdist_sorted_i[1]][1] * radii_sqdist[radii_sqdist_sorted_i[1]]
    props.ell_sqdist.medium_axis.lon = evecs_sqdist[radii_sqdist_sorted_i[1]][2] * radii_sqdist[radii_sqdist_sorted_i[1]]

    props.ell_sqdist.minor_axis.lev = evecs_sqdist[radii_sqdist_sorted_i[0]][0] * radii_sqdist[radii_sqdist_sorted_i[0]]
    props.ell_sqdist.minor_axis.lat = evecs_sqdist[radii_sqdist_sorted_i[0]][1] * radii_sqdist[radii_sqdist_sorted_i[0]]
    props.ell_sqdist.minor_axis.lon = evecs_sqdist[radii_sqdist_sorted_i[0]][2] * radii_sqdist[radii_sqdist_sorted_i[0]]
    
    # compare different ell strats again. PC probably most easy.
    
    
def gen_feature_descriptions(data_chunk: xr.Dataset, in_wts, id_reference):
    objs = []
    
    id_reference.config.data_resolution = abs(float(data_chunk.latitude.values[1] - data_chunk.latitude.values[0]))
    
    for wt in in_wts:
        
        o = id_reference.get_new_feature()
        o.id = wt.feature.id
        
        # copy properties from WT desc to PV desc over
        copy_wt_props(wt.feature.properties, o.properties)
        obj_props = o.properties
        
        generate_pv_desc(data_chunk, o, id_reference.config)
        

        # add to features if keep
        # if not self.keep_wavetrough(o.properties):
        #     continue

        objs.append(o)
    
    
    return objs
    
    
    
    
