import numpy as np
from skimage.morphology import remove_small_holes, flood
from skimage.measure import label
import skfmm
from multiprocessing.pool import ThreadPool
import multiprocessing

# ThreadPool/Lock initializations


def init_lock(l):
    global lock
    lock = l


l = multiprocessing.Lock()
pool = ThreadPool(initializer=init_lock, initargs=(l,))


def pool_map(fct, arg_list, *const_args):
    return pool.map(fct, arg_list)


from scipy.ndimage import generate_binary_structure

struct2d1 = generate_binary_structure(2, 1)
struct2d2 = generate_binary_structure(2, 2)
struct3d1 = generate_binary_structure(3, 1)
struct3d2 = generate_binary_structure(3, 2)


# preprocess pvu2 field: remove small holes on every layer to make it more smooth and less noisy
def preprocess_pvu2(field_3d, config):
    pp3d = np.copy(field_3d)
    
    def level_preprocess(i):
        pp3d[i] = remove_small_holes(pp3d[i], area_threshold=config.alg.remove_holes_maxarea)

    pool.map(level_preprocess, range(pp3d.shape[0]))
    return pp3d


# create distance map -> distances in projected field. field3d resembles the masked boundary field.
def create_dist_map_outer(field_3d, config, distance_map):
    speed_map = 1.0 / distance_map

    # nans and infs (out of scope, borders) to 1 to have valid values (shouldnt be reached anyways)
    speed_map[speed_map == np.inf] = 1
    np.nan_to_num(speed_map, nan=1, copy=False)

    skd = skfmm.travel_time(field_3d, speed_map, config.alg.dH)
    skd_data = np.nan_to_num(skd.data, copy=False)

    return skd_data


# get pole of reservoir: point with max distance from any boundary -> "core" of biggest reservoir
def get_data_pole(distance_map):
    # Calculate pole (center of reservoir): max distance from boundary. if not in skeleton, too small.
    max_dist = np.amax(distance_map)
    max_dist_ilist = np.where(distance_map == max_dist)

    first_max_idx = tuple([max_dist_ilist[i][0] for i in range(3)])
    return first_max_idx


# flood reservoir: might have multiple: take biggest reservoir, this is the one starting at pole_index
def flood_reservoir(inner_3d, pp_3d, pole_index):
    inner_reservoir = flood(inner_3d, footprint=struct3d1, seed_point=pole_index)
    isolated_cutoffs = flood(pp_3d, footprint=struct3d1, seed_point=pole_index) ^ pp_3d

    # 2D harder regarding reservoirs, here easy
    return inner_reservoir, isolated_cutoffs


# starting from outer boundary of inner3d reservoir, grow into pp3d fulfilling boundary conditions by data
def create_dist_map_inner(inner3d, pp3d, config, distance_map):
    start = np.ones(shape=pp3d.shape)
    start[inner3d] = 0  # res 0, dil 1 -> zero contour

    speed_map = 1.0 / distance_map

    speed_map[speed_map == np.inf] = 1
    np.nan_to_num(speed_map, nan=1, copy=False)

    # mask additionally non-strati areas
    start_mask = np.ma.array(start, mask=(~pp3d))  # from reservoir edge, but mask dil3d to get actual in-data distances
    skd = skfmm.travel_time(start_mask, speed_map, config.alg.dH)  # , dx=[1000, 1, 1])

    return skd


# label the areas by connectivity
# also compute the sizes for each feature (pixel count) - sizes[i] = size of feature labeled i in areas
def label_areas(streamer_areas):
    labeled_areas = label(streamer_areas, connectivity=2) # return_num=True,
    return labeled_areas

# filter by features. iterate over them and see if they fulfill certain criteria
def filter_feature_based(stereo_ds, obj_list, config):
    # get level if the user wants to extract features only for certain height.
    levels_list = stereo_ds.coords[config.vertical_dim].values
    try:
        if config.sel_layer is not None:
            if type(config.sel_layer) is tuple:
                sel_layer_id_lower = levels_list.tolist().index(config.sel_layer[0])
                sel_layer_id_upper = levels_list.tolist().index(config.sel_layer[1])
                if sel_layer_id_lower > sel_layer_id_upper:
                    sel_layer_id_lower, sel_layer_id_upper = sel_layer_id_upper, sel_layer_id_lower
            else:  # int/float
                sel_layer_id_lower = levels_list.tolist().index(config.sel_layer)
                sel_layer_id_upper = levels_list.tolist().index(config.sel_layer)
        else:
            sel_layer_id_lower, sel_layer_id_upper = -1, -1
    except ValueError:
        print("Selected level " + str(config.sel_layer) + " not in dataset.")

    features_passed = []

    thr = config.centroid_lat_thr
    labeled_areas = stereo_ds['streamer'].values

    for obj in obj_list:
        mask_index = obj.id
        obj_props = obj.properties

        obj_area = np.where(labeled_areas == mask_index)
        # if size zero: discard
        if len(obj_area[0]) == 0:
            print("Size zero feature...")
            continue

        # if not in user sel layer range, discard
        objvmin, objvmax = np.min(obj_area[0]), np.max(obj_area[0])
        if (sel_layer_id_lower != -1) and (objvmax < sel_layer_id_lower or objvmin > sel_layer_id_upper):
            labeled_areas[obj_area] = 0
            continue

        # volume threshold
        if config.dims == 3 and obj_props.volume_km2K < config.alg.min_streamer_volume:
            labeled_areas[obj_area] = 0
            continue

        # 2D: min streamer length.
        if config.dims == 2 and obj_props.length < config.alg.min_streamer_len:
            labeled_areas[obj_area] = 0
            continue

        cent_lat = obj_props.centroid.lat

        if thr is not None and cent_lat < thr:
            print(cent_lat)
            labeled_areas[obj_area] = 0
            continue

        # feature survived the filtering
        features_passed.append(obj)

    # pool.map(filter_feature_3d, range(1, nums + 1))
    return stereo_ds, features_passed


# squish streamer IDs because due to filtering some inbetween have size 0
def squish(stereo_ds, obj_list):
    labeled_areas = stereo_ds['streamer'].values

    for i in range(len(obj_list)):  # i+1 are final indices: squish indices in 1-num range
        obj_id = obj_list[i].id
        obj = obj_list[i]
        obj_area = np.where(labeled_areas == obj_id)

        obj_list[i].id = i + 1
        labeled_areas[obj_area] = i + 1

    return stereo_ds, obj_list


# filter detected areas area-based, not based on features. therefore done before labeling
def filter_area_based(str_areas_bin, pvu2_bin, config):
    # postprocessing for visualization only - dont change feature information
    start_str_areas_bin = str_areas_bin
    # min depth bascially. need to be >1 to survive. also regions of bigger streamers affected by this
    selem = np.zeros(shape=(3, 3, 3))
    selem[1, 1, 1] = 1
    selem[0, 1, 1] = 1

    # amount of times we thin (mainly from above)
    bridge_levels = int(config.alg.bridge_K / config.res_z)

    from skimage.morphology import binary_dilation, binary_erosion

    for i in range(bridge_levels):
        str_areas_bin = binary_erosion(str_areas_bin, footprint=selem)

    for i in range(bridge_levels):
        str_areas_bin = binary_dilation(str_areas_bin, footprint=selem) & start_str_areas_bin

    return str_areas_bin
