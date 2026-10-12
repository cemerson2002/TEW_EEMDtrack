import xarray as xr
import numpy as np
import logging

def get_pv_field_name(ds):
    v_names = ["pv", "PV", "pv_avg200-500", "PV_averaged"]
    for v_name in v_names:
        if v_name in ds.data_vars:
            return v_name
    return None


def pv_to_pvu(ds, config):
    pv_str = get_pv_field_name(ds)

    if pv_str is None:
        logging.error("Could not detect PV field in dataset. Expected pv, PV")
        exit(1)

    if config.pv_data_unit == 'pv':
        ds = ds.assign(pvu=lambda d: d[pv_str] * 1e6)
    elif config.pv_data_unit == 'pvu':
        ds = ds.assign(pvu=lambda d: d[pv_str] * 1)
    else:
        raise Exception("Unknown pv_data_unit.")
    return ds


def pvu_to_pvu2(ds):
    # logic: pvu>-500(def) AND ((>0N and pvu>2) OR (<0N and pvu<-2))
    ds['pvu2'] = np.logical_and(ds.pvu > -500.0, np.logical_or(ds.pvu > 2.0, ds.pvu < -2.0))
    return ds


def compute_global_fields(dataset: xr.Dataset, config):
    # from PV -> PVU -> PVU2 (binary thresholding 2pvu)
    dataset = pv_to_pvu(dataset, config)
    dataset = pvu_to_pvu2(dataset)

    return dataset


def filter_dataset(dataset: xr.Dataset, theta_range, layer_2d):
    from feature.util.enstools_utils import get_latitude_dim, get_vertical_dim
    l_dim = get_latitude_dim(dataset)

    # also make sure top is north pole
    dataset = dataset.sortby(l_dim)

    if dataset.coords[l_dim].values.min() < 0:
        print("Emit southern hemisphere, currently only concepted for NH.")

    dataset = dataset.sel({l_dim: slice(0, 90)})
    # dataset = dataset.sortby(l_dim, ascending=False)
    
    z_dim = get_vertical_dim(dataset)
    if theta_range is not None:
        dataset = dataset.sel({z_dim: slice(*theta_range)})
    elif layer_2d is not None:
        dataset = dataset.sel({z_dim: layer_2d})

    return dataset
