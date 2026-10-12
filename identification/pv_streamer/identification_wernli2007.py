from ..identification import IdentificationStrategy
from feature.identification.pv_streamer.data_util import get_pv_field_name

import xarray as xr
import os
import numpy as np
from shutil import copyfile

# wrapper calling Wernli Sprenger 2007 identification written in Fortran.
class PVWernliSprenger2007(IdentificationStrategy):

    def __init__(self, unit='pv', level=0, **kwargs):
        # put custom args
        self.pv_data_unit = unit
        self.level = level
        # Constructor. Called from example_template.py, can set parameters.
        pass

    def precompute(self, dataset: xr.Dataset, **kwargs):
        # Precomputation, done once for the entire dataset.
        # Also data fields need to be added here in case they are needed and populated later on.
        pv_dim = get_pv_field_name(dataset)
        dataset = dataset.rename({pv_dim: 'PV'})
        dataset['ws2007_areas'] = xr.zeros_like(dataset['PV'], dtype=bool)
        return dataset

    def identify(self, dataset: xr.Dataset, **kwargs):
        # here your code ... identify features for this subset.

        # save subset to file to be processed.

        import dask
        hash = dask.base.tokenize(dataset)
        wernli_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "wernli2007")
        #if hash != "dd4e5a30043f80f41c8a1a740a4f602e":
        #    return dataset, []

        tmp_filepath = os.path.join(wernli_path, "test.nc")

        if dataset.PV.data.ndim == 3:
            level_index = dataset.th_level.values.tolist().index(self.level)
            fil_ds = dataset.sel(th_level=self.level)
            fil_ds = fil_ds.expand_dims({'th_level': [self.level]})
        else:
            level_index = self.level
            fil_ds = dataset.expand_dims({'th_level': [self.level]})


        if self.pv_data_unit == 'pv':
            fil_ds['PV'] = fil_ds.PV * 1e6

        fil_ds.to_netcdf(tmp_filepath) # save for sprenger
        tmp_filepath_abs = os.path.abspath(tmp_filepath)

        ds_sprenger = self.contour_classify(fil_ds, tmp_filepath_abs, cached=False)

        sprenger_3d_values = np.zeros_like(fil_ds['ws2007_areas'].values)
        data_in_values = ds_sprenger.STREAMER.isel(time=0).values # spenger adds time=0, remove. also
        if sprenger_3d_values.shape[2] == data_in_values.shape[2]: # if data values does not have -180 and +180, add 180
            sprenger_3d_values = data_in_values
        else:
            sprenger_3d_values[:,:,:-1] = data_in_values
            sprenger_3d_values[:,:,-1] = sprenger_3d_values[:,:,0] # add +180

        fil_ds['ws2007_areas'].values = sprenger_3d_values
        dataset['ws2007_areas'].values = fil_ds['ws2007_areas'].values[level_index]

        # plotting 2d
        from .wernli2007.plotting import plot_ws_streamers
        fil_ds = fil_ds.squeeze()
        plot_ws_streamers(fil_ds)
        exit()

        # dataset.loc[dict({'th_level': self.level})] = fil_ds
        # merged = xr.merge([fil_ds, dataset], compat='override')

        # TODO delete temp file

        feature_properties = []

        # load feature structure as defined in template.proto
        # if you create your own my.proto, import my_pb2
        from .._proto_gen import template_pb2
        from feature.util.pb2_properties_api import FeatureProperties

        # return the dataset (can be changed), and a list of features
        return dataset, feature_properties

    def generate_streamer_netcdf_file(self, filepath, levels):
        current_wd = os.getcwd()
        # filename = os.path.basename(filepath)

        # data_path = os.path.join(os.path.expanduser("~"), 'phd/data/tracked/S2S_test/S2S-ECMWF_EXTP_pv_avg200-500_hcst_19980507_00Z.nc')
        wernli_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "wernli2007")
        os.chdir(wernli_path)

        os.putenv('levels', ' '.join(levels))

        command = "sh run.sh " + filepath  # from python, spaces inbetween "330,335,340"
        print("Executing Fortran routine from Sprenger et al. ...")
        print("Command: " + str(command))

        import subprocess
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        # execute bash with levels, name of file
        print(output.decode("utf-8"))

        os.chdir(current_wd)

        return str(filepath) + ".streamer"

    # Read sprenger streamer file and load it into xarray dataset
    # dataset.sprenger
    def contour_classify(self, ds, filenamepath, cached=False):

        filedir = os.path.dirname(filenamepath)

        from feature.util.enstools_utils import get_latitude_dim, get_longitude_dim, get_vertical_dim
        lat_str = get_latitude_dim(ds)
        lon_str = get_longitude_dim(ds)

        level_str = get_vertical_dim(ds)
        if level_str is not None: # 2D or 3D
            levels = [str(s) for s in ds.coords[level_str].values.tolist()]
        else:
            levels = ["0"]

        path_result = self.generate_streamer_netcdf_file(filenamepath, levels)

        # load result
        result_2d_nh = xr.load_dataset(path_result + ".nh")
        result_2d_sh = xr.load_dataset(path_result + ".sh")
        result_2d = xr.zeros_like(result_2d_nh)
        result_2d['STREAMER'] = result_2d_nh.STREAMER + result_2d_sh.STREAMER
        # xr.merge([result_2d_nh, result_2d_sh], compat='override')
        print(result_2d)

        return result_2d

    def load_streamer_content(self, streamer_file):

        # TODO file handling, where generated...

        with open(streamer_file) as f:
            content = f.readlines()
        return content

    def postprocess(self, dataset: xr.Dataset, pb2_desc, **kwargs):
        # Here post process: called once after processing the whole dataset.
        # If needed, the dataset or features can be changed here.
        return dataset, pb2_desc