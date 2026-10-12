import dask.array
import google
import numpy as np
from abc import ABC, abstractmethod
import xarray as xr
from datetime import datetime
import bisect, threading
import logging
from tqdm import tqdm
from tqdm.dask import TqdmCallback
from feature.util.data_utils import SplitDimension
from itertools import product
from feature.util.data_utils import get_split_dimensions, get_valid_time_split_dimension, valid_time_to_dt


class IdentificationStrategy(ABC):
    """
    Base abstract class for new feature identification algorithms. Need to implement abstract methods precompute,
    identify and postprocess. An identification strategy gets called for a dataset. The dataset is split along
    dimensions to parallelize the identification. The abstract method identify() is called for each spatial block of the
    dataset (2D, 3D), for example for each timestamp and for each ensemble member, and if processing_mode is set to
    '2d' also for each level. Beforehand, precompute() is called once, here one-time precomputations can be done.
    """

    def __init__(self):
        self.workers = None
        pass

    @abstractmethod
    def precompute(self, dataset: xr.Dataset, **kwargs):
        """
        This is called before the identification. Here, precomputations can be done. If the user wants
        to add fields to the dataset (for example as additional output fields), it should be added here. 
        The parallelized `identify()` needs to know the structure of the final dataset beforehand.

        Parameters
        ----------
        dataset : xarray.Dataset
                The dataset to process.

        Returns
        -------
        The dataset with additional initialized fields which will be populated in identify().
        """
        # self.precomputed_stuff = self.foo()...
        # dataset['bar'] = xr.zeros_like(dataset['foo'], dtype=int)
        # ...
        return dataset

    @abstractmethod
    def identify(self, process_data: xr.Dataset, index_accessor, **kwargs):
        """
        Abstract identification method. This gets called in parallel for each spatial block.

        Parameters
        ----------
        process_data : xarray.Dataset
                The subset of data which is processed in parallel,
                datasets should contain same amount of dimensions as imput dataset, but parallelized ones have length 1
                dataarrays ins dataset are squeezed accordingly to 2D or 3D fields.
        
        index_accessor : xr.DataArray with one item
                Contains a data array, with size 1 along all parallelized dimensions, thus, the
                time, ensemble member etc. of this block can be extracted. The field contains a tuple
                (set_id, time_id) containing indices to the set and time frame of this chunk. This needs 
                to get passed to `self.get_new_feature(index_accessor)` to get unique IDs for the features.

        Returns
        -------
        process_data : xarray.Dataset
                The (altered) dataset
        features : iterable of pb2.Feature
            list of features of corresponding protobuf type, see template for an example
        """
        features = []
        return process_data, features

    def identify_block(self, data_block: xr.Dataset, index_accessor, **kwargs):
        """
        Internal intermediate method which gets called for every data block and calls identify() on it. It also does
        some preparation and feature result handling. It adds the identified features into the protobuf struct.

        Parameters
        ----------
        data_block : xarray.Dataset
                The data block to be processed.
        index_accessor: xarray.DataArray
                DataArray of tuples specifying the set and timestep indices of this block's pb2 features
        Returns
        -------
        The (altered) data block.
        """
        
        # get order of dimensions to re-order them after
        dim_orders = dict()
        for v in data_block.data_vars:
            dim_orders[v] = data_block[v].dims

        # called in parallel from map_blocks
        split_dims = kwargs.keys()
        split_string = ''
        for dim in split_dims:
            dim_val = data_block.coords[dim].data[0]
            if isinstance(dim_val, np.datetime64):
                dt = datetime.utcfromtimestamp(dim_val.astype('O') / 1e9)
                dim_val = dt.replace(microsecond=0).isoformat()
            split_string += str(dim) + ":" + str(dim_val) + " \t"
            
#        if split_string == '':
#            tqdm.write("Start processing the singular data block...")
#        else:
#            tqdm.write("Start processing data block defined by:\t" + split_string)

        # get mapping which data fields' coords are getting squeezed (time, ens, ...). some fields may not have them all.
        squeeze_da_dim = []
        for split_dim in split_dims:
            for da_str in data_block.data_vars:
                if split_dim in data_block[da_str].coords:
                    squeeze_da_dim.append((da_str, split_dim))
                    data_block[da_str] = data_block[da_str].squeeze(dim=[split_dim])

        access_indices = index_accessor.values.item()
    
        data_block, feature_block = self.identify(data_block, index_accessor=index_accessor, split_dims=split_dims)
        self.pb_dataset.sets[access_indices[0]].timesteps[access_indices[1]].features.extend(feature_block)

        for unsqueeze_da, unsqueeze_dim in squeeze_da_dim:
            data_block[unsqueeze_da] = data_block[unsqueeze_da].expand_dims(dim=unsqueeze_dim)

        # restore dim order
        for v in data_block.data_vars:
            data_block[v] = data_block[v].transpose(*(dim_orders[v]))
        
        return data_block

    def execute(self, dataset: xr.Dataset, **kwargs):
        """
        Execute the identification of the features in the dataset. The properties of each feature are collected via the
        abstract identify() in parallel and here collected.

        Parameters
        ----------
        dataset : xarray.Dataset
                The dataset.

        Returns
        -------
        pb_dataset : pb2_structure.DatasetDescription
                The description struct containing all detected feature descriptions.
        ds_mapped : xarray.Dataset
                Dataset after being mapped via identify()
        """
        # build meta data for protobuf structure

        if not hasattr(self, 'pb_reference'):
            print("The protobuf type has not been set in the ID strategy.")
            print("Check the templates, the IdentificationStrategy.__init__() needs to set the pb2 type.")
            exit(1)

        self.pb_dataset = self.pb_reference.DatasetDescription()

        # precomputations according to strategy - may alter the dataset
        prec_ds = self.precompute(dataset)
        if prec_ds is not None:
            dataset = prec_ds
        print("Precomputation done.")

        # from enstools.misc import get_ensemble_dim, get_time_dim
        # from feature.util.enstools_utils import get_vertical_dim, get_init_time_dim

        # dimension names along which data is split (init_time, level, ens, ...)
        split_dimensions = get_split_dimensions(dataset,
                                                self.processing_mode)  # {ENUM.init_time: "init_time", len, is_scalar}
        valid_time_sd = get_valid_time_split_dimension(dataset)

        # make sure is sorted so indices work out at the end
        dataset = dataset.sortby([sd.name for sd in split_dimensions])

        # field of indices into descriptions
        rechunk_coords = split_dimensions  # [dataset.coords[sd.name] for sd in split_dimensions]
        if not valid_time_sd.is_scalar:
            rechunk_coords.append(valid_time_sd)
        index_field = xr.DataArray(coords=[dataset.coords[sd.name] for sd in rechunk_coords]).astype(dtype=tuple)

        # rechunk along parallelized dimensions
        ds_rechunked = dataset.chunk(dict((sd.name, 1) for sd in rechunk_coords))  # sizes: 1

        # init pb_dataset. set member + valid time for each element in lists if existent

        if self.processing_mode != '2d':
            level_dim_len = 1

        # get the split dimensions out of the dimensions in the data set
        split_dims_set = [sd for sd in split_dimensions if sd.dim != SplitDimension.SplitDimensionDim.VALID_TIME]
        split_dims_set_enum = [sd.dim for sd in split_dims_set]
        prod_set = [range(sd.size) for sd in split_dims_set]

        iter_prod = product(*prod_set)

        # for each split dimension search according dimension in enum
        for elem in iter_prod:
            cur_indices = dict()
            s_i = self.pb_dataset.sets.add()

            # init time
            init_time_cur = None
            if SplitDimension.SplitDimensionDim.INIT_TIME in split_dims_set_enum:
                init_time_idx = split_dims_set_enum.index(SplitDimension.SplitDimensionDim.INIT_TIME)
                init_time_cur = dataset.coords[split_dims_set[init_time_idx].name].data[elem[init_time_idx]]
                s_i.init_time = str(np.datetime_as_string(init_time_cur, unit='s'))
                cur_indices[split_dims_set[init_time_idx].name] = s_i.init_time

            if SplitDimension.SplitDimensionDim.ENSEMBLE_MEMBER in split_dims_set_enum:
                ens_idx = split_dims_set_enum.index(SplitDimension.SplitDimensionDim.ENSEMBLE_MEMBER)
                ens_array = dataset.coords[split_dims_set[ens_idx].name].data

                if not np.issubdtype(ens_array.dtype.type, np.integer):
                    print("Ensemble dimension not of integer type, parsing to np.int32.")
                    dataset = dataset.assign_coords(coords={split_dims_set[ens_idx].name: ens_array.astype(dtype=np.int32)})

                s_i.member = dataset.coords[split_dims_set[ens_idx].name].data[elem[ens_idx]]
                cur_indices[split_dims_set[ens_idx].name] = s_i.member

            if SplitDimension.SplitDimensionDim.LEVEL in split_dims_set_enum:
                level_idx = split_dims_set_enum.index(SplitDimension.SplitDimensionDim.LEVEL)
                s_i.level = dataset.coords[split_dims_set[level_idx].name].data[elem[level_idx]]
                cur_indices[split_dims_set[level_idx].name] = s_i.level

            for t in range(valid_time_sd.size):
                timestep = s_i.timesteps.add()

                valid_time_data = dataset.coords[valid_time_sd.name].data[t] if not valid_time_sd.is_scalar else dataset[
                    valid_time_sd.name]
                valid_time_dt = valid_time_to_dt(valid_time_data, init_time_cur)

                timestep.time = str(np.datetime_as_string(valid_time_dt, unit='s'))
                if not valid_time_sd.is_scalar:
                    cur_indices[valid_time_sd.name] = valid_time_data

                # ID to access this feature block in protobuf set
                sets_id = len(self.pb_dataset.sets) - 1
                tl_id = len(s_i.timesteps) - 1
                index_field.loc[cur_indices] = (sets_id, tl_id)

        # dict of values of the split dimensions -> to access them within each block
        dict_split_arrays = dict((dim.name, dataset.coords[dim.name]) for dim in split_dimensions)
        
        # remember the next available feature id for parallel access later
        self.next_id = [1] * len(self.pb_dataset.sets)
        self.feature_id_lock = threading.Lock()
        
        # map the blocks, computation here
        mb = xr.map_blocks(self.identify_block, ds_rechunked, args=[index_field], kwargs=dict_split_arrays,
                           template=ds_rechunked)
        
        with TqdmCallback(desc="identify()"):
            ds_mapped = mb.compute(num_workers=self.workers)

        # postprocess
        ds_mapped, self.pb_dataset = self.postprocess(ds_mapped, self.pb_dataset)

        return self.pb_dataset, ds_mapped

    @abstractmethod
    def postprocess(self, dataset: xr.Dataset, pb2_desc, **kwargs):
        """
        Abstract method for postprocessing operations.
        It takes as input the dataset and the corresponding descriptions, and returns them in an arbitrary altered
        state.

        Parameters
        ----------
        dataset : xarray.Dataset
                The dataset.
        pb2_desc : pb2_structure.DatasetDescription
                The description.
        kwargs

        Returns
        -------
        As inputs, but can be altered.
        """

        return dataset, pb2_desc


    def get_new_feature(self, index_accessor=None):
        """
        Create a new feature to be populated

        Parameters
        ----------
        index_accessor : Should be forwarded from identify() signature
            Used to extract the current set, so we have unique IDs therein.

        Returns
        -------
        The new feature that can be populated
        """
        if index_accessor is None:
            obj = self.pb_reference.Feature()
            obj.id = -1
            return obj
            
        
        # From index accessor get current set
        this_set = index_accessor.values.item()[0]
        
        with self.feature_id_lock:
            obj_id = self.next_id[this_set]
            self.next_id[this_set] += 1

        # infer from type
        obj = self.pb_reference.Feature()
        # set ID if given
        obj.id = obj_id
        return obj

    
    def add_voxel_rep(self, obj, float_mask):
        """
        Add to the feature (obj) description its voxel representation. Thus, every voxel in the data block
        will be extracted and added as list, where the mask is not np.nan. This method is expected to be
        called by the user from self.identify() after getting a new Feature with self.get_new_feature().
        
        Parameters
        ----------
        obj : Feature
            The feature to add the voxel representation.
        float_mask : DataArray
            The DataArray to take the values and coordinates from. Expects np.nan where the feature is not
            located and float values where the feature is located. Also writes these float values to the object.
            
        Returns
        -------
        """
        float_mask_data = float_mask.data
        
        # get numpy indices and values where not nan
        indices = np.where(~np.isnan(float_mask_data))
        values = float_mask_data[indices]
        
        # init object
        voxelrep = obj.voxel_rep
        voxelrep.desc = "Voxel Representation"
        voxel_data = voxelrep.voxel_data
        
        dim_values_per_dim = []
        # get dimension values from indices
        for dim_idx, dim in enumerate(float_mask.dims):
            voxelrep.dims.append(dim)
            dim_values = float_mask[dim].values[indices[dim_idx]]
            dim_values_per_dim.append(dim_values)
            
        coord_list = [list(i) for i in zip(*dim_values_per_dim)]
        
        # and fill the voxels
        for coord_idx, coord in enumerate(coord_list):
            voxel = voxel_data.add()
            voxel.coordinate.extend(coord)
            voxel.value = values[coord_idx]
        
        
        
            
        