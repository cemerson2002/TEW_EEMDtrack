from ..identification import IdentificationStrategy

import operator
import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from skimage.segmentation import flood_fill


class DoubleThresholdIdentification(IdentificationStrategy):

    def __init__(self, field, outer_threshold, inner_threshold, comparison_operator,
                 processing_mode="2d", **kwargs):
        self.field = field
        self.outer_threshold = outer_threshold
        self.inner_threshold = inner_threshold
        # Save comparison operator name and get associated comparison function
        self.comparison_operator = comparison_operator
        if comparison_operator in { ">=", "ge" }:
            self.compare = operator.ge
        elif comparison_operator in { ">", "gt" }:
            self.compare = operator.gt
        elif comparison_operator in { "<=", "le" }:
            self.compare = operator.le
        elif comparison_operator in { "<", "lt" }:
            self.compare = operator.lt
        else:
            raise ValueError(f"unknown comparison operator '{comparison_operator}'")
        # Check that thresholds are properly sorted for the given comparison
        # operator. Always allow equality to make single threshold
        # identification possible.
        assert self.inner_threshold == self.outer_threshold or self.compare(self.inner_threshold, self.outer_threshold)
        # Specify processing mode 2d or 3d: In 2d, identification will be performed on 2d levels,
        #   in 3d the identification will be performed per 3d block
        assert processing_mode in { "2d", "3d" }
        self.processing_mode = processing_mode

    def precompute(self, dataset: xr.Dataset, **kwargs):
        print(dataset)
        
        assert self.field in dataset
        field = dataset[self.field]
        
        inner_thresh_mask = self.compare(field, self.inner_threshold)
        outer_thresh_mask = self.compare(field, self.outer_threshold)
        
        dataset['inner_thresh_mask'] = xr.zeros_like(field, dtype=int)
        dataset['outer_thresh_mask'] = xr.zeros_like(field, dtype=int)
        
        dataset['inner_thresh_mask'].data = self.compare(field, self.inner_threshold)
        dataset['outer_thresh_mask'].data = self.compare(field, self.outer_threshold)

        dataset['feature_field'] = xr.zeros_like(field, dtype=int)
        
        return dataset

    def identify(self, data_block: xr.Dataset, index_accessor, **kwargs):
        field = data_block[self.field].values
        
        if self.processing_mode == "2d":
            field = field.reshape((1,) + field.shape)
            
        
        inner_field = data_block.inner_thresh_mask
        outer_field = data_block.outer_thresh_mask
        
        # Label features in the inner threshold mask
        labeled_inner, num_features_inner = ndi.label(inner_field)
        # Label features in the outer threshold mask
        labeled_outer, num_features_outer = ndi.label(outer_field)
        
        obj_list = []

        # Expand inner features to the outer mask
        for label in range(1, num_features_inner + 1):
            inner_feature_indices = np.argwhere(labeled_inner == label)
            
            mid_idx = int(len(inner_feature_indices)/2)
            # take some middle index, should be the fastest to fill from
            sample_inner_obj_idx = tuple(inner_feature_indices[mid_idx])
            
            # fill one from inner into outer
            filled_obj = flood_fill(labeled_outer, sample_inner_obj_idx, -1)
            
            obj = self.get_new_feature(index_accessor)
            obj_id = obj.id
            
            data_block['feature_field'].data[np.where(filled_obj == -1)] = obj_id
            obj_list.append(obj)
            # expanded_feature = ndimage.binary_dilation(inner_feature, structure=np.ones((3, 3)))
            # outer_thresh_mask[expanded_feature] = True

        # return the dataset (can be changed here), and the list of features
        # TODO check netcdf file if this works now. also single thr.
        return data_block, obj_list

    def postprocess(self, dataset: xr.Dataset, pb2_desc, **kwargs):
        # Here post process: called once after processing the whole dataset.
        # If needed, the dataset or features can be changed here.

        # features are already in finished protobuf description, cf. the JSON output.
        # pb2_desc.sets is list of trackable sets, each one contains the timesteps list, where each entry is a valid time and contains a list of features.
        return dataset, pb2_desc


class SingleThresholdIdentification(DoubleThresholdIdentification):
    """
    Single threshold identification is realized (suboptimally) here as a special case of double threshold
    identification, with twice the same threshold.
    
    Identifies features based on a threshold.
    Creates a feature_field by default, realized as mask of the data field.
    """

    def __init__(self, field, threshold, comparison_operator, *args, **kwargs):
        super().__init__(field, threshold, threshold, comparison_operator, *args, **kwargs)


