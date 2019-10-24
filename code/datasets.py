import os
import re
import collections

# from medicaltorch import transforms as mt_transforms

from tqdm import tqdm
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset
import torch
from torch._six import string_classes, int_classes

from PIL import Image
from utils import find



__numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

class TensorDataset(Dataset):
    def __init__(self, nodules, malignancy, transforms=None):
        self.nodules = nodules
        self.malignancy = malignancy
        self.transforms = transforms
    def __len__(self):
        return self.nodules.shape[0]
    def __getitem__(self, index):
        nodule = self.nodules[index]
        if self.transforms:
            nodule = self.transforms(nodule)
        malignancy = self.malignancy[index]
        return nodule, malignancy


class SampleMetadata(object):
    def __init__(self, d=None):
        self.metadata = {} or d

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def __getitem__(self, key):
        return self.metadata[key]

    def __contains__(self, key):
        return key in self.metadata

    def keys(self):
        return self.metadata.keys()


class BatchSplit(object):
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        batch_len = len(self.batch["input"])
        for i in range(batch_len):
            single_sample = {k: v[i] for k, v in self.batch.items()}
            single_sample['index'] = i
            yield single_sample
        raise StopIteration


class SegmentationPair2D(object):
    """This class is used to build 2D segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).

    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """
    def __init__(self, input_filename, gt_filename, cache=True,
                 canonical=False):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.canonical = canonical
        self.cache = cache
        self.error = False

        self.input_handle = nib.load(self.input_filename)


        # Unlabeled data (inference time)
        if self.gt_filename is None:
            self.gt_handle = None
        else:
            self.gt_handle = nib.load(self.gt_filename)

        if len(self.input_handle.shape) > 3:
            raise RuntimeError("4-dimensional volumes not supported.")

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()
        # print(input_shape, gt_shape)

        if self.gt_handle is not None:
            if not np.allclose(input_shape, gt_shape):
                #raise RuntimeError('Input and ground truth with different dimensions.')
                self.error = True
                print("Skipping #", input_filename, input_shape, gt_shape)
             
        if self.canonical:
            self.input_handle = nib.as_closest_canonical(self.input_handle)

            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.gt_handle)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.input_handle.header.get_data_shape()

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape, gt_shape

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'
        input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32)

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        self.input_handle.uncache()
        self.gt_handle.uncache()

        return input_data, gt_data


class MRI2DSegmentationDataset(Dataset):
    """This is a generic class for 2D (slice-wise) segmentation datasets.

    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth filename).
    :param slice_axis: axis to make the slicing (default axial).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """
    def __init__(self, filename_pairs, slice_axis=2, cache=True,
                 transform=None, slice_filter_fn=None, canonical=False):
        self.filename_pairs = filename_pairs
        self.handlers = []
        self.indexes = []
        self.transform = transform
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.canonical = canonical
        
    def set_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.

        :param transform: the new transformation
        """
        self.transform = transform

    # def compute_mean_std(self, verbose=False):
    #     """Compute the mean and standard deviation of the entire dataset.

    #     :param verbose: if True, it will show a progress bar.
    #     :returns: tuple (mean, std dev)
    #     """
    #     sum_intensities = 0.0
    #     numel = 0

    #     with DatasetManager(self,
    #                         override_transform=mt_transforms.ToTensor()) as dset:
    #         pbar = tqdm(dset, desc="Mean calculation", disable=not verbose)
    #         for sample in pbar:
    #             input_data = sample['input']
    #             sum_intensities += input_data.sum()
    #             numel += input_data.numel()
    #             pbar.set_postfix(mean="{:.2f}".format(sum_intensities / numel),
    #                              refresh=False)

    #         training_mean = sum_intensities / numel

    #         sum_var = 0.0
    #         numel = 0

    #         pbar = tqdm(dset, desc="Std Dev calculation", disable=not verbose)
    #         for sample in pbar:
    #             input_data = sample['input']
    #             sum_var += (input_data - training_mean).pow(2).sum()
    #             numel += input_data.numel()
    #             pbar.set_postfix(std="{:.2f}".format(np.sqrt(sum_var / numel)),
    #                              refresh=False)

    #     training_std = np.sqrt(sum_var / numel)
    #     return training_mean.item(), training_std.item()

    def __len__(self):
        """Return the dataset size."""
        return len(self.filename_pairs)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).

        :param index: slice index.
        """
        # nodule_id = value.split(".nrrd.nii.gz")[0].split("Annotation")[1]
        # patient_id = key.split("LIDC-IDRI-")[1][0:4]


        # filepaths to the nrrds files
        input_filename, gt_filename = self.filename_pairs[index]

        # below code get me lables
        patient_id = input_filename.split("LIDC-IDRI-")[1][0:4]
        nodule_id = gt_filename.split(".nrrd.nii.gz")[0].split("Annotation")[1]
        labels = find(patient_id, nodule_id)



        segpair = SegmentationPair2D(input_filename, gt_filename, self.cache, self.canonical)
        if segpair.error:
            return None
        input_img, gt = segpair.get_pair_data()
        
        #print(input_img.shape)

           
        # 512 x 512 x n
        #print(input_img.shape)
        #x = input_img.shape[2]//2
        #input_img = input_img[:, :, x-45:x+45]
        #if input_img.shape[-1] < 90:
            #padding = (90 - input_img.size(-1))
            #second_tensor = torch.zeros((512,512,padding), dtype=torch.int32)
            #input_img = torch.cat((input_img, second_tensor), 0)        
        

        data_dict = {
            'filename': segpair.input_filename,
            'gt_filename': segpair.gt_filename,
            'input': input_img,
            'gt': gt,
            'input_metadata': None,
            'gt_metadata': None,
        }

        if self.transform is not None:
            data_dict = self.transform(data_dict)
        
        return data_dict
        
        # return input_img, torch.tensor([labels[-1]-1])


class DatasetManager(object):
    def __init__(self, dataset, override_transform=None):
        self.dataset = dataset
        self.override_transform = override_transform
        self._transform_state = None

    def __enter__(self):
        if self.override_transform:
            self._transform_state = self.dataset.transform
            self.dataset.transform = self.override_transform
        return self.dataset

    def __exit__(self, *args):
        if self._transform_state:
            self.dataset.transform = self._transform_state


def mt_collate(batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        stacked = torch.stack(batch, 0)
        return stacked
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return __numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: mt_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [mt_collate(samples) for samples in transposed]

    return batch
