from statistics import mode
from cProfile import label
from torch.utils.data import Dataset
from medpy.io import load as medload
import skimage.transform
from albumentations.augmentations.transforms import CropNonEmptyMaskIfExists, RandomCrop, PadIfNeeded
from albumentations import Compose as AlbuCompose
from albumentations import BboxParams
import numpy as np
from copy import deepcopy
from skimage.measure import label as find_connected_regions
from collections import OrderedDict

from univread import read as imread

def convert_id_to_3d(id, shapes):
    """
    Converts linear id to the slice taken across one of the axes.
    """
    if id < shapes[0]:
        internal_ax = 0
        internal_id = id
    elif id < shapes[0] + shapes[1]:
        internal_ax = 1
        internal_id = id - shapes[0]
    else:
        internal_ax = 2
        internal_id = id - (shapes[0] + shapes[1])

    return internal_ax, internal_id

def get_new_sampling_probabilities(is_positive, negative_probability=0.1, normed=False):
    ppo = is_positive.mean()
    ppn = 1 - negative_probability

    sampling_probabilities = np.zeros(is_positive.shape)
    sampling_probabilities[is_positive == 0] = (1 - ppn) / (1 - ppo)
    sampling_probabilities[is_positive == 1] = ppn / ppo

    if normed:
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    return sampling_probabilities

internal_medload = lambda addr: medload(addr)[0]

class ExpandedPaddedSegmentation:
    """
    Could be used to hold only RoI in memory, instead of the whole array.
    Args:
        data: np.array with markup itself.
        original_shape: tuple of the shapes of the image to be matched with segmentation.
        If left unattended, this class almost replicate the simple np.array with some tweaks for indexing and checking markup.
        boundaries: list of tuples to indicate to which region in original volume RoI patch belongs.
        mode_3d: bool indicator of supporting linear slicing from axes other then 0.
    """
    def __init__(self, data, original_shape=None, boundaries=None, mode_3d=False):
        PendingDeprecationWarning('Using ExpandedPaddedSegmentation will be deprecated, as this class doesn"t look usable anymore. Please, write maintainers if you disagree.')

        if isinstance(data, str):
            data = imread(data)

        if original_shape is None:
            self.is_expanded = False

            self.original_shape = data.shape
            self.boundaries = []
            for ax in range(len(data.shape)):
                axes_to_sum = tuple([i for i in range(len(data.shape)) if i!=ax])
                is_zero_along_axis = ((data > 0).sum(axes_to_sum))
                f,t = np.where(is_zero_along_axis)[0][[0, -1]]
                self.boundaries.append((f, t))

            self.data = data
        else:
            self.is_expanded = True

            self.original_shape = original_shape
            self.boundaries = boundaries

            shapes_original = original_shape if boundaries is None else map(lambda p: p[1]-p[0], boundaries)
            self.data = skimage.transform.resize(data, shapes_original, order=0, preserve_range=True)

        self.mode_3d = mode_3d
        self.len = sum(self.original_shape) if mode_3d else self.original_shape[0]

    def __len__(self):
        return self.len

    def _contains_markup(self):
        is_marked = []
        axes_to_search = [0, 1, 2] if self.mode_3d else [0]
        for ax in axes_to_search:
            local_marked = np.zeros(self.original_shape[ax])
            local_marked[self.boundaries[ax][0]: self.boundaries[ax][1]] = 1
            is_marked.append(local_marked)

        return np.concatenate(is_marked)


    def _is_empty(self, id, axis=None):
        if axis is None:
            axis, id = convert_id_to_3d(id, self.original_shape)
        if (id < self.boundaries[axis][0]) or (id >= self.boundaries[axis][1]):
            return True
        else:
            return False

    def _expand_markup(self, axis, id):
        if self.is_expanded:
            new_shape = [s for i,s in enumerate(self.original_shape) if i != axis]
            new_image = np.zeros(tuple(new_shape))

            if self._is_empty(id):
                return new_image

            new_patch_position = tuple([slice(*s) for i,s in enumerate(self.boundaries) if i!= axis])
            if axis == 0:
                markup_patch = self.data[id - self.boundaries[axis][0]]
            else:
                markup_patch = self.data.take(id - self.boundaries[axis][0], axis)

            new_image[new_patch_position] = markup_patch
        else:
            if axis == 0:
                new_image = self.data[id]
            else:
                new_image = self.data.take(id, axis)

        return new_image

    def __getitem__(self, id):
        internal_ax, internal_id = convert_id_to_3d(id, self.original_shape)
        return self._expand_markup(internal_ax, internal_id)

class SlicedSegmentation:
    def __init__(self, data, mode_3d=False, use_ram=True, label_converter=None):
        self.mode_3d = mode_3d
        self.use_ram = use_ram 
        self.label_converter = label_converter

        if isinstance(data, str):
            self.data = imread(data, lazy=(not self.use_ram))
        else:
            self.data = data
        if self.label_converter is not None:
            if self.use_ram or not isinstance(data, str):
                self.data = np.vectorize(self.label_converter.get)(self.data)
                self.label_converter = None # already converted, removing to avoid dynamically converting it

        self.original_shape = self.data.shape #left for compatibility with ExpandedPaddedSegmentation should probably be deprecated later

        self.len = sum(self.data.shape) if mode_3d else self.data.shape[0]
    
    def __len__(self):
        return self.len
    
    def _contains_markup(self):
        is_marked = []
        axes_to_search = [0, 1, 2] if self.mode_3d else [0]
        for ax in axes_to_search:
            all_axes = [0, 1, 2]
            all_axes.pop(ax)
            is_marked.append(self.data.sum(tuple(all_axes)) > 0)

        return np.concatenate(is_marked)

    def _is_empty(self, id, axis=None):
        DeprecationWarning('method _is_empty() is deprecated and will be removed in future releases. Please, use _contains_markup()')
        if axis is None:
            id, axis = convert_id_to_3d(id, self.data.shape)
        return self.data.take(id, axis).sum() > 0

    def __getitem__(self, id):
        internal_ax, internal_id = convert_id_to_3d(id, self.original_shape)

        if internal_ax == 0:
            slc = self.data[internal_id]
        else:
            slc = self.data.take(internal_id, internal_ax)
        
        if self.label_converter is not None:
            slc = np.vectorize(self.label_converter.get)(slc)
        
        return slc

class ConvertableBoundingBoxes:
    def __init__(self, bboxes, shapes, coord_format='int'):
        self.bboxes = bboxes
        self.shapes = shapes

    def _convert_int_to_format(self, bbox, coord_format):
        if coord_format == 'int':
            return bbox

        bbox_pascal = [((bb[1][0], bb[0][0], bb[1][1], bb[0][1]), c) for bb,c in bbox]

        if coord_format == 'pascal_voc':
            return bbox_pascal
        else:
            raise NotImplementedError('Conversion to other formats of coordinates is not implemented yet!')

    @classmethod
    def from_segmentation(cls, segmentation):
        bboxes = []
        for lbl in np.unique(segmentation):
            if lbl == 0:
                continue # background label. Probably should be more flexible on background selection.

            sub_label = (segmentation == lbl)
            # generate separate bounding boxes for each isolated area
            connected_components = find_connected_regions(sub_label)
            for region_id in np.unique(connected_components)[1:]:
                current_bbox = []
                region = (connected_components == region_id)
                for ax in (0, 1, 2):
                    axes_to_sum = tuple([i for i in (0, 1, 2) if i!= ax])
                    current_bbox.append(np.where(region.sum(axes_to_sum) > 0)[0][[0, -1]])
                bboxes.append((current_bbox, lbl))
        #TODO: apply NMS of some kind
        return cls(bboxes, segmentation.shape)

    def __getitem__(self, id, return_format='pascal_voc'):
        internal_ax, internal_id = convert_id_to_3d(id, self.shapes)
        bboxes_2d = []
        for bb, i in self.bboxes:
            if (internal_id >= bb[internal_ax][0]) and (internal_id < bb[internal_ax][1]):
                bboxes_2d.append(([boundaries for j, boundaries in enumerate(bb) if j != internal_ax], i))

        if len(bboxes_2d) > 0:
            return self._convert_int_to_format(bboxes_2d, return_format)
        else:
            return None


class OneVolume:
    """
    Class to wrap around different types of volumetric data to be processed.
    Args:
        data: string or np.array with address of the data or data itself respectively
        use_ram: bool if set to false while using tiff format will not load whole volume to ram, will directly read slices from drive.
        pixel_transform: callable for pixel-wise transformation, e.g. to cast to another type or scale-shift values
    """
    def __init__(self, data, mode_3d=False, use_ram=True, pixel_transform=None):
        self.is_memmapped = False
        if isinstance(data, str):
            if use_ram:
                self.image = imread(data)
            else:
                if data.endswith('.tif') or data.endswith('.tiff'):
                    self.file_addr = data
                    self.is_memmapped = True
                    self.shapes = imread(self.file_addr, lazy=True).shape
                else:
                    self.image = internal_medload(data)
        else:
            self.image = data

        if not self.is_memmapped:
            self.shapes = self.image.shape

        self.mode_3d = mode_3d
        if mode_3d:
            self.len = sum(self.shapes)
        else:
            self.len = self.shapes[0]

        if self.is_memmapped:
            self.pixel_transform = pixel_transform
        else:
            if pixel_transform is not None:
                self.image = pixel_transform(self.image)

    def __len__(self):
        return self.len

    def __getitem__(self, id):
        internal_ax, internal_id = convert_id_to_3d(id, self.shapes)

        img = imread(self.file_addr, lazy=True) if self.is_memmapped else self.image
        if internal_ax == 0:
            slc = img[internal_id] # this acts way faster for large arrays
        else:
            slc = img.take(internal_id, internal_ax)

        if self.is_memmapped and (self.pixel_transform is not None):
            slc = self.pixel_transform(slc)

        return slc

def get_mrnn_by_mask(mask):
    zone = find_connected_regions(mask)
    boxes, labels, masks = [], [], []
    for i, s in zip(*np.unique(zone, return_counts=True)):
        c = mask[zone==i][0]
        if c == 0:
            continue # skipping background
        if s <= 10:
            continue # skipping noisy zones
        box_y, box_x = np.where(zone == i) # I guess...
        box = [box_x.min(), box_y.min(), box_x.max()+1, box_y.max()+1]
        boxes.append(box)
        
        labels.append(c)
        
        masks.append((zone==i).astype(np.uint8))
    
    return {'boxes': np.array(boxes).astype(np.float32), 
            'labels': np.array(labels).astype(np.int64), 
            'masks': np.array(masks)}

class VolumeSlicingDataset(Dataset):
    """
    Dataset class which works for all possible cases: 2d and 3d classification, segmentation and localisation.
    It allows both slicing from the 0 axis and from all axes.

    Args:
        volume: [np.array, str, OneVolume] is the image itself. If array or string is passed will be converted to the OneVolume with usage of one_volume_kwargs param.
        volume_kwargs: [dict] is the parameters to convert address or np.array to the OneVolume
        mode_3d: [bool] if set to True will override this parameter for both volume and segmentation. Will allow slicing along all different axes, not only 0 axis will be sliced.
        augmentations: [albumentations.Compose] -- set of albumentation augmentations, will be applied for all available targets.
        crop_size: [int, tuple(int)] size of the image crop to be done from each slice. If one int passed will be square crop.
        localised_crop: [bool] if True, will localise the crop around the segmentation labels/bounding box if available on current slice.
        class_label_3d: single or tuple of values to be returned universally for each slice taken from this volume.
        class_label_2d: [np.array, list] classes for each possible slice (either of length volume.shape[0] or sum(volume.shape) if mode_3d is True), to be returned for slices specifically.
        bounding_box: list(tuple(list(tuple), int)) -- 3D bounding boxes list. Each bounding box in list should be pair of list of coordinates of start and end across each axis and class.
        segmentation:[np.array, str, ExpandedPaddedSegmentation] -- either address, volume or special class of segmentation. If address or volume passed, will be converted to the ExpandedPaddedSegmentation.
        segmentation_kwargs: [dict] -- paramters to convert volume or address of segmentation masks to the ExpandedPaddedMarkup.
        task: [str] -- either 'auto', which means everything passed to the class will be returned, or possible tasks from list of 'bbox,segmentation,class_2d,class_3d' separated by comma.
            Some conversions are possible. if bounding boxes or segmentation masks are provided, labels for 2d classification would be generated. If segmentation is provided, bounding_boxes will be generated.
    """
    def __init__(self, volume, volume_kwargs=None, 
                 mode_3d=False, use_ram=True,
                 augmentations=None,
                 crop_size=None, localised_crop=False,
                 class_label_3d=None, class_label_2d=None, bounding_box=None,
                 segmentation=None, segmentation_kwargs=None, 
                 task='auto',
                 additional_slices_aside=0, step_slices_aside=1):

        volume_kwargs = volume_kwargs or {}
        segmentation_kwargs = segmentation_kwargs or {}

        default_parameters = {'mode_3d': mode_3d, 'use_ram': use_ram}
        
        volume_kwargs = default_parameters | volume_kwargs
        segmentation_kwargs = default_parameters | segmentation_kwargs

        if isinstance(volume, OneVolume):
            self.volume = volume
        else:
            if volume_kwargs is None:
                volume_kwargs = {}
            if mode_3d:
                volume_kwargs['mode_3d'] = True
            self.volume = OneVolume(volume, **volume_kwargs)

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        if crop_size is not None:
            if localised_crop:
                self.cropper = AlbuCompose([PadIfNeeded(*crop_size, always_apply=True), CropNonEmptyMaskIfExists(*crop_size, always_apply=True)], bbox_params=BboxParams('pascal_voc', label_fields=['bbox_labels']))
            else:
                self.cropper = AlbuCompose([PadIfNeeded(*crop_size, always_apply=True), RandomCrop(*crop_size, always_apply=True)], bbox_params=BboxParams('pascal_voc', label_fields=['bbox_labels']))
        else:
            self.cropper = None

        self.augmentations = augmentations

        self.asa = additional_slices_aside
        self.ssa = step_slices_aside

        if self.asa > 0:
            self.cropper.add_targets({f'image{i}':'image' for i in range(self.asa*2)})
            self.augmentations.add_targets({f'image{i}':'image' for i in range(self.asa*2)})

        self.class_label_2d = None
        self.class_label_3d = None
        self.bounding_box = None
        self.segmentation = None
        self.values_to_return = []

        if class_label_2d is not None:
            self.class_label_2d = class_label_2d
            if task == 'auto':
                self.values_to_return.append('class_2d')

        if class_label_3d is not None:
            self.class_label_3d = class_label_3d
            if task == 'auto':
                self.values_to_return.append('class_3d')

        if bounding_box is not None:
            self.bounding_box = ConvertableBoundingBoxes(bounding_box, self.volume.shapes)
            if task == 'auto':
                self.values_to_return.append('bbox')

        if segmentation is not None:
            if isinstance(segmentation, (ExpandedPaddedSegmentation, SlicedSegmentation)):
                self.segmentation = segmentation
            else:
                if segmentation_kwargs is None:
                    segmentation_kwargs = {}
                if mode_3d:
                    segmentation_kwargs['mode_3d'] = True
                if ('original_shape' in segmentation_kwargs) and (segmentation_kwargs['original_shape'] is not None):
                    self.segmentation = ExpandedPaddedSegmentation(segmentation, **segmentation_kwargs)
                else:
                    self.segmentation = SlicedSegmentation(segmentation, **segmentation_kwargs)

            if task == 'auto':
                self.values_to_return.append('segmentation')

        if task != 'auto':
            self.values_to_return += task.split(',')

            if 'class_3d' in self.values_to_return:
                if self.class_label_3d is None:
                    raise ValueError('To return class_3d we need class_labels_3d passed!')

            if 'segmentation' in self.values_to_return:
                if self.segmentation is None:
                    raise ValueError('To return segmentation we need segmentation passed!')

            if 'bbox' in self.values_to_return:
                if self.bounding_box is not None:
                    pass
                elif self.segmentation is not None:
                    self.bounding_box = ConvertableBoundingBoxes.from_segmentation(self.segmentation.data) #TODO: probably neede workaround for padded data
                else:
                    raise ValueError('To return bbox we need either bbox or segmentation!')

            if 'class_2d' in self.values_to_return:
                if self.class_label_2d is not None:
                    pass
                elif self.bounding_box is not None:
                    self.class_label_2d = self._convert_bounding_box_to_classes(self.bounding_box.bboxes) #TODO: add here max_class
                elif self.segmentation is not None:
                    self.class_label_2d = self._convert_segmentation_to_classes(self.segmentation) #TODO: add here max_class
                else:
                    raise ValueError('To return class_2d we need either class_2d, bounding_box or segmentation!')

            if 'mask_rcnn' in self.values_to_return:
                if self.segmentation is None:
                    raise ValueError('To return MaskRCNN labels we need segmentation passed!')

    def _convert_bounding_box_to_classes(self, bounding_boxes, max_class=None):
        if self.volume.mode_3d:
            axes_to_scan = [0, 1, 2]
        else:
            axes_to_scan = [0]

        if max_class is None:
            max_class = np.unique([i[1] for i in bounding_boxes]).max()

        classes = np.zeros((len(self.volume), max_class))

        for box, lbl in bounding_boxes:
            shift = 0
            for ax in axes_to_scan:
                classes[box[ax][0]+shift:box[ax][1]+shift, lbl-1] = 1
                shift += self.volume.shapes[ax]

        return classes

    def _convert_segmentation_to_classes(self, segmentation, max_class=None):
        if max_class is None:
            max_class = np.unique(segmentation.data).max()

        classes = np.zeros((len(self.volume), max_class))

        for i in range(len(segmentation)):
            for lbl in np.unique(segmentation[i])[1:]:
                classes[i, lbl-1] = 1

        return classes

    def __len__(self):
        return len(self.volume)

    def _augment_parallel(self, aug, image, **other_targets):
        image_as_dict = OrderedDict()
        keys = ['image'] + [f'image{i}' for i in range(len(image)-1)]
        for k,v in zip(keys, image):
            image_as_dict[k] = v
        
        augmented = aug(**image_as_dict, **other_targets)
        augmented_image = np.stack([augmented.pop(i) for i in image_as_dict.keys()])
        augmented['image'] = augmented_image
        return augmented

    def _get_augmentation(self, augmentation, image, bbox, mask):
        dict_input = {'image': image}

        if bbox is not None:
            dict_input['bboxes'], dict_input['bbox_labels'] = zip(*bbox)
        else:
            dict_input['bboxes'] = []
            dict_input['bbox_labels'] = []
        if mask is not None:
            dict_input['mask'] = mask
        
        if self.asa > 0:
            augmented = self._augment_parallel(augmentation, **dict_input)
        else:
            augmented = augmentation(**dict_input)

        if bbox is not None:
            bbox = zip(augmented['bboxes'], augmented['bbox_labels'])
        if mask is not None:
            mask = augmented['mask']
        image = augmented['image']

        return image, bbox, mask

    def _get_atrous_slices(self, id):
        slices = np.array(list(range(id-self.asa*self.ssa, id+self.asa*self.ssa+1, self.ssa)))
        slices = np.clip(slices, 0, len(self.volume)-1)
        return np.stack([self.volume[i] for i in slices])

    def __getitem__(self, id):
        if self.asa > 0:
             img = self._get_atrous_slices(id)
        else:
            img = self.volume[id]
        lbl_2d = None if self.class_label_2d is None else self.class_label_2d[id]
        lbl_3d = self.class_label_3d
        bbox = self.bounding_box[id] if self.bounding_box is not None else None
        segm = self.segmentation[id] if self.segmentation is not None else None


        if self.cropper is not None:
            if isinstance(self.cropper.transforms[1], CropNonEmptyMaskIfExists):
                crop_mask = np.zeros_like(img)
                if segm is not None:
                    crop_mask = deepcopy(segm)
                if bbox is not None:
                    for bb, i in bbox:
                        crop_mask[bb[1]:bb[3], bb[0]:bb[2]] = i
                temp_mask = np.stack([segm, crop_mask], -1) #TODO: will not work for multichannel masks now

                img, bbox, temp_mask = self._get_augmentation(self.cropper, img, bbox, temp_mask)
                segm = temp_mask[..., 0]

            elif isinstance(self.cropper.transforms[1], RandomCrop):
                img, bbox, segm = self._get_augmentation(self.cropper, img, bbox, segm)

            else:
                raise ValueError(f'Unknown type of cropper! Got {type(self.cropper)}.')

        if self.augmentations is not None:
            img, bbox, segm = self._get_augmentation(self.augmentations, img, bbox, segm)

        if img.ndim == 2:
            img = img[None, ...]
        if (segm is not None) and (segm.ndim == 2):
            segm = segm[None, ...]


        values_dict = {'class_3d': lbl_3d, 'class_2d': lbl_2d, 'bbox': bbox, 'segmentation': segm}
        if 'mask_rcnn' in self.values_to_return:
            # nothing else will be returned. Pots zochvatchen!!1
            return (img, get_mrnn_by_mask(segm[0]), segm) # last one for measuring IoU

        return [img] + [values_dict[val] for val in self.values_to_return]
