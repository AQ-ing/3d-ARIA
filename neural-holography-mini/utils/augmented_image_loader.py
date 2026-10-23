import os
import torch
import random
import numpy as np
import cv2
import torch.nn.functional as F
from imageio.v2 import imread
from skimage.transform import resize

import utils.utils as utils


class ImageLoader:
    """Loads images a folder with augmentation for generator training

    Class initialization parameters
    -------------------------------
    data_path: folder containing images
    channel: color channel to load (0, 1, 2 for R, G, B, None for all 3),
        default None
    batch_size: number of images to pass each iteration, default 1
    image_res: 2d dimensions to pad/crop the image to for final output, default
        (1080, 1920)
    homography_res: 2d dims to scale the image to before final crop to image_res
        for consistent resolutions (crops to preserve input aspect ratio),
        default (880, 1600)
    shuffle: True to randomize image order across batches, default True
    vertical_flips: True to augment with vertical flipping, default True
    horizontal_flips: True to augment with horizontal flipping, default True
    idx_subset: for the iterator, skip all but these images. Given as a list of
        indices corresponding to sorted filename order. Forces shuffle=False and
        batch_size=1. Defaults to None to not subset at all.
    crop_to_homography: if True, only crops the image instead of scaling to get
        to target homography resolution, default False
    direct_resize: resize directly to image_res, bypassing homography crop/pad,
        default False
    depth_path: optional depth file or directory paired by image filename;
        requires direct_resize=True and both flips disabled. load_batch then
        returns a fourth value: depths with shape [batch, 1, height, width].

    Usage
    -----
    To be used as an iterator:

    >>> image_loader = ImageLoader(...)
    >>> for ims, input_resolutions, filenames in image_loader:
    >>>     ...

    ims: images in the batch after transformation and conversion to linear
        amplitude, with dimensions [batch, channel, height, width]
    input_resolutions: list of length batch_size containing tuples of the
        original image height/width before scaling/cropping
    filenames: list of input image filenames, without extension

    Alternatively, can be used to manually load a single image:

    >>> ims, input_resolutions, filenames = image_loader.load_image(idx)

    idx: the index for the image to load, indices are alphabetical based on the
        file path.
    """

    def __init__(self, data_path, channel=None, batch_size=1,
                 image_res=(1080, 1920), homography_res=(880, 1600),
                 shuffle=True, vertical_flips=True, horizontal_flips=True,
                 idx_subset=None, crop_to_homography=False,
                 direct_resize=False, depth_path=None):
        if not os.path.isdir(data_path) and not os.path.isfile(data_path):
            raise NotADirectoryError(f'Data folder: {data_path}')
        self.data_path = data_path
        self.channel = channel
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_res = image_res
        self.homography_res = homography_res
        self.subset = idx_subset
        self.crop_to_homography = crop_to_homography
        self.direct_resize = direct_resize
        self.depth_path = depth_path
        if depth_path and (not direct_resize or vertical_flips or horizontal_flips):
            raise ValueError('Paired depth loading requires direct_resize=True and flips disabled')

        self.augmentations = []
        if vertical_flips:
            self.augmentations.append(self.augment_vert)
        if horizontal_flips:
            self.augmentations.append(self.augment_horz)
        # store the possible states for enumerating augmentations
        self.augmentation_states = [fn() for fn in self.augmentations]

        self.im_names = get_image_filenames(data_path) if os.path.isdir(data_path) else [data_path]
        self.im_names.sort()
        if not self.im_names:
            raise ValueError(f'No images in: {data_path}')
        self.depth_names = []
        if depth_path:
            self.depth_names = [os.path.join(depth_path, os.path.basename(path))
                                if os.path.isdir(depth_path) else depth_path for path in self.im_names]
            if not os.path.isdir(depth_path) and len(self.im_names) != 1:
                raise ValueError('A depth directory is required for multiple RGB images')
            for path in self.depth_names:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f'Missing paired depth: {path}')

        # if subsetting indices, force no randomization and batch size 1
        if self.subset is not None:
            self.shuffle = False
            self.batch_size = 1

        # create list of image IDs with augmentation state
        self.order = ((i,) for i in range(len(self.im_names)))
        for aug_type in self.augmentations:
            states = aug_type()  # empty call gets possible states
            # augment existing list with new entry to states tuple
            self.order = ((*prev_states, s)
                          for prev_states in self.order
                          for s in states)
        self.order = list(self.order)

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __next__(self):
        if self.subset is not None:
            while self.ind not in self.subset and self.ind < len(self.order):
                self.ind += 1

        if self.ind < len(self.order):
            batch_ims = self.order[self.ind:self.ind+self.batch_size]
            self.ind += self.batch_size
            return self.load_batch(batch_ims)
        else:
            raise StopIteration

    def __len__(self):
        if self.subset is None:
            return len(self.order)
        else:
            return len(self.subset)

    def load_batch(self, images):
        im_res_name = [self.load_image(*im_data) for im_data in images]
        ims = torch.stack([im for im, _, _ in im_res_name], 0)
        result = (ims,
                [res for _, res, _ in im_res_name],
                [name for _, _, name in im_res_name])
        if self.depth_path:
            depths = []
            for (index, *_), (_, source_res, _) in zip(images, im_res_name):
                depth = cv2.imread(self.depth_names[index], cv2.IMREAD_GRAYSCALE)
                if depth is None:
                    raise ValueError(f'Could not read depth: {self.depth_names[index]}')
                if depth.shape != source_res:
                    raise ValueError(f'RGB/depth dimensions differ: {self.im_names[index]}')
                depth = torch.from_numpy(utils.im2float(depth))[None, None]
                depths.append(F.interpolate(depth, self.image_res, mode='bilinear', align_corners=False)[0])
            result += (torch.stack(depths),)
        return result

    def load_image(self, filenum, *augmentation_states):
        im = imread(self.im_names[filenum])

        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)  # augment channels for gray images

        if self.channel is None:
            im = im[..., :3]  # remove alpha channel, if any
        else:
            # select channel while keeping dims
            im = im[..., self.channel, np.newaxis]

        im = utils.im2float(im, dtype=np.float64)  # convert to double, max 1

        # 复用 utils 的 sRGB 转换，再将线性强度转为振幅。
        im = np.sqrt(utils.srgb_gamma2lin(im))

        # move channel dim to torch convention
        im = np.transpose(im, axes=(2, 0, 1))

        # apply data augmentation
        for fn, state in zip(self.augmentations, augmentation_states):
            im = fn(im, state)

        # normalize resolution
        input_res = im.shape[-2:]
        if self.direct_resize:
            im = torch.from_numpy(np.ascontiguousarray(im)).float()[None]
            im = F.interpolate(im, self.image_res, mode='bilinear', align_corners=False)[0]
            return im, input_res, os.path.splitext(self.im_names[filenum])[0]
        if self.crop_to_homography:
            im = pad_crop_to_res(im, self.homography_res)
        else:
            im = resize_keep_aspect(im, self.homography_res)
        im = pad_crop_to_res(im, self.image_res)

        return (torch.from_numpy(im).float(),
                input_res,
                os.path.splitext(self.im_names[filenum])[0])

    def augment_vert(self, image=None, flip=False):
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1, :]
        return image

    def augment_horz(self, image=None, flip=False):
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1]
        return image


def get_image_filenames(dir):
    """Returns all files in the input directory dir that are images"""
    image_types = ('jpg', 'jpeg', 'tiff', 'tif', 'png', 'bmp', 'gif')
    files = os.listdir(dir)
    exts = (os.path.splitext(f)[1] for f in files)
    images = [os.path.join(dir, f)
              for e, f in zip(exts, files)
              if e[1:].lower() in image_types and os.path.isfile(os.path.join(dir, f))]
    return images


def resize_keep_aspect(image, target_res, pad=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # finds the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res, pytorch=False)

    # switch to numpy channel dim convention, resize, switch back
    image = np.transpose(image, axes=(1, 2, 0))
    image = resize(image, target_res, mode='reflect')
    return np.transpose(image, axes=(2, 0, 1))


def pad_crop_to_res(image, target_res):
    """Pads with 0 and crops as needed to force image to be target_res

    image: an array with dims [..., channel, height, width]
    target_res: [height, width]
    """
    return utils.crop_image(utils.pad_image(image,
                                            target_res, pytorch=False),
                            target_res, pytorch=False)
