"""Base segmentation dataset"""
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480,pad_size=None,scale_ratio=0.75):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.pad_size = pad_size
        self.scale_ratio = scale_ratio

    def _val_sync_transform(self, img, mask):

        if self.pad_size:
            img = img.resize((self.pad_size[0],self.pad_size[1]),Image.BILINEAR)
            mask = mask.resize((self.pad_size[0],self.pad_size[1]),Image.BILINEAR)
        scale_ratio = self.scale_ratio
        if scale_ratio != None:
            w,h = img.size
            img = img.resize((int(w*scale_ratio),int(h*scale_ratio)),Image.BILINEAR)
            mask = mask.resize((int(w*scale_ratio),int(h*scale_ratio)),Image.BILINEAR)
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        if self.pad_size:
            img = img.resize((self.pad_size[0],self.pad_size[1]),Image.BILINEAR)
            mask = mask.resize((self.pad_size[0],self.pad_size[1]),Image.BILINEAR)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        if self.base_size:
            w, h = img.size 
            short = min(w,h)
            short_size = random.randint(int(short * 0.5), int(short * 2.0))
            if h > w:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                oh = short_size
                ow = int(1.0 * w * oh / h)
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)

        scale_ratio = self.scale_ratio
        if scale_ratio != None:
            w,h = img.size
            img = img.resize((int(w*scale_ratio),int(h*scale_ratio)),Image.BILINEAR)
            mask = mask.resize((int(w*scale_ratio),int(h*scale_ratio)),Image.BILINEAR)

        if self.crop_size:
            if short_size < crop_size:
                padh = crop_size - oh if oh < crop_size else 0
                padw = crop_size - ow if ow < crop_size else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

            w, h = img.size
            x1 = random.randint(0, w - crop_size)
            y1 = random.randint(0, h - crop_size)
            img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
