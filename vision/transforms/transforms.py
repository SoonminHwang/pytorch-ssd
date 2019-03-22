# from https://github.com/amdegroot/ssd.pytorch


import torch
from torchvision import transforms
import cv2
import numpy as np
import types
# from numpy import random
import random

import pdb
import math
from PIL import Image

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[:,2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:,:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


# class ConvertFromInts(object):
#     def __call__(self, image, boxes=None, labels=None):
#         return image.astype(np.float32), boxes, labels


# class SubtractMeans(object):
#     def __init__(self, mean):
#         self.mean = np.array(mean, dtype=np.float32)

#     def __call__(self, image, boxes=None, labels=None):
#         image = image.astype(np.float32)
#         image -= self.mean
#         return image.astype(np.float32), boxes, labels

class Normalize(object):
    def __init__(self, mean, std):
        self.func = transforms.Normalize(mean, std)

    def __call__(self, image, boxes=None, labels=None):
        image = self.func(image) 
        return image, boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):

        try:
            if all(boxes.shape):
                # height, width, channels = image.shape
                height = image.height
                width = image.width
            
                boxes[:, 0] /= width
                boxes[:, 2] /= width
                boxes[:, 1] /= height
                boxes[:, 3] /= height
        except:
            import pdb
            pdb.set_trace()
        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300, max_size=1000, random_interpolation=False):

        self.size = (size, size) if isinstance(size, int) else size
        self.max_size = max_size
        self.random_interpolation = random_interpolation

    def __call__(self, image, boxes=None, labels=None):
        w, h = image.size
    
        ow, oh = self.size
        sw = float(ow) / w
        sh = float(oh) / h

        # if isinstance(self.size, int):
        #     size_min = min(w,h)
        #     size_max = max(w,h)
        #     sw = sh = float(self.size) / size_min
        #     if sw * size_max > self.max_size:
        #         sw = sh = float(self.max_size) / size_max
        #     ow = int(w * sw + 0.5)
        #     oh = int(h * sh + 0.5)
        # else:
        #     ow, oh = self.size
        #     sw = float(ow) / w
        #     sh = float(oh) / h

        method = random.choice([
            Image.BOX,
            Image.NEAREST,
            Image.HAMMING,
            Image.BICUBIC,
            Image.LANCZOS,
            Image.BILINEAR]) if self.random_interpolation else Image.BILINEAR

        image = image.resize((ow,oh), method)

        if boxes is not None:
            # boxes = boxes * torch.tensor([sw,sh,sw,sh])
            boxes = boxes * np.array([[sw,sh,sw,sh]])

        return image, boxes, labels

        # image = image.astype(np.uint8)
        # image = cv2.resize(image, (self.size, self.size))
        
        # image = image.astype(np.float32)
        # import pdb
        # pdb.set_trace()
        # image = np.array( Image.fromarray(image.astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR), dtype=np.float32 )
        # image = np.array( Image.fromarray(image).resize((self.size, self.size), Image.BICUBIC), dtype=np.float32 )
        return image, boxes, labels


# class RandomSaturation(object):
#     def __init__(self, lower=0.5, upper=1.5):
#         self.lower = lower
#         self.upper = upper
#         assert self.upper >= self.lower, "contrast upper must be >= lower."
#         assert self.lower >= 0, "contrast lower must be non-negative."

#     def __call__(self, image, boxes=None, labels=None):
#         if np.random.randint(2):
#             image[:, :, 1] *= random.uniform(self.lower, self.upper)

#         return image, boxes, labels


# class RandomHue(object):
#     def __init__(self, delta=18.0):
#         assert delta >= 0.0 and delta <= 360.0
#         self.delta = delta

#     def __call__(self, image, boxes=None, labels=None):
#         if np.random.randint(2):
#             image[:, :, 0] += random.uniform(-self.delta, self.delta)
#             image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
#             image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
#         return image, boxes, labels


# class RandomLightingNoise(object):
#     def __init__(self):
#         self.perms = ((0, 1, 2), (0, 2, 1),
#                       (1, 0, 2), (1, 2, 0),
#                       (2, 0, 1), (2, 1, 0))

#     def __call__(self, image, boxes=None, labels=None):
#         if np.random.randint(2):
#             swap = self.perms[np.random.randint(len(self.perms))]
#             shuffle = SwapChannels(swap)  # shuffle channels
#             image = shuffle(image)
#         return image, boxes, labels


# class ConvertColor(object):
#     def __init__(self, current, transform):
#         self.transform = transform
#         self.current = current

#     def __call__(self, image, boxes=None, labels=None):
#         image = np.array( Image.fromarray(image.astype(np.uint8)).convert(self.transform), dtype=np.float32 )
#         # if self.current == 'BGR' and self.transform == 'HSV':
#         #     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         # elif self.current == 'RGB' and self.transform == 'HSV':
#         #     image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#         # elif self.current == 'BGR' and self.transform == 'RGB':
#         #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # elif self.current == 'HSV' and self.transform == 'BGR':
#         #     image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
#         # elif self.current == 'HSV' and self.transform == "RGB":
#         #     image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
#         # else:
#         #     raise NotImplementedError
#         return image, boxes, labels

# ## From https://github.com/kuangliu/torchcv/blob/master/torchcv/transforms/random_distort.py
# class RandomContrast(object):
#     def __init__(self, delta=0.5):
#         self.delta = delta
#         self.func = transforms.ColorJitter(contrast=delta)
    
#     def __call__(self, image, boxes=None, labels=None):
#         if np.random.randint(2):
#             image = self.func(image)
#         return image, boxes, labels


# class RandomBrightness(object):
#     def __init__(self, delta=32/255.):
#         self.delta = delta
#         self.func = transforms.ColorJitter(brightness=delta)

#     def __call__(self, image, boxes=None, labels=None):
#         if np.random.randint(2):
#             img = self.func(image)
#         return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __init__(self):
        self.func = transforms.ToTensor()

    def __call__(self, pilimage, boxes=None, labels=None):
        # return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels
        return self.func(pilimage), boxes, labels

# from vision.utils.box_utils import iou_of
def box_clamp(boxes, xmin, ymin, xmax, ymax):
    '''Clamp boxes.
    Args:
      boxes: (np.array) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.
    Returns:
      (tensor) clamped boxes.
    '''
    boxes[:,(0,2)] = np.clip(boxes[:,(0,2)], a_min=xmin, a_max=xmax)
    boxes[:,(1,3)] = np.clip(boxes[:,(1,3)], a_min=ymin, a_max=ymax)
    
    return boxes

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, min_scale=0.3, max_aspect_ratio=2.):
        self.min_scale = min_scale
        self.max_aspect_ratio = max_aspect_ratio
        # self.sample_options = (
        #     # using entire original input image
        #     None,
        #     # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
        #     (0.1, None),
        #     (0.3, None),
        #     (0.7, None),
        #     (0.9, None),
        #     # randomly sample a patch
        #     (None, None),
        # )

    def __call__(self, image, boxes=None, labels=None):
        imw, imh = image.size
        params = [(0, 0, imw, imh)]  # crop roi (x,y,w,h) out
        for min_iou in (0, 0.1, 0.3, 0.5, 0.7, 0.9):
            for _ in range(50):
                scale = random.uniform(self.min_scale, 1)
                aspect_ratio = random.uniform(
                    max(1/self.max_aspect_ratio, scale*scale),
                    min(self.max_aspect_ratio, 1/(scale*scale)))
                w = int(imw * scale * math.sqrt(aspect_ratio))
                h = int(imh * scale / math.sqrt(aspect_ratio))

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)

                # roi = torch.tensor([[x,y,x+w,y+h]], dtype=torch.float)
                roi = np.array([[x,y,x+w,y+h]], dtype=np.float32)
                # ious = iou_of(boxes, roi)
                ious = jaccard_numpy(boxes, roi)
                if ious.min() >= min_iou:
                    params.append((x,y,w,h))
                    break

        x,y,w,h = random.choice(params)
        image = image.crop((x,y,x+w,y+h))

        center = (boxes[:,:2] + boxes[:,2:]) / 2
        mask = (center[:,0]>=x) & (center[:,0]<=x+w) \
             & (center[:,1]>=y) & (center[:,1]<=y+h)
        if mask.any():
            # boxes = boxes[mask] - torch.tensor([x,y,x,y], dtype=torch.float)
            boxes = boxes[mask] - np.array([x,y,x,y], dtype=np.float32)
            boxes = box_clamp(boxes,0,0,w,h)
            labels = labels[mask]
        else:
            # boxes = torch.tensor([[0,0,0,0]], dtype=torch.float)
            # labels = torch.tensor([0], dtype=torch.long)
            # boxes = np.array([[]], dtype=np.float32)
            # labels = np.array([], dtype=np.long)
            boxes = np.zeros((0,4), dtype=np.float32)
            labels = np.zeros((0), dtype=np.long)
        return image, boxes, labels


        # height, width, _ = image.shape
        # while True:
        #     # randomly choose a mode
        #     mode = random.choice(self.sample_options)
        #     if mode is None:
        #         return image, boxes, labels

        #     min_iou, max_iou = mode
        #     if min_iou is None:
        #         min_iou = float('-inf')
        #     if max_iou is None:
        #         max_iou = float('inf')

        #     # max trails (50)
        #     for _ in range(50):
        #         current_image = image

        #         w = random.uniform(0.3 * width, width)
        #         h = random.uniform(0.3 * height, height)

        #         # aspect ratio constraint b/t .5 & 2
        #         if h / w < 0.5 or h / w > 2:
        #             continue

        #         left = random.uniform(width - w)
        #         top = random.uniform(height - h)

        #         # convert to integer rect x1,y1,x2,y2
        #         rect = np.array([int(left), int(top), int(left+w), int(top+h)])

        #         # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
        #         overlap = jaccard_numpy(boxes, rect)

        #         # is min and max overlap constraint satisfied? if not try again
        #         if overlap.min() < min_iou and max_iou < overlap.max():
        #             continue

        #         # cut the crop from the image
        #         # current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
        #         #                               :]
        #         current_image = current_image.crop(rect)

        #         # keep overlap with gt box IF center in sampled patch
        #         centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

        #         # mask in all gt boxes that above and to the left of centers
        #         m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

        #         # mask in all gt boxes that under and to the right of centers
        #         m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

        #         # mask in that both m1 and m2 are true
        #         mask = m1 * m2

        #         # have any valid boxes? try again if not
        #         if not mask.any():
        #             continue

        #         # take only matching gt boxes
        #         current_boxes = boxes[mask, :].copy()

        #         # take only matching gt labels
        #         current_labels = labels[mask]

        #         # should we use the box left and top corner or the crop's
        #         current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
        #                                           rect[:2])
        #         # adjust to crop (by substracting crop's left,top)
        #         current_boxes[:, :2] -= rect[:2]

        #         current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
        #                                           rect[2:])
        #         # adjust to crop (by substracting crop's left,top)
        #         current_boxes[:, 2:] -= rect[:2]

        #         return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean, max_ratio=4):
        self.mean = tuple(mean)
        self.max_ratio = max_ratio

    def __call__(self, image, boxes, labels):

        w, h = image.size
        ratio = random.uniform(1, self.max_ratio)
        ow, oh = int(w*ratio), int(h*ratio)

        if ow == w or oh == h:
            return image, boxes, labels

        canvas = Image.new('RGB', (ow,oh), self.mean)

        x = np.random.randint(0, ow - w)
        y = np.random.randint(0, oh - h)
        canvas.paste(image, (x,y))

        if boxes is not None:
            # boxes = boxes + torch.tensor([x,y,x,y], dtype=torch.float)
            boxes = boxes + np.array([[x,y,x,y]], dtype=np.float32)


        # if np.random.randint(2):
        #     return image, boxes, labels



        # height, width, depth = image.shape
        # ratio = random.uniform(1, 4)
        # left = random.uniform(0, width*ratio - width)
        # top = random.uniform(0, height*ratio - height)

        # expand_image = np.zeros(
        #     (int(height*ratio), int(width*ratio), depth),
        #     dtype=image.dtype)
        # expand_image[:, :, :] = self.mean
        # expand_image[int(top):int(top + height),
        #              int(left):int(left + width)] = image
        # image = expand_image

        # boxes = boxes.copy()
        # boxes[:, :2] += (int(left), int(top))
        # boxes[:, 2:] += (int(left), int(top))

        return canvas, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
                
        if np.random.randint(2):            
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w = image.width
            if boxes is not None:
                xmin = w - boxes[:,2]
                xmax = w - boxes[:,0]
                boxes[:,0] = xmin
                boxes[:,2] = xmax

        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self, brightness_delta=32/255., contrast_delta=0.5, saturation_delta=0.5, hue_delta=0.1):
        self.pd = [
            transforms.ColorJitter(contrast=contrast_delta),
            transforms.ColorJitter(saturation=saturation_delta),
            transforms.ColorJitter(hue=hue_delta),
            transforms.ColorJitter(contrast=contrast_delta)            
        ]

        self.rand_brightness = transforms.ColorJitter(brightness=brightness_delta)

        self.distort_csh = self.pd[:-1]
        self.distort_shc = self.pd[1:]
        # self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        # im = image.copy()
        # im, boxes, labels = self.rand_brightness(im, boxes, labels)
        image = self.rand_brightness(image)
        if np.random.randint(2):
            distort = self.distort_csh
        else:
            distort = self.distort_shc

        for t in distort:
            image = t(image)
        
        # return self.rand_light_noise(im, boxes, labels)
        return image, boxes, labels

