from ..transforms.transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0, mean_tensor=None, stds_tensor=None):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        # self.mean = mean
        # self.std = std
        # self.size = size
        self.augment = Compose([
            # ConvertFromInts(),
            PhotometricDistort(),
            Expand(mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(size),
            ToTensor(),
            Normalize(mean_tensor, stds_tensor),
            # SubtractMeans(self.mean),
            # Lambda(lambda img, boxes=None, labels=None: (img / std, boxes, labels)),                        
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0, mean_tensor=None, stds_tensor=None):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            # SubtractMeans(mean),
            # lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
            Normalize(mean_tensor, stds_tensor)
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean_tensor=None, stds_tensor=None):
        self.transform = Compose([
            Resize(size),
            ToTensor(),
            Normalize(mean_tensor, stds_tensor)
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image