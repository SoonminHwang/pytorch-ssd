import numpy as np
import pathlib
import xml.etree.ElementTree as ET
import cv2
from PIL import Image

class VOCDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_names = ('BACKGROUND',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        )
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        # if not self.keep_difficult:
        #     boxes = boxes[is_difficult == 0]
        #     labels = labels[is_difficult == 0]

        labels[ is_difficult == 1 ] = -100

        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            bbox = object.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = object.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        if len(boxes) == 0:
            import pdb
            pdb.set_trace()
            return (np.zeros( (0,4), dtype=np.float32),
                np.zeros( (0,1), dtype=np.int64),
                np.zeros( (0,1), dtype=np.uint8)),                

        else:
            return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        # image = cv2.imread(str(image_file))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open( str(image_file) )
        # return np.array(image)
        return image


if __name__ == '__main__':    
    
    import sys, os, pdb
    CUR_DIR = os.path.abspath( os.path.join( os.path.dirname(__file__), '..', '..' ) )
    
    if CUR_DIR not in sys.path:
        sys.path.insert(0, CUR_DIR)

    from vision.ssd.data_preprocessing import TrainAugmentation
    from vision.ssd.ssd import MatchPrior
    from vision.ssd.config import vgg_ssd_config as config

    dataset_path = '/raid/datasets/pascal_voc/VOC2007/'

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std, config.image_mean_tensor, config.image_stds_tensor)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)


    # Profiling
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
        
    for ii, blob in enumerate(dataset):                
        if (ii+1) % 1000 == 0:
            print('{:}/{:}'.format(ii, len(dataset)))
    
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

