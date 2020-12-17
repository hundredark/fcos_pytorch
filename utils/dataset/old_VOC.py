import os
import random
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
from config import DefaultConfig


class VOCDataset(Dataset):
    def __init__(self, root_dir, config=DefaultConfig, resize_size=[800, 1333], split='trainval',
                 use_difficult=False, is_train=True, transforms=None):
        self.root = root_dir
        self.imgset = split
        self.use_difficult = use_difficult
        self.train = is_train
        self.transforms = transforms
        self.resize_size = resize_size
        self.config = config

        self._annopath = os.path.join(self.root, "Annotations", "{}.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "{}.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "{}.txt")

        with open(self._imgsetpath.format(self.imgset)) as f:
            self.img_ids = f.readlines()
        self.img_ids = [x.strip() for x in self.img_ids]

        self.name2id = dict(zip(self.config.class_name, range(len(self.config.class_name))))
        self.id2name = {v: k for k, v in self.name2id.items()}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        r = random.random()
        if True: #not self.is_train or r <= 0.5:
            img, boxes, classes = self.load_img_and_boxes(index)
        else:
            img, boxes, classes = self.load_cutmix_image_and_boxes(index)
        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)
 
        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': img,
                    'bboxes': boxes,
                    'labels': classes
                })

                if len(sample['bboxes']) > 0:
                    img = sample['image']
                    boxes = torch.Tensor(sample['bboxes'])
                    classes = torch.LongTensor(sample['labels'])
                    break
        return img, boxes, classes

    def load_img_and_boxes(self, index):
        def get_xml_label(xml_path):
            root = ET.parse(xml_path).getroot()
            boxes = []
            classes = []

            for obj in root.iter("object"):
                if not self.use_difficult and int(obj.find("difficult").text):
                    continue

                label = obj.find("name").text.lower().strip()
                classes.append(self.name2id[label])

                box = []
                # VOC datasets starts from (1, 1) instead of (0, 0)
                TO_REMOVE = 1
                _box = obj.find("bndbox")
                for pos in ["xmin", "ymin", "xmax", "ymax"]:
                    box.append(float(_box.find(pos).text) - TO_REMOVE)
                boxes.append(box)

            boxes = np.array(boxes, dtype=np.float32)
            return boxes, classes

        img_id = self.img_ids[index]
        img_path = self._imgpath.format(img_id)
        xml_path = self._annopath.format(img_id)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.uint8)  # BGR2RGB
        boxes, classes = get_xml_label(xml_path)
        return img, boxes, classes

    # TODO: cutmix augmentation
    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []
        result_labels = []

        for i, index in enumerate(indexes):
            image, boxes, labels = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            result_labels.append(labels)

        result_boxes = np.concatenate(result_boxes, 0)
        result_labels = np.concatenate(result_labels, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        condition = np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)
        result_boxes = result_boxes[condition]
        result_labels = result_labels[condition]
        return result_image, result_boxes, result_labels

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        min_side, max_side = input_ksize
        h, w = image.shape[:-1]

        smallest_side = min(w, h)
        largest_side = max(w, h)

        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))  # cv.resize(img, (w, h))

        boxes = boxes * scale
        return image_resized, boxes

    # the size of data in a batch should be same
    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)

        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []
        max_num = np.array([box.shape[0] for box in boxes_list]).max()
        max_h = np.array([int(s.shape[1]) for s in imgs_list]).max()
        max_w = np.array([int(s.shape[2]) for s in imgs_list]).max()
        pad_h = 32 - max_h % 32
        pad_w = 32 - max_w % 32
        max_h += pad_h
        max_w += pad_w

        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(
                    torch.nn.functional.pad(img, [0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])], value=0.))
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], [0, 0, 0, max_num - boxes_list[i].shape[0]], value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], [0, max_num - classes_list[i].shape[0]], value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)
        return batch_imgs, batch_boxes, batch_classes
