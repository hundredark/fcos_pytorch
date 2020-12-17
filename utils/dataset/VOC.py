import os
import cv2
import random
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from config import DefaultConfig


class VOCDatasetMS(Dataset):
    def __init__(self, config, root_dir, split='trainval',
                 use_difficult=False, is_train=True, transforms=None):
        self.config = config
        self.root = root_dir
        self.imgset = split
        self.use_difficult = use_difficult
        self.train = is_train
        self.transforms = transforms(self.config.mean, self.config.std)

        if self.config.scale_mode == 'single':
            self.scales = [[640, 800]]
        elif self.config.scale_mode == 'multi':
            self.scales = [[480, 800], [512, 800], [544, 800], [576, 800], [608, 800], [640, 800]]

        self._annopath = os.path.join(self.root, "Annotations", "{}.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "{}.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "{}.txt")

        with open(self._imgsetpath.format(self.imgset)) as f:
            self.img_ids = f.readlines()
        self.img_ids = [x.strip() for x in self.img_ids]

        self.name2id = dict(zip(self.config.class_names, range(len(self.config.class_names))))
        self.id2name = {v: k for k, v in self.name2id.items()}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        if True:  # not self.train or random.random() <= 0.5:
            img, boxes, classes = self.load_img_and_boxes(index)

        img_size = random.choice(self.scales)
        img, boxes = self.preprocess_img_boxes(img, boxes, img_size)


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
                # Make coords indexes 0-based
                TO_REMOVE = 1
                _box = obj.find("bndbox")
                for pos in ["xmin", "ymin", "xmax", "ymax"]:
                    box.append(float(_box.find(pos).text) - TO_REMOVE)
                boxes.append(box)

            boxes = np.array(boxes, dtype=np.float32)
            classes = np.array(classes, dtype=np.float32)
            return boxes, classes

        img_id = self.img_ids[index]
        img_path = self._imgpath.format(img_id)
        xml_path = self._annopath.format(img_id)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.uint8)
        boxes, classes = get_xml_label(xml_path)
        return img, boxes, classes

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        min_side, max_side = input_ksize
        h, w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        resize_w, resize_h = int(scale * w), int(scale * h)

        resize_w += 32 - resize_w % 32
        resize_h += 32 - resize_h % 32

        image_resized = cv2.resize(image, (resize_w, resize_h))
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (resize_w * 1. / w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (resize_h * 1. / h)
        return image_resized, boxes

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
