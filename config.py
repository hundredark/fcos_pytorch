import os
from utils.dataset.public_dataset_setting import *
import torch.optim as optim


class DefaultConfig:
    # path
    pretrained = "pretrained/darknet19-deepBakSu-e1b3ec1e.pth"
    assert os.path.exists(pretrained) or pretrained == False
    trainval_dir = "/home/fbc/Datasets/VOC/VOCdevkit/VOC2012"
    test_dir = "/home/fbc/Datasets/VOC/VOCdevkit/VOC2007"
    exp_dir = "exps"

    # dataset
    mean = imagenet_mean # [0.5, 0.5, 0.5]
    std = imagenet_std # [1., 1., 1.]
    # mean = [0.5, 0.5, 0.5]
    # std = [1., 1., 1.]
    scale_mode = "multi"
    assert scale_mode in ["single", "multi"]
    class_names = voc_class_name
    class_names = class_names if "__background__ " in class_names \
                            else class_names.insert(0, "__background__ ")
    class_num = len(class_names)-1

    # train
    SEED = 0
    BATCH_SIZE = 4
    EPOCHS = 24
    LR_INIT = 2e-3
    optimizer_class = optim.SGD
    optimizer_params = dict(lr=LR_INIT, momentum=0.9, weight_decay=1e-4)
    save_best = "loss"
    assert save_best in ["accuracy", "loss"]
    verbose_interval = 1
    # model
    backbone = "darknet19"
    assert backbone in "darknet19" or "resnet50"
    # fpn
    fpn_out_channels = 256
    # head
    prior = 0.01  # focal loss bias init
    add_centerness = True
    use_GN_head = False  # group norm
    cnt_on_reg = True  # cnt on reg or cls
    # target
    # ===== fcos =====
    strides = [8, 16, 32, 64, 128]
    limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
    assert len(strides) == len(limit_range)

    # inference
    score_threshold = 0.05
    nms_iou_threshold = 0.3
    max_detection_boxes_num = 150
