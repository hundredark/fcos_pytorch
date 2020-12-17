import torch
import torch.nn as nn

from utils.backbone.darknet19 import Darknet19
from utils.backbone.resnet import resnet50
from utils.det_frames.fpn import FPN
from utils.det_frames.head import ClsCntRegHead
from utils.det_frames.fcos_loss import GenTargets, LOSS
from utils.utils.targets import coords_fmap2orig
from config import DefaultConfig


class FCOSDetector(nn.Module):
    def __init__(self, mode="training", config=None):
        super().__init__()
        self.config = config if config is not None else DefaultConfig
        assert mode in ["training", "inference"]
        self.mode = mode
        self.model = FCOS(config=self.config)

        if mode == "training":
            self.target_layer = GenTargets(self.config)
            self.loss_layer = LOSS(self.config)

        elif mode == "inference":
            self.detection_head = DetectHead(self.config)
            self.clip_boxes = ClipBoxes()

    def forward(self, inputs):
        if self.mode == "training":
            batch_imgs, batch_boxes, batch_classes = inputs
            out = self.model(batch_imgs)
            targets = self.target_layer([out, batch_boxes, batch_classes])
            losses = self.loss_layer(out, targets)
            return losses

        elif self.mode == "inference":
            batch_imgs = inputs
            out = self.model(batch_imgs)
            scores, classes, boxes = self.detection_head(out)
            boxes = self.clip_boxes(batch_imgs, boxes)
            return scores, classes, boxes


class FCOS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.backbone == "resnet50":
            self.backbone = resnet50(pretrained=self.config.pretrained)
        elif config.backbone == "darknet19":
            self.backbone = Darknet19(pretrained=self.config.pretrained)
            
        self.neck = FPN(self.config.fpn_out_channels, backbone=self.config.backbone)
        
        self.head = ClsCntRegHead(self.config.fpn_out_channels,
                                  self.config.class_num,
                                  self.config.use_GN_head,
                                  self.config.cnt_on_reg,
                                  self.config.prior)

    def train(self, mode=True):
        super().train(mode=mode)

    def forward(self, x):
        C3, C4, C5 = self.backbone(x)
        all_P = self.neck([C3, C4, C5])
        cls_logits, cnt_logits, reg_preds = self.head(all_P)
        return cls_logits, cnt_logits, reg_preds


class DetectHead(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else DefaultConfig

        self.score_threshold = self.config.score_threshold
        self.nms_iou_threshold = self.config.nms_iou_threshold
        self.max_detection_boxes_num = self.config.max_detection_boxes_num
        self.strides = self.config.strides

    def forward(self, inputs):
        '''
        inputs  [cls_logits, cnt_logits, reg_preds]
        cls_logits  [batch_size, class_num, h, w] * 5
        cnt_logits  [batch_size, 1, h, w] * 5
        reg_preds   [batch_size, 4, h, w] * 5
        '''
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size, sum(h*w), class_num]
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size, sum(h*w), 1]
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)  # [batch_size, sum(h*w), 4]

        cls_preds = cls_logits.sigmoid_()
        cnt_preds = cnt_logits.sigmoid_()

        coords = coords.cuda() if torch.cuda.is_available() else coords

        cls_scores, cls_classes = torch.max(cls_preds, dim=-1)  # [batch_size, sum(h*w)]
        if self.config.add_centerness:
            cls_scores = torch.sqrt(cls_scores*(cnt_preds.squeeze(dim=-1)))  # [batch_size, sum(h*w)]
        cls_classes = cls_classes+1  # [batch_size, sum(h*w)]

        boxes = self._coords2boxes(coords, reg_preds)  # [batch_size, sum(h*w), 4]

        # select topk
        max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size, max_num]
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num, 4]
        cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size, max_num]
        cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size, max_num]
        boxes_topk = torch.stack(_boxes, dim=0)  # [batch_size, max_num, 4]
        assert boxes_topk.shape[-1] == 4
        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        '''
        cls_scores_topk [batch_size, max_num]
        cls_classes_topk [batch_size, max_num]
        boxes_topk [batch_size, max_num, 4]
        '''
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]  # [?]
            _cls_classes_b = cls_classes_topk[batch][mask]  # [?]
            _boxes_b = boxes_topk[batch][mask]  # [?, 4]
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores = torch.stack(_cls_scores_post, dim=0)
        classes = torch.stack(_cls_classes_post, dim=0)
        boxes = torch.stack(_boxes_post, dim=0)
        return scores, classes, boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx+1]
        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):
            pred = pred.permute(0, 2, 3, 1)
            coord = coords_fmap2orig(pred, stride).to(device=pred.device)
            pred = torch.reshape(pred, [batch_size, -1, c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_imgs, batch_boxes):
        h, w = batch_imgs.shape[2:]

        batch_boxes = batch_boxes.clamp_(min=0)
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max=w-1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max=h-1)
        return batch_boxes
