import torch
import torch.nn as nn


def compute_cls_loss(preds, targets, mask):
    '''
    :param preds: [batch_size, class_num, h, w]*5
    :param targets: [batch_size, sum(h*w), 1]
    :param mask: [batch_size, sum(h*w)]
    :return loss: [batch_size]
    '''
    batch_size = targets.shape[0]
    preds_reshape = []
    class_num = preds[0].shape[1]
    mask = mask.unsqueeze(dim=-1)  # [batch_size, sum(h*w), 1]
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # [batch_size,]

    # [batch_size, class_num, h, w] -> [batch_size, sum(h*w), class_num]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, class_num])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)  # [batch_size,sum(_h*_w),class_num]
    assert preds.shape[:2] == targets.shape[:2]

    loss = []
    # compute the loss from both positive and negative samples
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index]  # [sum(h*w), class_num]
        target_pos = targets[batch_index]  # [sum(h*w), 1]
        # one-hot encoding : [sum(h*w), 1] -> [sum(h*w), class_num]:
        target_pos = (torch.arange(1, class_num+1, device=target_pos.device)[None, :] == target_pos).float()
        loss.append(compute_focal_loss(pred_pos, target_pos).view(1))
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


def compute_cnt_loss(preds, targets, mask):
    '''
     :param preds: [batch_size, 1, h, w]*5
     :param targets: [batch_size, sum(h*w), 1]
     :param mask: [batch_size, sum(h*w)]
     '''
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    mask = mask.unsqueeze(dim=-1)  # [batch_size, sum(h*w), 1]
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # [batch_size,]

    # [batch_size, 1, h, w] -> [batch_size, sum(h*w), 1]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape == targets.shape

    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]
        target_pos = targets[batch_index][mask[batch_index]]
        assert len(pred_pos.shape) == 1
        # In general, neg samples have low cls scores, the inference confidence is still low even if cnt score is high
        # so cnt loss of neg samples is not considered
        loss.append(nn.functional.binary_cross_entropy_with_logits(input=pred_pos, target=target_pos, reduction='sum').view(1))
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


def compute_reg_loss(preds, targets, mask, mode='giou'):
    '''
    :param preds: [batch_size, 4, h, w]*5
    :param targets: [batch_size, sum(h*w), 4]
    :param mask: [batch_size, sum(h*w)]
    :param mode: "iou" or "giou"
    :return loss: [batch_size]
    '''
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    num_pos = torch.sum(mask, dim=1).clamp_(min=1).float()  # [batch_size,]

    # [batch_size, 4, h, w] -> [batch_size, sum(h*w), 4]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape == targets.shape

    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b, 4]
        target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b, 4]
        assert len(pred_pos.shape) == 2

        if mode == 'iou':
            loss.append(compute_iou_loss(pred_pos, target_pos).view(1))
        elif mode == 'giou':
            loss.append(compute_giou_loss(pred_pos, target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou', 'giou']")

    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


# TODO compute iou loss for different format
def compute_iou_loss(preds, targets, format="ltrb"):
    '''
    :param preds: [n, 4] ltrb
    :param targets: [n, 4]
    :return loss: []
    '''
    lt = torch.min(preds[:, :2], targets[:, :2])  # [n, 2]
    rb = torch.min(preds[:, 2:], targets[:, 2:])
    wh = (rb + lt).clamp(min=0)

    overlap = wh[:, 0] * wh[:, 1]  # [n]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])

    iou = overlap / (area1 + area2 - overlap)
    loss = -iou.clamp(min=1e-6).log()
    return loss.sum()


# TODO compute giou loss for different format
def compute_giou_loss(preds, targets, format="ltrb"):
    '''
    :param preds: [n, 4] ltrb
    :param targets: [n, 4]
    :return loss: []
    '''
    lt_min = torch.min(preds[:, :2], targets[:, :2])  # [n, 2]
    rb_min = torch.min(preds[:, 2:], targets[:, 2:])
    wh_min = (rb_min + lt_min).clamp(min=0)

    overlap = wh_min[:, 0] * wh_min[:, 1]  # [n]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    union = (area1 + area2 - overlap)
    iou = overlap / union

    # smallest enclosing box
    lt_max = torch.max(preds[:, :2], targets[:, :2])
    rb_max = torch.max(preds[:, 2:], targets[:, 2:])
    wh_max = (rb_max + lt_max).clamp(0)
    G_area = wh_max[:, 0] * wh_max[:, 1]  # [n]

    giou = iou - (G_area - union) / G_area.clamp(1e-10)
    loss = 1. - giou
    return loss.sum()


def compute_focal_loss(preds, targets, gamma=2.0, alpha=0.25):
    '''
    :param preds: [n, class_num]
    :param targets: [n, class_num]
    :return loss: []
    '''
    preds = preds.sigmoid()
    pt = preds * targets + (1.0 - preds) * (1.0 - targets)
    w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = -w * torch.pow((1.0 - pt), gamma) * pt.log()
    return loss.sum()
