from utils.utils.loss import *
from utils.utils.targets import coords_fmap2orig


class GenTargets(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.strides = self.config.strides
        self.limit_range = self.config.limit_range
        assert len(self.strides) == len(self.limit_range)

    def forward(self, inputs):
        '''
        [0] [cls_logits, cnt_logits, reg_preds]
        cls_logits   [batch_size, class_num, h, w] * 5
        cnt_logits   [batch_size, 1, h, w] * 5
        reg_preds    [batch_size, 4, h, w] * 5
        [1] gt_boxes [batch_size, m, 4]  FloatTensor
        [2] classes  [batch_size, m]  LongTensor
        Returns
        cls_targets: [batch_size, sum(h*w), 1]
        cnt_targets: [batch_size, sum(h*w), 1]
        reg_targets: [batch_size, sum(h*w), 4]
        '''
        out, gt_boxes, classes = inputs
        cls_logits, cnt_logits, reg_preds = out
        assert len(self.strides) == len(cls_logits)

        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []
        for level in range(len(cls_logits)):
            level_out = [cls_logits[level], cnt_logits[level], reg_preds[level]]
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes,
                                                    self.strides[level], self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])

        cls_targets = torch.cat(cls_targets_all_level, dim=1)
        cnt_targets = torch.cat(cnt_targets_all_level, dim=1)
        reg_targets = torch.cat(reg_targets_all_level, dim=1)
        return cls_targets, cnt_targets, reg_targets

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        '''
        build targets for single fmap
        :param out: [[batch_size, class_num, h, w], [batch_size, 1, h, w], [batch_size, 4, h, w]]
        :param gt_boxes: [batch_size, m, 4]
        :param classes: [batch_size, m]
        :param stride: int
        :param limit_range:[min, max]
        return: cls_targets, cnt_targets, reg_targets
        '''
        cls_logits, cnt_logits, reg_preds = out
        batch_size, class_num = cls_logits.shape[0:2]

        cls_logits = cls_logits.permute(0, 2, 3, 1)  # [batch_size, h, w, class_num]
        coords = coords_fmap2orig(cls_logits, stride).to(device=gt_boxes.device)  # [h*w, 2]
        cls_logits = cls_logits.reshape((batch_size,-1,class_num))  # [batch_size,h*w,class_num]
        cnt_logits = cnt_logits.permute(0, 2, 3, 1)
        cnt_logits = cnt_logits.reshape((batch_size, -1, 1))
        reg_preds = reg_preds.permute(0, 2, 3, 1)
        reg_preds = reg_preds.reshape((batch_size, -1, 4))
        h_mul_w = cls_logits.shape[1]

        # 4 directions off to reach gt boxes edges for anchor points
        x = coords[:, 0]
        y = coords[:, 1]
        l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
        r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)  # [batch_size,h*w,m,4]

        off_min = torch.min(ltrb_off, dim=-1)[0]  # [batch_size, h*w, m]
        off_max = torch.max(ltrb_off, dim=-1)[0]  # [batch_size, h*w, m]
        # condition 1: filter anchor points outside the gt boxes
        mask_in_gtboxes = off_min > 0
        # condition 2: fmaps from different levels should predict boxes of different size
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])

        radiu = stride * sample_radiu_ratio
        # [batch_size, m]
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2.
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2.
        # 4 directions off to reach gt boxes centers for anchor points
        # [1, h*w, 1] - [batch_size, 1, m] -> [batch_size, h*w, m]
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]
        c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)  # [batch_size,h*w,m,4]
        c_off_max = torch.max(c_ltrb_off, dim=-1)[0]
        # condition 3: keep anchor points in the center of gt boxes
        mask_center = c_off_max < radiu
        # intersection of 3 conditions
        mask_pos = mask_in_gtboxes & mask_in_level & mask_center  # [batch_size, h*w, m]

        # if an anchor point corresponds to multiple gt boxes, pick a small gt box
        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])  # [batch_size,h*w,m]
        areas[~mask_pos] = 99999999
        areas_min_ind = torch.min(areas, dim=-1)[1]  # [batch_size, h*w]

        # reg
        # [batch_size, h*w, m, 4][batch_size, h*w, m] -> [batch_size*h*w, m]
        reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]  # [batch_size*h*w,4]
        reg_targets = torch.reshape(reg_targets, (batch_size, -1, 4))  # [batch_size, h*w, 4]

        # cls
        classes = torch.broadcast_tensors(classes[:, None, :], areas.long())[0]  # [batch_size, h*w, m]
        cls_targets = classes[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        cls_targets = torch.reshape(cls_targets, (batch_size, -1, 1))  # [batch_size, h*w, 1]

        # cnt
        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])  #[batch_size, h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        cnt_targets = ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(dim=-1)  # [batch_size,h*w,1]

        assert reg_targets.shape == (batch_size, h_mul_w, 4)
        assert cls_targets.shape == (batch_size, h_mul_w, 1)
        assert cnt_targets.shape == (batch_size, h_mul_w, 1)

        # filter anchor points without gt boxes
        mask_pos_2 = mask_pos.long().sum(dim=-1)  # [batch_size, h*w]
        mask_pos_2 = mask_pos_2 >= 1
        assert mask_pos_2.shape == (batch_size, h_mul_w)

        cls_targets[~mask_pos_2] = 0   # [batch_size, h*w, 1]
        cnt_targets[~mask_pos_2] = -1  # [batch_size, h*w, 1]
        reg_targets[~mask_pos_2] = -1  # [batch_size, h*w, 4]
        
        return cls_targets, cnt_targets, reg_targets


class LOSS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, preds, targets):
        '''
        [0] [cls_logits, cnt_logits, reg_preds]
        cls_logits   [batch_size, class_num, h, w] * 5
        cnt_logits   [batch_size, 1, h, w] * 5
        reg_preds    [batch_size, 4, h, w] * 5
        [1]
        targets :   [[batch_size, sum(h*w), 1],
                     [batch_size, sum(h*w), 1],
                     [batch_size, sum(h*w), 4]]
        '''
        cls_logits, cnt_logits, reg_preds = preds
        cls_targets, cnt_targets, reg_targets = targets

        # to mask pos and neg samples, only compute the cls loss of neg samples
        # negative samples positions are filled with -1 when build targets
        mask_pos = (cnt_targets > -1).squeeze(dim=-1)  # [batch_size,sum(h*w)]
        cls_loss = compute_cls_loss(cls_logits, cls_targets, mask_pos).mean()  # []
        cnt_loss = compute_cnt_loss(cnt_logits, cnt_targets, mask_pos).mean()
        reg_loss = compute_reg_loss(reg_preds, reg_targets, mask_pos).mean()

        cnt_loss_flag = 1. if self.config.add_centerness else 0.
        total_loss = cls_loss + cnt_loss * cnt_loss_flag + reg_loss
        return cls_loss, cnt_loss, reg_loss, total_loss



