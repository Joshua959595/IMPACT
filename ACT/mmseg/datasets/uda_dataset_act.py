# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Based on: https://github.com/lhoyer/DAFormer (UDADataset)
# Modifications:
# - Forward the per-target-image weight map (from the IMPACT Generation stage)
#   to the UDA model as `target_weight_map`, used by ACT.
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from .builder import DATASETS
from .uda_dataset import UDADataset, get_crop_bbox


@DATASETS.register_module()
class UDADatasetACT(UDADataset):
    """UDADataset that additionally passes the target weight map."""

    @staticmethod
    def _merge(s1, s2):
        out = {
            **s1, 'target_img_metas': s2['img_metas'],
            'target_img': s2['img']
        }
        if 'valid_pseudo_mask' in s2:
            out['valid_pseudo_mask'] = s2['valid_pseudo_mask']
        if 'weight_map' in s2:
            out['target_weight_map'] = s2['weight_map']
        return out

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        s1 = self.source[idx // len(self.target)]
        s2 = self.target[idx % len(self.target)]
        s1, s2 = self.synchronized_crop(s1, s2)
        return self._merge(s1, s2)

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                s1 = self.source[i1]
        i2 = np.random.choice(range(len(self.target)))
        s2 = self.target[i2]
        s1, s2 = self.synchronized_crop(s1, s2)
        return self._merge(s1, s2)

    def synchronized_crop(self, s1, s2):
        if self.sync_crop_size is None:
            return s1, s2
        orig_crop_size = s1['img'].data.shape[1:]
        crop_y1, crop_y2, crop_x1, crop_x2 = get_crop_bbox(
            orig_crop_size, self.sync_crop_size)
        for s in [s1, s2]:
            for key in ['img', 'gt_semantic_seg', 'valid_pseudo_mask',
                        'weight_map']:
                if key not in s:
                    continue
                s[key] = DC(
                    s[key].data[:, crop_y1:crop_y2, crop_x1:crop_x2],
                    stack=s[key]._stack)
        return s1, s2
