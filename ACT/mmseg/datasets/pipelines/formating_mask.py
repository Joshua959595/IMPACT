# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support valid_pseudo_mask and weight_map with negative stride handling

from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES
from .formating import DefaultFormatBundle, to_tensor, Collect


@PIPELINES.register_module()
class DefaultFormatBundleMask(DefaultFormatBundle):
    """Default formatting bundle for Mask Prior (handles weight_map)."""

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if self.keys is None:
            results_tmp = [results]
        else:
            results_tmp = []
            for k in self.keys:
                results_tmp.append(results[k])
        
        for i, r in enumerate(results_tmp):
            if 'img' in r:
                img = r['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                results_tmp[i]['img'] = DC(to_tensor(img), stack=True)
            if 'gt_semantic_seg' in r:
                # convert to long
                results_tmp[i]['gt_semantic_seg'] = DC(
                    to_tensor(r['gt_semantic_seg'][None, ...].astype(np.int64)),
                    stack=True)
            if 'valid_pseudo_mask' in r:
                results_tmp[i]['valid_pseudo_mask'] = DC(
                    to_tensor(r['valid_pseudo_mask'][None, ...]), stack=True)
            
            # Handle weight_map
            if 'weight_map' in r:
                # FIX: Handle negative strides by copying the array
                weight_map = r['weight_map']
                if weight_map.strides[0] < 0 or weight_map.strides[1] < 0:
                    weight_map = weight_map.copy()
                
                results_tmp[i]['weight_map'] = DC(
                    to_tensor(weight_map[None, ...]), stack=True)
        
        if self.keys is None:
            results = results_tmp[0]
        else:
            for i, k in enumerate(self.keys):
                results[k] = results_tmp[i]
        
        return results


@PIPELINES.register_module()
class CollectMask(Collect):
    """Collect data from the loader relevant to the specific task."""
    
    def __call__(self, results):
        if self.parts is None:
            results_tmp = [results]
            data_tmp = [{}]
        else:
            results_tmp = []
            data_tmp = []
            for k in self.parts:
                results_tmp.append(results[k])
                data_tmp.append({})

        for i, r in enumerate(results_tmp):
            img_meta = {}
            for key in self.meta_keys:
                if key in r:
                    img_meta[key] = r[key]
            if self.parts is None:
                data_tmp[i]['img_metas'] = DC(img_meta, cpu_only=True)
                for key in self.keys:
                    if isinstance(key, list):
                        for k in key:
                            if k in r:
                                data_tmp[i][k] = r[k]
                    else:
                        if key in r:
                            data_tmp[i][key] = r[key]
            else:
                data_tmp[i]['img_metas'] = DC(img_meta, cpu_only=True)
                # Handle list keys if necessary in parts mode (not used here but for safe keeping)
                
        return data_tmp[0] if self.parts is None else data_tmp
