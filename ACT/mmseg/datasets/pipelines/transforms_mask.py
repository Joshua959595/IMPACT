# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Support processing of additional image fields (e.g., weight_map) in ResizeMask, RandomFlipMask, RandomCropMask

import mmcv
import numpy as np
from mmseg.datasets.builder import PIPELINES
from .transforms import Resize, RandomFlip, RandomCrop

@PIPELINES.register_module()
class ResizeMask(Resize):
    """Resize images & seg & mask (weight_map)."""
    
    def _resize_img(self, results):
        """Resize images including weight_map with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
            
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

        # Resize other image fields (including weight_map)
        for key in results.get('img_fields', ['img']):
            if key == 'img':
                continue
            if self.keep_ratio:
                results[key] = mmcv.imrescale(results[key], results['scale'])
            else:
                results[key] = mmcv.imresize(results[key], results['scale'])
        
        return results

@PIPELINES.register_module()
class RandomFlipMask(RandomFlip):
    """Flip the image & seg & mask (weight_map)."""

    def __call__(self, results):
        if self.keys is None:
            results_tmp = [results]
        else:
            results_tmp = []
            for k in self.keys:
                results_tmp.append(results[k])
        
        for i, r in enumerate(results_tmp):
            if 'flip' not in r:
                flip = True if np.random.rand() < self.prob else False
                results_tmp[i]['flip'] = flip
            if 'flip_direction' not in r:
                results_tmp[i]['flip_direction'] = self.direction
            if results_tmp[i]['flip']:
                # flip image
                results_tmp[i]['img'] = mmcv.imflip(
                    r['img'], direction=results_tmp[i]['flip_direction'])

                
                # Flip other img fields (including weight_map)
                for key in r.get('img_fields', ['img']):
                     if key == 'img':
                         continue
                     results_tmp[i][key] = mmcv.imflip(
                        r[key], direction=results_tmp[i]['flip_direction'])
                
                # flip segs
                for key in r.get('seg_fields', []):
                    results_tmp[i][key] = mmcv.imflip(
                        r[key], direction=results_tmp[i]['flip_direction']).copy()
        
        if self.keys is None:
            results = results_tmp[0]
        else:
            for i, k in enumerate(self.keys):
                results[k] = results_tmp[i]

        return results

@PIPELINES.register_module()
class RandomCropMask(RandomCrop):
    """Random crop the image & seg & mask (weight_map)."""

    def __call__(self, results):
        if self.keys is None:
            results_tmp = [results]
        else:
            results_tmp = []
            for k in self.keys:
                results_tmp.append(results[k])
        
        for i, r in enumerate(results_tmp):
            img = r['img']
            crop_bbox = self.get_crop_bbox(img)
            if self.cat_max_ratio < 1.:
                for _ in range(10):
                    if 'gt_semantic_seg' in r:
                        seg_temp = self.crop(r['gt_semantic_seg'], crop_bbox)
                    elif 'weight_map' in r:
                        # Target domain: build a temporary label (background 0, smoke 1) from the weight map
                        w_temp = self.crop(r['weight_map'], crop_bbox)
                        seg_temp = (w_temp > -1.0).astype(np.uint8)
                    else:
                        break
                        
                    labels, cnt = np.unique(seg_temp, return_counts=True)
                    cnt = cnt[labels != self.ignore_index]
                    if len(cnt) > 1 and np.max(cnt) / np.sum(
                            cnt) < self.cat_max_ratio:
                        break
                    crop_bbox = self.get_crop_bbox(img)

            # crop the image
            img = self.crop(img, crop_bbox)
            img_shape = img.shape
            results_tmp[i]['img'] = img
            results_tmp[i]['img_shape'] = img_shape

            
            # Crop other img fields (including weight_map)
            for key in r.get('img_fields', ['img']):
                 if key == 'img':
                     continue
                 results_tmp[i][key] = self.crop(r[key], crop_bbox)

            # crop semantic seg
            for key in r.get('seg_fields', []):
                results_tmp[i][key] = self.crop(r[key], crop_bbox)

        if self.keys is None:
            results = results_tmp[0]
        else:
            for i, k in enumerate(self.keys):
                results[k] = results_tmp[i]
        
        return results