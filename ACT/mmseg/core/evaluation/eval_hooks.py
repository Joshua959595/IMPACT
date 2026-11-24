# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Report a binary image-level fire/no-fire F1-score alongside mIoU
# - Load the GT seg maps once per evaluation

import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


def _compute_fire_f1_score(results, dataset, gt_seg_maps=None):
    """Compute binary fire/no-fire F1-score from segmentation results.

    Args:
        results: List of segmentation prediction arrays
        dataset: Dataset with GT annotations
        gt_seg_maps: Optional pre-loaded ground-truth segmentation maps
            (as returned by ``dataset.get_gt_seg_maps()``). Pass this in to
            reuse GT maps already loaded for the standard mIoU evaluation,
            instead of loading the full validation label set into memory a
            second time.

    Returns:
        dict: Contains TP, FP, TN, FN, Precision, Recall, F1-score
    """
    try:
        tp = fp = tn = fn = 0.0
        ignore_index = getattr(dataset, 'ignore_index', 255)
        if gt_seg_maps is None:
            gt_seg_maps = dataset.get_gt_seg_maps(efficient_test=False)

        if len(results) != len(gt_seg_maps):
            mmcv.print_log(
                f'Fire F1: result length ({len(results)}) != gt length ({len(gt_seg_maps)}). '
                f'Using min length for evaluation.',
                'mmseg')

        for pred_result, gt_mask in zip(results, gt_seg_maps):
            # pre_eval mode returns tuples instead of segmentation masks.
            if isinstance(pred_result, tuple):
                continue

            if isinstance(pred_result, str):
                pred_mask = np.load(pred_result)
            elif isinstance(pred_result, (list, tuple)):
                pred_mask = np.asarray(pred_result[0])
            else:
                pred_mask = np.asarray(pred_result)

            gt_mask = np.asarray(gt_mask)
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[..., 0]
            if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                pred_mask = pred_mask[0]

            valid_gt = gt_mask != ignore_index
            gt_fire = int(np.any((gt_mask > 0) & valid_gt))
            pred_fire = int(np.any(pred_mask > 0))

            if pred_fire == 1 and gt_fire == 1:
                tp += 1
            elif pred_fire == 1 and gt_fire == 0:
                fp += 1
            elif pred_fire == 0 and gt_fire == 0:
                tn += 1
            else:
                fn += 1
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'fire_TP': tp,
            'fire_FP': fp,
            'fire_TN': tn,
            'fire_FN': fn,
            'fire_precision': precision,
            'fire_recall': recall,
            'fire_f1': f1,
        }
    except Exception as e:
        mmcv.print_log(f'Error computing fire F1-score: {e}', 'mmseg')
        return {}


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, distributed_eval=False, pre_eval=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.pre_eval = pre_eval

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        torch.cuda.empty_cache()
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            show=False,
            efficient_test=self.efficient_test,
            pre_eval=self.pre_eval)
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

        # Load GT seg maps once (if applicable) and reuse them for both the
        # fire F1-score and the standard mIoU evaluation.
        gt_seg_maps = None
        if not self.pre_eval and mmcv.is_list_of(results, np.ndarray):
            gt_seg_maps = self.dataloader.dataset.get_gt_seg_maps(efficient_test=False)

        fire_metrics = _compute_fire_f1_score(results, self.dataloader.dataset, gt_seg_maps=gt_seg_maps)
        if fire_metrics:
            runner.log_buffer.output['fire_precision'] = fire_metrics['fire_precision']
            runner.log_buffer.output['fire_recall'] = fire_metrics['fire_recall']
            runner.log_buffer.output['fire_f1'] = fire_metrics['fire_f1']
            mmcv.print_log(
                f"[Fire Classification] TP={int(fire_metrics['fire_TP'])} FP={int(fire_metrics['fire_FP'])} "
                f"TN={int(fire_metrics['fire_TN'])} FN={int(fire_metrics['fire_FN'])} | "
                f"Precision={fire_metrics['fire_precision']:.4f} Recall={fire_metrics['fire_recall']:.4f} "
                f"F1={fire_metrics['fire_f1']:.4f}",
                'mmseg'
            )

        if gt_seg_maps is not None:
            self.eval_kwargs['gt_seg_maps'] = gt_seg_maps
        try:
            key_score = self.evaluate(runner, results)
        finally:
            self.eval_kwargs.pop('gt_seg_maps', None)
            del gt_seg_maps

        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, distributed_eval=False, pre_eval=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.pre_eval = pre_eval

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        torch.cuda.empty_cache()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            efficient_test=self.efficient_test,
            pre_eval=self.pre_eval)

        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            # Load GT seg maps once (if applicable) and reuse them for both the
            # fire F1-score and the standard mIoU evaluation.
            gt_seg_maps = None
            if not self.pre_eval and mmcv.is_list_of(results, np.ndarray):
                gt_seg_maps = self.dataloader.dataset.get_gt_seg_maps(efficient_test=False)

            fire_metrics = _compute_fire_f1_score(results, self.dataloader.dataset, gt_seg_maps=gt_seg_maps)
            if fire_metrics:
                runner.log_buffer.output['fire_precision'] = fire_metrics['fire_precision']
                runner.log_buffer.output['fire_recall'] = fire_metrics['fire_recall']
                runner.log_buffer.output['fire_f1'] = fire_metrics['fire_f1']
                mmcv.print_log(
                    f"[Fire Classification] TP={int(fire_metrics['fire_TP'])} FP={int(fire_metrics['fire_FP'])} "
                    f"TN={int(fire_metrics['fire_TN'])} FN={int(fire_metrics['fire_FN'])} | "
                    f"Precision={fire_metrics['fire_precision']:.4f} Recall={fire_metrics['fire_recall']:.4f} "
                    f"F1={fire_metrics['fire_f1']:.4f}",
                    'mmseg'
                )

            if gt_seg_maps is not None:
                self.eval_kwargs['gt_seg_maps'] = gt_seg_maps
            try:
                key_score = self.evaluate(runner, results)
            finally:
                self.eval_kwargs.pop('gt_seg_maps', None)
                del gt_seg_maps

            if self.save_best:
                self._save_ckpt(runner, key_score)
