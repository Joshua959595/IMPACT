# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os
import os.path as osp
import shutil

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm
import torch
import numpy as np
import mmcv


def _resolve_allowed_filenames(vis_dir):
    """Resolve a vis_dir config value into an allow-list of basenames.

    Returns None when unset (no filtering, save every val image, matching
    legacy behavior). Returns a (possibly empty) set when vis_dir is set,
    so a missing/empty directory saves zero overlays instead of silently
    reverting to saving everything.
    """
    if not vis_dir:
        return None
    if not osp.isdir(vis_dir):
        mmcv.print_log(f'[vis] vis_dir={vis_dir} not found; saving 0 overlays this run.', 'mmseg')
        return set()
    names = set(os.listdir(vis_dir))
    if not names:
        mmcv.print_log(f'[vis] vis_dir={vis_dir} is empty; saving 0 overlays this run.', 'mmseg')
    return names


def _save_overlays_from_dir(model, vis_dir, out_dir, dataset_pipeline):
    """Run inference directly on every image found in vis_dir and save a
    red-overlay visualization for each one.

    Unlike the (legacy) allow-list filtering above, this does not require the
    vis_dir images to be part of the validation split at all - it loads and
    predicts on them directly, so curated "visualization" folders that live
    outside the validation set still get overlays.
    """
    if not vis_dir or not osp.isdir(vis_dir):
        return 0
    fnames = sorted(
        f for f in os.listdir(vis_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    if not fnames:
        return 0

    from mmcv.parallel import collate, scatter
    from mmseg.apis.inference import LoadImage
    from mmseg.datasets.pipelines import Compose

    test_pipeline = Compose([LoadImage()] + dataset_pipeline.transforms[1:])
    real_model = model.module if hasattr(model, 'module') else model
    device = next(real_model.parameters()).device
    was_training = real_model.training
    real_model.eval()

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    try:
        for fname in fnames:
            img_path = osp.join(vis_dir, fname)
            orig = mmcv.imread(img_path)
            if orig is None:
                continue
            data = test_pipeline(dict(img=img_path))
            data = collate([data], samples_per_gpu=1)
            if next(real_model.parameters()).is_cuda:
                data = scatter(data, [device])[0]
            else:
                data['img_metas'] = [i.data[0] for i in data['img_metas']]
            with torch.no_grad():
                result = real_model(return_loss=False, rescale=True, **data)
            pred_mask = np.asarray(result[0])
            if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                pred_mask = pred_mask[0]

            h, w = orig.shape[:2]
            pred_resized = mmcv.imresize(
                pred_mask.astype(np.uint8), (w, h), interpolation='nearest')
            base = orig.astype(np.float32) * 0.5
            overlay = base.copy()
            smoke = pred_resized > 0
            overlay[smoke] = base[smoke] + np.array(
                [0, 0, 255], dtype=np.float32) * 0.5
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            stem = osp.splitext(fname)[0]
            mmcv.imwrite(overlay, osp.join(out_dir, f'{stem}_overlay.jpg'))
            saved += 1
    finally:
        if was_training:
            real_model.train()
    return saved


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

    def __init__(self, *args, by_epoch=False, efficient_test=False, distributed_eval=False, pre_eval=False, vis_dir=None, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.pre_eval = pre_eval
        self.vis_dir = vis_dir

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        return self._evaluate_and_save(runner)

    def _evaluate_and_save(self, runner):
        from mmseg.apis import single_gpu_test#, multi_gpu_test
        torch.cuda.empty_cache()
        is_final = (runner.iter + 1 >= runner.max_iters)
        # Only render/save prediction overlays for the final iteration (or when
        # tracking a best checkpoint, which needs the images to persist).
        save_images = bool(self.save_best) or is_final
        out_dir = osp.join(runner.work_dir, f'val_results/val_{runner.iter}') if save_images else None
        allowed_filenames = _resolve_allowed_filenames(self.vis_dir) if save_images else None
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            show=False,
            out_dir=out_dir,
            efficient_test=self.efficient_test,
            pre_eval=self.pre_eval,
            allowed_filenames=allowed_filenames)
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

        if save_images and self.vis_dir:
            try:
                vis_out_dir = osp.join(out_dir, 'vis_overlays')
                n_vis = _save_overlays_from_dir(
                    runner.model, self.vis_dir, vis_out_dir,
                    self.dataloader.dataset.pipeline)
                mmcv.print_log(
                    f'[vis] Saved {n_vis} overlays from vis_dir={self.vis_dir} '
                    f'to {vis_out_dir}', 'mmseg')
            except Exception as e:
                mmcv.print_log(f'[vis] vis_dir overlay saving failed: {e}', 'mmseg')

        # Load GT seg maps once (if applicable) and reuse for both the fire
        # F1-score and the standard mIoU evaluation below, instead of
        # loading the full validation label set into memory twice per eval.
        gt_seg_maps = None
        if not self.pre_eval and mmcv.is_list_of(results, np.ndarray):
            gt_seg_maps = self.dataloader.dataset.get_gt_seg_maps(efficient_test=False)

        # Compute fire F1-score
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
        else:
            # No best checkpoint: remove this eval's val_results unless it's the final iter
            if not is_final and out_dir and osp.exists(out_dir):
                try:
                    shutil.rmtree(out_dir)
                except Exception:
                    pass

    def _save_ckpt(self, runner, key_score):
        work_dir = runner.work_dir
        try:
            before = set(os.listdir(work_dir)) if osp.exists(work_dir) else set()
        except Exception:
            before = set()

        # call parent implementation which may create best_* artifacts
        try:
            super()._save_ckpt(runner, key_score)
        except Exception:
            return

        try:
            after = set(os.listdir(work_dir)) if osp.exists(work_dir) else set()
        except Exception:
            after = set()

        new = after - before
        # if a new "best*" artifact appeared, keep only the current val results
        best_new = [n for n in new if n.startswith('best')]
        val_root = osp.join(work_dir, 'val_results')
        current_val = osp.join(val_root, f'val_{runner.iter}')
        is_final = (runner.iter + 1 >= runner.max_iters)
        if best_new:
            if osp.exists(val_root):
                for name in os.listdir(val_root):
                    path = osp.join(val_root, name)
                    if path != current_val:
                        try:
                            if osp.isdir(path):
                                shutil.rmtree(path)
                            else:
                                os.remove(path)
                        except Exception:
                            mmcv.print_log(f'Failed to remove {path}', 'mmseg')
        else:
            # Not a new best: remove this eval's overlay dir, unless it's the final iteration
            if not is_final and osp.exists(current_val):
                try:
                    shutil.rmtree(current_val)
                except Exception:
                    mmcv.print_log(f'Failed to remove {current_val}', 'mmseg')


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

    def __init__(self, *args, by_epoch=False, efficient_test=False, distributed_eval=False, pre_eval=False, vis_dir=None, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.pre_eval = pre_eval
        self.vis_dir = vis_dir

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
        is_final = (runner.iter + 1 >= runner.max_iters)
        # Only render/save prediction overlays for the final iteration (or when
        # tracking a best checkpoint, which needs the images to persist).
        save_images = bool(self.save_best) or is_final
        out_dir = osp.join(runner.work_dir, f'val_results/val_{runner.iter}') if save_images else None
        allowed_filenames = _resolve_allowed_filenames(self.vis_dir) if save_images else None
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            out_dir=out_dir,
            efficient_test=self.efficient_test,
            pre_eval=self.pre_eval,
            allowed_filenames=allowed_filenames)

        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            if save_images and self.vis_dir:
                try:
                    vis_out_dir = osp.join(out_dir, 'vis_overlays')
                    n_vis = _save_overlays_from_dir(
                        runner.model, self.vis_dir, vis_out_dir,
                        self.dataloader.dataset.pipeline)
                    mmcv.print_log(
                        f'[vis] Saved {n_vis} overlays from vis_dir={self.vis_dir} '
                        f'to {vis_out_dir}', 'mmseg')
                except Exception as e:
                    mmcv.print_log(f'[vis] vis_dir overlay saving failed: {e}', 'mmseg')

            # Load GT seg maps once (if applicable) and reuse for both the
            # fire F1-score and the standard mIoU evaluation below, instead
            # of loading the full validation label set into memory twice.
            gt_seg_maps = None
            if not self.pre_eval and mmcv.is_list_of(results, np.ndarray):
                gt_seg_maps = self.dataloader.dataset.get_gt_seg_maps(efficient_test=False)

            # Compute fire F1-score
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
            else:
                if not is_final and out_dir and osp.exists(out_dir):
                    try:
                        shutil.rmtree(out_dir)
                    except Exception:
                        pass

    def _save_ckpt(self, runner, key_score):
        work_dir = runner.work_dir
        try:
            before = set(os.listdir(work_dir)) if osp.exists(work_dir) else set()
        except Exception:
            before = set()

        # call parent implementation which may create best_* artifacts
        try:
            super()._save_ckpt(runner, key_score)
        except Exception:
            return

        try:
            after = set(os.listdir(work_dir)) if osp.exists(work_dir) else set()
        except Exception:
            after = set()

        new = after - before
        # if a new "best*" artifact appeared, keep only the current val results
        best_new = [n for n in new if n.startswith('best')]
        val_root = osp.join(work_dir, 'val_results')
        current_val = osp.join(val_root, f'val_{runner.iter}')
        is_final = (runner.iter + 1 >= runner.max_iters)
        if best_new:
            if osp.exists(val_root):
                for name in os.listdir(val_root):
                    path = osp.join(val_root, name)
                    if path != current_val:
                        try:
                            if osp.isdir(path):
                                shutil.rmtree(path)
                            else:
                                os.remove(path)
                        except Exception:
                            mmcv.print_log(f'Failed to remove {path}', 'mmseg')
        else:
            # Not a new best: remove this eval's overlay dir, unless it's the final iteration
            if not is_final and osp.exists(current_val):
                try:
                    shutil.rmtree(current_val)
                except Exception:
                    mmcv.print_log(f'Failed to remove {current_val}', 'mmseg')
