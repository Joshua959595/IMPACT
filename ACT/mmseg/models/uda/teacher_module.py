# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Ported from: https://github.com/lhoyer/MIC
# (seg/mmseg/models/uda/teacher_module.py)
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. Licensed under the Apache
# License, Version 2.0. A copy of the license is available at
# resources/license_dacs.txt (DAFormer/MIC lineage acknowledgment).
# Modifications:
# - Adapted debug hooks to this repo's EncoderDecoder API, which exposes
#   `debug` / `debug_output` on `decode_head` rather than on the model
#   itself (mmseg 0.16.0-based HRDA/DAFormer lineage used here).

from copy import deepcopy

import numpy as np
import torch
from timm.models.layers import DropPath
from torch.nn import Module
from torch.nn.modules.dropout import _DropoutNd

from mmseg.models import build_segmentor
from mmseg.models.uda.uda_decorator import get_module


class EMATeacher(Module):
    """Standalone EMA teacher used by MaskingConsistencyModule.

    Only instantiated when the masking consistency loss needs a teacher
    that is independent from the host UDA method's own EMA model, i.e.
    when `mask_alpha` or `mask_pseudo_threshold` differ from the host's
    `alpha` / `pseudo_threshold`.
    """

    def __init__(self, use_mask_params, cfg):
        super(EMATeacher, self).__init__()
        prefix = 'mask_' if use_mask_params else ''
        self.alpha = cfg[f'{prefix}alpha']
        if self.alpha == 'same':
            self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg[f'{prefix}pseudo_threshold']
        if self.pseudo_threshold == 'same':
            self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']

        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        self.debug = False
        self.debug_output = {}

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self, model):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_debug_state(self):
        self.get_ema_model().decode_head.debug = self.debug

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        if self.pseudo_threshold is not None:
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            ps_size = np.size(np.array(pseudo_label.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_prob.shape, device=logits.device)
        else:
            pseudo_weight = torch.ones(pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model)
        if iter > 0:
            self._update_ema(model, iter)

    def __call__(self, target_img, target_img_metas, valid_pseudo_mask):
        self.update_debug_state()

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().generate_pseudo_label(
            target_img, target_img_metas)

        pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
            ema_logits)
        del ema_logits

        pseudo_weight = self.filter_valid_pseudo_region(
            pseudo_weight, valid_pseudo_mask)

        self.debug_output = self.get_ema_model().decode_head.debug_output

        return pseudo_label, pseudo_weight
