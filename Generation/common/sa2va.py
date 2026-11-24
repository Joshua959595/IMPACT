"""Wraps ByteDance/Sa2VA-Qwen3-VL-4B for referring segmentation: the model
predicts a [SEG] token whose hidden state conditions a bundled SAM mask
decoder, returning pixel-level masks.

Custom `trust_remote_code=True` architecture (not a standard
ForConditionalGeneration class), so it is driven through `predict_forward()`
per the model card.
"""
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class Sa2VASegmenter:
    def __init__(self, model_id: str, device: str = "cuda:0", use_flash_attn: bool = False):
        self.model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
        ).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
        self.device = device

    def segment(
        self, image: Image.Image, referring_expression: str, context_hint: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Returns a boolean (H, W) mask (union of every segmented instance
        the model returns for the referring expression), or None if the
        model didn't emit any mask (e.g. it decided the phenomenon isn't
        present in the image).

        `context_hint` (sample plan + the T2I prompt used to generate the
        image) gives extra grounding for what/where the phenomenon should be
        in this specific sample."""
        instruction = f"Please segment the {referring_expression} in this image."
        if context_hint:
            text_prompt = f"<image>{context_hint}\n\n{instruction}"
        else:
            text_prompt = f"<image>{instruction}"

        input_dict = {
            "image": image,
            "text": text_prompt,
            "past_text": "",
            "mask_prompts": None,
            "processor": self.processor,
        }
        with torch.no_grad():
            result = self.model.predict_forward(**input_dict)

        masks: List[np.ndarray] = result.get("prediction_masks") or []
        if not masks:
            return None

        union = None
        for m in masks:
            m = np.asarray(m)
            if m.ndim == 3:
                m = m[0]
            m_bool = m > 0.5
            union = m_bool if union is None else (union | m_bool)
        return union
