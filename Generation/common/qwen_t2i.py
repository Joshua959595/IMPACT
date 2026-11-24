"""Wraps Qwen-Image-Edit-2509 (QwenImageEditPlusPipeline) behind a
generate_image_t2i(prompt, condition_img) call.

The reference/condition image is used as a real image-conditioning input
(not just text) to preserve scene identity.

Loaded with enable_model_cpu_offload(): only the submodules needed for the
current call are moved onto the GPU and released right after. It is called
once per self-refinement iteration (vs. ~7 calls to the resident VLM), so
paying an offload/reload cost here -- instead of keeping ~40GB resident --
lets both models share a single 80GB GPU.
"""
import io
from typing import Optional

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


class QwenT2I:
    def __init__(self, model_id: str, offload: bool = True, device: str = "cuda:0"):
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

    def generate_image_t2i(
        self,
        prompt: str,
        condition_img: Image.Image,
        negative_prompt: str = " ",
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> bytes:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        kwargs = {}
        if width is not None and height is not None:
            kwargs["width"] = width
            kwargs["height"] = height

        out = self.pipe(
            image=condition_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
            **kwargs,
        )
        img = out.images[0]

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
