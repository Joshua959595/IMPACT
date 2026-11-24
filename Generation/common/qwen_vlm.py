"""Wraps a Qwen3-VL model behind a single call_mm_text(...) interface used
by the Multilevel Planner (GGP / SGP generators) and by every
self-refinement step (DVQ generation/evaluation, prompt editing, comparator).

Kept resident on GPU for the whole run (loaded once): it is called far more
often per self-refinement iteration (DVQ eval, targeted edit, implicit
improve, merge, verify, comparator x2 = 7 calls) than the T2I model (1 call).

Loaded in 4-bit (bitsandbytes NF4) from the standard bf16 checkpoint
(~18GB), which leaves headroom for Qwen-Image-Edit-2509 on the same GPU and
works with torch 2.5.1 (the FP8 checkpoint's fast kernel path needs
torch>=2.6).

Defaults to a -Thinking checkpoint's sampling recipe, but auto-detects
-Instruct checkpoints (by model_id) and skips the <think>...</think>
stripping for those, since Instruct models answer directly.
"""
from typing import List, Optional, Tuple

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

# Reasonable general-purpose sampling defaults for Qwen3-VL chat models
# (used for both -Thinking and -Instruct checkpoints here); the caller's
# `temperature` still scales sampling per call.
_TOP_P = 0.95
_TOP_K = 20


class QwenVLM:
    def __init__(self, model_id: str, device: str = "cuda:0", max_new_tokens: int = 16384):
        self.is_thinking = "thinking" in model_id.lower()
        self.processor = AutoProcessor.from_pretrained(model_id)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map={"": device},
            quantization_config=quant_config,
        )
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _strip_thinking(text: str) -> str:
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        # No closing tag means generation was cut off by max_new_tokens while
        # still inside the reasoning block -- surface this loudly instead of
        # silently handing back thinking prose for the caller's JSON/tag
        # parser to fail on.
        print(f"[QwenVLM WARNING] no </think> found (likely hit max_new_tokens); head={text[:150]!r}")
        return text.strip()

    def call_mm_text(
        self,
        user_text: str,
        images: Optional[List[Tuple[str, Image.Image]]] = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        content = []
        if images:
            for label, img in images:
                content.append({"type": "text", "text": label})
                content.append({"type": "image", "image": img})
        if user_text:
            content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": content})

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            do_sample=True,
            temperature=max(temperature, 0.05),
            top_p=_TOP_P,
            top_k=_TOP_K,
        )

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, out_ids)
        ]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return self._strip_thinking(raw) if self.is_thinking else raw.strip()
