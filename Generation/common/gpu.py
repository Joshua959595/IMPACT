"""Import before torch/transformers/diffusers in every entrypoint script.

Select the GPU from the shell, e.g. `CUDA_VISIBLE_DEVICES=0 python ...`;
all models are placed on `cuda:0` of the visible devices.
"""
import os

# Route around a DeepGEMM/Blackwell-oriented fast path in recent transformers
# releases that is not needed here: force the Triton fallback for FP8 linear
# layers.
os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")

# transformers' fp8_linear() calls torch.cuda.get_device_properties() with no
# argument, which torch 2.5.1 does not accept. Default it to the current
# device.
import torch  # noqa: E402

_orig_get_device_properties = torch.cuda.get_device_properties


def _get_device_properties_default_device(device=None):
    if device is None:
        device = torch.cuda.current_device()
    return _orig_get_device_properties(device)


torch.cuda.get_device_properties = _get_device_properties_default_device
