# ACT: Adaptive Confidence Thresholding

Downstream unsupervised domain adaptation for smoke segmentation:
HRDA (DAFormer decoder, MiT-B5 encoder) with DACS self-training, where the
pseudo-label confidence threshold is adapted per pixel by the weight maps
produced by [`../Generation`](../Generation).

- Source (labeled): SMOKE5K
- Target (unlabeled): IMPACT-generated images + weight maps
- Validation / test: Boreal Forest Fire

## Environment

```bash
conda create -n impact_act python=3.8 -y
conda activate impact_act
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install -r requirements.txt
```

## Data and pretrained weights

All paths are relative to this `ACT/` directory. Masks are single-channel
PNGs with values `{0: background, 1: smoke}`.

```
dataset/
├── source/              # SMOKE5K
│   ├── images/*.jpg
│   └── masks/*.png      # same basename as the image
├── target/              # written by ../Generation (stages 1-4)
│   ├── images/*.png
│   └── weight_maps/*.png  # same basename as the image
└── val/                 # Boreal Forest Fire
    ├── images/*.jpg
    └── masks/*.png
pretrained/
└── mit_b5.pth           # ImageNet-pretrained MiT-B5 (SegFormer)
```

The MiT-B5 weights are provided by
[SegFormer](https://github.com/NVlabs/SegFormer) (ImageNet-1K pretrained
`mit_b5.pth`).

## Training

```bash
cd ACT
conda activate impact_act
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/train.py configs/act/act_hrda_mitb5.py
```

- 40k iterations, batch size 2, 640×640 crops, AdamW (lr 6e-5, head ×10),
  linear warmup (1.5k iters) + poly decay; validation (mIoU, fire F1) every 4k
  iterations.
- Outputs go to `work_dirs/act_hrda_mitb5/` (log `*.log`, `*.log.json`,
  checkpoints `iter_*.pth`, `latest.pth`).
- Useful options: `--work-dir <dir>`, `--seed <n>`, `--deterministic`,
  `--resume-from <ckpt>`, `--load-from <ckpt>`.

## Evaluation

```bash
cd ACT
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/test.py \
    configs/act/act_hrda_mitb5.py work_dirs/act_hrda_mitb5/latest.pth --eval mIoU
```

Add `--show-dir <dir>` to save prediction visualizations.

## Method

ACT is implemented in [`mmseg/models/uda/act.py`](mmseg/models/uda/act.py)
(UDA type `ACT`); the weight maps are loaded by `LoadWeightMap`
([`mmseg/datasets/pipelines/weight_map.py`](mmseg/datasets/pipelines/weight_map.py))
and passed to the model by `UDADatasetACT`
([`mmseg/datasets/uda_dataset_act.py`](mmseg/datasets/uda_dataset_act.py)).

The weight map `W ∈ [-1, 1]` (positive inside the generated smoke region,
negative outside, growing with distance from the boundary) shifts the base
threshold `τ = pseudo_threshold = 0.968` per pixel between
`τ_min = act_tau_min = 0.5` and `τ_max = act_tau_max = 1.0`:

| EMA-teacher prediction | `W ≥ 0` | `W < 0` |
| --- | --- | --- |
| smoke | `τ − (τ − τ_min)·W` (relaxed) | `τ − (τ_max − τ)·W` (tightened) |
| background | `τ + (τ_max − τ)·W` (tightened) | `τ + (τ − τ_min)·W` (relaxed) |

Accepted pixels are weighted by their confidence in the pseudo-label loss.

## License

Released under [CC BY-NC 4.0](LICENSE.txt). Built on DAFormer, HRDA, MIC,
DACS and MMSegmentation; see [`resources/`](resources/) for third-party
licenses.
