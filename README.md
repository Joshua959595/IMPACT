# IMPACT: A Single Non-fire UAV Image-based Unsupervised Domain Adaptation for Wildfire Smoke Segmentation

IMPACT adapts a wildfire smoke segmentation model to a new UAV target domain
using only **a single non-fire UAV image** from that domain. From this image,
a Multilevel Planner (MP) and plan-guided self-refinement synthesize a
target-domain training set with smoke, and the segmentation model is then
adapted with **Adaptive Confidence Thresholding (ACT)**.

```
single non-fire UAV image + base user prompt
        │
        ▼
Generation/                                         ACT/
  1. Multilevel Planner (MP)                         HRDA (DAFormer, MiT-B5) self-training
       GGP Generator → Global Generation Plan          source: SMOKE5K (labeled)
       SGP Generator → Sample Generation Plans         target: IMPACT images + weight maps
  2. Plan-guided self-refinement (Qwen-Image-Edit)     ACT: per-pixel pseudo-label
  3. Referring segmentation masks (Sa2VA)                   threshold from the weight map
  4. Weight maps                     ──────────►     ACT/dataset/target/
```

## Repository structure

```
IMPACT/
├── Generation/                 # MP + self-refinement + masks + weight maps
│   ├── 01_generate_plans.py    # stage 1: Multilevel Planner (GGP, SGPs)
│   ├── 02_generate_images.py   # stage 2: plan-guided self-refinement
│   ├── 03_generate_masks.py    # stage 3: Sa2VA referring segmentation
│   ├── 04_generate_weightmaps.py  # stage 4: weight maps for ACT
│   ├── common/                 # model wrappers, config, utils
│   ├── configs/impact.yaml     # single config for all four stages
│   ├── reference/              # the single non-fire UAV image (image.jpg)
│   ├── requirements_gen.txt    # env for stages 1-2
│   └── requirements_seg.txt    # env for stages 3-4
└── ACT/                        # downstream UDA training with ACT
    ├── configs/act/act_hrda_mitb5.py
    ├── mmseg/                  # ACT: mmseg/models/uda/act.py
    ├── tools/train.py, tools/test.py
    ├── dataset/{source,target,val}/
    ├── pretrained/             # mit_b5.pth
    └── requirements.txt
```

## Environments

Three separate environments are used, because Sa2VA's remote code requires
`transformers==4.57.0` while stages 1-2 need `transformers==5.14.1`, and ACT
is built on mmcv 1.x. The versions below are the ones we used (CUDA 12.x
driver, H100 80GB).

**1. `impact_gen`: Generation stages 1-2**

```bash
conda create -n impact_gen python=3.11 -y
conda activate impact_gen
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r Generation/requirements_gen.txt
```

**2. `impact_seg`: Generation stages 3-4**

```bash
conda create -n impact_seg python=3.11 -y
conda activate impact_seg
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r Generation/requirements_seg.txt
```

**3. `impact_act`: ACT training/evaluation**

```bash
conda create -n impact_act python=3.8 -y
conda activate impact_act
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install -r ACT/requirements.txt
```

> If pip pulls a newer `nvidia-nvjitlink-cu12` into the cu121 environments and
> torch fails with `undefined symbol: __nvJitLinkGetErrorLogSize_12_x`, pin it:
> `pip install "nvidia-nvjitlink-cu12==12.1.105" --index-url https://download.pytorch.org/whl/cu121`.

Models are downloaded from the Hugging Face Hub on first use:
[Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct),
[Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509),
[ByteDance/Sa2VA-Qwen3-VL-4B](https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-4B).

## Datasets

Datasets are not included in this repository.

The SMOKE5K and original Boreal Forest Fire dataset is publicly available at
https://github.com/SiyuanYan1/Transmission-BVM and
https://etsin.fairdata.fi/dataset/1dce1023-493a-4d63-a906-f2a44f831898,
respectively. The extended Boreal Forest Fire subset used in this study is
publicly available as part of MultiNatSmoke at
https://github.com/henryzhao0615/MultiNatSmoke.

Arrange the data as follows. Masks are single-channel PNGs with values
`{0: background, 1: smoke}`.

```
Generation/reference/image.jpg       # a single non-fire UAV frame from the target domain

ACT/dataset/
├── source/                          # SMOKE5K (labeled source domain)
│   ├── images/*.jpg
│   └── masks/*.png                  # same basename as the image
├── target/                          # written by Generation (stages 1-4)
│   ├── images/*.png
│   ├── weight_maps/*.png
│   └── plans/, prompts/, masks/, masks_0_255/
└── val/                             # Boreal Forest Fire (target domain, evaluation)
    ├── images/*.jpg
    └── masks/*.png
```

## Usage

### 1. Generation (MP + self-refinement)

Edit `Generation/configs/impact.yaml` if needed (non-fire reference image, base user
prompt, number of samples, ...). By default every stage writes into
`ACT/dataset/target/`.

```bash
cd Generation

# stages 1-2
conda activate impact_gen
CUDA_VISIBLE_DEVICES=0 python 01_generate_plans.py  --config configs/impact.yaml
CUDA_VISIBLE_DEVICES=0 python 02_generate_images.py --config configs/impact.yaml

# stages 3-4
conda activate impact_seg
CUDA_VISIBLE_DEVICES=0 python 03_generate_masks.py      --config configs/impact.yaml
python 04_generate_weightmaps.py --config configs/impact.yaml
```

| Stage | Output (under `output_dir`) |
| --- | --- |
| 1. Multilevel Planner | `plans/global_plan.txt` (GGP), `plans/sample_plans.json` (SGPs: `sample_plan`, `start_prompt`) |
| 2. Self-refinement | `images/NNN.png`, `prompts/NNN.txt` (final refined prompt) |
| 3. Masks | `masks/NNN.png` ({0,1}), `masks_0_255/NNN.png` |
| 4. Weight maps | `weight_maps/NNN.png` |

Set `resume_on: true` in the config to only generate samples whose outputs
are still missing. Logs are written to `Generation/logs/`.

### 2. ACT training

Download the ImageNet-pretrained MiT-B5 weights provided by
[SegFormer](https://github.com/NVlabs/SegFormer) and place them at
`ACT/pretrained/mit_b5.pth`.

```bash
cd ACT
conda activate impact_act
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/train.py configs/act/act_hrda_mitb5.py
```

Training runs 40k iterations (batch size 2, 640×640 crops) and validates on
`dataset/val` every 4k iterations; checkpoints and logs go to
`work_dirs/act_hrda_mitb5/`.

### 3. Evaluation

```bash
cd ACT
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/test.py \
    configs/act/act_hrda_mitb5.py work_dirs/act_hrda_mitb5/latest.pth --eval mIoU
```

## Adaptive Confidence Thresholding (ACT)

Implemented in [`ACT/mmseg/models/uda/act.py`](ACT/mmseg/models/uda/act.py).
The weight map `W ∈ [-1, 1]` (positive inside the generated smoke region,
negative outside, growing with distance from the boundary) shifts the base
pseudo-label threshold `τ = 0.968` per pixel between `τ_min = 0.5` and
`τ_max = 1.0`:

| EMA-teacher prediction | `W ≥ 0` | `W < 0` |
| --- | --- | --- |
| smoke | `τ − (τ − τ_min)·W` (relaxed) | `τ − (τ_max − τ)·W` (tightened) |
| background | `τ + (τ_max − τ)·W` (tightened) | `τ + (τ − τ_min)·W` (relaxed) |

Pseudo-labels that agree with the weight map are accepted more easily, those
that contradict it need higher confidence; accepted pixels are weighted by
their confidence.

## Acknowledgments

ACT is built on [DAFormer](https://github.com/lhoyer/DAFormer),
[HRDA](https://github.com/lhoyer/HRDA), [MIC](https://github.com/lhoyer/MIC)
[DACS](https://github.com/vikolss/DACS) and
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) (v0.16.0),
using the codebase of CISS (Sakaridis et al., Condition-Invariant Semantic
Segmentation) as a starting point. The MiT backbone is from
[SegFormer](https://github.com/NVlabs/SegFormer).
We thank the authors for releasing their code.

## License

`ACT/` is derived from code released under the
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license
(see [`ACT/LICENSE.txt`](ACT/LICENSE.txt) and the third-party licenses in
[`ACT/resources/`](ACT/resources/)) and is released under the same license.
