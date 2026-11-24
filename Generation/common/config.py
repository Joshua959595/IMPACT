"""Loads the shared YAML config and derives every filesystem path from it.

All four Generation stages read the same config (configs/impact.yaml by
default). Relative paths in the config are resolved against the Generation/
directory, so the scripts can be launched from anywhere.

Output layout (written directly into ACT's target-domain folder by default):

    <output_dir>/
        plans/global_plan.txt     # GGP  (stage 1)
        plans/sample_plans.json   # SGPs (stage 1)
        images/000.png ...        # synthetic target images (stage 2)
        prompts/000.txt ...       # final refined T2I prompts (stage 2)
        masks_0_255/000.png ...   # masks in {0, 255} (stage 3)
        masks/000.png ...         # masks in {0, 1}   (stage 3)
        weight_maps/000.png ...   # ACT weight maps   (stage 4)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

GENERATION_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (GENERATION_ROOT / path).resolve()


@dataclass
class ModelConfig:
    vlm_id: str
    t2i_id: str
    seg_id: str


@dataclass
class PipelineConfig:
    reference_image: Path
    output_dir: Path
    resume_on: bool
    base_user_prompt: str
    referring_expression: str
    n_samples: int
    max_iters: int
    patience: int
    image_size: List[int]  # [W, H]
    models: ModelConfig

    # --- derived paths ---
    plan_dir: Path = field(init=False)
    image_dir: Path = field(init=False)
    prompt_dir: Path = field(init=False)
    mask_255_dir: Path = field(init=False)
    mask_dir: Path = field(init=False)
    weight_map_dir: Path = field(init=False)

    def __post_init__(self):
        self.plan_dir = self.output_dir / "plans"
        self.image_dir = self.output_dir / "images"
        self.prompt_dir = self.output_dir / "prompts"
        self.mask_255_dir = self.output_dir / "masks_0_255"
        self.mask_dir = self.output_dir / "masks"
        self.weight_map_dir = self.output_dir / "weight_maps"

    @property
    def global_plan_path(self) -> Path:
        return self.plan_dir / "global_plan.txt"

    @property
    def sample_plans_path(self) -> Path:
        return self.plan_dir / "sample_plans.json"

    def ensure_dirs(self, *, plans=False, images=False, masks=False, weight_maps=False):
        if plans:
            self.plan_dir.mkdir(parents=True, exist_ok=True)
        if images:
            self.image_dir.mkdir(parents=True, exist_ok=True)
            self.prompt_dir.mkdir(parents=True, exist_ok=True)
        if masks:
            self.mask_255_dir.mkdir(parents=True, exist_ok=True)
            self.mask_dir.mkdir(parents=True, exist_ok=True)
        if weight_maps:
            self.weight_map_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> PipelineConfig:
    with open(_resolve(str(path)), "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return PipelineConfig(
        reference_image=_resolve(raw["reference_image"]),
        output_dir=_resolve(raw["output_dir"]),
        resume_on=bool(raw.get("resume_on", False)),
        base_user_prompt=raw["base_user_prompt"].strip(),
        referring_expression=str(raw.get("referring_expression", "wildfire smoke")),
        n_samples=int(raw["n_samples"]),
        max_iters=int(raw["max_iters"]),
        patience=int(raw["patience"]),
        image_size=list(raw["image_size"]),
        models=ModelConfig(**raw["models"]),
    )


def build_arg_parser(default_config: str = "configs/impact.yaml") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=default_config)
    return p
