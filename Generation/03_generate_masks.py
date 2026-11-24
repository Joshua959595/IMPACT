"""Stage 3: referring segmentation masks for the generated images.

Uses Sa2VA-Qwen3-VL-4B (see common/sa2va.py): the model predicts a [SEG]
token whose hidden state conditions a bundled SAM mask decoder. Each image
is segmented with the configured referring expression plus a context hint
(its SGP sample_plan and the final T2I prompt used to generate it).

Two mask folders are written per sample:
  masks_0_255/  -- values in {0, 255}, easy to eyeball
  masks/        -- values in {0, 1}, input to stage 4

Sa2VA's remote code requires transformers==4.57.0, so this stage runs in its
own environment (see README).

Usage:
    CUDA_VISIBLE_DEVICES=0 python 03_generate_masks.py --config configs/impact.yaml
"""
import common.gpu  # noqa: F401

import json as _json
import time
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from common.config import build_arg_parser, load_config
from common.run_logging import setup_logging
from common.sa2va import Sa2VASegmenter

MAX_RETRY = 3


def build_context_hint(sample_plan: str, used_prompt: str) -> str:
    return (
        f"[SAMPLE PLAN]\n{sample_plan}\n\n"
        f"[PROMPT USED TO GENERATE THIS IMAGE]\n{used_prompt}\n"
    )


def run_generation(seg: Sa2VASegmenter, cfg, indices) -> int:
    target_size = tuple(cfg.image_size)  # (W, H)
    packs = _json.loads(cfg.sample_plans_path.read_text(encoding="utf-8"))

    generated_count = 0
    for idx in indices:
        idx_str = f"{idx:03d}"
        try:
            img_path = cfg.image_dir / f"{idx_str}.png"
            input_img = Image.open(img_path).convert("RGB")

            sample_plan = packs[idx]["sample_plan"]
            used_prompt = (cfg.prompt_dir / f"{idx_str}.txt").read_text(encoding="utf-8")
            context_hint = build_context_hint(sample_plan, used_prompt)

            mask_bool = None
            last_err = None
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    mask_bool = seg.segment(input_img, cfg.referring_expression, context_hint=context_hint)
                    break
                except Exception as e:
                    last_err = e
                    print(f"[Sa2VA retry {attempt}/{MAX_RETRY}] {idx_str} failed: {type(e).__name__}: {e}")
                    time.sleep(0.6)
            else:
                raise RuntimeError(f"Sa2VA segmentation failed after retries. Last error: {last_err}")

            if mask_bool is None:
                # No target found -> an all-black mask is the correct output.
                mask_arr = np.zeros(input_img.size[::-1], dtype=np.uint8)  # (H, W)
                fg_note = "nothing detected"
            else:
                mask_arr = (mask_bool.astype(np.uint8)) * 255
                fg_note = f"{int(mask_bool.sum())} px"

            # NEAREST resize keeps the mask exactly {0,255}.
            mask_255_img = Image.fromarray(mask_arr, mode="L").resize(target_size, Image.NEAREST)
            mask_255_img.save(cfg.mask_255_dir / f"{idx_str}.png")

            arr01 = (np.array(mask_255_img) > 0).astype(np.uint8)
            Image.fromarray(arr01, mode="L").save(cfg.mask_dir / f"{idx_str}.png")

            print(f"[OK] {idx_str} ({fg_note})")
            generated_count += 1

        except Exception as e:
            print(f"[FAIL] {idx_str} {type(e).__name__}: {e}")

    return generated_count


def find_available_indices(cfg) -> List[int]:
    """Sample indices that have both a generated image and its prompt saved
    (i.e. 02_generate_images.py finished them), regardless of mask state."""
    available = []
    for idx in range(cfg.n_samples):
        idx_str = f"{idx:03d}"
        img_path = cfg.image_dir / f"{idx_str}.png"
        prompt_path = cfg.prompt_dir / f"{idx_str}.txt"
        if img_path.exists() and prompt_path.exists():
            available.append(idx)
    return available


def find_missing_generation_indices(cfg) -> List[int]:
    return [
        idx for idx in find_available_indices(cfg)
        if not (cfg.mask_255_dir / f"{idx:03d}.png").exists()
    ]


# Final audit (non-mutating): reports mask value sets so a bad batch is easy
# to spot.

def _read_mask_as_array(p: Path) -> np.ndarray:
    img = Image.open(p)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img)


def check_masks_binary(mask_dir: Path, pattern: str = "*.png", max_print: int = 30):
    files = sorted(mask_dir.glob(pattern))
    if not files:
        print(f"[INFO] No files in {mask_dir}")
        return

    summary = Counter()
    bad_files = []
    for p in files:
        arr = _read_mask_as_array(p)
        u = np.unique(arr)
        u_key = tuple(u.tolist()) if len(u) <= 16 else ("many", int(len(u)), int(u.min()), int(u.max()))
        summary[u_key] += 1

        good = (
            set(u.tolist()).issubset({0, 1})
            or set(u.tolist()).issubset({0, 255})
            or len(u) == 1
        )
        if not good:
            bad_files.append((p.name, u[:20].tolist(), int(u.min()), int(u.max()), int(len(u))))

    print(f"[DONE] total={len(files)}")
    print("---- Unique-set summary (top) ----")
    for i, (k, v) in enumerate(summary.most_common(max_print), 1):
        print(f"{i:02d}) count={v} unique={k}")

    if bad_files:
        print("\n---- BAD (not subset of {0,1} or {0,255}) ----")
        for name, head_u, umin, umax, k in bad_files[:max_print]:
            print(f"- {name}: unique_head={head_u}  min={umin} max={umax}  n_unique={k}")
        print(f"[WARN] bad_files={len(bad_files)} (showing first {min(max_print, len(bad_files))})")
    else:
        print("[OK] All masks look binary-ish ({0,1} or {0,255} or constant).")


def main():
    setup_logging("03_generate_masks")

    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    cfg.ensure_dirs(masks=True)
    print(f"Reading images / writing masks under: {cfg.output_dir}")

    print(f"[Setup] Loading segmentation model: {cfg.models.seg_id}")
    seg = Sa2VASegmenter(cfg.models.seg_id)

    t_start = time.time()

    if cfg.resume_on:
        target_idxs = find_missing_generation_indices(cfg)
        print(f"[resume_on=true] {len(target_idxs)} mask(s) still missing: {target_idxs}")
    else:
        target_idxs = find_available_indices(cfg)
        print(f"[resume_on=false] generating masks for all {len(target_idxs)} available sample(s)")

    generated_count = 0
    if target_idxs:
        generated_count = run_generation(seg, cfg, target_idxs)
    else:
        print("[OK] Nothing to generate.")

    print("\n===== FINAL AUDIT (masks_0_255) =====")
    check_masks_binary(cfg.mask_255_dir)

    print("\n===== FINAL AUDIT (masks) =====")
    check_masks_binary(cfg.mask_dir)

    elapsed = time.time() - t_start
    print(f"[TIMING] total run: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[SUMMARY] masks generated this run: {generated_count}")


if __name__ == "__main__":
    main()
