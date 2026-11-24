"""Stage 1: Multilevel Planner (MP).

Two-level planning with Qwen3-VL (no image is generated in this stage):
  1. GGP Generator -> ONE Global Generation Plan (GGP) for the whole dataset,
     from the base user prompt + the reference image.
  2. SGP Generator -> one Sample Generation Plan (SGP) + start_prompt per
     sample, instantiated from the GGP.

Writes <output_dir>/plans/global_plan.txt and <output_dir>/plans/sample_plans.json,
which stage 2 (02_generate_images.py) reads back.

If both files already exist, this script does nothing; pass --force to
regenerate.

Usage:
    CUDA_VISIBLE_DEVICES=0 python 01_generate_plans.py --config configs/impact.yaml [--force]
"""
import common.gpu  # noqa: F401

import json as _json
import time
from typing import Dict, List, Tuple

from PIL import Image

from common.config import build_arg_parser, load_config
from common.io_utils import atomic_write_json
from common.qwen_vlm import QwenVLM
from common.run_logging import setup_logging
from common.text_utils import extract_json_object, render

MAX_RETRY = 3

# =========================================================================
# System prompts & templates
# =========================================================================

GGP_SYSTEM_PROMPT = """You are a Domain Adaptation Expert designing a global dataset plan for synthetic image generation.


Goal:
- Use the base user prompt as the primary specification of what the dataset should contain and what should vary.
- Use the reference/condition image to preserve scene identity/context.
- Define dataset-level criteria that encourage robustness through realistic variation.

Task:
- Produce ONE global plan for the entire set of samples.
- The plan should state the main criteria that will guide diversity across the dataset, derived from the base user prompt and extended when needed.
- Possible criteria may include, but are not limited to, capture geometry, target phenomenon appearance, scene context, environmental conditions, and lighting.
- Add any additional criteria that are important for representing the user’s intended dataset realistically and comprehensively, rather than limiting the plan to only the example categories.
- Write the global_plan as a numbered list inside the string:
  1. ...
  2. ...
  3. ...


Important:
- The global plan defines WHAT kinds of variation exist across the dataset.
- Do not assign criteria to specific samples.


Output JSON only with keys:
{
  "global_plan": "1. ...\\n2. ...\\n3. ..."
}
""".strip()

GGP_USER_TEMPLATE = """Base user prompt:
{base_user_prompt}

A reference/condition image is provided.

Now produce:
- global_plan: a numbered list of dataset-level criteria that follow the base user prompt and expand it with additional realistic variation when helpful.
  Format the global_plan string as:
  1. ...
  2. ...
  3. ...

Return JSON only.
""".strip()


SGP_SYSTEM_PROMPT = """You are a Domain Adaptation Expert producing sample-wise plans from a global dataset plan.


Goal:
- Use the global dataset plan as the governing specification for diversity across the full sample set.
- Convert that global plan into concrete, realistic sample-level capture situations.
- Use the reference/condition image to preserve scene identity/context while allowing plausible variation required by the base user prompt.


Task:
- You are given a global plan for the full set and a list of sample indices.
- For EACH sample index:
  1) Write a sample_plan that instantiates the global plan into a concrete capture situation for this sample.
     The plan should reflect what will vary and what will remain consistent for this sample.
  2) Specify anchors for where key phenomena should appear in the frame using approximate regions
     (e.g., left-lower quadrant, right-middle area, top third).
  3) Produce ONE start_prompt that applies prompt engineering to the sample_plan and is directly usable for image generation.
     The start_prompt MUST explicitly include the anchors defined in the sample_plan and restate their placements in the image.


Important:
- Keep each output realistic for natural capture (natural lighting, no cinematic color grading, avoid dramatic or artistic mood words).
- Each output should represent a single realistic image (not a collage, grid, or multi-panel layout).
- Think of a drone moving through the area and capturing different views; each sample reflects a different capture moment.
- The set of sample plans should remain balanced across the global criteria; avoid repeatedly generating the same type of situation.
- If any diversity criterion (including environmental conditions) would require the target phenomenon to be absent, only implied, or in a precursor state, do not apply that criterion for this sample.
- Avoid making the target look uniformly prominent in every sample.
- Every sample_plan must specify a target phenomenon anchor.



Output JSON only with the following schema:
{
  "packs": [
    {
      "sample_index": 0,
      "sample_plan": "...",
      "start_prompt": "..."
    }
  ]
}
""".strip()

SGP_USER_TEMPLATE = """Global plan (for the full set):
{global_plan}

Sample indices:
{sample_indices}

A reference/condition image is provided.

Now produce packs for ALL indices in the required JSON schema.
Return JSON only.
""".strip()


# =========================================================================
# GGP / SGP generators; `vlm` is set up in main() after the config is loaded.
# =========================================================================

vlm: QwenVLM


def generate_ggp(base_user_prompt: str, condition_img: Image.Image, max_retry: int = MAX_RETRY) -> str:
    """GGP Generator: one Global Generation Plan for the whole dataset."""
    user_text = render(GGP_USER_TEMPLATE, base_user_prompt=base_user_prompt)

    last_err = None
    for attempt in range(1, max_retry + 1):
        out = vlm.call_mm_text(
            user_text=user_text,
            images=[("[Condition image]", condition_img)],
            system_prompt=GGP_SYSTEM_PROMPT,
            temperature=0.7,
            max_new_tokens=16384,
        )
        try:
            data = _json.loads(extract_json_object(out))
            gp = data.get("global_plan", "")
            if not isinstance(gp, str) or not gp.strip():
                raise ValueError("Missing/invalid key: global_plan")
            return gp.strip()
        except Exception as e:
            last_err = e
            print(f"[GGP retry {attempt}/{max_retry}] parse failed. head={out[:120]!r}")
            time.sleep(0.6)

    raise RuntimeError(f"GGP generation failed after retries. Last error: {last_err}")


def generate_sgps(
    condition_img: Image.Image,
    global_plan: str,
    sample_indices: List[int],
    max_retry: int = MAX_RETRY,
    temperature: float = 0.7,
) -> List[Dict]:
    """SGP Generator: one Sample Generation Plan + start_prompt per index."""
    user_text = render(
        SGP_USER_TEMPLATE,
        global_plan=global_plan,
        sample_indices=", ".join(str(i) for i in sample_indices),
    )

    last_err = None
    for attempt in range(1, max_retry + 1):
        out = vlm.call_mm_text(
            user_text=user_text,
            images=[("[Condition image]", condition_img)],
            system_prompt=SGP_SYSTEM_PROMPT,
            temperature=temperature,
            # ~200-400 tokens of JSON per sample on top of the base budget.
            max_new_tokens=min(8192 + 500 * len(sample_indices), 32768),
        )
        try:
            data = _json.loads(extract_json_object(out))
            packs = data.get("packs", None)
            if not isinstance(packs, list) or len(packs) == 0:
                raise ValueError("Missing/invalid key: packs")

            out_packs = []
            seen = set()
            for p in packs:
                if not isinstance(p, dict):
                    raise ValueError("Pack item is not an object")

                si = p.get("sample_index", None)
                if not isinstance(si, int):
                    raise ValueError("Missing/invalid sample_index")
                if si in seen:
                    raise ValueError(f"Duplicate sample_index: {si}")
                seen.add(si)

                for k in ["sample_plan", "start_prompt"]:
                    if k not in p or not isinstance(p[k], str) or not p[k].strip():
                        raise ValueError(f"Missing/invalid key in pack[{si}]: {k}")

                out_packs.append({
                    "sample_index": si,
                    "sample_plan": p["sample_plan"].strip(),
                    "start_prompt": p["start_prompt"].strip(),
                })

            missing = sorted(set(sample_indices) - set(pp["sample_index"] for pp in out_packs))
            if missing:
                raise ValueError(f"Missing packs for indices: {missing[:10]}{'...' if len(missing) > 10 else ''}")

            out_packs.sort(key=lambda x: x["sample_index"])
            return out_packs

        except Exception as e:
            last_err = e
            print(f"[SGP retry {attempt}/{max_retry}] parse failed. head={out[:120]!r}")
            time.sleep(0.6)

    raise RuntimeError(f"SGP generation failed after retries. Last error: {last_err}")


def multilevel_plan(
    base_user_prompt: str,
    condition_img: Image.Image,
    n_samples: int,
    max_retry: int = MAX_RETRY,
    batch_chunk: int = 15,
) -> Tuple[str, List[Dict]]:
    """GGP first, then all SGPs in one call (falls back to chunks of
    `batch_chunk` indices, all conditioned on the same GGP, if the one-shot
    call fails)."""
    global_plan = generate_ggp(base_user_prompt, condition_img, max_retry=max_retry)

    all_indices = list(range(n_samples))
    try:
        packs = generate_sgps(condition_img, global_plan, all_indices, max_retry=max_retry)
        return global_plan, packs
    except Exception as e:
        print("[SGP] One-shot batch failed -> fallback to chunking. err:", repr(e))

    packs_map = {}
    for s in range(0, n_samples, batch_chunk):
        chunk = all_indices[s:s + batch_chunk]
        for p in generate_sgps(condition_img, global_plan, chunk, max_retry=max_retry):
            packs_map[p["sample_index"]] = p

    return global_plan, [packs_map[i] for i in all_indices]


def main():
    setup_logging("01_generate_plans")

    parser = build_arg_parser()
    parser.add_argument("--force", action="store_true", help="Regenerate even if plan files already exist.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs(plans=True)

    global_path = cfg.global_plan_path
    packs_path = cfg.sample_plans_path

    if not args.force and global_path.exists() and packs_path.exists():
        print(f"[SKIP] Plan files already exist:\n  {global_path}\n  {packs_path}")
        print("Pass --force to regenerate.")
        return

    global vlm
    print(f"[Setup] Loading VLM: {cfg.models.vlm_id}")
    vlm = QwenVLM(cfg.models.vlm_id)

    t_start = time.time()
    condition_img = Image.open(cfg.reference_image).convert("RGB")

    print("Generating GGP and SGPs...")
    global_plan, sample_plans = multilevel_plan(
        base_user_prompt=cfg.base_user_prompt, condition_img=condition_img, n_samples=cfg.n_samples
    )

    global_path.write_text(global_plan, encoding="utf-8")
    atomic_write_json(packs_path, sample_plans)
    print("GGP saved:", global_path)
    print("SGPs saved:", packs_path)

    elapsed = time.time() - t_start
    print(f"[TIMING] total run: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
