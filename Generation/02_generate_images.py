"""Stage 2: plan-guided self-refinement image generation.

For every sample, the SGP (sample_plan + start_prompt) produced by the
Multilevel Planner (01_generate_plans.py) drives an iterative
self-refinement loop:
  - Qwen3-VL-32B-Instruct  -> DVQ generation/evaluation, prompt editing,
                              pairwise comparator
  - Qwen-Image-Edit-2509   -> image generation, conditioned on the
                              reference image
The loop runs up to `max_iters` iterations and stops early after `patience`
consecutive non-improving iterations; the best image/prompt is kept.

Usage:
    CUDA_VISIBLE_DEVICES=0 python 02_generate_images.py --config configs/impact.yaml
"""
import common.gpu  # noqa: F401

import io
import json as _json
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image

from common.config import build_arg_parser, load_config
from common.qwen_t2i import QwenT2I
from common.qwen_vlm import QwenVLM
from common.run_logging import setup_logging
from common.text_utils import extract_all_tagged, extract_first_tag, extract_json_object, render

random.seed(42)

MAX_RETRY = 3

# =========================================================================
# System prompts & templates
# =========================================================================

DVQ_SYSTEM_PROMPT = """You are an expert at converting a user's image-generation request into Decomposed Visual Questions (DVQs).
Your job is to produce a compact set of YES/NO visual questions that can be answered by a vision-language model by looking at a generated image.

GROUNDING:
- DVQs must be derived from the provided intent text.
- When a reference/condition image is provided, you MUST jointly consider BOTH the intent text and the image.
- The image clarifies ambiguity/context but must NOT introduce new constraints absent from the intent text.
- If the intent requires adding elements not present in the condition image, DVQs MUST include those required elements.
- If the intent text specifies where an element should appear in the image, DVQs should include questions verifying that the element appears in the intended spatial region.
- DVQs represent stable intent constraints for this sample; later prompt optimization must not change DVQs.

REQUIREMENTS:
- Each question tests exactly ONE visual constraint.
- Concrete, visually verifiable, unambiguous; YES/NO answerable.
- Avoid subjective aesthetics unless explicitly requested.
- Prefer: subject presence, attributes, counts, spatial relations, viewpoint/camera, lighting, environment, style (if implied).
- If the intent text makes an explicit, checkable claim about an attribute of the target phenomenon (e.g., size/prominence, color, density, shape), you MUST include a DVQ question that verifies the rendered result actually matches that claim - not just that the phenomenon is present.

OUTPUT JSON ONLY:
{ "dvqs": ["Q1 ...?", "Q2 ...?", ...] }

PROCESS:
1) Draft DVQs from joint interpretation of intent text + image.
2) Refine once: remove duplicates, fix ambiguity, ensure coverage.
Return ONLY the refined JSON.
""".strip()



DVQ_BATCH_REVIEW_SYSTEM_PROMPT = """You are an expert reviewer for generated images.

You are given:
- one generated image
- multiple yes/no visual questions

Your task:
- evaluate all questions in one pass
- keep the same order as the input questions
- for each question, provide:
  1) answer: yes or no
  2) rationalization: short reason grounded in the image
  3) suggestion: if answer is no, give a concrete prompt-edit suggestion; if answer is yes, write NO_CHANGE

Output format exactly:

<questions>Q1 / Q2 / Q3</questions>
<answers>yes / no / no</answers>
<rationalizations>... / ... / ...</rationalizations>
<suggestions>NO_CHANGE / ... / ...</suggestions>

Rules:
- Use exactly one entry per input question.
- Preserve the exact original question text and order.
- Separate entries only with " / ".
- Do not use " / " inside a single entry.
- Be strict but fair.
- Judge only from what is visually observable.
""".strip()

COMPARATOR_TEMPLATE = """You are an expert in critiquing images.

Given a user prompt and a sample generation plan, a text-to-image model generated two images ("Image A" and "Image B").
Pick the overall better image for this sample.

User prompt:
"{user_prompt}"

Sample generation plan (what this sample should emphasize):
{plan_text}

A reference/condition image is also provided. You MUST consider it when judging which image better satisfies the user prompt,
while preserving the condition image's scene identity/context.

Condition image: {condition_image}
Image A: {image_A}
Image B: {image_B}

Please explain the rationale before your final choice.
Finish with: <answer> X </answer>, where X is A or B.
"""

TARGETED_EDIT_TEMPLATE = """You are an expert in prompt engineering for text-to-image generation models.

My current prompt is:
'{best_prompt_so_far}'

A reviewer answered 'No' to one or more questions and provided feedback below:
{qa_feedback_block}

Now improve my prompt by incorporating all feedback points into {new_solutions} new prompt(s),
so that the reviewer will answer 'Yes' to the questions when generating a new image.

Bracket each new prompt between <PROMPT> and </PROMPT>.
"""

IMPLICIT_IMPROVE_TEMPLATE = """Hi, I want to create an image featuring the topic of '{user_prompt}'
Here is my current prompt:
'{best_prompt_so_far}'

I asked the T2I model to generate an image using the above prompt. Does it correctly reflect the topic?

Generated image (best so far) is provided.
A reference/condition image is also provided.

You MUST consider BOTH the original user prompt and the condition image when judging correctness and proposing improvements.
- Use the condition image to preserve identity/context, but do not treat it as a limit of what may appear.
- If the user prompt asks to add new content not present in the condition image, reflect that in the rewritten prompt.

If yes, respond with "<PROMPT> NO_CHANGE </PROMPT>".
If no, generate {n_prompts} new prompt(s) and bracket each in <PROMPT>...</PROMPT>.
"""

MERGE_TEMPLATE = """You are to produce ONE final improved prompt for a text-to-image model.

Original user prompt:
{user_prompt}

Current best prompt:
{best_prompt}

A reference/condition image is provided. You MUST consider it while merging suggestions.
- Use the condition image to preserve identity/context, but do not treat it as a limit of what may appear.
- If the original user prompt requires adding new content not present in the condition image, you must include it.
- Do NOT introduce new constraints solely based on the condition/reference image.

Below are TWO improvement suggestions.

[Targeted editing suggestion (based on failed DVQs + feedback)]
{p_target}

[Implicit improvement suggestion (holistic improvement)]
{p_impl}

Now write ONE merged prompt that best satisfies the original user prompt and addresses ALL issues.
If the two suggestions conflict, prioritize satisfying the ORIGINAL USER PROMPT and fixing failed DVQs.

Return ONLY the merged prompt text (no tags, no extra commentary).
"""

VERIFY_SELF_CORRECT_TEMPLATE = """Your task is to first verify whether my original prompt satisfies each of the constraints specified by the user:
{constraints}

Verify each constraint and answer "Yes" or "No" with a short explanation.
If any unmet constraint exists, revise the prompt to satisfy all constraints.
Only modify the part that caused violations.
Bracket the revised prompt between <answer> and </answer>.
If the original prompt satisfies all constraints, write "<answer> NO_CHANGE </answer>".
""".strip()



# =========================================================================
# Self-refinement components; `vlm`/`t2i` are module-level singletons set
# up in main() after the config is loaded.
# =========================================================================

vlm: QwenVLM
t2i: QwenT2I


def _generate_dvqs_from_intent(intent_text: str, condition_img: Image.Image, max_retry: int = MAX_RETRY) -> List[str]:
    user_text = f"Intent text for this sample:\n{intent_text}\n\nGenerate the refined DVQs now in JSON."

    last_err = None
    for attempt in range(1, max_retry + 1):
        out = vlm.call_mm_text(
            user_text=user_text,
            images=[("[Condition image]", condition_img)],
            system_prompt=DVQ_SYSTEM_PROMPT,
            temperature=0.7,
        )
        cand = extract_json_object(out)
        try:
            data = _json.loads(cand)
            dvqs = data.get("dvqs", None)
            if isinstance(dvqs, list) and all(isinstance(q, str) for q in dvqs) and len(dvqs) > 0:
                return [q.strip() for q in dvqs if q.strip()]
            last_err = ValueError("Bad dvqs field")
        except Exception as e:
            last_err = e

        print(f"[DVQGen retry {attempt}/{max_retry}] JSON parse failed. head={out[:120]!r}")
        time.sleep(0.6)

    raise RuntimeError(f"DVQ generation failed after retries. Last error: {last_err}")


def generate_dvqs(sample_plan: str, condition_img: Image.Image, max_retry: int = MAX_RETRY) -> List[str]:
    intent_text = (
        "The following is the guidance used to create this sample:\n"
        f"{sample_plan}\n\n"
        "You MUST refer to this guidance."
    )
    return _generate_dvqs_from_intent(intent_text, condition_img, max_retry)


def evaluate_dvqs_batch_with_suggestions(
    image: Image.Image, dvqs: List[str], max_retry: int = MAX_RETRY, temperature: float = 0.2
) -> List[Dict[str, str]]:
    if not dvqs:
        return []

    q_block = " / ".join(dvqs)
    user_text = f"""Evaluate the attached generated image for the following questions.

Questions: {q_block}
"""

    last_err = None
    for attempt in range(1, max_retry + 1):
        out = vlm.call_mm_text(
            user_text=user_text,
            images=[("[Generated image]", image)],
            system_prompt=DVQ_BATCH_REVIEW_SYSTEM_PROMPT,
            temperature=temperature,
        )
        try:
            qs_text = extract_first_tag(out, "questions") or ""
            ans_text = extract_first_tag(out, "answers") or ""
            rat_text = extract_first_tag(out, "rationalizations") or ""
            sug_text = extract_first_tag(out, "suggestions") or ""

            qs = [x.strip() for x in qs_text.split(" / ")] if qs_text else []
            ans = [x.strip().lower() for x in ans_text.split(" / ")] if ans_text else []
            rat = [x.strip() for x in rat_text.split(" / ")] if rat_text else []
            sug = [x.strip() for x in sug_text.split(" / ")] if sug_text else []

            if not (len(qs) == len(ans) == len(rat) == len(sug) == len(dvqs)):
                raise ValueError(
                    f"Parse mismatch: got lens qs={len(qs)}, ans={len(ans)}, rat={len(rat)}, sug={len(sug)}, expected={len(dvqs)}"
                )

            if qs != dvqs:
                raise ValueError("Returned questions do not exactly match input DVQs")

            results = []
            for i in range(len(dvqs)):
                answer = ans[i]
                if answer.startswith("y"):
                    answer = "yes"
                elif answer.startswith("n"):
                    answer = "no"
                else:
                    raise ValueError(f"Invalid answer: {ans[i]!r}")

                results.append({
                    "question": qs[i],
                    "answer": answer,
                    "rationalization": rat[i],
                    "suggestion": sug[i],
                })

            return results

        except Exception as e:
            last_err = e
            print(f"[DVQ-Batch retry {attempt}/{max_retry}] parse failed. head={out[:180]!r}")
            time.sleep(0.6)

    raise RuntimeError(f"DVQ batch review failed after retries. Last error: {last_err}")


def targeted_edit(best_prompt: str, failed_qs: List[str], feedbacks: List[str], new_solutions: int = 1) -> str:
    qa_block_lines = []
    for i, (q, fb) in enumerate(zip(failed_qs, feedbacks), start=1):
        qa_block_lines.append(f"Question {i}:\n{q}\nFeedback {i}:\n{fb}\n")
    qa_feedback_block = "\n".join(qa_block_lines).strip()

    user_text = render(
        TARGETED_EDIT_TEMPLATE,
        best_prompt_so_far=best_prompt,
        qa_feedback_block=qa_feedback_block,
        new_solutions=new_solutions,
    )

    out = vlm.call_mm_text(user_text=user_text, images=None, system_prompt="", temperature=0.7)
    prompts = extract_all_tagged(out, "PROMPT")
    if prompts:
        return prompts[0].strip()
    return out.strip()


def implicit_improve(
    user_prompt: str, best_prompt: str, best_img: Image.Image, condition_img: Image.Image, n_prompts: int = 1
) -> str:
    user_text = render(
        IMPLICIT_IMPROVE_TEMPLATE, user_prompt=user_prompt, best_prompt_so_far=best_prompt, n_prompts=n_prompts
    )

    out = vlm.call_mm_text(
        user_text=user_text,
        images=[("[Best generated image so far]", best_img), ("[Condition image]", condition_img)],
        system_prompt="",
        temperature=0.7,
    )
    prompts = extract_all_tagged(out, "PROMPT")
    if prompts:
        if prompts[0].strip() == "NO_CHANGE":
            return best_prompt
        return prompts[0].strip()
    return out.strip() or best_prompt


def merge_prompts(user_prompt: str, best_prompt: str, p_target: str, p_impl: str, condition_img: Image.Image) -> str:
    user_text = render(MERGE_TEMPLATE, user_prompt=user_prompt, best_prompt=best_prompt, p_target=p_target, p_impl=p_impl)

    out = vlm.call_mm_text(
        user_text=user_text, images=[("[Condition image]", condition_img)], system_prompt="", temperature=0.7
    )
    return out.strip() or best_prompt


def verify_self_correct(candidate_prompt: str, dvqs: List[str]) -> str:
    constraints = "\n".join([f"- {q}" for q in dvqs])
    user_text = render(VERIFY_SELF_CORRECT_TEMPLATE, constraints=constraints) + f"\n\nPrompt:\n{candidate_prompt}"
    out = vlm.call_mm_text(user_text=user_text, images=None, system_prompt="", temperature=0.7)
    ans = extract_first_tag(out, "answer")
    if ans is None:
        return candidate_prompt
    if ans.strip() == "NO_CHANGE":
        return candidate_prompt
    return ans.strip()


def comparator_judge_with_trace(
    user_prompt: str, plan_text: str, imgA: Image.Image, imgB: Image.Image, condition_img: Image.Image
) -> Tuple[str, str]:
    user_text = render(
        COMPARATOR_TEMPLATE,
        user_prompt=user_prompt,
        plan_text=plan_text,
        condition_image="(see condition image)",
        image_A="(see Image A)",
        image_B="(see Image B)",
    )
    out = vlm.call_mm_text(
        user_text=user_text,
        images=[("[Condition image]", condition_img), ("[Image A]", imgA), ("[Image B]", imgB)],
        system_prompt="",
        temperature=0.7,
    )
    raw_out = out.strip()
    ans = extract_first_tag(raw_out, "answer")
    if not ans:
        return "A", raw_out
    ans = ans.strip().upper()
    decision = "A" if "A" in ans else ("B" if "B" in ans else "A")
    return decision, raw_out


def duel_current_vs_best_with_trace(
    user_prompt: str, plan_text: str, current: Image.Image, best: Image.Image, condition_img: Image.Image
) -> Tuple[bool, Dict[str, str]]:
    votes_current = 0
    votes_best = 0

    first_choice, first_trace = comparator_judge_with_trace(user_prompt, plan_text, current, best, condition_img)
    if first_choice == "A":
        votes_current += 1
    else:
        votes_best += 1

    second_choice, second_trace = comparator_judge_with_trace(user_prompt, plan_text, best, current, condition_img)
    if second_choice == "B":
        votes_current += 1
    else:
        votes_best += 1

    if votes_current == votes_best:
        winner = random.random() < 0.5
    else:
        winner = votes_current > votes_best

    trace = {
        "judge_1_choice": first_choice,
        "judge_1_raw": first_trace,
        "judge_2_choice": second_choice,
        "judge_2_raw": second_trace,
        "votes_current": str(votes_current),
        "votes_best": str(votes_best),
        "winner": "current" if winner else "best",
    }
    return winner, trace



# =========================================================================
# Self-refinement loop
# =========================================================================

@dataclass
class RefineState:
    base_user_prompt: str
    sample_plan: str

    best_prompt: str
    best_image_bytes: bytes
    best_image: Image.Image
    dvqs: List[str]

    no_improve_streak: int = 0


def run_self_refinement(
    base_user_prompt: str,
    condition_img: Image.Image,
    sample_plan: str,
    start_prompt: str,
    max_iters: int = 5,
    patience: int = 3,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> RefineState:
    """DVQs come from the SGP's sample_plan, the initial prompt is the SGP's
    start_prompt; each iteration proposes a refined prompt and keeps it only
    if the comparator prefers its image over the best one so far."""
    dvqs = generate_dvqs(sample_plan, condition_img)
    p0 = start_prompt

    img0_bytes = t2i.generate_image_t2i(p0, condition_img, width=image_width, height=image_height)
    img0 = Image.open(io.BytesIO(img0_bytes)).convert("RGB")

    state = RefineState(
        base_user_prompt=base_user_prompt,
        sample_plan=sample_plan,
        best_prompt=p0,
        best_image_bytes=img0_bytes,
        best_image=img0,
        dvqs=dvqs,
        no_improve_streak=0,
    )

    plan_text_for_comparator = (
        f"Guidance used to create this sample:\n{state.sample_plan}\n\nYou MUST refer to this guidance when judging."
    )

    for _ in range(1, max_iters + 1):
        review_results = evaluate_dvqs_batch_with_suggestions(state.best_image, state.dvqs)

        failed_qs, feedbacks = [], []
        for item in review_results:
            if item["answer"] == "no":
                failed_qs.append(item["question"])
                feedbacks.append(f"Rationalization: {item['rationalization']}\nSuggestion: {item['suggestion']}")

        p_target = targeted_edit(state.best_prompt, failed_qs, feedbacks, new_solutions=1) if failed_qs else state.best_prompt

        p_impl = implicit_improve(
            user_prompt=state.base_user_prompt,
            best_prompt=state.best_prompt,
            best_img=state.best_image,
            condition_img=condition_img,
            n_prompts=1,
        )

        candidate = merge_prompts(
            user_prompt=state.base_user_prompt, best_prompt=state.best_prompt, p_target=p_target, p_impl=p_impl,
            condition_img=condition_img,
        )
        candidate = verify_self_correct(candidate, state.dvqs)

        cur_bytes = t2i.generate_image_t2i(candidate, condition_img, width=image_width, height=image_height)
        cur_img = Image.open(io.BytesIO(cur_bytes)).convert("RGB")

        current_wins, _ = duel_current_vs_best_with_trace(
            user_prompt=state.base_user_prompt,
            plan_text=plan_text_for_comparator,
            current=cur_img,
            best=state.best_image,
            condition_img=condition_img,
        )

        if current_wins:
            state.best_prompt = candidate
            state.best_image_bytes = cur_bytes
            state.best_image = cur_img
            state.no_improve_streak = 0
        else:
            state.no_improve_streak += 1

        if state.no_improve_streak >= patience:
            break

    return state


# =========================================================================
# Entry point
# =========================================================================

def main():
    setup_logging("02_generate_images")

    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)

    packs_path = cfg.sample_plans_path
    if not packs_path.exists():
        raise SystemExit(
            f"Sample plans not found: {packs_path}\n"
            f"Run 01_generate_plans.py first (same --config)."
        )
    cfg.ensure_dirs(images=True)

    global vlm, t2i
    print(f"[Setup] Loading VLM: {cfg.models.vlm_id}")
    vlm = QwenVLM(cfg.models.vlm_id)
    print(f"[Setup] Loading T2I: {cfg.models.t2i_id}")
    t2i = QwenT2I(cfg.models.t2i_id)

    t_start = time.time()

    condition_img = Image.open(cfg.reference_image).convert("RGB")
    sample_plans = _json.loads(packs_path.read_text(encoding="utf-8"))
    print(f"[IMPACT] plan-guided self-refinement -> {cfg.output_dir}")

    def _generate_one(idx: int) -> bool:
        for attempt in range(1, MAX_RETRY + 1):
            try:
                t_sample = time.time()
                pack = sample_plans[idx]
                state = run_self_refinement(
                    base_user_prompt=cfg.base_user_prompt,
                    condition_img=condition_img,
                    sample_plan=pack["sample_plan"],
                    start_prompt=pack["start_prompt"],
                    max_iters=cfg.max_iters,
                    patience=cfg.patience,
                    image_width=cfg.image_size[0],
                    image_height=cfg.image_size[1],
                )

                img_path = cfg.image_dir / f"{idx:03d}.png"
                prompt_path = cfg.prompt_dir / f"{idx:03d}.txt"
                img_path.write_bytes(state.best_image_bytes)
                prompt_path.write_text(state.best_prompt, encoding="utf-8")

                print(f"Saved: {img_path.name}  [TIMING] sample took {time.time()-t_sample:.1f}s")
                return True
            except Exception as e:
                print(f"Retry {attempt}/{MAX_RETRY} failed:", repr(e))
                time.sleep(0.8)
                if attempt == MAX_RETRY:
                    print("-> Giving up this sample and moving on.")
        return False

    if cfg.resume_on:
        target_indices = [
            idx for idx in range(cfg.n_samples)
            if not (cfg.image_dir / f"{idx:03d}.png").exists()
        ]
        print(f"[resume_on=true] {len(target_indices)}/{cfg.n_samples} sample(s) still missing: {target_indices}")
    else:
        target_indices = list(range(cfg.n_samples))
        print(f"[resume_on=false] generating all {cfg.n_samples} sample(s) from scratch")

    generated_count = 0
    for idx in target_indices:
        print(f"\n===== SAMPLE {idx + 1}/{cfg.n_samples} =====")
        if _generate_one(idx):
            generated_count += 1

    # One more pass over anything that is still missing.
    empty_idx = [idx for idx in range(cfg.n_samples) if not (cfg.image_dir / f"{idx:03d}.png").exists()]
    print("Missing image indices:", empty_idx)
    for idx in empty_idx:
        print(f"\n===== REGENERATE SAMPLE {idx}/{cfg.n_samples - 1} =====")
        if _generate_one(idx):
            generated_count += 1

    elapsed = time.time() - t_start
    print(f"[TIMING] total run: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[SUMMARY] images generated this run: {generated_count}")


if __name__ == "__main__":
    main()
