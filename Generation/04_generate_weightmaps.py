"""Stage 4: weight maps for Adaptive Confidence Thresholding (ACT).

Pure numpy/scipy processing of the stage-3 masks: connected components whose
inscribed radius D_max is < 10% of the largest one are suppressed, then a
boundary-guided signed distance is normalized per component to S in [-1, 1]
(S > 0 inside the target, S < 0 outside) and stored as uint8 W = (S+1)/2*255.
ACT maps W back to [-1, 1] when loading.

Usage:
    python 04_generate_weightmaps.py --config configs/impact.yaml
"""
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from common.config import build_arg_parser, load_config
from common.run_logging import setup_logging

STRUCTURE = np.ones((3, 3), dtype=np.uint8)  # 8-connectivity
MAJOR_THRESHOLD_RATIO = 0.10  # suppress components with D_max < 10% of the global max


def load_binary_mask(mask_path) -> np.ndarray:
    img = Image.open(mask_path).convert("L")
    arr = np.array(img)
    return arr > 0


def get_component_dmax(mask: np.ndarray):
    labels, num_labels = label(mask, structure=STRUCTURE)
    dmax_dict = {}
    for comp_id in range(1, num_labels + 1):
        comp_mask = labels == comp_id
        inside_dist = distance_transform_edt(comp_mask)
        dmax_dict[comp_id] = float(inside_dist.max())
    return labels, num_labels, dmax_dict


def filter_major_components(labels: np.ndarray, dmax_dict: dict, ratio: float = 0.10):
    if len(dmax_dict) == 0:
        return np.zeros_like(labels, dtype=np.int32), {}, 0.0

    global_max_d = max(dmax_dict.values())
    threshold = ratio * global_max_d
    keep_ids = [cid for cid, dmax in dmax_dict.items() if dmax >= threshold]

    new_labels = np.zeros_like(labels, dtype=np.int32)
    new_dmax_dict = {}
    new_id = 1
    for old_id in keep_ids:
        new_labels[labels == old_id] = new_id
        new_dmax_dict[new_id] = dmax_dict[old_id]
        new_id += 1

    return new_labels, new_dmax_dict, global_max_d


def build_boundary_label_map(labels: np.ndarray) -> np.ndarray:
    boundary_labels = np.zeros_like(labels, dtype=np.int32)
    comp_ids = np.unique(labels)
    comp_ids = comp_ids[comp_ids > 0]
    for comp_id in comp_ids:
        comp_mask = labels == comp_id
        eroded = binary_erosion(comp_mask, structure=STRUCTURE, border_value=0)
        boundary = comp_mask & (~eroded)
        boundary_labels[boundary] = comp_id
    return boundary_labels


def get_nearest_boundary_component(boundary_labels: np.ndarray) -> np.ndarray:
    if np.all(boundary_labels == 0):
        return np.zeros_like(boundary_labels, dtype=np.int32)

    non_boundary = boundary_labels == 0
    _, indices = distance_transform_edt(non_boundary, return_distances=True, return_indices=True)
    return boundary_labels[tuple(indices)].astype(np.int32)


def compute_weight_map(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.full(mask.shape, 127, dtype=np.uint8)

    labels_raw, num_raw, dmax_raw = get_component_dmax(mask)
    if num_raw == 0:
        return np.full(mask.shape, 127, dtype=np.uint8)

    labels, dmax_dict, global_max_d = filter_major_components(labels_raw, dmax_raw, ratio=MAJOR_THRESHOLD_RATIO)
    if len(dmax_dict) == 0:
        return np.full(mask.shape, 127, dtype=np.uint8)

    major_mask = labels > 0

    boundary_labels = build_boundary_label_map(labels)
    nearest_comp_map = get_nearest_boundary_component(boundary_labels)

    inside_dist = distance_transform_edt(major_mask)
    outside_dist = distance_transform_edt(~major_mask)

    signed_d = inside_dist.astype(np.float32)
    signed_d[~major_mask] = -outside_dist[~major_mask].astype(np.float32)

    max_label = int(labels.max())
    dmax_lut = np.zeros(max_label + 1, dtype=np.float32)
    for comp_id, dmax in dmax_dict.items():
        dmax_lut[comp_id] = dmax

    D_ref = dmax_lut[nearest_comp_map]
    inside_labels = labels[major_mask]
    D_ref[major_mask] = dmax_lut[inside_labels]

    S = np.full(mask.shape, -1.0, dtype=np.float32)
    inside = major_mask
    outside = ~major_mask

    valid_inside = inside & (D_ref > 0)
    S[valid_inside] = signed_d[valid_inside] / D_ref[valid_inside]

    valid_outside = outside & (D_ref > 0)
    d_out = signed_d[valid_outside]
    D_out = D_ref[valid_outside]
    S[valid_outside] = d_out / (D_out - d_out)

    S = np.clip(S, -1.0, 1.0)

    W = ((S + 1.0) / 2.0) * 255.0
    return np.clip(W, 0, 255).astype(np.uint8)


def main():
    setup_logging("04_generate_weightmaps")

    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    cfg.ensure_dirs(weight_maps=True)
    print(f"Reading masks / writing weight maps under: {cfg.output_dir}")

    mask_paths = sorted(cfg.mask_dir.glob("*.png"))
    if not mask_paths:
        print(f"[WARN] No PNG files found in: {cfg.mask_dir}")
    print(f"[INFO] Found {len(mask_paths)} mask files.")

    t_start = time.time()

    if cfg.resume_on:
        before = len(mask_paths)
        mask_paths = [p for p in mask_paths if not (cfg.weight_map_dir / p.name).exists()]
        print(f"[resume_on=true] {len(mask_paths)}/{before} weight map(s) still missing")

    generated_count = 0
    for mask_path in mask_paths:
        weight_map = compute_weight_map(load_binary_mask(mask_path))
        out_path = cfg.weight_map_dir / mask_path.name
        Image.fromarray(weight_map).save(out_path)
        print(f"[SAVED] {out_path}")
        generated_count += 1

    elapsed = time.time() - t_start
    print(f"[TIMING] total run: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[SUMMARY] weight maps generated this run: {generated_count}")


if __name__ == "__main__":
    main()
