"""
Generates static experiment configuration JSON files for all target models.
Budget: exactly half of decoder layers at INT8 and the rest at NF4 for all mixed configs.

Models:
- microsoft/Phi-3-mini-4k-instruct: 32 layers (16 INT8 / 16 NF4)
- Qwen/Qwen2.5-Coder-7B-Instruct: 28 layers (14 INT8 / 14 NF4)
- Qwen/Qwen2.5-Coder-3B-Instruct: 36 layers (18 INT8 / 18 NF4)
"""

import os
import json
import random

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
os.makedirs(CONFIGS_DIR, exist_ok=True)


def generate_static_suite_for_model(
    model_tag: str,
    num_layers: int,
    prefix: str = "",
    description_label: str = "",
):
    """
    Generates:
    - {prefix}fp16
    - {prefix}int8
    - {prefix}nf4
    - {prefix}first{half}
    - {prefix}last{half}
    - {prefix}rand_s0, {prefix}rand_s1, {prefix}rand_s2
    """
    half = num_layers // 2
    configs = {}

    # 1. Uniform
    configs[f"{prefix}fp16"] = {
        "name": f"{prefix}fp16",
        "description": f"{description_label} Upper reference: all {num_layers} layers FP16",
        "precision_map": {str(i): "fp16" for i in range(num_layers)},
        "nf4_blocksize": 64,
    }
    configs[f"{prefix}int8"] = {
        "name": f"{prefix}int8",
        "description": f"{description_label} Uniform baseline: all {num_layers} layers INT8",
        "precision_map": {str(i): "int8" for i in range(num_layers)},
        "nf4_blocksize": 64,
    }
    configs[f"{prefix}nf4"] = {
        "name": f"{prefix}nf4",
        "description": f"{description_label} Uniform baseline: all {num_layers} layers NF4 bs=64",
        "precision_map": {str(i): "nf4" for i in range(num_layers)},
        "nf4_blocksize": 64,
    }

    # 2. Contiguous Controls
    first_map = {str(i): "int8" if i < half else "nf4" for i in range(num_layers)}
    configs[f"{prefix}first{half}"] = {
        "name": f"{prefix}first{half}",
        "description": f"{description_label} Contiguous control: INT8 on layers 0-{half-1}, NF4 on {half}-{num_layers-1}",
        "precision_map": first_map,
        "nf4_blocksize": 64,
    }

    last_map = {str(i): "nf4" if i < half else "int8" for i in range(num_layers)}
    configs[f"{prefix}last{half}"] = {
        "name": f"{prefix}last{half}",
        "description": f"{description_label} Contiguous control: NF4 on layers 0-{half-1}, INT8 on {half}-{num_layers-1}",
        "precision_map": last_map,
        "nf4_blocksize": 64,
    }

    # 3. Random Controls (seeds 0, 1, 2)
    for seed in [0, 1, 2]:
        rng = random.Random(seed)
        int8_chosen = set(rng.sample(list(range(num_layers)), half))
        rand_map = {str(i): "int8" if i in int8_chosen else "nf4" for i in range(num_layers)}
        configs[f"{prefix}rand_s{seed}"] = {
            "name": f"{prefix}rand_s{seed}",
            "description": f"{description_label} Random control (seed {seed}): {half} random layers INT8, {half} NF4",
            "precision_map": rand_map,
            "nf4_blocksize": 64,
        }

    # Write configs
    for name, cfg in configs.items():
        filepath = os.path.join(CONFIGS_DIR, f"{name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    print(f"[CONFIGS] Generated {len(configs)} static configs for {description_label} (prefix='{prefix}')")
    return configs


def generate_all_configs():
    # 1. Phi-3-mini-4k-instruct (32 layers, half=16) - default prefix "" for canonical ROADMAP names
    generate_static_suite_for_model(
        model_tag="phi3",
        num_layers=32,
        prefix="",
        description_label="Phi-3-mini",
    )
    # Also with phi3_ prefix
    generate_static_suite_for_model(
        model_tag="phi3",
        num_layers=32,
        prefix="phi3_",
        description_label="Phi-3-mini",
    )

    # 2. Qwen2.5-Coder-7B-Instruct (28 layers, half=14)
    generate_static_suite_for_model(
        model_tag="qwen7b",
        num_layers=28,
        prefix="qwen7b_",
        description_label="Qwen2.5-Coder-7B",
    )

    # 3. Qwen2.5-Coder-3B-Instruct (36 layers, half=18)
    generate_static_suite_for_model(
        model_tag="qwen3b",
        num_layers=36,
        prefix="qwen3b_",
        description_label="Qwen2.5-Coder-3B",
    )
    # Legacy qwen_ prefix alias
    generate_static_suite_for_model(
        model_tag="qwen3b",
        num_layers=36,
        prefix="qwen_",
        description_label="Qwen2.5-Coder-3B",
    )


if __name__ == "__main__":
    generate_all_configs()
