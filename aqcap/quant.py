"""
AQCAP Dynamic Bit-Width Quantization (DBQ) Engine.

Handles:
1. Model-agnostic layer surgery supporting arbitrary decoder architectures (Phi-3, Qwen2.5, Llama).
2. Per-layer precision mapping ("fp16" | "int8" | "nf4").
3. Universal FP16 preservation for embed_tokens and lm_head across all variants.
4. Two-phase CPU-to-CUDA surgery and quantization materialization.
5. VRAM tracking and architectural auditing.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

_bnb_error = None
try:
    import bitsandbytes as bnb
except Exception as e:
    _bnb_error = e
    bnb = None

_hqq_error = None
try:
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
except Exception as e:
    _hqq_error = e
    HQQLinear = None
    BaseQuantizeConfig = None


def _patch_hqq_linear() -> None:
    """
    HQQ's official HQQLinear class has a dummy .to() method (literally '# TODO: later; return self')
    that does not migrate quantized weights (W_q) and metadata (scale, zero) across devices.
    This monkey-patches HQQLinear.to so that standard PyTorch model.to(...) and
    layer.to(...) migrations work properly on both CUDA and CPU.
    """
    global HQQLinear
    if HQQLinear is None:
        return

    try:
        from hqq.core.quantize import Quantizer

        def _custom_to(self, *args, **kwargs):
            device = None
            if args:
                if isinstance(args[0], (torch.device, str, int)):
                    device = args[0]
                elif isinstance(args[0], torch.dtype):
                    pass
            if device is None and "device" in kwargs:
                device = kwargs["device"]

            if device is not None:
                dev_str = str(device)
                if "cuda" in dev_str:
                    if hasattr(self, "cuda"):
                        self.cuda(device)
                else:
                    if hasattr(self, "W_q") and self.W_q is not None:
                        if isinstance(self.W_q, nn.parameter.Parameter):
                            self.W_q.data, self.meta = Quantizer.to_inplace(self.W_q.data, self.meta, device=device)
                        else:
                            self.W_q, self.meta = Quantizer.to_inplace(self.W_q, self.meta, device=device)
                    if hasattr(self, "bias") and self.bias is not None:
                        if isinstance(self.bias, nn.Parameter):
                            self.bias.data = self.bias.data.to(device)
                        elif isinstance(self.bias, torch.Tensor):
                            self.bias = self.bias.to(device)
                    self.device = device
                    self.in_gpu = False
            return self

        HQQLinear.to = _custom_to
    except Exception:
        pass


if HQQLinear is not None:
    _patch_hqq_linear()


from transformers import AutoModelForCausalLM, AutoTokenizer


def get_decoder_layers(model: Any) -> Any:
    """
    Model-agnostic discovery of transformer decoder layers.
    Works seamlessly across Phi-3, Qwen2.5, Llama, and Mistral.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    elif hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        f"Unable to find decoder layers in model architecture: {type(model).__name__}"
    )


def _create_nf4_linear(source: nn.Linear, blocksize: int = 64, compute_dtype: Any = torch.float16) -> Any:
    """Creates a bitsandbytes NF4 4-bit Linear layer cloned from source."""
    global bnb, _bnb_error
    if bnb is None:
        try:
            import importlib
            importlib.invalidate_caches()
            import bitsandbytes as bnb_imported
            bnb = bnb_imported
            _bnb_error = None
        except Exception as retry_e:
            raise ImportError(
                f"bitsandbytes is required for NF4 quantization. "
                f"Initial import error: {_bnb_error}. Retry error: {retry_e}. "
                f"Please run: !pip install -q --upgrade bitsandbytes"
            )
    has_bias = source.bias is not None
    layer = bnb.nn.Linear4bit(
        input_features=source.in_features,
        output_features=source.out_features,
        bias=has_bias,
        compute_dtype=compute_dtype,
        compress_statistics=True,
        quant_type="nf4",
    )
    layer.weight = bnb.nn.Params4bit(
        source.weight.data.clone(),
        requires_grad=False,
        compress_statistics=True,
        quant_type="nf4",
        blocksize=blocksize,
    )
    if has_bias:
        layer.bias = torch.nn.Parameter(source.bias.data.clone(), requires_grad=False)
    return layer


def _create_int8_linear(source: nn.Linear, threshold: float = 6.0) -> Any:
    """Creates a bitsandbytes INT8 Linear layer cloned from source."""
    global bnb, _bnb_error
    if bnb is None:
        try:
            import importlib
            importlib.invalidate_caches()
            import bitsandbytes as bnb_imported
            bnb = bnb_imported
            _bnb_error = None
        except Exception as retry_e:
            raise ImportError(
                f"bitsandbytes is required for INT8 quantization. "
                f"Initial import error: {_bnb_error}. Retry error: {retry_e}. "
                f"Please run: !pip install -q --upgrade bitsandbytes"
            )
    has_bias = source.bias is not None
    layer = bnb.nn.Linear8bitLt(
        input_features=source.in_features,
        output_features=source.out_features,
        bias=has_bias,
        has_fp16_weights=False,
        threshold=threshold,
    )
    layer.weight = bnb.nn.Int8Params(
        source.weight.data.clone(),
        requires_grad=False,
        has_fp16_weights=False,
    )
    if has_bias:
        layer.bias = torch.nn.Parameter(source.bias.data.clone(), requires_grad=False)
    return layer


def _create_hqq_linear(
    source: nn.Linear,
    nbits: int = 4,
    group_size: int = 64,
    del_orig: bool = True,
    compute_dtype: Any = torch.float16,
    device: Any = None,
) -> Any:
    """Creates an HQQ low-bit (4, 3, or 2 bit) Linear layer cloned from source."""
    global HQQLinear, BaseQuantizeConfig, _hqq_error
    if HQQLinear is None:
        try:
            import importlib
            importlib.invalidate_caches()
            from hqq.core.quantize import HQQLinear as hqq_lin, BaseQuantizeConfig as bq_cfg
            HQQLinear = hqq_lin
            BaseQuantizeConfig = bq_cfg
            _patch_hqq_linear()
            _hqq_error = None
        except Exception as retry_e:
            raise ImportError(
                f"hqq is required for HQQ low-bit quantization, but could not be imported. "
                f"Initial error: {_hqq_error}; Retry error: {retry_e}. "
                f"Please run: pip install hqq"
            )

    target_dev = device or (source.weight.device if hasattr(source, "weight") else "cuda")
    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=1)
    layer = HQQLinear(
        source,
        quant_config=quant_config,
        del_orig=del_orig,
        compute_dtype=compute_dtype,
        device=target_dev,
    )
    return layer


def _quantize_submodule_linears(
    module: nn.Module,
    precision: str,
    nf4_blocksize: int,
    compute_dtype: Any,
    hqq_device: Any = None,
) -> int:
    """
    Recursively finds and replaces all nn.Linear children inside a decoder block.
    Model-agnostic: handles q_proj, qkv_proj, gate_up_proj, down_proj, etc.
    Returns the count of replaced projections.

    hqq_device: if not None, HQQ quantization is performed on this device
    (typically 'cuda'). This is critical for 3-bit/2-bit where the HQQ
    optimization solver is ~100x faster on GPU than CPU.
    """
    replaced_count = 0
    p = precision.lower()
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, nn.Linear):
            if p == "nf4":
                new_layer = _create_nf4_linear(child_module, blocksize=nf4_blocksize, compute_dtype=compute_dtype)
                setattr(module, child_name, new_layer)
            elif p == "int8":
                new_layer = _create_int8_linear(child_module)
                setattr(module, child_name, new_layer)
            elif p in ["hqq4", "hqq_4bit", "4bit"]:
                # For HQQ: move source to hqq_device first so optimization runs on GPU
                src = child_module
                if hqq_device is not None and str(src.weight.device) != str(hqq_device):
                    src = child_module.to(hqq_device)
                new_layer = _create_hqq_linear(src, nbits=4, compute_dtype=compute_dtype, device=hqq_device)
                setattr(module, child_name, new_layer)
            elif p in ["hqq3", "hqq_3bit", "3bit"]:
                src = child_module
                if hqq_device is not None and str(src.weight.device) != str(hqq_device):
                    src = child_module.to(hqq_device)
                new_layer = _create_hqq_linear(src, nbits=3, compute_dtype=compute_dtype, device=hqq_device)
                setattr(module, child_name, new_layer)
            elif p in ["hqq2", "hqq_2bit", "2bit"]:
                src = child_module
                if hqq_device is not None and str(src.weight.device) != str(hqq_device):
                    src = child_module.to(hqq_device)
                new_layer = _create_hqq_linear(src, nbits=2, compute_dtype=compute_dtype, device=hqq_device)
                setattr(module, child_name, new_layer)
            elif p == "fp16":
                child_module.weight.data = child_module.weight.data.to(torch.float16)
                if child_module.bias is not None:
                    child_module.bias.data = child_module.bias.data.to(torch.float16)
            replaced_count += 1
        else:
            # Recurse into attention or MLP blocks
            replaced_count += _quantize_submodule_linears(
                child_module, precision, nf4_blocksize, compute_dtype, hqq_device=hqq_device
            )
    return replaced_count


def load_model(
    model_id: str,
    precision_map: Dict[int, str],
    nf4_blocksize: int = 64,
    compute_dtype: Any = torch.float16,
    device: str = "cuda",
) -> Tuple[Any, Any]:
    """
    Loads base model to CPU, performs in-place layer-wise surgery according to
    precision_map, and migrates to target device.

    Args:
        model_id: HuggingFace model ID or local directory path.
        precision_map: Dict mapping layer index (0 to num_layers-1) to 'fp16' | 'int8' | 'nf4'.
        nf4_blocksize: Block size for NF4 quant (typically 64 or 32).
        compute_dtype: Compute dtype for dequantization ops (default: torch.float16).
        device: Target execution device ('cuda' or 'cpu').

    Returns:
        (model, tokenizer) tuple.
    """
    print(f"\n[QUANT] Loading tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[QUANT] Loading base model weights to CPU in FP16 (may take 1-2 min)...")
    t0 = time.time()
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id)
    config.use_cache = False
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        if "rope_type" in config.rope_scaling and "type" not in config.rope_scaling:
            config.rope_scaling["type"] = config.rope_scaling["rope_type"]
        elif "type" in config.rope_scaling and "rope_type" not in config.rope_scaling:
            config.rope_scaling["rope_type"] = config.rope_scaling["type"]

    # Support multi-GPU for FP16 references (device_map="auto" across 2 T4s) so 7B/8B FP16 fits
    is_pure_fp16 = all(p.lower() == "fp16" for p in precision_map.values())
    use_multi_gpu_fp16 = is_pure_fp16 and (
        device == "auto" or (device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.device_count() > 1)
    )

    if use_multi_gpu_fp16:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"[QUANT] Pure FP16 reference detected. Loading directly with device_map='auto' across {n_gpus} GPU(s)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=False,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        print(f"[QUANT] Pure FP16 model loaded across GPUs in {time.time() - t0:.1f}s.")
        return model, tokenizer

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=False,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
    print(f"[QUANT] Base model loaded in {time.time() - t0:.1f}s.")

    layers = get_decoder_layers(model)
    num_layers = len(layers)

    # Validate precision_map
    valid_precisions = ("fp16", "int8", "nf4", "hqq4", "hqq3", "hqq2", "hqq_4bit", "hqq_3bit", "hqq_2bit", "4bit", "3bit", "2bit")
    for i in range(num_layers):
        if i not in precision_map:
            raise ValueError(f"Missing precision specification for layer {i} (total {num_layers} layers).")
        p = precision_map[i].lower()
        if p not in valid_precisions:
            raise ValueError(f"Invalid precision '{p}' for layer {i}. Must be one of {valid_precisions}.")

    # ── Phase 1: CPU Surgery (HQQ layers quantized directly on CUDA) ──────
    # HQQ 3-bit/2-bit optimization runs ~100x faster on GPU than CPU.
    # Detect if any layers need HQQ and pre-determine the surgery device.
    hqq_precisions = {"hqq4", "hqq3", "hqq2", "hqq_4bit", "hqq_3bit", "hqq_2bit", "4bit", "3bit", "2bit"}
    has_hqq_layers = any(precision_map[i].lower() in hqq_precisions for i in range(num_layers))
    target_device_for_surgery = "cuda" if (device == "auto" or (isinstance(device, str) and device.startswith("cuda"))) and torch.cuda.is_available() else None
    hqq_surgery_device = target_device_for_surgery if has_hqq_layers else None
    if hqq_surgery_device:
        print(f"[QUANT] HQQ layers detected — quantization will run directly on {hqq_surgery_device} (avoids CPU hang for 3-bit/2-bit).")

    print(f"\n[QUANT] Phase 1: Applying layer-by-layer surgery across {num_layers} decoder layers...")
    tier_counts = defaultdict(int)
    total_projections = 0

    for i in range(num_layers):
        p = precision_map[i].lower()
        block = layers[i]
        n_proj = _quantize_submodule_linears(block, p, nf4_blocksize, compute_dtype, hqq_device=hqq_surgery_device)
        tier_counts[p] += 1
        total_projections += n_proj

    tier_summary = " | ".join([f"{k.upper()}: {v} layers" for k, v in sorted(tier_counts.items())])
    print(f"[QUANT] Projections swapped: {total_projections} total.")
    print(f"  Summary: {tier_summary}")

    # ── Phase 1b: Preserve Embeddings and LM Head in FP16 ──────────────────
    print(f"[QUANT] Phase 1b: Enforcing FP16 on embed_tokens and lm_head...")
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.to(torch.float16)
    elif hasattr(model, "embed_tokens"):
        model.embed_tokens = model.embed_tokens.to(torch.float16)

    if hasattr(model, "lm_head"):
        model.lm_head = model.lm_head.to(torch.float16)

    # ── Phase 2: CUDA Migration & Materialization ──────────────────────────
    target_device = "cuda" if (device == "auto" or device.startswith("cuda")) and torch.cuda.is_available() else device
    if target_device.startswith("cuda") and torch.cuda.is_available():
        print(f"[QUANT] Phase 2: Migrating to CUDA ({target_device}) (materializing quantization compression)...")
        gc.collect()
        torch.cuda.empty_cache()
        t_cuda = time.time()

        # Step 2a: Move non-quantized parameters (embeddings, norms, lm_head, fp16 layers) to CUDA
        q_param_names = set()
        if bnb is not None:
            for mn, m in model.named_modules():
                if isinstance(m, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                    for pn, _ in m.named_parameters(recurse=False):
                        q_param_names.add(f"{mn}.{pn}" if mn else pn)
        if HQQLinear is not None:
            for mn, m in model.named_modules():
                if isinstance(m, HQQLinear):
                    for pn, _ in m.named_parameters(recurse=False):
                        q_param_names.add(f"{mn}.{pn}" if mn else pn)

        for n, p in model.named_parameters():
            if n not in q_param_names and not p.is_cuda:
                p.data = p.data.to(target_device)

        for n, b in model.named_buffers():
            if not b.is_cuda:
                b.data = b.data.to(target_device)

        # Step 2b: Move NF4 projections to CUDA
        if bnb is not None:
            for _, m in model.named_modules():
                if isinstance(m, bnb.nn.Linear4bit):
                    m.to(target_device)
                    if not m.weight.is_cuda:
                        m.weight = m.weight.cuda(target_device)
            torch.cuda.synchronize()

        # Step 2c: Move INT8 projections to CUDA
        if bnb is not None:
            for _, m in model.named_modules():
                if isinstance(m, bnb.nn.Linear8bitLt):
                    m.to(target_device)
                    if not m.weight.is_cuda:
                        m.weight = m.weight.cuda(target_device)
            torch.cuda.synchronize()

        # Step 2d: Move HQQ projections to CUDA (if not already on CUDA from Phase 1)
        if HQQLinear is not None:
            cuda_dev = target_device if target_device.startswith("cuda") else "cuda"
            migrated = 0
            for _, m in model.named_modules():
                if isinstance(m, HQQLinear):
                    # If already on CUDA (quantized directly on CUDA in Phase 1), skip
                    already_on_cuda = hasattr(m, "W_q") and m.W_q is not None and m.W_q.is_cuda
                    if not already_on_cuda:
                        if hasattr(m, "cuda"):
                            m.cuda(cuda_dev)
                        else:
                            m.to(target_device)
                        migrated += 1
            if migrated > 0:
                print(f"[QUANT] Migrated {migrated} HQQ layers to {cuda_dev}.")
            torch.cuda.synchronize()

        gc.collect()
        torch.cuda.empty_cache()
        print(f"[QUANT] CUDA migration completed in {time.time() - t_cuda:.1f}s.")

    model.config.use_cache = True
    model.eval()

    # Print memory report
    report = memory_report(model)
    if report["weights_bytes"] > 0:
        print(f"[QUANT] Post-surgery VRAM: {report['weights_mb']:,.1f} MB ({report['weights_gb']:.3f} GB)")

    return model, tokenizer


def memory_report(model: Any) -> Dict[str, Any]:
    """
    Returns exact memory allocation report across all active CUDA devices.
    Tracks weights allocated and peak allocated bytes across evaluation.
    """
    w_bytes = 0
    p_bytes = 0
    if torch is not None and torch.cuda.is_available():
        n_dev = torch.cuda.device_count()
        for d in range(n_dev):
            try:
                w_bytes += torch.cuda.memory_allocated(d)
                p_bytes += torch.cuda.max_memory_allocated(d)
            except Exception:
                pass
    
    if w_bytes == 0:
        # Fallback / CPU approximation via parameter tensors
        w_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        p_bytes = w_bytes

    return {
        "weights_bytes": w_bytes,
        "weights_mb": w_bytes / (1024**2),
        "weights_gb": w_bytes / (1024**3),
        "peak_bytes": p_bytes,
        "peak_mb": p_bytes / (1024**2),
        "peak_gb": p_bytes / (1024**3),
    }


def reset_peak_memory() -> None:
    """Resets CUDA peak memory tracking across all available GPUs for clean benchmark intervals."""
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        try:
            for d in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(d)
                except Exception:
                    try:
                        torch.cuda.reset_peak_memory_stats(torch.device(f"cuda:{d}"))
                    except Exception:
                        pass
        except Exception:
            pass


def audit_model(model: Any, precision_map: Dict[int, str]) -> Dict[str, Any]:
    """
    Verifies that every projection in every decoder layer matches the intended precision.
    Returns audit statistics and mismatch count.
    """
    layers = get_decoder_layers(model)
    audit_results = {
        "layers_audited": len(layers),
        "total_projections": 0,
        "counts": defaultdict(int),
        "mismatches": 0,
    }

    for i, block in enumerate(layers):
        expected_p = precision_map[i].lower()
        for _, module in block.named_modules():
            if bnb is not None and isinstance(module, bnb.nn.Linear4bit):
                audit_results["counts"]["nf4"] += 1
                audit_results["total_projections"] += 1
                if expected_p != "nf4":
                    audit_results["mismatches"] += 1
            elif bnb is not None and isinstance(module, bnb.nn.Linear8bitLt):
                audit_results["counts"]["int8"] += 1
                audit_results["total_projections"] += 1
                if expected_p != "int8":
                    audit_results["mismatches"] += 1
            elif HQQLinear is not None and isinstance(module, HQQLinear):
                nb = getattr(getattr(module, "quant_config", None), "nbits", 4)
                hqq_tag = f"hqq_{nb}bit"
                audit_results["counts"][hqq_tag] += 1
                audit_results["total_projections"] += 1
                if not (expected_p.startswith("hqq") or expected_p.startswith(str(nb)) or expected_p == f"{nb}bit"):
                    audit_results["mismatches"] += 1
            elif isinstance(module, nn.Linear):
                audit_results["counts"]["fp16"] += 1
                audit_results["total_projections"] += 1
                if expected_p != "fp16":
                    audit_results["mismatches"] += 1

    return audit_results
