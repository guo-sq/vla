#!/usr/bin/env python3
"""将 T5Gemma 2 的 Encoder 部分转换为 OpenPI Pi0 格式。

转换内容：
1. Vision Encoder (SigLIP 400M)
2. Vision Projection (SigLIP head: 1152 -> 640)
3. Encoder LLM (Gemma 3 270M) -> Pi0 主 LLM

注意：不转换 Decoder LLM，因为 OpenPI 的架构不需要它。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors library not found. Install with:")
    print("  pip install safetensors")
    sys.exit(1)


def load_safetensors_model(model_path: str) -> dict:
    """从 safetensors 文件加载模型参数。"""
    tensors = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 转换为 float32 numpy 数组
            tensors[key] = tensor.float().numpy()
    return tensors


def convert_t5gemma2_encoder_to_openpi(
    model_path: str = "/mnt/model/t5gemma2_270m/models--google--t5gemma-2-270m-270m/snapshots/7c38f16641f455ef0685b18431faf1b17722d5a1/model.safetensors",
    output_path: str = "/mnt/model/t5gemma2_encoder_openpi.npz",
):
    """转换 T5Gemma 2 Encoder 部分到 OpenPI Pi0 格式。"""
    print(f"Loading model from: {model_path}")
    print("-" * 60)

    param_dict = load_safetensors_model(model_path)
    print(f"Loaded {len(param_dict)} parameters")

    converted_params = {}
    stats = {"vision": 0, "vision_proj": 0, "llm": 0}

    # ========================================
    # 1. Vision Encoder (SigLIP 400M)
    # ========================================
    print("\n=== Converting Vision Encoder (SigLIP 400M) ===")

    # Patch embedding: PyTorch (out, in, h, w) -> Pi0 (h, w, in, out)
    if "model.encoder.vision_tower.vision_model.embeddings.patch_embedding.weight" in param_dict:
        w = param_dict["model.encoder.vision_tower.vision_model.embeddings.patch_embedding.weight"]
        # PyTorch: (out, in, h, w) -> Pi0: (h, w, in, out)
        converted_params["params/img/embedding/kernel"] = w.transpose(2, 3, 1, 0)
        print(f"  patch_embedding/kernel: {w.shape} -> {converted_params['params/img/embedding/kernel'].shape}")
        stats["vision"] += 1

    if "model.encoder.vision_tower.vision_model.embeddings.patch_embedding.bias" in param_dict:
        converted_params["params/img/embedding/bias"] = param_dict[
            "model.encoder.vision_tower.vision_model.embeddings.patch_embedding.bias"]
        stats["vision"] += 1

    # Position embedding
    if "model.encoder.vision_tower.vision_model.embeddings.position_embedding.weight" in param_dict:
        w = param_dict["model.encoder.vision_tower.vision_model.embeddings.position_embedding.weight"]
        converted_params["params/img/pos_embedding"] = w[np.newaxis, :]
        stats["vision"] += 1

    # 初始化 27 层的数组
    num_vision_layers = 27
    vision_num_heads = 16
    vision_head_dim = 72

    # Pre-allocate vision layer parameters
    vision_ln0_scale = np.zeros((num_vision_layers, 1152), dtype=np.float32)
    vision_ln0_bias = np.zeros((num_vision_layers, 1152), dtype=np.float32)
    vision_ln1_scale = np.zeros((num_vision_layers, 1152), dtype=np.float32)
    vision_ln1_bias = np.zeros((num_vision_layers, 1152), dtype=np.float32)

    # Q/K/V kernels: (num_layers, features, num_heads, head_dim) = (27, 1152, 16, 72)
    # Flax DenseGeneral with features=(num_heads, head_dim), axis=-1
    vision_q_kernel = np.zeros((num_vision_layers, 1152, vision_num_heads, vision_head_dim), dtype=np.float32)
    vision_q_bias = np.zeros((num_vision_layers, vision_num_heads, vision_head_dim), dtype=np.float32)
    vision_k_kernel = np.zeros((num_vision_layers, 1152, vision_num_heads, vision_head_dim), dtype=np.float32)
    vision_k_bias = np.zeros((num_vision_layers, vision_num_heads, vision_head_dim), dtype=np.float32)
    vision_v_kernel = np.zeros((num_vision_layers, 1152, vision_num_heads, vision_head_dim), dtype=np.float32)
    vision_v_bias = np.zeros((num_vision_layers, vision_num_heads, vision_head_dim), dtype=np.float32)
    # out kernel: (num_layers, num_heads, head_dim, features) = (27, 16, 72, 1152)
    # Flax MultiHeadDotProductAttention in scan mode expects this format
    vision_out_kernel = np.zeros((num_vision_layers, vision_num_heads, vision_head_dim, 1152), dtype=np.float32)
    vision_out_bias = np.zeros((num_vision_layers, 1152), dtype=np.float32)

    vision_mlp0_kernel = np.zeros((num_vision_layers, 1152, 4304), dtype=np.float32)
    vision_mlp0_bias = np.zeros((num_vision_layers, 4304), dtype=np.float32)
    vision_mlp1_kernel = np.zeros((num_vision_layers, 4304, 1152), dtype=np.float32)
    vision_mlp1_bias = np.zeros((num_vision_layers, 1152), dtype=np.float32)

    # 转换每个 vision encoder 层
    for i in range(num_vision_layers):
        prefix = f"model.encoder.vision_tower.vision_model.encoder.layers.{i}"

        # Layer norm 0
        key = f"{prefix}.layer_norm1.weight"
        if key in param_dict:
            vision_ln0_scale[i] = param_dict[key]
        key = f"{prefix}.layer_norm1.bias"
        if key in param_dict:
            vision_ln0_bias[i] = param_dict[key]

        # Layer norm 1
        key = f"{prefix}.layer_norm2.weight"
        if key in param_dict:
            vision_ln1_scale[i] = param_dict[key]
        key = f"{prefix}.layer_norm2.bias"
        if key in param_dict:
            vision_ln1_bias[i] = param_dict[key]

        # Self attention Q: PyTorch (1152, 1152) -> Flax scan (1152, 16, 72)
        # Flax DenseGeneral with features=(num_heads, head_dim), axis=-1
        # expects kernel shape (features, num_heads, head_dim)
        key = f"{prefix}.self_attn.q_proj.weight"
        if key in param_dict:
            w = param_dict[key]  # (1152, 1152) PyTorch Linear weight
            # Reshape to (features, num_heads, head_dim) = (1152, 16, 72)
            w = w.reshape(1152, vision_num_heads, vision_head_dim)
            vision_q_kernel[i] = w
        key = f"{prefix}.self_attn.q_proj.bias"
        if key in param_dict:
            w = param_dict[key].reshape(vision_num_heads, vision_head_dim)
            vision_q_bias[i] = w

        # Self attention K: PyTorch (1152, 1152) -> Flax scan (1152, 16, 72)
        key = f"{prefix}.self_attn.k_proj.weight"
        if key in param_dict:
            w = param_dict[key]  # (1152, 1152)
            w = w.reshape(1152, vision_num_heads, vision_head_dim)
            vision_k_kernel[i] = w
        key = f"{prefix}.self_attn.k_proj.bias"
        if key in param_dict:
            w = param_dict[key].reshape(vision_num_heads, vision_head_dim)
            vision_k_bias[i] = w

        # Self attention V: PyTorch (1152, 1152) -> Flax scan (1152, 16, 72)
        key = f"{prefix}.self_attn.v_proj.weight"
        if key in param_dict:
            w = param_dict[key]  # (1152, 1152)
            w = w.reshape(1152, vision_num_heads, vision_head_dim)
            vision_v_kernel[i] = w
        key = f"{prefix}.self_attn.v_proj.bias"
        if key in param_dict:
            w = param_dict[key].reshape(vision_num_heads, vision_head_dim)
            vision_v_bias[i] = w

        # Self attention out: PyTorch (1152, 1152) -> Flax scan (16, 72, 1152)
        # Flax MultiHeadDotProductAttention in scan mode expects (num_heads, head_dim, features)
        key = f"{prefix}.self_attn.out_proj.weight"
        if key in param_dict:
            w = param_dict[key]  # (1152, 1152) PyTorch Linear weight
            # Reshape to (num_heads, head_dim, features) = (16, 72, 1152)
            w = w.reshape(vision_num_heads, vision_head_dim, 1152)
            vision_out_kernel[i] = w
        key = f"{prefix}.self_attn.out_proj.bias"
        if key in param_dict:
            vision_out_bias[i] = param_dict[key]

        # MLP
        key = f"{prefix}.mlp.fc1.weight"
        if key in param_dict:
            vision_mlp0_kernel[i] = param_dict[key].T
        key = f"{prefix}.mlp.fc1.bias"
        if key in param_dict:
            vision_mlp0_bias[i] = param_dict[key]

        key = f"{prefix}.mlp.fc2.weight"
        if key in param_dict:
            vision_mlp1_kernel[i] = param_dict[key].T
        key = f"{prefix}.mlp.fc2.bias"
        if key in param_dict:
            vision_mlp1_bias[i] = param_dict[key]

    # 保存 vision encoder 参数
    converted_params["params/img/Transformer/encoderblock/LayerNorm_0/scale"] = vision_ln0_scale
    converted_params["params/img/Transformer/encoderblock/LayerNorm_0/bias"] = vision_ln0_bias
    converted_params["params/img/Transformer/encoderblock/LayerNorm_1/scale"] = vision_ln1_scale
    converted_params["params/img/Transformer/encoderblock/LayerNorm_1/bias"] = vision_ln1_bias

    converted_params["params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel"] = vision_q_kernel
    converted_params["params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias"] = vision_q_bias
    converted_params["params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel"] = vision_k_kernel
    converted_params["params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias"] = vision_k_bias
    converted_params["params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel"] = vision_v_kernel
    converted_params["params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias"] = vision_v_bias
    converted_params["params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel"] = vision_out_kernel
    converted_params["params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias"] = vision_out_bias

    converted_params["params/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel"] = vision_mlp0_kernel
    converted_params["params/img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias"] = vision_mlp0_bias
    converted_params["params/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel"] = vision_mlp1_kernel
    converted_params["params/img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias"] = vision_mlp1_bias

    # Encoder norm
    if "model.encoder.vision_tower.vision_model.post_layernorm.weight" in param_dict:
        converted_params["params/img/Transformer/encoder_norm/scale"] = param_dict[
            "model.encoder.vision_tower.vision_model.post_layernorm.weight"]
    if "model.encoder.vision_tower.vision_model.post_layernorm.bias" in param_dict:
        converted_params["params/img/Transformer/encoder_norm/bias"] = param_dict[
            "model.encoder.vision_tower.vision_model.post_layernorm.bias"]

    print(f"  Vision encoder layers: {num_vision_layers}")
    stats["vision"] += len([k for k in converted_params.keys() if k.startswith("params/img/")])

    # ========================================
    # 2. Vision Projection (SigLIP head: 1152 -> 640)
    # ========================================
    print("\n=== Converting Vision Projection (SigLIP head) ===")

    if "model.encoder.multi_modal_projector.mm_input_projection_weight" in param_dict:
        w = param_dict["model.encoder.multi_modal_projector.mm_input_projection_weight"]
        # T5Gemma 2 weight is already (1152, 640) format, same as JAX Dense kernel
        converted_params["params/img/head/kernel"] = w
        print(f"  head/kernel: {w.shape}")
        stats["vision_proj"] += 1

    if "model.encoder.multi_modal_projector.mm_input_projection_bias" in param_dict:
        converted_params["params/img/head/bias"] = param_dict[
            "model.encoder.multi_modal_projector.mm_input_projection_bias"]
        print(f"  head/bias: {converted_params['params/img/head/bias'].shape}")
        stats["vision_proj"] += 1

    # ========================================
    # 3. Encoder LLM (Gemma 3 270M) -> Pi0 主 LLM
    # ========================================
    print("\n=== Converting Encoder LLM (Gemma 3 270M) -> Pi0 Main LLM ===")

    num_layers = 18
    hidden_size = 640
    num_heads = 4
    head_dim = 256
    num_kv_heads = 1
    mlp_dim = 2048

    # Pre-allocate arrays (Gemma 3 架构)
    llm_q = np.zeros((num_layers, num_heads, hidden_size, head_dim), dtype=np.float32)
    llm_kv = np.zeros((num_layers, 2, num_kv_heads, hidden_size, head_dim), dtype=np.float32)
    llm_o = np.zeros((num_layers, num_heads, head_dim, hidden_size), dtype=np.float32)
    # MLP gating_einsum: (num_layers, 2, hidden_size, mlp_dim) = (18, 2, 640, 2048)
    # Format must match OpenPI's (2, features, hidden_dim) = (2, width, mlp_dim)
    llm_mlp_gate = np.zeros((num_layers, 2, hidden_size, mlp_dim), dtype=np.float32)
    # MLP linear (down_proj): (num_layers, mlp_dim, hidden_size) = (18, 2048, 640)
    # Flax Dense kernel format: (in_features, out_features)
    llm_mlp_linear = np.zeros((num_layers, mlp_dim, hidden_size), dtype=np.float32)
    llm_pre_attn_norm = np.zeros((num_layers, hidden_size), dtype=np.float32)
    llm_pre_ffw_norm = np.zeros((num_layers, hidden_size), dtype=np.float32)

    # Gemma 3 特有参数
    llm_q_norm = np.zeros((num_layers, head_dim), dtype=np.float32)
    llm_k_norm = np.zeros((num_layers, head_dim), dtype=np.float32)
    llm_post_attn_norm = np.zeros((num_layers, hidden_size), dtype=np.float32)
    llm_post_ffw_norm = np.zeros((num_layers, hidden_size), dtype=np.float32)

    # 转换 encoder 层
    for i in range(num_layers):
        prefix = f"model.encoder.layers.{i}"

        # Q projection: (1024, 640) -> (4, 640, 256)
        # 1024 = num_heads * head_dim = 4 * 256
        key = f"{prefix}.self_attn.q_proj.weight"
        if key in param_dict:
            w = param_dict[key]  # (1024, 640)
            w = w.reshape(num_heads, head_dim, hidden_size)  # (4, 256, 640)
            w = w.transpose(0, 2, 1)  # (4, 640, 256)
            llm_q[i] = w

        # K projection: (256, 640) -> (1, 640, 256)
        # 256 = num_kv_heads * head_dim = 1 * 256
        key = f"{prefix}.self_attn.k_proj.weight"
        if key in param_dict:
            w = param_dict[key]  # (256, 640)
            w = w.reshape(num_kv_heads, head_dim, hidden_size)  # (1, 256, 640)
            w = w.transpose(0, 2, 1)  # (1, 640, 256)
            llm_kv[i, 0] = w

        # V projection: (256, 640) -> (1, 640, 256)
        key = f"{prefix}.self_attn.v_proj.weight"
        if key in param_dict:
            w = param_dict[key]  # (256, 640)
            w = w.reshape(num_kv_heads, head_dim, hidden_size)  # (1, 256, 640)
            w = w.transpose(0, 2, 1)  # (1, 640, 256)
            llm_kv[i, 1] = w

        # O projection: (640, 1024) -> (4, 256, 640)
        # 1024 = num_heads * head_dim = 4 * 256
        key = f"{prefix}.self_attn.o_proj.weight"
        if key in param_dict:
            w = param_dict[key]  # (640, 1024)
            w = w.reshape(hidden_size, num_heads, head_dim)  # (640, 4, 256)
            w = w.transpose(1, 2, 0)  # (4, 256, 640)
            llm_o[i] = w

        # Gemma 3: q_norm, k_norm
        key = f"{prefix}.self_attn.q_norm.weight"
        if key in param_dict:
            llm_q_norm[i] = param_dict[key]
        key = f"{prefix}.self_attn.k_norm.weight"
        if key in param_dict:
            llm_k_norm[i] = param_dict[key]

        # Pre-attention norm
        key = f"{prefix}.pre_self_attn_layernorm.weight"
        if key in param_dict:
            llm_pre_attn_norm[i] = param_dict[key]

        # Gemma 3: post_attention_norm
        key = f"{prefix}.post_self_attn_layernorm.weight"
        if key in param_dict:
            llm_post_attn_norm[i] = param_dict[key]

        # MLP: gate_proj and up_proj need transpose from (mlp_dim, hidden_size) to (hidden_size, mlp_dim)
        # PyTorch Linear: (out_features, in_features) = (mlp_dim, hidden_size)
        # OpenPI format: (hidden_size, mlp_dim)
        key = f"{prefix}.mlp.gate_proj.weight"
        if key in param_dict:
            w = param_dict[key].T  # Transpose (2048, 640) -> (640, 2048)
            llm_mlp_gate[i, 0] = w

        key = f"{prefix}.mlp.up_proj.weight"
        if key in param_dict:
            w = param_dict[key].T  # Transpose (2048, 640) -> (640, 2048)
            llm_mlp_gate[i, 1] = w

        key = f"{prefix}.mlp.down_proj.weight"
        if key in param_dict:
            # PyTorch Linear: (out_features, in_features) = (640, 2048)
            # OpenPI format: (in_features, out_features) = (2048, 640)
            # Need to transpose!
            llm_mlp_linear[i] = param_dict[key].T  # Transpose (640, 2048) -> (2048, 640)

        # Pre-FFN norm
        key = f"{prefix}.pre_feedforward_layernorm.weight"
        if key in param_dict:
            llm_pre_ffw_norm[i] = param_dict[key]

        # Gemma 3: post_ffw_norm
        key = f"{prefix}.post_feedforward_layernorm.weight"
        if key in param_dict:
            llm_post_ffw_norm[i] = param_dict[key]

    # 保存 LLM 参数
    converted_params["params/llm/layers/attn/q_einsum/w"] = llm_q
    converted_params["params/llm/layers/attn/kv_einsum/w"] = llm_kv
    converted_params["params/llm/layers/attn/attn_vec_einsum/w"] = llm_o
    converted_params["params/llm/layers/mlp/gating_einsum"] = llm_mlp_gate
    converted_params["params/llm/layers/mlp/linear"] = llm_mlp_linear
    converted_params["params/llm/layers/pre_attention_norm/scale"] = llm_pre_attn_norm
    converted_params["params/llm/layers/pre_ffw_norm/scale"] = llm_pre_ffw_norm

    # Gemma 3 特有参数
    converted_params["params/llm/layers/attn/q_norm"] = llm_q_norm
    converted_params["params/llm/layers/attn/k_norm"] = llm_k_norm
    converted_params["params/llm/layers/post_attention_norm/scale"] = llm_post_attn_norm
    converted_params["params/llm/layers/post_ffw_norm/scale"] = llm_post_ffw_norm

    stats["llm"] += 11  # 增加了 Gemma 3 的 4 个参数

    print(f"  LLM layers: {num_layers}")

    # ========================================
    # 4. Embeddings and Final Norm
    # ========================================
    print("\n=== Converting Embeddings ===")

    # Embedding
    if "model.encoder.embed_tokens.weight" in param_dict:
        converted_params["params/llm/embedder/input_embedding"] = param_dict[
            "model.encoder.embed_tokens.weight"]
        print(f"  embedder/input_embedding: {converted_params['params/llm/embedder/input_embedding'].shape}")
        stats["llm"] += 1

    # Encoder final norm
    if "model.encoder.norm.weight" in param_dict:
        converted_params["params/llm/final_norm/scale"] = param_dict[
            "model.encoder.norm.weight"]
        print(f"  final_norm/scale: {converted_params['params/llm/final_norm/scale'].shape}")
        stats["llm"] += 1

    # ========================================
    # 5. Save
    # ========================================
    print("\n=== Saving ===")
    print(f"  Vision params: {stats['vision']}")
    print(f"  Vision proj params: {stats['vision_proj']}")
    print(f"  LLM params: {stats['llm']}")
    print(f"  Total: {len(converted_params)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, **converted_params)
    print(f"\nSaved to: {output_path}")

    return converted_params


if __name__ == "__main__":
    convert_t5gemma2_encoder_to_openpi()
