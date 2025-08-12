#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
计算和分析 Transformer 模型的参数量、内存占用和 FLOPs。
配置从外部 JSON 文件加载。
"""

import json
import math

CONFIG_FILE = "cs336_basics/model_configs.json"

def transformer_memory_accounting(
    vocab_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    d_model: int,
    d_ff: int,
    batch_size: int = 1,
    bytes_per_param: int = 4
):
    """
    计算 Transformer 模型的内存占用（参数、梯度、优化器状态、激活值）。
    严格遵循 CS336 Assignment 1 的描述。
    """
    bytes_for_optimizer_state = 2 * bytes_per_param  # AdamW 存储 m 和 v 两个动量

    # 1. 参数量计算
    params = {}
    params["embedding"] = vocab_size * d_model
    params["rope"] = 0  # RoPE 没有可学习参数

    params_per_block = {}
    params_per_block["mha"] = 4 * (d_model * d_model)  # Wq, Wk, Wv, Wo
    params_per_block["ffn_swiglu"] = 3 * (d_model * d_ff)  # W1, W2, W3
    params_per_block["rmsnorm"] = 2 * d_model  # 块内两个 RMSNorm
    
    params["transformer_blocks"] = num_layers * sum(params_per_block.values())
    params["final_rmsnorm"] = d_model
    params["lm_head"] = d_model * vocab_size
    
    total_params = sum(params.values())

    # 2. 内存占用计算
    memory = {}
    memory["parameters_bytes"] = total_params * bytes_per_param
    memory["gradients_bytes"] = total_params * bytes_per_param
    memory["optimizer_state_bytes"] = total_params * bytes_for_optimizer_state
    
    # 3. 激活值内存计算 (基于 PDF Problem adamwAccounting) [cite: 937-945]
    activations = {}
    activations_per_block = {}
    qkv_proj = 3 * batch_size * seq_len * d_model
    attention_scores = batch_size * num_heads * seq_len * seq_len
    weighted_sum = batch_size * seq_len * d_model
    activations_per_block["mha"] = qkv_proj + attention_scores + weighted_sum
    
    ffn_w1_w3 = 2 * batch_size * seq_len * d_ff
    ffn_w2 = batch_size * seq_len * d_model
    activations_per_block["ffn"] = ffn_w1_w3 + ffn_w2
    
    total_activations = num_layers * sum(activations_per_block.values())
    total_activations += (batch_size * seq_len * vocab_size)  # Final Logits
    
    memory["activations_bytes"] = total_activations * bytes_per_param
    memory["total_peak_bytes"] = sum(memory.values())

    return {
        "params_breakdown": params,
        "total_params": total_params,
        "memory_breakdown_bytes": memory
    }

def transformer_flops_accounting(
    vocab_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    d_model: int,
    d_ff: int,
    batch_size: int = 1
):
    """
    计算 Transformer 模型一次前向传播的 FLOPs。
    规则: A(m,n) @ B(n,p) 需要 2*m*n*p FLOPs [cite: 778]
    """
    flops = {}
    flops_per_block = {}
    d_k = d_model // num_heads

    # MHA Sub-layer FLOPs
    qkv_proj_flops = 3 * (2 * batch_size * seq_len * d_model * d_model)
    attn_scores_flops = 2 * batch_size * num_heads * seq_len * seq_len * d_k
    attn_output_flops = 2 * batch_size * num_heads * seq_len * seq_len * d_k
    output_proj_flops = 2 * batch_size * seq_len * d_model * d_model
    flops_per_block["mha"] = qkv_proj_flops + attn_scores_flops + attn_output_flops + output_proj_flops
    
    # FFN (SwiGLU) Sub-layer FLOPs
    ffn_w1_w3_flops = 2 * (2 * batch_size * seq_len * d_model * d_ff)
    ffn_w2_flops = 2 * batch_size * seq_len * d_ff * d_model
    flops_per_block["ffn_swiglu"] = ffn_w1_w3_flops + ffn_w2_flops
    
    # Total FLOPs for all blocks
    flops["transformer_blocks"] = num_layers * sum(flops_per_block.values())
    
    # Final LM Head
    flops["lm_head"] = 2 * batch_size * seq_len * d_model * vocab_size
    
    total_flops = sum(flops.values())
    
    return {
        "flops_breakdown": flops,
        "total_flops_forward_pass": total_flops,
        "flops_per_block": flops_per_block # 返回块内分解以便主函数使用
    }

def main():
    """
    主函数，读取模型配置，执行计算并打印结果。
    """
    try:
        with open(CONFIG_FILE, 'r') as f:
            all_configs = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        return

    for model_name, config in all_configs.items():
        print(f"\n{'='*20} Analysis for: {model_name.upper()} {'='*20}")

        # --- Memory and Parameter Calculation ---
        mem_results = transformer_memory_accounting(**config)
        total_params = mem_results["total_params"]
        params_mem_gb = mem_results["memory_breakdown_bytes"]["parameters_bytes"] / 1e9
        
        print(f"\n[Memory & Parameters]")
        print(f"Total Trainable Parameters: {total_params / 1e6:.2f} Million ({total_params / 1e9:.3f} Billion)")
        print(f"Memory for Parameters (float32): {params_mem_gb:.3f} GB")

        # --- FLOPs Calculation ---
        flops_results = transformer_flops_accounting(**config)
        total_flops = flops_results["total_flops_forward_pass"]
        flops_per_block = flops_results["flops_per_block"]
        flops_lm_head = flops_results["flops_breakdown"]["lm_head"]
        num_layers = config["num_layers"]
        
        # --- Detailed FLOPs Breakdown ---
        total_attn_flops_all_layers = num_layers * flops_per_block["mha"]
        total_ffn_flops_all_layers = num_layers * flops_per_block["ffn_swiglu"]
        
        attn_percentage = (total_attn_flops_all_layers / total_flops) * 100
        ffn_percentage = (total_ffn_flops_all_layers / total_flops) * 100
        lm_head_percentage = (flops_lm_head / total_flops) * 100
        
        print(f"\n[FLOPs Accounting (Forward Pass)]")
        print(f"Total FLOPs: {total_flops / 1e12:.3f} TFLOPs")
        print("\n--- Detailed FLOPs Proportions ---")
        print(f"Attention (all layers): {attn_percentage:>6.2f}%")
        print(f"FFN (all layers)      : {ffn_percentage:>6.2f}%")
        print(f"Final LM Head         : {lm_head_percentage:>6.2f}%")
        print(f"{'-'*35}")


if __name__ == '__main__':
    main()