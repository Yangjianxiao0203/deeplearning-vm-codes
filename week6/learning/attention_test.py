import numpy as np
import torch.nn as nn
import torch

from week6.learning.attention import SelfAttention

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)
def self_attention(x,
                   q_w,
                   q_b,
                   k_w,
                   k_b,
                   v_w,
                   v_b,
                   attention_output_weight,
                   attention_output_bias,
                   num_attention_heads,
                   hidden_size):
    # x.shape = max_len * hidden_size
    # q_w, k_w, v_w  shape = hidden_size * hidden_size
    # q_b, k_b, v_b  shape = hidden_size
    q = np.dot(x, q_w.T) + q_b  # shape: [max_len, hidden_size]      W * X + B lINER
    k = np.dot(x, k_w.T) + k_b  # shpae: [max_len, hidden_size]
    v = np.dot(x, v_w.T) + v_b  # shpae: [max_len, hidden_size]
    attention_head_size = int(hidden_size / num_attention_heads)
    # q.shape = num_attention_heads, max_len, attention_head_size
    q = transpose_for_scores(q, attention_head_size, num_attention_heads)
    # k.shape = num_attention_heads, max_len, attention_head_size
    k = transpose_for_scores(k, attention_head_size, num_attention_heads)
    # v.shape = num_attention_heads, max_len, attention_head_size
    v = transpose_for_scores(v, attention_head_size, num_attention_heads)
    '''matmul对于3维张量，相当于对最后两维进行矩阵乘法，前面的维度不变，batch的矩阵乘法'''
    # qk.shape = num_attention_heads, max_len, max_len
    qk = np.matmul(q, k.swapaxes(1, 2))
    qk /= np.sqrt(attention_head_size)
    qk = softmax(qk)
    # qkv.shape = num_attention_heads, max_len, attention_head_size
    qkv = np.matmul(qk, v)
    # qkv.shape = max_len, hidden_size
    qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
    # attention.shape = max_len, hidden_size
    # print("linear weight: ",attention_output_weight)
    # print("linear bias: ",attention_output_bias)
    print("before linear: ",qkv)
    attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias  # this is different in my class
    # print("function output ",attention)
    return attention


# 多头机制
def transpose_for_scores(x, attention_head_size, num_attention_heads):
    # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
    max_len, hidden_size = x.shape
    x = x.reshape(max_len, num_attention_heads, attention_head_size)
    x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
    return x

def compare_outputs(x):
    model = SelfAttention(embed_size=8, heads_num=2)

    # Extracting weights from the model.py
    q_w = model.w_q.weight.detach().numpy()
    q_b = model.w_q.bias.detach().numpy() if model.w_q.bias is not None else np.zeros_like(q_w[0])
    k_w = model.w_k.weight.detach().numpy()
    k_b = model.w_k.bias.detach().numpy() if model.w_k.bias is not None else np.zeros_like(k_w[0])
    v_w = model.w_v.weight.detach().numpy()
    v_b = model.w_v.bias.detach().numpy() if model.w_v.bias is not None else np.zeros_like(v_w[0])
    attention_output_weight = model.fc_out.weight.detach().numpy()
    attention_output_bias = model.fc_out.bias.detach().numpy() if model.fc_out.bias is not None else np.zeros_like(
        attention_output_weight[0])

    params = {
        "q_w": q_w, "q_b": q_b,
        "k_w": k_w, "k_b": k_b,
        "v_w": v_w, "v_b": v_b,
        "attention_output_weight": attention_output_weight,
        "attention_output_bias": attention_output_bias,
        "num_attention_heads": 2,
        "hidden_size": 8
    }
    model_output = model(x).detach().numpy()
    x = x.squeeze()
    function_output = self_attention(x.numpy(), **params)
    # print the weights
    print("q_w == model.py w_q", np.allclose(q_w, model.w_q.weight.detach().numpy()))
    print("k_w == model.py w_k ",np.allclose(k_w, model.w_k.weight.detach().numpy()))
    print("v_w == model.py w_v ",np.allclose(v_w, model.w_v.weight.detach().numpy()))
    print("fc_out_w == model.py fc_out ",np.allclose(attention_output_weight, model.fc_out.weight.detach().numpy()))
    print("fc_out_b == model.py fc_out ",np.allclose(attention_output_bias, model.fc_out.bias.detach().numpy()))
    # Calculating the difference between the outputs
    difference = np.abs(model_output - function_output).mean()
    print(f"Difference between the outputs: {difference:.5f}")
    return difference


# Mock input
x = torch.rand(1, 3, 8)  # 1 batch, 3 sequence length, 8 embedding size

compare_outputs(x)
