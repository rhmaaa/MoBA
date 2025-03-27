"""A clean version of efficient moba implementation with flash-attn"""

import torch
import time

from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from functools import lru_cache
from einops import rearrange


# def calculate_moba_recall(q, k, gate_top_k_idx, moba_chunk_size, top_k=None):
#     """
#     计算 MOBA 选择的 token 相对于真实 Top K attention score 的召回率
    
#     参数:
#     q: 查询张量，形状为 [seqlen, num_head, head_dim]
#     k: 键张量，形状为 [seqlen, num_head, head_dim]
#     gate_top_k_idx: MOBA 选择的块索引，形状为 [batch, num_head, moba_topk]
#     moba_chunk_size: 每个块的大小（包含的token数量）
#     top_k: 计算真实 Top K 时的 K 值，默认为 moba_chunk_size * gate_top_k_idx.shape[-1]
    
#     返回:
#     平均召回率
#     """
#     # 获取维度信息
#     seqlen, num_head, head_dim = q.shape
#     batch_size = gate_top_k_idx.shape[0]
#     moba_topk = gate_top_k_idx.shape[-1]
    
#     if top_k is None:
#         # 默认 top_k 为 MOBA 选择的总 token 数量（近似值）
#         top_k = min(moba_chunk_size * moba_topk, seqlen)
    
#     # 1. 计算真实的注意力分数矩阵（不应用softmax，只需要相对大小）
#     # 形状: [seqlen, num_head, seqlen]
#     attn_scores = torch.matmul(q, k.transpose(1, 2))
    
#     # 2. 对于每个查询 token 和每个头，获取真实的 Top K token 索引
#     # 形状: [seqlen, num_head, top_k]
#     true_topk_indices = torch.topk(attn_scores, k=top_k, dim=-1).indices
    
#     # 3. 从 gate_top_k_idx 计算 MOBA 选择的 token 索引
#     # 假设 gate_top_k_idx 中的值是块索引，范围是 [0, num_chunks)
    
#     # 创建一个空的集合列表，每个查询 token 和注意力头一个集合
#     moba_selected_tokens = [[set() for _ in range(num_head)] for _ in range(seqlen)]
#     true_topk_token_sets = [[set() for _ in range(num_head)] for _ in range(seqlen)]
    
#     # 填充真实 Top K token 集合
#     for q_idx in range(seqlen):
#         for h_idx in range(num_head):
#             true_topk_token_sets[q_idx][h_idx] = set(true_topk_indices[q_idx, h_idx].tolist())
    
#     # 从块索引计算对应的 token 索引并填充 MOBA 选择的 token 集合
#     for b_idx in range(batch_size):
#         for h_idx in range(num_head):
#             for q_idx in range(seqlen):
#                 # 获取当前查询选择的块索引
#                 selected_chunks = gate_top_k_idx[b_idx, h_idx].tolist()
                
#                 # 为每个块添加相应的 token 索引
#                 for chunk_idx in selected_chunks:
#                     chunk_start = chunk_idx * moba_chunk_size
#                     # 确保不超过序列长度
#                     chunk_end = min((chunk_idx + 1) * moba_chunk_size, seqlen)
#                     # 添加该块中的所有 token
#                     moba_selected_tokens[q_idx][h_idx].update(range(chunk_start, chunk_end))
    
#     # 4. 计算每个查询 token 和头的召回率
#     total_recall = 0.0
#     count = 0
    
#     for q_idx in range(seqlen):
#         for h_idx in range(num_head):
#             true_topk = true_topk_token_sets[q_idx][h_idx]
#             moba_selected = moba_selected_tokens[q_idx][h_idx]
            
#             if len(true_topk) > 0:  # 避免除零错误
#                 # 计算交集大小
#                 intersection = len(true_topk.intersection(moba_selected))
                
#                 # 计算召回率
#                 recall = intersection / len(true_topk)
#                 total_recall += recall
#                 count += 1
    
#     # 计算平均召回率
#     avg_recall = total_recall / count if count > 0 else 0.0
#     print(f"Average recall: {avg_recall}")
    
#     return avg_recall
# def calculate_moba_recall(q, k, gate_top_k_idx, moba_chunk_size, moba_topk,top_k=None, layer_name=None, print_results=True, verbose=False):
#     """
#     计算 MOBA 选择的 token 相对于真实 Top K attention score 的召回率
    
#     参数:
#     q: 查询张量，形状为 [seqlen, num_head, head_dim]
#     k: 键张量，形状为 [seqlen, num_head, head_dim]
#     gate_top_k_idx: MOBA 选择的块索引，形状为 [moba_topk - 1, num_head, seqlen]
#     moba_chunk_size: 每个块的大小（包含的token数量）
#     top_k: 计算真实 Top K 时的 K 值，默认为 moba_chunk_size * gate_top_k_idx.shape[-1]
#     layer_name: 当前层的名称或索引，用于输出结果，默认为None
#     print_results: 是否打印结果，默认为True
#     verbose: 是否打印详细的计算过程信息，默认为False
    
#     返回:
#     per_head_recall: 每个注意力头的平均召回率
#     avg_recall: 所有头的平均召回率
#     """
    
#     # 记录哪些block是被丢掉了  两者之间不一样的地方
#     # import ipdb; ipdb.set_trace()
#     start_time = time.time()
    
#     # 获取维度信息
#     seqlen, num_head, head_dim = q.shape
#     num_selected_chunks = gate_top_k_idx.shape[0]  # moba_topk
    
#     # 1. 将gate_top_k_idx转换为实际的token indices
#     # gate_top_k_idx shape: [moba_topk-1, HEAD, SEQ]
#     # 创建一个mask来标记被选中的tokens
#     selected_tokens_mask = torch.zeros((seqlen, num_head, seqlen), dtype=torch.bool, device=q.device)
    
#     # 首先，为每个query位置添加其对应的当前chunk
#     for seq_pos in range(seqlen):
#         current_chunk_id = seq_pos // moba_chunk_size
#         chunk_start = current_chunk_id * moba_chunk_size
#         chunk_end = min((current_chunk_id + 1) * moba_chunk_size, seqlen)
#         # 标记当前chunk中的所有token（对所有head都一样）
#         selected_tokens_mask[seq_pos, :, chunk_start:chunk_end] = True
    
#     # 然后添加gate_top_k_idx选择的chunks
#     for chunk_idx in range(num_selected_chunks):
#         selected_chunks = gate_top_k_idx[chunk_idx]  # [HEAD, SEQ]
#         for h in range(num_head):
#             for seq_pos in range(seqlen):
#                 chunk_id = selected_chunks[h, seq_pos]
#                 start_pos = chunk_id * moba_chunk_size
#                 end_pos = min((chunk_id + 1) * moba_chunk_size, seqlen)
#                 # 标记这个chunk中的所有token
#                 selected_tokens_mask[seq_pos, h, start_pos:end_pos] = True
    
#     if top_k is None:
#         # 计算每个位置实际选中的token数量的平均值
#         avg_selected_tokens = selected_tokens_mask.sum(-1).float().mean().item()
#         print(f"Average selected tokens per position: {avg_selected_tokens:.1f}")
#         # 使用实际选中token数量的2%作为top_k
#         top_k = max(int(avg_selected_tokens * 0.02), 1)
#         print(f"Using top_k = {top_k}")
#     top_k = max(int(avg_selected_tokens * 0.02), 1)
#     print("moba_chunk_size", moba_chunk_size)
#     print("moba_topk", moba_topk)
#     print("seqlen", seqlen)
#     # 1. 计算真实的注意力分数矩阵
#     attn_scores = torch.matmul(q, k.transpose(1, 2))  # [seqlen, num_head, seqlen]
    
#     # 2. 对于每个查询token和每个头，获取真实的Top K token索引
#     true_topk_values, true_topk_indices = torch.topk(attn_scores, k=top_k, dim=-1)  # [seqlen, num_head, top_k]
    
#     # 3. 计算召回率
#     # 创建一个mask来标记真实的top-k tokens
#     true_topk_mask = torch.zeros_like(selected_tokens_mask)
#     for q_idx in range(seqlen):
#         for h_idx in range(num_head):
#             true_topk_mask[q_idx, h_idx, true_topk_indices[q_idx, h_idx]] = True
    
#     # 计算每个查询和每个头的交集
#     intersection = (selected_tokens_mask & true_topk_mask).sum(dim=-1)  # [seqlen, num_head]
    
#     # 计算每个头的平均召回率
#     per_head_recall = (intersection.float() / top_k).mean(dim=0)  # [num_head]
    
#     # 计算总体平均召回率
#     avg_recall = per_head_recall.mean().item()
    
#     # 计算每个头选择的token数量
#     selected_tokens_per_head = selected_tokens_mask.sum(-1).float().mean(0)  # [num_head]
    
#     # 计算理论上的计算节省
#     full_attn_pairs = seqlen * seqlen
#     moba_attn_pairs = selected_tokens_per_head.mean().item() * seqlen
#     reduction = (1 - moba_attn_pairs/full_attn_pairs) * 100
    
#     # 计算耗时
#     elapsed_time = time.time() - start_time
    
#     # 打印每层的召回率结果
#     if print_results:
#         layer_info = f"Layer {layer_name}" if layer_name is not None else "Current layer"
#         print(f"\n{layer_info} Recall Results:")
#         print(f"Average recall across all heads: {avg_recall:.4f}")
#         if verbose:
#             print("Per-head recall rates:")
#             for h_idx, recall in enumerate(per_head_recall):
#                 print(f"  Head {h_idx}: {recall:.4f}")
#             print(f"Computation time: {elapsed_time:.4f} seconds")
        
#             # 额外详细信息
#             moba_token_count = sum(len(moba_selected_tokens[0][h]) for h in range(num_head)) / num_head
#             print(f"MOBA selected ~{moba_token_count:.1f} tokens per head on average")
#             print(f"True Top-K used: {top_k}")
            
#             # 计算理论上的计算节省
#             full_attn_pairs = seqlen * seqlen
#             moba_attn_pairs = moba_token_count * seqlen
#             reduction = (1 - moba_attn_pairs/full_attn_pairs) * 100
#             print(f"Computational reduction: ~{reduction:.1f}% compared to full attention")
    
#     return per_head_recall, avg_recall


def calculate_moba_recall(q, k, gate_top_k_idx, moba_chunk_size, moba_topk, top_k=None, layer_name=None, print_results=True, verbose=False):
    """
    计算 MOBA 选择的 token 相对于真实 Top K attention score 的召回率
    
    参数:
    q: 查询张量，形状为 [seqlen, num_head, head_dim]
    k: 键张量，形状为 [seqlen, num_head, head_dim]
    gate_top_k_idx: MOBA 选择的块索引，形状为 [moba_topk - 1, num_head, seqlen]
    moba_chunk_size: 每个块的大小（包含的token数量）
    moba_topk: 总共选择的块数量（包括当前块）
    top_k: 计算真实 Top K 时的 K 值，默认为None（会基于选中的令牌数自动计算）
    layer_name: 当前层的名称或索引，用于输出结果，默认为None
    print_results: 是否打印结果，默认为True
    verbose: 是否打印详细的计算过程信息，默认为False
    
    返回:
    per_head_recall: 每个注意力头的平均召回率
    avg_recall: 所有头的平均召回率
    """
    
    # 移除调试断点
    import time
    start_time = time.time()
    moba_topk = moba_topk + 1
    
    # 获取维度信息
    seqlen, num_head, head_dim = q.shape
    num_chunks = (seqlen + moba_chunk_size - 1) // moba_chunk_size  # 总块数（向上取整）
    
    # 确认gate_top_k_idx形状正确
    assert gate_top_k_idx.shape[0] == moba_topk - 1, f"gate_top_k_idx应有{moba_topk - 1}个选择的块，但实际有{gate_top_k_idx.shape[0]}个"
    assert gate_top_k_idx.shape[1] == num_head, f"gate_top_k_idx的头数应为{num_head}，但实际为{gate_top_k_idx.shape[1]}"
    assert gate_top_k_idx.shape[2] == seqlen, f"gate_top_k_idx的序列长度应为{seqlen}，但实际为{gate_top_k_idx.shape[2]}"
    
    # 1. 创建一个mask来标记被MOBA选中的tokens
    selected_tokens_mask = torch.zeros((seqlen, num_head, seqlen), dtype=torch.bool, device=q.device)
    
    # 首先，为每个query位置添加其对应的当前chunk
    for seq_pos in range(seqlen):
        current_chunk_id = seq_pos // moba_chunk_size
        chunk_start = current_chunk_id * moba_chunk_size
        chunk_end = min((current_chunk_id + 1) * moba_chunk_size, seqlen)
        # 标记当前chunk中的所有token（对所有head都一样）
        selected_tokens_mask[seq_pos, :, chunk_start:chunk_end] = True
    
    # 然后添加gate_top_k_idx选择的chunks
    for chunk_idx in range(moba_topk - 1):  # gate_top_k_idx中有moba_topk-1个额外选择的块
        selected_chunks = gate_top_k_idx[chunk_idx]  # [num_head, seqlen]
        for h in range(num_head):
            for seq_pos in range(seqlen):
                chunk_id = selected_chunks[h, seq_pos]
                # 确保chunk_id是有效的
                if 0 <= chunk_id < num_chunks:
                    start_pos = chunk_id * moba_chunk_size
                    end_pos = min((chunk_id + 1) * moba_chunk_size, seqlen)
                    # 标记这个chunk中的所有token
                    selected_tokens_mask[seq_pos, h, start_pos:end_pos] = True
    
    # 计算每个位置实际选中的token数量的平均值
    avg_selected_tokens = selected_tokens_mask.sum(-1).float().mean().item()
    top_k = max(int(seqlen * 0.02), 1)
    if top_k is None:
        print(f"Average selected tokens per position: {avg_selected_tokens:.1f}")
        # 使用实际选中token数量的2%作为top_k
        top_k = max(int(avg_selected_tokens * 0.02), 1)
    print(f"Using top_k = {top_k}")
    
    if verbose:
        print("moba_chunk_size:", moba_chunk_size)
        print("moba_topk:", moba_topk)
        print("seqlen:", seqlen)
        print("num_chunks:", num_chunks)
    
    # 2. 计算真实的注意力分数矩阵
    # 调整维度顺序为 [num_head, seqlen, head_dim]
    q = q.transpose(0, 1)  # [num_head, seqlen, head_dim]
    k = k.transpose(0, 1)  # [num_head, seqlen, head_dim]
    
    # 计算注意力分数：Q @ K^T
    # attn_scores: [num_head, seqlen, seqlen]
    attn_scores = torch.matmul(q, k.transpose(-2, -1))
    
    # 3. 对每个查询token和每个头，获取真实的Top K token索引
    # 确保 top_k 不超过序列长度
    top_k = min(top_k, attn_scores.size(-1))
    
    # 在最后一个维度（键的维度）上取 top_k
    # 输出的 true_topk_indices: [num_head, seqlen, top_k]
    true_topk_values, true_topk_indices = torch.topk(attn_scores, k=top_k, dim=-1)
    
    # 调整回原始维度顺序 [seqlen, num_head, top_k]
    true_topk_indices = true_topk_indices.transpose(0, 1)
    
    # 4. 计算召回率
    # 创建一个mask来标记真实的top-k tokens
    true_topk_mask = torch.zeros_like(selected_tokens_mask)
    for q_idx in range(seqlen):
        for h_idx in range(num_head):
            true_topk_mask[q_idx, h_idx, true_topk_indices[q_idx, h_idx]] = True
    
    # 计算每个查询和每个头的交集
    intersection = (selected_tokens_mask & true_topk_mask).sum(dim=-1)  # [seqlen, num_head]
    
    # 计算每个头的平均召回率
    per_head_recall = (intersection.float() / top_k).mean(dim=0)  # [num_head]
    
    # 计算总体平均召回率
    avg_recall = per_head_recall.mean().item()
    
    # 计算每个头选择的token数量
    selected_tokens_per_head = selected_tokens_mask.sum(-1).float().mean(0)  # [num_head]
    
    # 计算理论上的计算节省
    full_attn_pairs = seqlen * seqlen
    moba_attn_pairs = selected_tokens_per_head.mean().item() * seqlen
    reduction = (1 - moba_attn_pairs/full_attn_pairs) * 100
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    
    # 打印每层的召回率结果
    # if print_results:
    #     layer_info = f"Layer {layer_name}" if layer_name is not None else "Current layer"
    #     print(f"\n{layer_info} Recall Results:")
    #     print(f"Average recall across all heads: {avg_recall:.4f}")
    #     if verbose:
    #         print("Per-head recall rates:")
    #         for h_idx, recall in enumerate(per_head_recall):
    #             print(f"  Head {h_idx}: {recall:.4f}")
    #         print(f"Computation time: {elapsed_time:.4f} seconds")
            
    #         # 额外详细信息
    #         print(f"MOBA selected ~{avg_selected_tokens:.1f} tokens per head on average")
    #         print(f"True Top-K used: {top_k}")
            
    #         # 计算理论上的计算节省
    #         print(f"Computational reduction: ~{reduction:.1f}% compared to full attention")
    import csv
    import os
    
    # 创建文件名
    path="/public/dolma/result/"
    filename = f"{path}{seqlen}_{moba_chunk_size}_{moba_topk}.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # 如果文件不存在，写入标题行
        if not file_exists:
            headers = ['average_recall']
            writer.writerow(headers)
        
        # 只写入平均召回率
        writer.writerow([round(avg_recall, 4)])
    return per_head_recall, avg_recall


@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """calc chunks that needs moba attention"""

    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size
    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    # total chunk ( for all batch )
    num_chunk = cu_num_chunk[-1]
    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    """ filter chunks that need moba attn """

    # filter chunks ( remove last chunk of each batch )
    # filtered_chunk_indices: chunk index list that excludes the last chunk of each batch
    chunk_to_remove = cu_num_chunk[1:] - 1
    chunk_to_remain = torch.ones(
        (num_chunk,), dtype=torch.bool, device=cu_seqlen.device
    )
    chunk_to_remain[chunk_to_remove] = False
    # 这里保存的就是对应的chunk的序列  
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
    num_filtered_chunk = len(filtered_chunk_indices)

    return (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    )


class MixedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    ):
        ctx.max_seqlen = max_seqlen
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        # self attn
        _, _, _, _, self_attn_out_sh, self_attn_lse_hs, _, _ = (
            _flash_attn_varlen_forward(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self_attn_cu_seqlen,
                cu_seqlens_k=self_attn_cu_seqlen,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=True,
                dropout_p=0.0,
            )
        )

        # moba attn
        _, _, _, _, moba_attn_out, moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )

        # convert lse shape hs -> sh ( follow the legacy mix attn logic )
        self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
        moba_attn_lse = moba_attn_lse_hs.t().contiguous()

        # output buffer [S, H, D], same shape as q
        output = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        # flatten vS & H for index ops
        output_2d = output.view(-1, q.shape[2])

        # calc mixed_lse
        # minus max lse to avoid exp explosion
        max_lse_1d = self_attn_lse_sh.view(-1)
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
        moba_attn_lse = (
            moba_attn_lse.view(-1)
            .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
            .reshape_as(moba_attn_lse)
        )

        mixed_attn_se_sh = self_attn_lse_sh.exp()
        moba_attn_se = moba_attn_lse.exp()

        mixed_attn_se_sh.view(-1).index_add_(
            0, moba_q_sh_indices, moba_attn_se.view(-1)
        )
        mixed_attn_lse_sh = mixed_attn_se_sh.log()

        # add attn output
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ vS, H ]
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

        # add moba output
        mixed_attn_lse = (
            mixed_attn_lse_sh.view(-1)
            .index_select(0, moba_q_sh_indices)
            .view_as(moba_attn_lse)
        )
        factor = (moba_attn_lse - mixed_attn_lse).exp()  # [ vS, H ]
        moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
        raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
        output = output.to(q.dtype)
        # add back max lse
        mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)
        ctx.save_for_backward(
            output,
            mixed_attn_lse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        )

        return output

    @staticmethod
    def backward(ctx, d_output):

        max_seqlen = ctx.max_seqlen
        moba_chunk_size = ctx.moba_chunk_size
        softmax_scale = ctx.softmax_scale

        (
            output,
            mixed_attn_vlse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        ) = ctx.saved_tensors

        d_output = d_output.contiguous()

        dq, dk, dv, _ = _flash_attn_varlen_backward(
            dout=d_output,
            q=q,
            k=k,
            v=v,
            out=output,
            softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
            dq=None,
            dk=None,
            dv=None,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
            dropout_p=0.0,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        headdim = q.shape[-1]
        d_moba_output = (
            d_output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )
        moba_output = (
            output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )

        mixed_attn_vlse = (
            mixed_attn_vlse_sh.view(-1).index_select(0, moba_q_sh_indices).view(1, -1)
        )

        dmq, dmk, dmv, _ = _flash_attn_varlen_backward(
            dout=d_moba_output,
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=None,
            dk=None,
            dv=None,
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        dmkv = torch.stack((dmk, dmv), dim=1)
        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None


def moba_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
    print_recall: bool = False,
    layer_name: str = None,
    verbose: bool = False
) -> torch.Tensor:
    """An efficient version of moba implementation with triton kernels and flash-attn, the core logic:
    1. Calculate the chunks and the number of chunks, n = floor(data_size / chunk_size)
       - tokens in the tail chunk are reserved for self attn
       - tokens in other chunks will be processed in later steps
    2. K in each chunk will calculate mean value as the representative k, and Q will attend to these representative
    k to get the gate logit, which will be used to select topk chunks
    3. Select the topk chunks and get the dense q for each kv chunk pair and do the varlen attention
    4. Combine the varlen attn and self attn results via online softmax to get the final result

    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): the cumulative sequence length tensor, same definition in flash attn
        max_seqlen (int): the max sequence length of the batch, same definition in flash attn

    Returns:
        attn_output (torch.Tensor): [seqlen, head, head_dim]
    """
    # import ipdb; ipdb.set_trace()
    kv = torch.stack((k, v), dim=1)

    """ some basic variables """
    # qkv shape = [ S, H, D ]
    seqlen, num_head, head_dim = q.shape

    """ prepare chunk meta """
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    ) = calc_chunks(cu_seqlens, moba_chunk_size)

    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    moba_topk = min(moba_topk - 1, num_filtered_chunk)
    need_moba_attn = moba_topk > 0

    # corner case: if no moba attn needed, just return self attn
    if not need_moba_attn:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )

    self_attn_cu_seqlen = cu_chunk

    # filtered_kv is a dense matrix that only contains filtered chunk of kv
    filtered_kv_indices = torch.arange(
        0, moba_chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

    """ calc key_gate_weight and gate """

    # key_gate_weight [ F_N_CHUNK, HEAD, HEAD_DIM ]
    key_gate_weight = (
        filtered_kv[:, 0]
        .view(num_filtered_chunk, moba_chunk_size, num_head, head_dim)
        .mean(dim=1)
        .float()
    )
    q = q.type(torch.float32)  # float logit on the fly for better gate logit perception
    key_gate_weight = key_gate_weight.type(
        torch.float32
    )  # float logit for better gate logit perception
    gate = torch.einsum(
        "nhd,shd->nhs", key_gate_weight, q
    )  # gate [ F_N_CHUNK, HEAD, SEQ ]
    key_gate_weight = key_gate_weight.type_as(k)
    q = q.type_as(k)

    # pose process gate, masking unchosen batch and apply causal mask to current chunk
    gate_seq_idx = torch.arange(0, seqlen, device=q.device, dtype=torch.int32)[
        None, :
    ].repeat(num_filtered_chunk, 1)
    chunk_end = cu_chunk[filtered_chunk_indices + 1]
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1]
    gate_chunk_end_mask = gate_seq_idx < chunk_end[:, None]
    gate_batch_end_mask = gate_seq_idx >= batch_end[:, None]
    gate_inf_mask = gate_chunk_end_mask | gate_batch_end_mask
    gate.masked_fill_(gate_inf_mask.unsqueeze(1), -float("inf"))

    """ find moba q that needs moba attn """
    # find topk chunks
    # gate_mask [ N_CHUNK, HEAD, SEQ ], true indicates that needs attention
    _, gate_top_k_idx = torch.topk(gate, k=moba_topk, dim=0, largest=True, sorted=False)
    # 添加召回率计算
    if print_recall:
        per_head_recall, avg_recall = calculate_moba_recall(
            q, k, gate_top_k_idx, moba_chunk_size, moba_topk,
            layer_name=layer_name, verbose=verbose
        )
    # apply causal mask
    gate_mask = torch.logical_not(gate.isinf())
    # select topk chunks
    gate_idx_mask = torch.zeros(gate_mask.shape, dtype=torch.bool, device=q.device)
    gate_idx_mask = gate_idx_mask.scatter_(dim=0, index=gate_top_k_idx, value=True)
    gate_mask = torch.logical_and(gate_mask, gate_idx_mask)

    # varlen trick: combining all q index that needs moba attn
    # the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1).nonzero(as_tuple=True)[
        -1
    ]  # [ HS indices ] * N
    # moba_seqlen_q indicates that how many q chunks are selected for each kv chunk - head
    moba_seqlen_q = gate_mask.sum(dim=-1).flatten()
    # select all q that needs moba attn based on the moba_q_indices
    moba_q = rearrange(q, "s h d -> ( h s ) d").index_select(
        0, moba_q_indices
    )  # [ selected_S, D ]
    moba_q = moba_q.unsqueeze(1)
    # moba_q_sh_indices represents the position in the origin q tensor of each q token inside moba_q
    moba_q_sh_indices = moba_q_indices % seqlen * num_head + moba_q_indices // seqlen

    """ prepare moba kv """
    # Since moba_q is organized as HS * N, we need to reorganize kv to adapt to q

    # cut off zero experts
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    # only keep the kv that has q select > 0
    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]
    # moba cu_seqlen for flash attn
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0,
    ).to(torch.int32)
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d")
    moba_kv = moba_kv.split(moba_chunk_size, dim=1)
    moba_kv = torch.cat(moba_kv, dim=0)
    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count
        moba_kv = moba_kv[
            valid_expert_mask
        ]  # cut off zero Q expert from kv , or the grad may be nan
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)
    moba_cu_seqlen_kv = (
        torch.arange(
            0,
            num_filtered_chunk * num_head + 1 - zero_expert_count,
            dtype=torch.int32,
            device=q.device,
        )
        * moba_chunk_size
    )

    # Shape check
    assert (
        moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"

    # Wrapping up the flash attn call and online softmax dlse inside MixedAttention class
    return MixedAttention.apply(
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    )
