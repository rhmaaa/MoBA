import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from moba import register_moba, MoBAConfig
import json
import random
import numpy as np
import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # 6K true 
    # attn topk 2%
    # 



    # parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--model", type=str, default="/mnt/Meta-Llama-3-8B")
    parser.add_argument("--moba-chunk-size", type=int, default=4)
    parser.add_argument("--moba-topk", type=int, default=2)
    parser.add_argument(
        "--attn",
        default="moba",
        help="choose attention backend",
        choices=["flash_attention_2", "moba", "moba_naive"],
    )
    parser.add_argument("--print-recall", type=bool, default=True)
    parser.add_argument("--layer-names", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument('--input_file', type=str, default='/public/dolma/data/arxiv-0000.json', help='输入文件路径，token_ids的列表的json')
    parser.add_argument('--seq_start', type=int, default=1000, help='序列长度起始值')
    parser.add_argument('--seq_end', type=int, default=10000, help='序列长度结束值')
    parser.add_argument('--seq_step', type=int, default=1000, help='序列长度步长')
    parser.add_argument('--test_size', type=int, default=1, help='随机抽取的样本数量')
    parser.add_argument('--seq_len', type=int, default=1000, help='序列长度')
    
    args = parser.parse_args()
    seq_len = args.seq_len
    if args.print_recall:
        print("print recall")
        register_moba(MoBAConfig(args.moba_chunk_size, 
                                    args.moba_topk,
                                    print_recall=args.print_recall, 
                                    layer_names=args.layer_names, 
                                    verbose=args.verbose))
    else:
        register_moba(MoBAConfig(args.moba_chunk_size, 2))
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=args.attn,
    )
    tknz = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # prompt = "how are you?"*5
    # 读取输入文件，直接获取token_ids列表
    with open(args.input_file, 'r') as f:
        all_token_ids = json.load(f)
    # filtered_token_ids = [token_ids for token_ids in all_token_ids if len(token_ids) >= seq_len]
    # import ipdb; ipdb.set_trace()
    # 随机抽取test_size个样本，截取到seq_len
    # samples = []
    # for _ in range(test_size):
    #     token_ids = random.choice(filtered_token_ids)
    #     samples.append(token_ids[:seq_len])
# 根据参数设置序列长度范围
    # for seq_len in range(args.seq_start, args.seq_end + 1, args.seq_step):
    filtered_token_ids = [token_ids for token_ids in all_token_ids if len(token_ids) >= seq_len]
    # 随机抽取test_size个样本，截取到seq_len
    samples = []
    for _ in range(args.test_size):
        token_ids = random.choice(filtered_token_ids)
        samples.append(token_ids[:seq_len])
    for i in range(0, len(samples)):
        # import ipdb; ipdb.set_trace()
        input_token_ids = samples[i]
        input_length = len(input_token_ids)
        input_ids = torch.tensor([input_token_ids], device=model.device)  # Add batch dimension
        attention_mask = torch.ones_like(input_ids)
        # with torch.cuda.amp.autocast():  # 使用混合精度训练
        tokens = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=seq_len+128,
            do_sample=False
        )
            

   




    # for i in range(0, len(samples)):
    #     import ipdb; ipdb.set_trace()
    # input_tokens = tknz.encode(prompt)
    # input_ids = torch.tensor([input_tokens], device=model.device)
        # tokens = model.generate(input_ids, max_length=32, do_sample=False)
# 根据参数设置序列长度范围
    # for seq_len in range(args.seq_start, args.seq_end + 1, args.seq_step):
    #     best_topk = run_topk(model, tokenizer, all_token_ids, seq_len,
    #         test_size=args.test_size, x=args.x, max_tokens_count=args.max_tokens_count)
        
    #     # 以追加模式打开文件，写入当前结果
    #     with open(args.output_file, 'a') as f:
    #         f.write(f"{seq_len},{args.x},{best_topk}\n")  # 写入x值
    
    # print(tokens)
    # print(tknz.decode(tokens.squeeze().tolist()))
