import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from moba import register_moba, MoBAConfig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    if args.print_recall:
        print("print recall")
        register_moba(MoBAConfig(args.moba_chunk_size, 
                                    args.moba_topk,
                                    print_recall=args.print_recall, 
                                    layer_names=args.layer_names, 
                                    verbose=args.verbose))
    else:
        register_moba(MoBAConfig(args.moba_chunk_size, args.moba_topk))
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=args.attn,
    )
    tknz = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompt = "how are you?"*5
    input_tokens = tknz.encode(prompt)
    input_ids = torch.tensor([input_tokens], device=model.device)
    tokens = model.generate(input_ids, max_length=32, do_sample=False)
    print(tokens)
    print(tknz.decode(tokens.squeeze().tolist()))
