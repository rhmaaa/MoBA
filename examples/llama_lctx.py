import torch
import os
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from moba import register_moba, MoBAConfig
from datasets import load_dataset, load_from_disk

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=3,
        help="Number of samples to use from LongContex"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=8192,
        help="Maximum sequence length for generation"
    )

    args = parser.parse_args()
    
    # Load LongContex dataset for evaluation
    print("Loading LongContex dataset...")
    try:
        # First try loading from local disk
        dataset_path = os.getenv('LONGCONTEX_PATH', 'data/longcontex')
        try:
            dataset = load_from_disk(dataset_path)
            print(f"Successfully loaded LongContex dataset from {dataset_path}")
        except Exception as local_e:
            # If local load fails, try loading from HF hub
            print(f"Could not load from local path: {str(local_e)}")
            dataset = load_dataset("LongContex/LongContex", split="validation")
            print("Successfully loaded LongContex dataset from HF hub")
        
        if dataset and len(dataset) > 0:
            print(f"Dataset contains {len(dataset)} examples")
        else:
            raise ValueError("Dataset is empty")
    except Exception as e:
        print(f"Error loading LongContex dataset: {str(e)}")
        print("Falling back to default input...")
        # Create a simple dataset with long repetitive text for testing
        default_text = "Tell me about artificial intelligence. " * 100
        dataset = [{"text": default_text}] * args.num_samples
        print(f"Created default dataset with {len(dataset)} examples")
        print("To use actual dataset, set LONGCONTEX_PATH env var to point to the dataset directory")
    
    # Configure MoBA settings
    moba_config = MoBAConfig(
        args.moba_chunk_size, 
        args.moba_topk, 
        print_recall=args.print_recall
    )
    register_moba(moba_config)
    
    # 加载模型和分词器
    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=args.attn,
    )
    tknz = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # 准备输入
    if dataset is not None and len(dataset) > 0:
        # 从LongContex选择随机样本
        sample_indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
        
        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            
            # LongContex数据集包含文档和问题
            document = sample['document']
            question = sample['question']
            
            # 构建提示
            prompt = f"Document:\n{document}\n\nQuestion: {question}\n\nAnswer:"
            
            print(f"\n\nProcessing LongContex sample {i+1}/{len(sample_indices)}")
            print(f"Question: {question}")
            print(f"Document length: {len(document)} characters")
            
            # 编码输入
            input_tokens = tknz.encode(prompt)
            print(f"Input length: {len(input_tokens)} tokens")
            
            # 如果输入太长，截断
            if len(input_tokens) > args.max_length - 100:
                print(f"Input too long ({len(input_tokens)} tokens), truncating...")
                input_tokens = input_tokens[:args.max_length - 100]
                
            input_ids = torch.tensor([input_tokens], device=model.device)
            
            # 生成输出
            print("Generating answer...")
            tokens = model.generate(
                input_ids, 
                max_length=min(len(input_tokens) + 200, args.max_length),
                do_sample=False
            )
            
            # 打印结果
            output = tknz.decode(tokens.squeeze().tolist())
            print(f"\nOutput:\n{output}")
            
            # 打印实际答案进行比较
            if 'short_answers' in sample and sample['short_answers']:
                print(f"\nReference answer: {sample['short_answers'][0]}")
    else:
        # 使用默认输入
        prompt = "how are you?" * 5
        input_tokens = tknz.encode(prompt)
        input_ids = torch.tensor([input_tokens], device=model.device)
        tokens = model.generate(input_ids, max_length=32, do_sample=False)
        print(tokens)
        print(tknz.decode(tokens.squeeze().tolist()))