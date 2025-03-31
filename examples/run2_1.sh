#!/bin/bash


# 设置实验参数
CHUNK_SIZES=(4 8 16 32 64 128 256 512 1024)
TOPK_VALUES=(128)
SEQ_LENs=(11000 13000 15000 16000)

# 创建结果目录
RESULTS_DIR="moba_results"
mkdir -p $RESULTS_DIR

# 记录实验开始时间
echo "开始MOBA参数扫描实验: $(date)" > $RESULTS_DIR/experiment_log.txt

# 遍历所有参数组合
for seq_len in "${SEQ_LENs[@]}"; do
  for chunk_size in "${CHUNK_SIZES[@]}"; do
    for topk in "${TOPK_VALUES[@]}"; do
      echo "运行参数组合: seq_len=$seq_len, chunk_size=$chunk_size, topk=$topk"
      
      # 创建实验特定的输出文件
    OUTPUT_FILE="$RESULTS_DIR/moba_cs${chunk_size}_tk${topk}.log"
    
    # 运行实验
    echo "开始实验: chunk_size=$chunk_size, topk=$topk, 时间: $(date)" >> $RESULTS_DIR/experiment_log.txt
    
    model_path="/public/Qwen/Qwen2.5-1.5B-Instruct"
    
    # 执行命令并记录输出
    python llama2.py \
      --moba-chunk-size $chunk_size \
      --moba-topk $topk \
      --seq_len $seq_len \
      --model $model_path
      

    s
    # 检查是否成功
    if [ $? -eq 0 ]; then
      echo "实验完成: chunk_size=$chunk_size, topk=$topk, 时间: $(date)" >> $RESULTS_DIR/experiment_log.txt
    else
      echo "实验失败: chunk_size=$chunk_size, topk=$topk, 时间: $(date)" >> $RESULTS_DIR/experiment_log.txt
    fi
  done
done
done

echo "所有实验完成: $(date)" >> $RESULTS_DIR/experiment_log.txt

# 创建结果汇总
echo "汇总结果..."
echo "chunk_size,topk,avg_recall" > $RESULTS_DIR/summary_results.csv

for seq_len in "${SEQ_LENs[@]}"; do
  for chunk_size in "${CHUNK_SIZES[@]}"; do
    for topk in "${TOPK_VALUES[@]}"; do
      # 提取平均召回率（根据您的输出格式可能需要调整）
      OUTPUT_FILE="$RESULTS_DIR/moba_cs${chunk_size}_tk${topk}.log"
      
      # 假设输出中包含"Average recall across all heads: X.XXXX"格式的行
    if [ -f "$OUTPUT_FILE" ]; then
      recall=$(grep "Average recall across all heads:" "$OUTPUT_FILE" | tail -1 | awk '{print $5}')
      echo "$chunk_size,$topk,$recall" >> $RESULTS_DIR/summary_results.csv
    else
      echo "$chunk_size,$topk,failed" >> $RESULTS_DIR/summary_results.csv
    fi
  done
  done
done

echo "实验完成! 结果保存在 $RESULTS_DIR 目录"