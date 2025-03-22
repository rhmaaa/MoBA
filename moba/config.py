from dataclasses import dataclass


@dataclass
class MoBAConfig:
    moba_chunk_size: int
    moba_topk: int
    # 分析与调试参数
    print_recall: bool = False
    layer_names: bool = True
    verbose: bool = False
