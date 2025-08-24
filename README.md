<h1 align="center">Multimodal-Search-R1: Incentivizing LMMs to Search</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2506.20670">Paper</a> ｜ 
  <a href="https://www.lmms-lab.com/posts/mmsearch_r1">Blog</a> ｜
  <a href="https://huggingface.co/lmms-lab/MMSearch-R1-7B">Model</a> ｜
  <a href="https://huggingface.co/datasets/lmms-lab/FVQA">Data</a>
</p>

## Overview
<p align="center">
  <img src="assets/mmsearch_r1_overview.png" alt="Overview of MMSearch-R1" width="800">
</p>

**MMSearch-R1** is an end-to-end RL framework that enables LMMs to perform on-demand, multi-turn search with real-world multimodal search tools.

[cache data位置](XLDDD/FVQA_Cache)

## Table of Content
- [Installation](#installation)
- [Multimodal Search Tool Implementation](#multimodal-search-tool-implemention)
- [Data Construction](#data-construction)
- [Train & Eval](#train--eval)

## Installation
```bash
# Clone this repo with submodules
git clone --recurse-submodules https://github.com/EvolvingLMMs-Lab/multimodal-search-r1.git
cd multimodal-search-r1
# Init Conda Env
conda create -n mmsearch_r1 python==3.10 -y
conda activate mmsearch_r1
# Install Dependencies
pip3 install -e ./verl
pip3 install vllm==0.8.2
pip3 install transformers==4.51.0
pip3 install flash-attn==2.7.4.post1
# Init wandb
pip3 install wandb
export WANDB_API_KEY="XXX"
wandb login $WANDB_API_KEY
```
就照着这样装，不过wandb可能要翻墙

## Multimodal Search Tool Implemention
- **Image Search Tool:** 下载[cache data](https://huggingface.co/datasets/XLDDD/FVQA_Cache)即可
- **Text Search Tool:** 
- 如果有联网搜索的条件，可以先使用免费的ddgs搜索（需翻墙）
- 如果没有，则参照[Search-R1](https://github.com/PeterGriffinJin/Search-R1)的本地检索工具搭建，参考[搜索搭建](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md) ，推荐e5，装好index,corpus, retriever模型后运行local_dense_retriever里的start即可
- 两种搜索的切换切换在 /multimodal/rollout/vllm_rollout_spmd.py第317 行


## Train & Eval
We recommend use the command below for unified training and evaluation:
```bash
bash mmsearch_r1/scripts/run_mmsearch_r1_grpo.sh
```
We highlight the important configurations for training the Multi-Round Search LMMs:
- `actor_rollout_ref.rollout.name`: should be `vllm_multiturn_mmsearch` for multi-turn search rollout;
- `actor_rollout_ref.actor.use_multi_turn_response_mask`: should be `True`, as we use it to refine the original `response_mask` for accurate loss calculation.
- `actor_rollout_ref.rollout.max_gen_round`: The max number of turns during rollout;
- `data.max_response_length`: The max response length for each turn;
- `actor_rollout_ref.rollout.response_length_total`: The max conversation length for all turns (except the user prompt in the first turn);

For evaluation only, configure these parameters in the above script:
```bash
...
trainer.val_files=${path_to_val_data} \
trainer.val_only=True \
trainer.val_only_save_dir=${path_to_save_dir} \
trainer.val_generations_to_log_to_wandb=64 # num of val generations to log, this should be larger than the size of val dataset for complete saving
```
The model's responses will be saved in JSON format under `${path_to_save_dir}`, which can be used for subsequent analysis and evaluation.
trainer.rollout_data_dir：打印训练时的输出

训练脚本相关：
 8 卡 80g内存的话 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu只能开到 8
 如果train batch或者roll out开大时推理有问题，更改verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
165 行 enable_prefix_caching=False


## Acknowledgement
We sincerely thank these repositories for providing helpful open-source resources: [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [veRL](https://github.com/volcengine/verl), [OpenDeepResearcher](https://github.com/mshumer/OpenDeepResearcher), [cfpark00/verl](https://github.com/cfpark00/verl/tree/multi_turn_rollout), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [MMSearch](https://github.com/CaraJ7/MMSearch).

## Citation
```
@article{wu2025mmsearch,
  title={MMSearch-R1: Incentivizing LMMs to Search},
  author={Wu, Jinming and Deng, Zihao and Li, Wei and Liu, Yiding and You, Bo and Li, Bo and Ma, Zejun and Liu, Ziwei},
  journal={arXiv preprint arXiv:2506.20670},
  year={2025}
}
```
