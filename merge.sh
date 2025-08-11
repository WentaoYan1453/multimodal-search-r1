python /mnt/workspace/yanwentao/code/mmrl/verl/scripts/legacy_model_merger.py merge\
 --backend fsdp \
 --local_dir /mnt/workspace/yanwentao/code/multimodal-search-r1/checkpoints/mmsearch_r1_grpo_new/mmsearch_r1_grpo_newprompt2_batch32_topk3_no_warmup_em_search_score2_search_penalty0_newtextprompt/global_step_200/actor \
 --target_dir /oss-tanxin/yanwentao/model/mmsearch_r1_grpo/mmsearch_r1_grpo_newprompt2_batch32_topk3_no_warmup_em_search_score2_search_penalty0_newtextprompt_200