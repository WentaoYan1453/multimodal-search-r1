python /mnt/workspace/yanwentao/code/mmrl/verl/scripts/legacy_model_merger.py merge\
 --backend fsdp \
 --local_dir /mnt/workspace/yanwentao/code/multimodal-search-r1/checkpoints/mmsearch_r1_grpo_new/batchsize_128_rollout_8_search_penalty_0_no_warmup_new_score_3_new_prompt/global_step_150/actor\
 --target_dir /oss-tanxin/yanwentao/model/mmsearch_r1_grpo/dense_search_step_150