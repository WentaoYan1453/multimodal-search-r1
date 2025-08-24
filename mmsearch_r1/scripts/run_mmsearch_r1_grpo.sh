# This script is for single-node running test
export WANDB_API_KEY="your_wandb_api_key"
# export WANDB_BASE_URL=https://api.bandw.top
export RETRIEVAL_SERVICE_URL="http://0.0.0.0:8000/retrieve"
export TIMEOUT=300
export FVQA_TRAIN_CACHE_PATH="/nas/dmcv/yanwentao/dataset/FVQA/fvqa_train_cache"
export FVQA_TEST_CACHE_PATH="/nas/dmcv/yanwentao/dataset/FVQA/fvqa_test_cache"
TRAIN_DATA_PATH="/nas/dmcv/yanwentao/dataset/FVQA_new/fvqa_train.parquet"
VAL_DATA_PATH="/nas/dmcv/yanwentao/dataset/FVQA_new/fvqa_test.parquet"
WANDB_PROJECT_NAME="mmsearch_r1_grpo"
WANDB_EXP_NAME="web_search_mmsearch_128batch_rollout_8_search_penalty_1_no_warmup_ori_score_top5"

cd /mnt/workspace/yanwentao/code/multimodal-search-r1;


/mnt/workspace/yanwentao/mmverl/bin/python3 -m mmsearch_r1.trainer.multimodal.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.image_key=images \
    data.user_prompt_round_1=/mnt/workspace/yanwentao/code/multimodal-search-r1/mmsearch_r1/prompts/new_prompt_reason.pkl \
    data.user_prompt_after_image_search=/mnt/workspace/yanwentao/code/multimodal-search-r1/mmsearch_r1/prompts/after_image_search_prompt_qwenvl.pkl \
    data.user_prompt_after_text_search=/mnt/workspace/yanwentao/code/multimodal-search-r1/mmsearch_r1/prompts/after_text_search_prompt_qwenvl.pkl \
    actor_rollout_ref.model.path=/oss-tanxin/yanwentao/model/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup=False \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_ratio=0.65 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup_steps=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_multiturn_mmsearch \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_gen_round=3 \
    actor_rollout_ref.rollout.response_length_total=8192 \
    actor_rollout_ref.rollout.search.topk=3 \
    actor_rollout_ref.rollout.search.image_search_limit=1 \
    actor_rollout_ref.rollout.search.text_search_limit=2 \
    actor_rollout_ref.rollout.search.parallel_tool_call=True \
    actor_rollout_ref.rollout.search.parallel_tool_call_threads=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.rollout_data_dir=/nas/dmcv/yanwentao/output/mmsearch_r1_grpo_1_7B_70 \
    trainer.total_epochs=8 \
    +trainer.search_penalty=0.1 \
    +trainer.format_penalty=0.1 \
    +trainer.reward_mode="EM" \
    +trainer.val_before_train=False \
    +algorithm.filter_groups.enable=False 

    #trainer.val_only=True \
    # trainer.val_only_save_dir=/nas/dmcv/yanwentao/output/qwen_val_only_new_prompt \
    # trainer.val_generations_to_log_to_wandb=300 \


