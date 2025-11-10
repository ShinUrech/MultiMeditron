import yaml
import argparse

DESIRED_KEY = [
    "trainer.nnodes",
    "data.max_prompt_length",
    "data.max_response_length",
    "actor_rollout_ref.model.use_remove_padding",
    "data.train_batch_size",
    "actor_rollout_ref.rollout.gpu_memory_utilization",
    "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "actor_rollout_ref.rollout.data_parallel_size",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
    "actor_rollout_ref.rollout.max_num_batched_tokens",
    "actor_rollout_ref.rollout.free_cache_engine",
    "actor_rollout_ref.rollout.max_num_seqs",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    "actor_rollout_ref.actor.ppo_mini_batch_size",
    "actor_rollout_ref.actor.fsdp_config.param_offload",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
    "actor_rollout_ref.actor.fsdp_config.strategy",
]

def main():
    parser = argparse.ArgumentParser(description="Filter YAML file by key-value pair.")
    parser.add_argument("input_file", help="Path to the input YAML file.")
    parser.add_argument("--tabular", action="store_true", help="Output in tabular format.")
    args = parser.parse_args()

    print("=== ", args.input_file, " ===")
    with open(args.input_file, 'r') as file:
        data = yaml.safe_load(file)
    
    out_data = []
    for key in DESIRED_KEY:
        keys = key.split('.')
        
        current_level = data
        for k in keys:
            current_level = current_level[k]

        if not args.tabular:
            print(f"{key}: {current_level}")
        out_data.append(current_level)
    
    if args.tabular:
        def func(x):
            key, x = x
            if key == "actor_rollout_ref.rollout.gpu_memory_utilization":
                # Fraction to percentage
                return f"{int(x * 100)}%"

            if isinstance(x, str):
                return x
            elif isinstance(x, bool):
                return ["False", "True"][x]
            else:
                return str(x)

        print(','.join(map(func, zip(DESIRED_KEY, out_data))))


if __name__ == "__main__":
    main()
