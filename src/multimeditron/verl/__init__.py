from functools import partial
import ray
import random
from omegaconf import OmegaConf
from recipe.spin.fsdp_workers import SPINRolloutRefWorker
from transformers import PreTrainedTokenizer
from verl.experimental.dataset.sampler import AbstractSampler
from verl.utils.config import validate_config
from verl.utils.import_utils import load_extern_type
from verl.workers.reward_manager.registry import get_reward_manager_cls
from multimeditron.verl.service import initialize_services
import logging

@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self,
            cfg,
            trust_remote_code: bool = False,
            verbose: bool = False,
            dryrun: bool = False):
        from transformers import AutoTokenizer
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        
        # Resolve and instantiate the configuration
        cfg = OmegaConf.create(cfg)
        OmegaConf.resolve(cfg)

        # Instantiate tokenizer
        tokenizer_path = cfg.actor_rollout_ref.model.get("tokenizer_path", None)
        if tokenizer_path is None:
            tokenizer_path = cfg.actor_rollout_ref.model.path
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=trust_remote_code
        )
        
        # Initialize services from the configuration
        if cfg.get("services", None) is not None:
            print("Initializing services...")
            self.services = initialize_services(cfg.services)
        
        # For each strategy, we would have different training loops
        # assert cfg.actor_rollout_ref.model.strategy == cfg.critic.model.strategy, "Currently only support same strategy for actor and critic"
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = RayWorkerGroup

        # Setup Ray
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            # Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }
        global_pool_id = "global_pool"
        reward_pool_id = "reward_pool"
        resource_pool_spec = {
            global_pool_id: [cfg.trainer.n_gpus_per_node] * cfg.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        # Validate the config
        validate_config(
            config=cfg,
            use_reference_policy=True,
            use_critic=False,
        )

        # We should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data

        # If we want reward model
        # if config.reward_model.enable:
        #     if config.reward_model.strategy == "fsdp":
        #         from verl.workers.fsdp_workers import RewardModelWorker
        #     elif config.reward_model.strategy == "megatron":
        #         from verl.workers.megatron_workers import RewardModelWorker
        #     else:
        #         raise NotImplementedError
        #     role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        #     mapping[Role.RewardModel] = global_pool_id
        
        # Reference model
        # if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        if cfg.algorithm.use_kl_in_reward:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        
        if cfg.reward_model.enable:
            from verl.workers.roles import RewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

            if cfg.reward_model.enable_resource_pool:
                resource_pool_spec[reward_pool_id] = [cfg.reward_model.n_gpus_per_node] * cfg.reward_model.nnodes
                resource_pool_spec[global_pool_id] = [cfg.trainer.n_gpus_per_node] * (cfg.trainer.nnodes - cfg.reward_model.nnodes)

            mapping[Role.RewardModel] = reward_pool_id if cfg.reward_model.enable_resource_pool else global_pool_id

        # Create reward function
        reward_manager_name = cfg.reward_model.get("reward_manager", "naive")
        reward_manager_cls = get_reward_manager_cls(reward_manager_name)

        reward_kwargs = cfg.reward_model.get("reward_kwargs", {})
        # compute_score_fn = partial(compute_score, **reward_kwargs)

        # compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=cfg.data.reward_fn_key,
            **reward_kwargs,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            reward_fn_key=cfg.data.reward_fn_key,
            **reward_kwargs,
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Create train rl smapler
        train_dataset = create_rl_dataset(
            cfg.data.train_files,
            cfg.data,
            tokenizer,
            processor=None,
            is_train=True,
            max_samples=cfg.data.get("max_train_samples", -1),
        )
        val_dataset = create_rl_dataset(
            cfg.data.val_files,
            cfg.data,
            tokenizer,
            processor=None,
            is_train=False,
            max_samples=cfg.data.get("max_val_samples", -1),
        )
        train_sampler = create_rl_sampler(cfg.data, train_dataset)
        

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.utils.dataset.rl_dataset import collate_fn
        trainer = RayPPOTrainer(
            config=cfg,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            collate_fn=collate_fn,
            train_dataset=train_dataset,
            train_sampler=train_sampler,
            val_dataset=val_dataset,
        )
        trainer.init_workers()

        if not dryrun:
            trainer.fit()
        else:
            print("Dry run complete. Exiting without training.")
    

def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset

def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    # reward_router_address: str,
    # reward_model_tokenizer: PreTrainedTokenizer,
):
    # print(extra_info)
    # if extra_info is None or "question" not in extra_info or "url" not in extra_info:
    #     raise ValueError("Extra info is required and must contain 'question' and 'url'")
    
    # do_print = False
    # if random.randint(0, 512) == 1:  
    #     do_print = True
    # if do_print:
    #     print(f"Response Case: {solution_str}, Question: {extra_info['question']}, GT: {ground_truth}")

    print(data_source)
    print(solution_str)
    print(ground_truth)
    print(extra_info)

    response = solution_str
    response_lower = response.lower()
    score = response_lower.count("a") / len(response_lower) if len(response_lower) > 0 else 0
    print(f"Score: {score}")

    return {
        "score": score,
        "acc": 0.0,
        "pred": "Maybe",
    }
