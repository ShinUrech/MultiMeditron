from typing import Optional
from multimeditron.cli import EPILOG, CONFIG_PATH, main_cli
from .utils import split_host_port
import yaml
import click
import os
import logging
import ray
import sys


logger = logging.getLogger(__name__)

@main_cli.command(epilog=EPILOG)
@click.option("--output", "-o", type=click.Path(), help="Path to save the final configuration used for training (in YAML format).")
def verl_config(output: Optional[str] = None):
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf

    verl_path = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(verl_path, "../../../third-party/verl/verl/trainer/config")
    config_dir = os.path.abspath(config_dir)
    print(config_dir)
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="ppo_trainer")

    if output is not None:
        logger.info(f"Saving final configuration to {output}...")
        with open(output, "w") as f:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, sort_keys=False)
        logger.info(f"Final configuration saved to {output}")
    else:
        print(OmegaConf.to_yaml(cfg))


@main_cli.command(epilog=EPILOG, context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to the configuration file(s) in YAML format.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode.")
@click.option("--trust-remote-code/--no-trust-remote-code", default=False, help="Whether to trust remote code when loading models from HuggingFace.")
@click.option("--dryrun", is_flag=True, help="Perform a dry run without executing the training.")
@click.option("--config-out", "-o", type=click.Path(), help="Path to save the final configuration used for training (in YAML format).")
@click.option("--only-config", is_flag=True, help="Only output the final configuration and exit.")
@click.pass_context
def verl(ctx,
         config: Optional[str] = None,
         trust_remote_code: bool = False,
         verbose: bool = False,
         debug: bool = False,
         dryrun: bool = False,
         config_out: Optional[str] = None,
         only_config: bool = False):
    from hydra import initialize_config_dir, compose
    from torch.cuda import is_available as cuda_is_available

    if config is None:
        with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.2"):
            cfg = compose(config_name="verl_trainer", overrides=ctx.args)
    else:
        config_dir = os.path.dirname(os.path.abspath(config))
        config_name = os.path.basename(config)
        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            cfg = compose(config_name=config_name, overrides=ctx.args)

    if hasattr(cfg.ray, "debug") and cfg.ray.debug is not None:
        if not debug:
            logger.info("Overriding debug mode from command line to configuration file.")
            debug = cfg.ray.debug
        
    if hasattr(cfg.ray, "verbose") and cfg.ray.verbose is not None:
        if not verbose:
            logger.info("Overriding verbose mode from command line to configuration file.")
            verbose = cfg.ray.verbose

    # Save final configuration if needed
    if config_out is not None:
        logger.info(f"Saving final configuration to {config_out}...")
        from omegaconf import OmegaConf
        with open(config_out, "w") as f:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, sort_keys=False)
        logger.info(f"Final configuration saved to {config_out}")
    elif only_config:
        from omegaconf import OmegaConf
        print(yaml.dump(OmegaConf.to_container(cfg, resolve=True)))
    
    if only_config:
        logger.info("Only configuration output requested. Exiting.")
        return
    
    # If dryrun, we just print the configuration and exit
    if dryrun:
        logger.info("Dry run enabled. The training will not be executed.")

    # Setup the trust remote code globally

    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")


    # 1 - `verl` initializes a `SGLangRollout` module during rollout, which is used to evaluate/generate samples
    # 2 - `SGLangRollout` will initialize `Engine`, and further initialize a `torch.distributed.DeviceMesh`, used to support Tensor Parallelism (TP)
    # 3 - `DeviceMesh.init()` internally checks the free GPU memory of all participating devices. If the difference is too large (more than ~10%),
    #      it raises an error and aborts the execution.
    os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"

    if not ray.is_initialized():
        kwargs = {
            "runtime_env": {
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "INFO" if debug else "WARN",
                    "VLLM_LOGGING_LEVEL": "INFO" if debug else "ERROR",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1",
                },
                "py_executable": sys.executable, # Use the same Python executable (notably for venvs)
            },
        }

        if (
            cuda_is_available()
            and cfg.global_profiler.tool == "nsys"
            and cfg.global_profiler.get("steps", None) is not None
            and len(cfg.global_profiler.steps.get("steps", [])) > 0
        ):
            from verl.utils.import_utils import is_nvtx_available

            assert is_nvtx_available(), "nvtx is required for nsys profiling, please install it via `pip install nvtx`"
            nsight_options = OmegaConf.to_container(
                cfg.global_profiler.global_tool_config.nsys.controller_nsight_options, resolve=True
            )
            logger.info(f"Enabling nsight profiling with options: {nsight_options}")
            kwargs["runtime_env"]["nsight"] = nsight_options
        
        if cfg.ray.get("timeline_json_file", None) is not None:
            logger.info(f"Ray timeline will be saved to {cfg.ray.timeline_json_file}")
            ray.timeline(filename=cfg.ray.timeline_json_file)

        if cfg.ray.num_cpus is not None:
            kwargs["num_cpus"] = cfg.ray.num_cpus

        if debug:
            logger.info("Ray debug mode is enabled.")
            kwargs["runtime_env"]["env_vars"]["RAY_DEBUG"] = "1"
            kwargs["runtime_env"]["env_vars"]["RAY_DEBUG_POST_MORTEM"] = "1"
        else:
            logger.info("Ray debug mode is disabled.")
            kwargs["runtime_env"]["env_vars"]["RAY_DEBUG"] = "0"
            kwargs["runtime_env"]["env_vars"]["RAY_DEBUG_POST_MORTEM"] = "0"

        if cfg.ray.dashboard is not None:
            host, port = split_host_port(cfg.ray.dashboard, default_port=8265)
            kwargs["dashboard_host"] = host
            kwargs["dashboard_port"] = port
            kwargs["include_dashboard"] = True
        else:
            kwargs["include_dashboard"] = False

        ray.init(
            **kwargs
        )
    else:
        logger.warning("Ray is already initialized. Skipping ray.init(), ray configuration will be partially ignored.")
        
    from multimeditron.verl import TaskRunner
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(cfg, trust_remote_code=trust_remote_code, verbose=verbose, dryrun=dryrun))

