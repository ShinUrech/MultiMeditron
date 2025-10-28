from typing import Optional
from multimeditron.cli import EPILOG, CONFIG_PATH, main_cli
from multimeditron.tools.python_exec import init_nsjail_python_executor, NsJailPythonExecutorPool
from omegaconf import OmegaConf
from ray import serve
from fastapi import Request
import os
import sys
import ray
import click
import logging

logger = logging.getLogger(__name__)


@serve.deployment(num_replicas=2)  # scale horizontally if needed
class PyExecService:
    def __init__(self, cfg, executor_pool: NsJailPythonExecutorPool):
        # create a single NsJailExecutor actor for each replica
        self.executor_pool = executor_pool

    async def __call__(self, request: Request):
        """
        HTTP handler:
        - expects POST with JSON body {"code": "print('hello')", "timeout": 5}
        - runs code in nsjail
        - returns JSON result
        """
        data = await request.json()
        try:
            code = data.get("code", "")
            stdin = data.get("stdin", None)

            # execute code in nsjail
            result = await self.executor_pool.execute.remote(
                user_code=code,
                stdin=stdin,
            )

            return result
        except Exception as e:
            logger.exception("Error executing code")
            return {"error": str(e)}

@main_cli.command("serve-python", epilog=EPILOG, context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to the configuration file(s) in YAML format.")
@click.pass_context
def serve_nsjail_python(ctx, config: Optional[str] = None):
    """
    Create and run a Ray Serve service that executes Python code snippets in isolated nsjail environments.
    """

    from hydra import initialize_config_dir, compose

    if config is None:
        with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.2"):
            cfg = compose(config_name="nsjail-python-exec-pool", overrides=ctx.args)
    else:
        config_dir = os.path.dirname(os.path.abspath(config))
        config_name = os.path.basename(config)
        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            cfg = compose(config_name=config_name, overrides=ctx.args)
    
    # Start ray if not already running
    if not ray.is_initialized():
        kwargs = {
            "runtime_env": {
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                },
                "py_executable": sys.executable, # Use the same Python executable (notably for venvs)
            }
        }
        ray.init(address="auto", namespace="serve", **kwargs)

    # Deploy service
    executor_pool = init_nsjail_python_executor(cfg)
    app = PyExecService.bind(cfg, executor_pool)
    serve.run(app, blocking=True)

    print("🚀 Ray Serve running at http://127.0.0.1:8000/PyExecService")
