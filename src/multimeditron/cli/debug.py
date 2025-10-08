from multimeditron.cli import EPILOG, CONFIG_PATH, main_cli
from multimeditron.tools.python_exec import init_nsjail_python_executor, NsJailPythonExecutorPool
from multimeditron.utils import get_torch_dtype
from datasets import load_dataset
from omegaconf import OmegaConf
from ray import serve
from fastapi import Request
import ray
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

@main_cli.command("serve")
def _serve():
    cfg = OmegaConf.create({
        "python": {
            "path": "/opt/conda/envs/py312/bin/python3.12",  # path to python binary inside the jail
        },
        "nsjail": {
            "path": "/usr/bin/nsjail",
            "max_rlimit_as": 64 * 1024 * 1024,  # 64MB
            "max_rlimit_cpu": 2,  # 2 seconds of CPU time
            "max_time_limit": 5, # 5 seconds of wall time
            "max_open_fds": 16,  # max number of open file descriptors is 16
            "allow_network": False,
            "ro_mounts": [
                "/lib",
                "/lib64",
                "/usr/lib",
                "/usr/lib64",
                "/bin/sh",
                "/dev/random",
                "/etc/ld.so.cache",
            ],
            "envs": {
                "LANG": "en_US.UTF-8",
                "OMP_NUM_THREADS": "1",  # limit to 1 thread
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONIOENCODING": "utf-8:strict",
            },
        },
        "pool": {
            "parallelism": 4,  # number of concurrent executors in the pool
            "cpu_per_job": 1,  # number of CPUs per executor
        },
    })

    # Start ray if not already running
    ray.init(address="auto", namespace="serve")

    # Deploy service
    executor_pool = init_nsjail_python_executor(cfg)
    app = PyExecService.bind(cfg, executor_pool)
    serve.run(app, blocking=True)

    print("🚀 Ray Serve running at http://127.0.0.1:8000/PyExecService")
