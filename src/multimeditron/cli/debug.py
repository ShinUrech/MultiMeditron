from multimeditron.cli import EPILOG, CONFIG_PATH, main_cli
from multimeditron.tools import NsJailExecutor
from multimeditron.utils import get_torch_dtype
from datasets import load_dataset
from omegaconf import OmegaConf
import ray
import logging
from ray import serve
from fastapi import Request

logger = logging.getLogger(__name__)


@serve.deployment(num_replicas=2)  # scale horizontally if needed
class PyExecService:
    def __init__(self, cfg):
        # create a single NsJailExecutor actor for each replica
        self.executor = NsJailExecutor.remote(cfg)

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

            # execute code in nsjail
            result = await self.executor.execute.remote(user_code=code,)

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
        }
    })

    # Start ray if not already running
    ray.init(address="auto", namespace="serve")

    # Deploy service
    app = PyExecService.bind(cfg)
    serve.run(app, blocking=True)

    print("🚀 Ray Serve running at http://127.0.0.1:8000/PyExecService")
