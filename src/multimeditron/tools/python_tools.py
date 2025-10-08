from multimeditron.tools.python_exec import get_nsjail_python_executor_pool, NsJailPythonExecutorPool
from omegaconf import OmegaConf
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op
from typing import Optional, Tuple
from uuid import uuid4

class PythonTools(BaseTool):
    """
    _tool_schema = OpenAIFunctionToolSchema({
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Execute python code in a secure sandboxed environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to execute."
                    },
                },
                "required": ["code"]
            },
        }
    })
    """
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_map = {}
        self._executor_pool: NsJailPythonExecutorPool = get_nsjail_python_executor_pool()

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_map[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict,
        **kwargs
    ) -> Tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)
        
        result = await self._executor_pool.execute.remote(code)
        self._instance_map[instance_id]["response"] = result
        return ToolResponse(text=result), None, None

    async def calc_reward(self, instance_id, **kwargs) -> float:
        return self._instance_map[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_map[instance_id]

