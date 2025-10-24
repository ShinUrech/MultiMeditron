from verl.utils.reward_score import gsm8k
from verl.interactions.base import BaseInteraction
from uuid import uuid4
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MultiTurnInteraction(BaseInteraction):
    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "turns": [],
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id
    
    async def generate_response(
        self,
        instance_id,
        messages,
        **kwargs
    ) -> Tuple[bool, str, float, dict]:
        content = ""

        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item["role"] == "assistant":
                content = item.get('content')
                break
        
        reward = await self.calculate_score(instance_id)

        if reward == 1.0:
            response = "Your response is correct!"
            should_terminate_sequence = True
        else:
            response = "Your response is incorrect. Please try again."
            should_terminate_sequence = False
        
        return should_terminate_sequence, response, reward, {}

    async def calculate_score(self, instance_id: str) -> float:
        return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]