from typing import Dict, List, Union
from transformers import PreTrainedTokenizer
import aiohttp
import json
import logging

logger = logging.getLogger(__name__)


async def generate_aiohttp(
    router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
    prompts: Union[str, List[Dict[str, str]]], 
    sampling_params: dict
):
    # Construct conversation from prompts
    if isinstance(prompts, str):
        prompt_dict = {"role": "user", "content": prompts}
    else:
        assert all(all(k in ['role', 'content'] and v in ['user', 'system', 'assistant'] if v == 'role' else True\
                       for (k,v) in x) for x in prompts), "Each prompt must have 'role' and 'content'"
        prompt_dict = prompts
    
    # Tokenize prompts
    token_ids = reward_model_tokenizer.apply_chat_template(prompt_dict, tokenize=True, add_generation_prompt=True)

    payload = {
        "input_ids": token_ids,
        "sampling_params": sampling_params,
    }
    url = f"http://{router_address}/generate"
    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        async with session.post(url, json=payload) as resp:
            output = await resp.text()
            try:
                output = json.loads(output)
                return output
            except Exception:
                logger.error(f"Failed to parse JSON response: {output}")
                return {}
    finally:
        await session.close()