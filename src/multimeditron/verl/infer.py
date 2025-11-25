import requests
import time
import asyncio
import os
import logging
import openai

logger = logging.getLogger(__name__)

def get_address(required: bool = False) -> str:
    node_address = os.getenv("SG_INFER_ADDRESS")
    if node_address is None and required:
        logger.error("Environment variable 'SG_INFER_ADDRESS' is not set. Please set it to the address of the SGLang inference server.")
        raise ValueError("Environment variable 'SG_INFER_ADDRESS' is not set.")
    return node_address

def get_api_key(required: bool = False) -> str:
    api_key = os.getenv("SG_INFER_API_KEY")
    if api_key is None and required:
        logger.error("Environment variable 'SG_INFER_API_KEY' is not set. Please set it to the API key for the SGLang inference server.")
        raise ValueError("Environment variable 'SG_INFER_API_KEY' is not set.")
    return api_key

def wait_for_sglang_server(address: str, timeout: int = 600) -> None:
    start_time = time.time()
    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"http://{address}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("SGLang server is healthy and ready.")
                    return
            except requests.RequestException as e:
                logger.debug(f"SGLang server health check failed: {e}")

            if time.time() - start_time > timeout:
                logger.error(f"Timeout reached while waiting for SGLang server at {address} to become healthy.")
                raise TimeoutError(f"SGLang server at {address} did not become healthy within {timeout} seconds.")

            time.sleep(2)

async def create_async_client(required: bool = True) -> openai.AsyncClient:
    # Get the SGLang server address from environment variable
    address = get_address(required=required)
    if address is None:
        return None
    
    kwargs = {}

    api_key = get_api_key()
    if api_key is not None:
        kwargs["api_key"] = api_key

    # Create the OpenAI client
    client = openai.AsyncClient(base_url=f"http://{address}/v1", **kwargs)
    return client

def create_sync_client(required: bool = True) -> openai.Client:
    # Get the SGLang server address from environment variable
    address = get_address(required=required)
    if address is None:
        return None
    
    kwargs = {}

    api_key = get_api_key()
    if api_key is not None:
        kwargs["api_key"] = api_key

    # Create the OpenAI client
    client = openai.Client(base_url=f"http://{address}/v1", **kwargs)
    return client
