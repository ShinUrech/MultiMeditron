import time
import asyncio

def _initialize_python_executor_pool(cfg):
    from multimeditron.tools.python_exec import init_nsjail_python_executor
    init_nsjail_python_executor(cfg)
    time.sleep(10)

    # Test code execution
    from multimeditron.tools.python_exec import get_nsjail_python_executor_pool
    pool = get_nsjail_python_executor_pool()
    _elem = pool.execute.remote({
        "code": "print('Hello from nsjail!')",
    })
    elem = asyncio.get_event_loop().run_until_complete(_elem)
    print(elem)

SERVICE_INITIALIZERS = {
    "nsjail-python-exec-pool": _initialize_python_executor_pool,
}

def initialize_services(cfg):
    for service_cfg in cfg:
        name = service_cfg.name
        if not name in SERVICE_INITIALIZERS:
            raise NotImplementedError(f"No initializer for service {name}. Available: {list(SERVICE_INITIALIZERS.keys())}")
        
        print(f"Initializing service {name} with config {service_cfg}")
        init_fn = SERVICE_INITIALIZERS[name]
        init_fn(service_cfg)