import time

def _initialize_python_executor_pool(cfg):
    from multimeditron.tools.python_exec import init_nsjail_python_executor
    return init_nsjail_python_executor(cfg)


SERVICE_INITIALIZERS = {
    "nsjail-python-exec-pool": _initialize_python_executor_pool,
}

def initialize_services(cfg):
    service_handles = {}
    for service_cfg in cfg:
        name = service_cfg.name
        if not name in SERVICE_INITIALIZERS:
            raise NotImplementedError(f"No initializer for service {name}. Available: {list(SERVICE_INITIALIZERS.keys())}")
        
        print(f"Initializing service {name} with config {service_cfg}")
        init_fn = SERVICE_INITIALIZERS[name]
        service_handles[name] = init_fn(service_cfg)
        time.sleep(1)  # Give some time for the service to start up properly

    # Wait a bit to ensure all services are up and running
    return service_handles