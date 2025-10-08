from typing import Optional, Tuple
from omegaconf import OmegaConf
from itertools import cycle
import asyncio
import tempfile
import subprocess
import shutil
import ray
import time
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NsJailPythonExecutor:
    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        self.nsjail_path = NsJailPythonExecutor._ensure_str(cfg.nsjail.path)
        self.python_path = NsJailPythonExecutor._ensure_str(cfg.python.path)
        self.env_vars = {
            NsJailPythonExecutor._ensure_str(k): NsJailPythonExecutor._ensure_str(v)
            for k, v in cfg.nsjail.envs.items()
        }
        self.readonly_mounts = cfg.nsjail.ro_mounts

        self.max_limits = NsJailPythonExecutor._extract_hard_limits(cfg)
        self.allow_network = cfg.nsjail.get("allow_network", False)

        self._ensure_path_executable(self.nsjail_path)
        self._ensure_path_executable(self.python_path)

        # Exec python getsitepackages to get them
        try:
            stdout = subprocess.check_output(
                [self.python_path, "-c", "import sys; print(sys.base_prefix)"],
                text=True,
                timeout=5
            )
            python_base_prefix = stdout.strip()
            print("Python prefix inside jail:", python_base_prefix)
            self.readonly_mounts.append(python_base_prefix)
        except Exception as e:
            logger.error("Error getting python base_prefix: %s", str(e))
            raise RuntimeError(f"Error getting python base_prefix: {str(e)}")
        
        # Get a list of all site-packages dirs
        try:
            stdout = subprocess.check_output(
                [self.python_path, "-c", "import site; print('\\n'.join(site.getsitepackages()))"],
                text=True,
                timeout=5
            )
            site_packages = stdout.strip().splitlines()
            print("Site-packages inside jail:", site_packages)
            self.readonly_mounts.extend(site_packages)
        except Exception as e:
            logger.error("Error getting python site-packages: %s", str(e))
            raise RuntimeError(f"Error getting python site-packages: {str(e)}")
        
        # Exec python path to get sys-executable real path (not a symlink)
        try:
            stdout = subprocess.check_output(
                [self.python_path, "-c", "import sys; print(sys.executable)"],
                text=True,
                timeout=5
            )
            self.python_path = stdout.strip()
            print("Python executable inside jail:", self.python_path)
            self.readonly_mounts.append(self.python_path)
            self.env_vars["PYTHONPATH"] = os.pathsep.join(site_packages)
        except Exception as e:
            logger.error("Error getting python executable: %s", str(e))
            raise RuntimeError(f"Error getting python executable: {str(e)}")

        self.readonly_mounts = [NsJailPythonExecutor._ensure_str(p) for p in self.readonly_mounts]
        print("Final readonly mounts:", self.readonly_mounts)
        print("Final env vars:", self.env_vars)

    @staticmethod
    def _ensure_str(x):
        if not isinstance(x, str):
            return str(x)
        return x
    
    @staticmethod
    def _extract_hard_limits(cfg) -> dict:
        return {
            "max_rlimit_as": cfg.nsjail.get("max_rlimit_as", 256 * 1024 * 1024),  # 256MB
            "max_rlimit_cpu": cfg.nsjail.get("max_rlimit_cpu", 2),  # 2 seconds of CPU time
            "max_time_limit": cfg.nsjail.get("max_time_limit", 5), # 5 seconds of wall time
            "max_open_fds": cfg.nsjail.get("max_open_fds", 16),  # max number of open file descriptors is 16
        }

    @staticmethod
    def _compute_limits(max_rlimit_as: int, max_rlimit_cpu: float, max_time_limit: float, max_open_fds: int,
                       rlimit_as: Optional[int] = None,
                       rlimit_cpu: Optional[float] = None,
                       time_limit: Optional[float] = None,
                       open_fds: Optional[int] = None) -> dict:
        """
        Estimate resources based on cfg and execute_args.
        This is a stub, you can implement your own logic.
        """
        
        rlimit_as = min(rlimit_as, max_rlimit_as) if rlimit_as is not None else max_rlimit_as
        rlimit_cpu = min(rlimit_cpu, max_rlimit_cpu) if rlimit_cpu is not None else max_rlimit_cpu
        time_limit = min(time_limit, max_time_limit) if time_limit is not None else max_time_limit
        open_fds = min(open_fds, max_open_fds) if open_fds is not None else max_open_fds

        return {
            "rlimit_as": int(rlimit_as),
            "rlimit_cpu": int(rlimit_cpu),
            "time_limit": float(time_limit),
            "open_fds": int(open_fds),
        }

    def _build_nsjail_cmd(self,
                          workdir: str,
                          code_filename: str,
                          rlimit_as: Optional[int] = None,
                          rlimit_cpu: Optional[int] = None,
                          time_limit: Optional[int] = None,
                          open_fds: Optional[int] = None) -> list:
        """
        Build nsjail CLI args; You may need to adapt bindmount/chroot paths based on your environment. This
        uses a conservative set of flags most nsjail builds support.
        """
        # Use provided or default values
        limits = NsJailPythonExecutor._compute_limits(
            **self.max_limits,
            rlimit_as=rlimit_as, rlimit_cpu=rlimit_cpu, time_limit=time_limit, open_fds=open_fds
        )

        # Example flags:
        #  --chroot: we set chroot to the workdir (empty dir)
        #  --user/--group: drop priveleges inside the jail (use nobody/nogroup uid/gid)
        #  --disable_proc: avoid mounting /proc inside the jail (safer)
        #  --rlimit_as / --rlimit_cpu: limit memory and CPU time
        #  --time_limit: limit wall time (in seconds)
        #  --rlimit_fsize: limit max file size (5MB here)
        #  --cwd: set working directory inside the jail (we set to / which is the chroot)
        #  --   : end of nsjail args, beginning of command to run inside the jail
        uid = 65534  # nobody
        gid = 65534  # nogroup

        cmd = [
            self.nsjail_path,
            "--mode", "o", # "once" mode
            "--chroot", workdir,
            "--cwd", "/",
            "--user", str(uid),
            "--group", str(gid),
            "--disable_proc",
            "--rlimit_as", str(limits['rlimit_as']),
            "--rlimit_cpu", str(limits['rlimit_cpu']),
            "--time_limit", str(limits['time_limit']),
            "--rlimit_fsize", str(5 * 1024 * 1024),  # 5MB
            "--rlimit_nofile", str(limits['open_fds']),
            "--keep_caps", # keep capabilities false? nsjail may drop them anyway
        ]

        # Bind-mount the code file into the chroot so python inside the jail can access it.
        # We mount it at /code.py inside the chroot.
        src_code_path = os.path.join(workdir, code_filename)  # absolute on host
        cmd += ["--bindmount", f"{src_code_path}:/code.py:ro"]

        # Bind-mount the python interpreter (if needed). If interpreter is available inside chroot, skip this.
        # Note: bind-mounting interpreter may not be sufficient if shared libs are needed (more bind mounts required).
        for ro_path in self.readonly_mounts:
            if os.path.exists(ro_path):
                cmd += ["--bindmount", f"{ro_path}:{ro_path}:ro"]
            else:
                logger.warning("Readonly mount path does not exist and will be skipped: %s", ro_path)
                print("Readonly mount path does not exist and will be skipped:", ro_path)

        # Environment variables to pass inside the jail
        for k, v in self.env_vars.items():
            if not isinstance(v, str):
                v = str(v)
            cmd += ["--env", f"{k}={v}"]

        # As final part, the program to execute inside the jail
        cmd += ["--", f"{self.python_path}", "/code.py"]
        return cmd
    
    @staticmethod
    def _prepare_workdir(user_code: str) -> Tuple[str, str]:
        """
        Create a temporary directory and write code to a file.
        We will chroot into that directory. Keep it minimal.
        Returns (workdir_abs_path, filename).
        """
        tmpdir = tempfile.mkdtemp(prefix="nsjail_exec_")
        # make a code filename with safe name
        code_filename = "code.py"
        code_path = os.path.join(tmpdir, code_filename)
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(user_code)
        # ensure minimal permissions
        os.chmod(code_path, 0o444)  # read-only r--r--r--
        return tmpdir, code_filename

    @staticmethod
    def _cleanup_workdir(path: str):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    @staticmethod
    def _ensure_path_executable(path: str):
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"The path '{path}' is not an executable file or is not accessible. Install it and ensure it is accessable from the worker nodes.")

    def execute(self, user_code: str, *,
                stdin: Optional[str] = None,
                rlimit_as: Optional[int] = None,
                rlimit_cpu: Optional[int] = None,
                time_limit: Optional[int] = None,
                open_fds: Optional[int] = None,
                wall_timeout: Optional[int] = None) -> dict:
        """
        Execute user_code in nsjail. Returns a dictionary:
          { "exit_code": int,
            "stdout": str,
            "stderr": str,
            "timed_out": bool,
            "duration_s": float,
            "meta": { ... } }
        - wall_timeout (seconds) is enforced by this function as a python-level timeout
          (nsjail --time_limit is still applied inside the jail).
        """
        start_ts = time.time()
        workdir, code_filename = self._prepare_workdir(user_code)
        cmd = self._build_nsjail_cmd(workdir, code_filename,
                                     rlimit_as=rlimit_as,
                                     rlimit_cpu=rlimit_cpu,
                                     time_limit=time_limit,
                                     open_fds=open_fds)
        timed_out = False

        try:
            # Start nsjail process
            logger.debug("Running nsjail command: %s", " ".join(cmd))
            print("Running nsjail command:", " ".join(cmd))
            proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid # helps killing the whole process group on timeout
            )

            try:
                stdout, stderr = proc.communicate(input=stdin, timeout=wall_timeout)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                # Kill the whole process group
                try:
                    os.killpg(proc.pid, 9) # SIGKILL
                except Exception:
                    proc.kill()
                stdout = ""
                stderr = "killed by wall timeout"
                exit_code = -1
            
        except Exception as e:
            stderr = f"executor error: {str(e)}"
            stdout = ""
            exit_code = -1
            logger.exception("Error running nsjail: %s", str(e))
            print("Error running nsjail:", str(e))
        finally:
            duration = time.time() - start_ts
            # best-effort cleanup
            self._cleanup_workdir(workdir)
            logger.debug("Cleaned up workdir %s", workdir)
            print("Cleaned up workdir", workdir)
                
        return {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "duration_s": duration,
            "meta": {
                "nsjail_path": self.nsjail_path,
                "python_path": self.python_path,
                "env_vars": self.env_vars,
                "ro_mounts": self.readonly_mounts,
                "rlimit_as": rlimit_as,
                "rlimit_cpu": rlimit_cpu,
                "time_limit": time_limit,
                "open_fds": open_fds,
            }
        }

@ray.remote(name="nsjail-python-exec-pool", namespace="default")
class NsJailPythonExecutorPool:
    def __init__(self, executors: list, cfg):
        
        # List of NsJailPythonExecutor actors
        assert len(executors) > 0, "At least one executor is required"
        self.executors = executors
        
        # Simple round-robin iterator over executors
        self.rr_cycle = cycle(executors)
        self.semaphore = asyncio.Semaphore(len(executors))
    
    async def execute(self,
                      *args,
                      **kwargs) -> dict:
        async with self.semaphore:
            # Pick an actor
            executor = next(self.rr_cycle)

            # Dynamically adjust launch options
            # Ray doesn't allow changing resources on existing actor methods, so we launch as a new task with custom resources
            # request
            task = executor.execute.remote(*args, **kwargs)

            # Await result
            return await task

    async def batch_execute(self, tasks):
        """Run a batch of tasks concurrently."""
        coros = [self.execute(**task) for task in tasks]
        results = await asyncio.gather(*coros)
        return results
    
    def shutdown(self):
        for executor in self.executors:
            try:
                ray.kill(executor, no_restart=True)
            except Exception as e:
                print("Error killing executor", executor, ":", str(e))

def init_nsjail_python_executor(cfg) -> NsJailPythonExecutorPool:
    """
    Initialize a Ray-based pool of NsJailPythonExecutor workers.
    """
    # Check if an instance of the actor already exists
    limits = NsJailPythonExecutor._extract_hard_limits(cfg)

    parallelism = int(cfg.pool.parallelism)
    cpu_per_job = float(cfg.pool.cpu_per_job)
    mem_per_job = int(cfg.pool.get("mem_per_job", 200)) * 1024 * 1024  # in MB
    mem_per_job += limits['max_rlimit_as']  # add max memory limit to actor memory

    # Create Ray actor replicas
    executors = [
        ray.remote(NsJailPythonExecutor).options(
            name=f"nsjail-python-exec-{i}",
            num_cpus=cpu_per_job,
            memory=mem_per_job,
            max_concurrency=1,
            max_restarts=3, # restart up to 3 times on failure
            max_task_retries=1,  # retry once
        ).remote(cfg) for i in range(parallelism)
    ]

    pool = NsJailPythonExecutorPool.remote(executors, cfg)
    ray.put(pool)
    return pool

def get_nsjail_python_executor_pool():
    try:
        actor = ray.get_actor("nsjail-python-exec-pool", namespace="default")
        return actor
    except ValueError as e:
        print(e)
        return None
