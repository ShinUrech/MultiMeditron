from typing import Optional, Tuple
import tempfile
import subprocess
import random
import shutil
import ray
import time
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@ray.remote
class NsJailExecutor:
    def __init__(self, cfg):
        self.nsjail_path = cfg.nsjail.path
        self.python_path = cfg.python.path
        self.readonly_mounts = [
            # self.python_path,
            "/lib",
            "/lib64",
            "/usr/lib",
            "/usr/local/lib",
            "/bin/sh",
            "/dev/random",
            "/etc/ld.so.cache",
        ]

        self.max_rlimit_as = cfg.nsjail.get("max_rlimit_as", 256 * 1024 * 1024)  # 256MB
        self.max_rlimit_cpu = cfg.nsjail.get("max_rlimit_cpu", 2)  # 2 seconds of CPU time
        self.max_time_limit = cfg.nsjail.get("max_time_limit", 5) # 5 seconds of wall time
        self.max_open_fds = cfg.nsjail.get("max_open_fds", 16)  # max number of open file descriptors is 16

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
            logger.info("Python prefix inside jail: %s", python_base_prefix)
            print(python_base_prefix)
            self.readonly_mounts.append(python_base_prefix)
        except Exception as e:
            logger.error("Error getting python base_prefix: %s", str(e))
            raise RuntimeError(f"Error getting python base_prefix: {str(e)}")



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
        rlimit_as = rlimit_as if rlimit_as is not None else self.max_rlimit_as
        rlimit_cpu = rlimit_cpu if rlimit_cpu is not None else self.max_rlimit_cpu
        time_limit = time_limit if time_limit is not None else self.max_time_limit
        open_fds = open_fds if open_fds is not None else self.max_open_fds

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
            "--rlimit_as", str(rlimit_as),
            "--rlimit_cpu", str(rlimit_cpu),
            "--time_limit", str(time_limit),
            "--rlimit_fsize", str(5 * 1024 * 1024),  # 5MB
            "--rlimit_nofile", str(open_fds),
            "--keep_caps", # keep capabilities false? nsjail may drop them anyway
        ]

        # Bind-mount the code file into the chroot so python inside the jail can access it.
        # We mount it at /code.py inside the chroot.
        src_code_path = os.path.join(workdir, code_filename)  # absolute on host
        cmd += ["--bindmount", f"{src_code_path}:/code.py:ro"]

        # Bind-mount the python interpreter (if needed). If interpreter is available inside chroot, skip this.
        # Note: bind-mounting interpreter may not be sufficient if shared libs are needed (more bind mounts required).
        if os.path.exists(self.python_path):
            # python_path = os.path.dirname(os.path.abspath(self.python_path))
            # cmd += [
            #     "--bindmount", f"{self.python_path}:/usr/local/bin/python:ro",
            # ]
            for ro_path in self.readonly_mounts:
                if os.path.exists(ro_path):
                    cmd += ["--bindmount", f"{ro_path}:{ro_path}:ro"]
                else:
                    logger.warning("Readonly mount path does not exist and will be skipped: %s", ro_path)

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
            proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid # helps killing the whole process group on timeout
            )

            try:
                stdout, stderr = proc.communicate(timeout=wall_timeout)
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
        finally:
            duration = time.time() - start_ts
            # best-effort cleanup
            self._cleanup_workdir(workdir)
            logger.debug("Cleaned up workdir %s", workdir)
                
        return {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "duration_s": duration,
            "meta": {
                # "nsjail_path": self.nsjail_path,
                # "python_path": self.python_path,
                "rlimit_as": rlimit_as,
                "rlimit_cpu": rlimit_cpu,
                "time_limit": time_limit,
                "open_fds": open_fds,
            }
        }

