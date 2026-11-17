"""Stdlib-only entry point copied into a sandbox Job's read-only control mount.

Runs in the tool image, which must provide python3 and /bin/sh. No application
imports, credentials, or Kubernetes client are present in the experiment.
"""

import json
import os
import resource
import signal
import subprocess
import sys


def main(control="/runner", output="/runner-output", workdir="/work"):
    with open(os.path.join(control, "request.json"), encoding="utf-8") as stream:
        request = json.load(stream)
    # RLIMIT_NPROC complements the cluster's kubelet podPidsLimit. Operators
    # must still configure that cgroup limit on sandbox nodes.
    resource.setrlimit(resource.RLIMIT_NPROC, (request["pids"], request["pids"]))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    size = request["max_file_bytes"]
    resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
    with open(os.path.join(output, "stdout"), "wb") as stdout, open(
        os.path.join(output, "stderr"), "wb"
    ) as stderr:
        try:
            process = subprocess.Popen(
                ["/bin/sh", "-lc", request["script"]],
                cwd=workdir,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        except OSError as exc:
            stderr.write(
                f"Sandbox launcher could not start the command: {exc}\n".encode()
            )
            return 126

        def terminate(_signal, _frame):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise SystemExit(143)

        signal.signal(signal.SIGTERM, terminate)
        try:
            code = process.wait(timeout=request["timeout_seconds"])
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return 124
        finally:
            # A script must not leave background processes mutating artifacts.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
