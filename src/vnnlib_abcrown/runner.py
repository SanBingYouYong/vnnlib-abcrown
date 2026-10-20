"""Running α,β-CROWN on a VNN-LIB 1.0 problem."""

import os
import re
import subprocess
import sys

# Used when the user passes no `--config` of their own.
#
# Patches mode does not support MaxPool, and the networks this was written
# for are small enough for matrix mode.
DEFAULT_CONFIG = """\
general:
  device: cuda
  conv_mode: matrix
"""

# For few inputs, splitting the input space is far more effective than
# splitting ReLUs (ACAS Xu property 2: 3.5 s instead of a timeout). This is
# α,β-CROWN's own VNN-COMP configuration for ACAS Xu.
INPUT_SPLIT_CONFIG = """\
general:
  device: cuda
  conv_mode: matrix
solver:
  batch_size: 16384
  bound_prop_method: crown
bab:
  branching:
    method: naive
    input_split:
      enable: true
      reorder_bab: true
  clip_n_verify:
    clip_input_domain:
      enabled: true
attack:
  pgd_order: after
  pgd_restarts: 10000
  pgd_restart_when_stuck: true
"""

# Problems with at most this many input variables use INPUT_SPLIT_CONFIG.
INPUT_SPLIT_MAX_INPUTS = int(os.environ.get("VNNLIB_ABCROWN_INPUT_SPLIT_MAX", "16"))


# Extra arguments for the first run and for each retry after CUDA out-of-memory.
OOM_RETRIES = [
    [],
    ["--batch_size", "64", "--no_auto_enlarge_batch_size"],
    ["--batch_size", "8", "--no_auto_enlarge_batch_size"],
]


def abcrown_dir():
    if "ABCROWN_DIR" in os.environ:
        return os.environ["ABCROWN_DIR"]
    root = os.environ.get("ABCROWN_ROOT", os.path.expanduser("~/alpha-beta-CROWN"))
    return os.path.join(root, "complete_verifier")


class Runner:
    def __init__(self, work_dir, extra_args):
        self.work_dir = work_dir
        self.extra_args = list(extra_args)
        self.python = os.environ.get("ABCROWN_PYTHON", sys.executable)

    def __call__(self, onnx_path, vnnlib_path, num_inputs):
        results_file = os.path.join(self.work_dir, "result.txt")
        cex_file = os.path.join(self.work_dir, "cex.txt")
        log_file = os.path.join(self.work_dir, "abcrown.log")
        args = self.extra_args
        if not any(a == "--config" or a.startswith("--config=") for a in args):
            config_path = os.path.join(self.work_dir, "config.yaml")
            with open(config_path, "w") as f:
                f.write(INPUT_SPLIT_CONFIG if num_inputs <= INPUT_SPLIT_MAX_INPUTS
                        else DEFAULT_CONFIG)
            args = ["--config", config_path] + args
        root = abcrown_dir()
        cmd = [self.python, os.path.join(root, "abcrown.py")] + args + [
            "--onnx_path", onnx_path,
            "--vnnlib_path", vnnlib_path,
            "--results_file", results_file,
            "--cex_path", cex_file,
            "--save_adv_example",
        ]
        # α,β-CROWN enlarges its batch size until memory runs out, and the GPU
        # may be shared. On CUDA out-of-memory, retry with smaller fixed batches.
        for attempt, extra in enumerate(OOM_RETRIES):
            if attempt:
                print("[vnnlib-abcrown] α,β-CROWN ran out of GPU memory; retrying with "
                      + " ".join(extra), file=sys.stderr, flush=True)
            if os.path.exists(results_file):
                os.remove(results_file)
            with open(log_file, "w") as lf:
                proc = subprocess.run(cmd + extra, cwd=root, stdout=lf, stderr=subprocess.STDOUT)
            if os.path.exists(results_file):
                break
            with open(log_file) as lf:
                if "CUDA out of memory" not in lf.read():
                    break

        status = None
        if os.path.exists(results_file):
            with open(results_file) as rf:
                status = rf.readline().strip()
        if status is None:
            with open(log_file) as lf:
                tail = lf.read()[-3000:]
            raise RuntimeError(
                f"α,β-CROWN exited with code {proc.returncode} and wrote no result.\n{tail}")

        if status in ("unsat", "safe"):
            return "unsat", None
        if status in ("sat", "unsafe"):
            with open(cex_file) as cf:
                content = cf.read()
            xs = {int(m.group(1)): float(m.group(2))
                  for m in re.finditer(r"\(X_(\d+)\s+([-+eE0-9.]+)\)", content)}
            return "sat", [xs[i] for i in range(num_inputs)]
        print(f"[vnnlib-abcrown] α,β-CROWN returned '{status}'; reporting 'timeout'",
              file=sys.stderr, flush=True)
        return "timeout", None
