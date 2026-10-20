"""The α,β-CROWN runner, with a fake abcrown.py."""

import os
import sys

from vnnlib_abcrown.runner import Runner

FAKE = r'''
import sys
args = sys.argv[1:]
res = args[args.index("--results_file") + 1]
cex = args[args.index("--cex_path") + 1]
open("calls.txt", "a").write(" ".join(args) + "\n")
if "--batch_size" not in args or args[args.index("--batch_size") + 1] != "8":
    print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1 GiB")
    sys.exit(1)
open(res, "w").write("sat\n")
open(cex, "w").write("((X_0 0.25)\n(X_1 -1.5e-3)\n(Y_0 1.0))\n")
'''


def test_retries_with_smaller_batches_after_cuda_oom(tmp_path, monkeypatch):
    root = tmp_path / "abcrown"
    root.mkdir()
    (root / "abcrown.py").write_text(FAKE)
    monkeypatch.setenv("ABCROWN_DIR", str(root))
    monkeypatch.setenv("ABCROWN_PYTHON", sys.executable)
    work = tmp_path / "work"
    work.mkdir()
    status, z = Runner(str(work), [])("net.onnx", "spec.vnnlib", 2)
    assert (status, z) == ("sat", [0.25, -1.5e-3])
    calls = (root / "calls.txt").read_text().splitlines()
    assert len(calls) == 3
    assert "--batch_size 64 --no_auto_enlarge_batch_size" in calls[1]
    # Two inputs: the input-splitting configuration is used.
    config = [a for a in calls[0].split() if a.endswith("config.yaml")][0]
    assert "input_split" in open(config).read()


GIVE_UP = r'''
import sys
args = sys.argv[1:]
open("calls.txt", "a").write(" ".join(args) + "\n")
open(args[args.index("--results_file") + 1], "w").write("timeout\n")
'''


def test_timeout_is_shared_by_runs_and_distinguished_from_giving_up(tmp_path, monkeypatch):
    root = tmp_path / "abcrown"
    root.mkdir()
    (root / "abcrown.py").write_text(GIVE_UP)
    monkeypatch.setenv("ABCROWN_DIR", str(root))
    monkeypatch.setenv("ABCROWN_PYTHON", sys.executable)
    work = str(tmp_path)
    # α,β-CROWN stops long before the deadline: it gave up.
    assert Runner(work, [], timeout=600)("net.onnx", "spec.vnnlib", 2) == ("unknown", None)
    args = (root / "calls.txt").read_text().split()
    assert 0 < float(args[args.index("--timeout") + 1]) <= 600
    # No time left: α,β-CROWN is not started.
    assert Runner(work, [], timeout=0)("net.onnx", "spec.vnnlib", 2) == ("timed-out", None)
    assert len((root / "calls.txt").read_text().splitlines()) == 1
    # Without a timeout of our own, α,β-CROWN's `timeout` is taken at its word.
    assert Runner(work, [])("net.onnx", "spec.vnnlib", 2) == ("timed-out", None)
