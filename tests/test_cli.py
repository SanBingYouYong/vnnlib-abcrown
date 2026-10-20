"""The VNN-LIB 2.0 command-line interface, without running α,β-CROWN."""

import pytest

from vnnlib_abcrown import cli
from vnnlib_abcrown.translate import Solution


@pytest.mark.parametrize("capability", list(cli.SUPPORTS))
def test_supports_answers_every_capability(capability, capsys):
    assert cli.main(["supports", capability]) == 0
    assert capsys.readouterr().out.splitlines() == cli.SUPPORTS[capability]


def test_supports_answers(capsys):
    cli.main(["supports", "--vnnlib-versions"])
    cli.main(["supports", "--optimised-disjunctive-reasoning"])
    assert capsys.readouterr().out == "2.0\n2.0\nfalse\n"


def test_supports_rejects_unknown_capability(capsys):
    assert cli.main(["supports", "--teleportation"]) == 1
    assert cli.main(["supports"]) == 1
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("status, legacy, printed", [
    ("unsat", False, "unsat"), ("timed-out", False, "timed-out"), ("unknown", False, "unknown"),
    ("unsat", True, "unsat"), ("timed-out", True, "timeout"), ("unknown", True, "timeout"),
])
def test_status_words(status, legacy, printed, capsys):
    cli.print_solution(Solution(status), query=None, legacy=legacy)
    assert capsys.readouterr().out == printed + "\n"


def test_unsupported_query_exits_with_an_error(tmp_path, capsys):
    q = tmp_path / "q.vnnlib"
    q.write_text("(vnnlib-version <2.0>)\n(declare-network n (declare-input n_X float32 [1]) "
                 "(declare-output n_Y float32 [1]))\n(assert (<= n_Y[0] 0.0))\n")
    assert cli.main(["verify", str(q), "--network", "n=missing.onnx"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "only real is supported" in captured.err
