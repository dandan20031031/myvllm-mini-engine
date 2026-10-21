from mini_engine.__main__ import main


def test_cli_help_lists_cuda_prefill(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["mini_engine", "--help"])
    try:
        main()
    except SystemExit as error:
        assert error.code == 0
    output = capsys.readouterr().out
    assert "{auto,flash,math,cuda}" in output
