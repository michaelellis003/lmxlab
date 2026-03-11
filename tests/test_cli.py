"""Tests for the lmt-metal CLI."""

import argparse

from lmt_metal.cli import ARCHITECTURES, cmd_count, cmd_info, cmd_list


class TestCmdList:
    """Test the 'list' command."""

    def test_lists_all_architectures(self, capsys):
        args = argparse.Namespace()
        cmd_list(args)
        output = capsys.readouterr().out

        assert "Available architectures:" in output
        for name in ARCHITECTURES:
            assert name in output

    def test_shows_attention_type(self, capsys):
        args = argparse.Namespace()
        cmd_list(args)
        output = capsys.readouterr().out

        assert "attention=" in output


class TestCmdInfo:
    """Test the 'info' command."""

    def test_shows_config(self, capsys):
        args = argparse.Namespace(arch="gpt", tiny=False)
        cmd_info(args)
        output = capsys.readouterr().out

        assert "Architecture: gpt" in output
        assert "d_model:" in output
        assert "attention:" in output

    def test_tiny_flag(self, capsys):
        args = argparse.Namespace(arch="llama", tiny=True)
        cmd_info(args)
        output = capsys.readouterr().out

        assert "llama (tiny)" in output

    def test_unknown_arch_exits(self):
        import pytest

        args = argparse.Namespace(arch="nonexistent", tiny=False)
        with pytest.raises(SystemExit):
            cmd_info(args)

    def test_all_architectures(self, capsys):
        """Every architecture should be displayable."""
        for name in ARCHITECTURES:
            args = argparse.Namespace(arch=name, tiny=True)
            cmd_info(args)
            output = capsys.readouterr().out
            assert name in output


class TestCmdCount:
    """Test the 'count' command."""

    def test_counts_parameters(self, capsys):
        args = argparse.Namespace(arch="gpt", tiny=True, detail=False)
        cmd_count(args)
        output = capsys.readouterr().out

        assert "parameters" in output

    def test_detail_flag(self, capsys):
        args = argparse.Namespace(arch="llama", tiny=True, detail=True)
        cmd_count(args)
        output = capsys.readouterr().out

        assert "embed" in output
        assert "blocks" in output
        assert "%" in output

    def test_unknown_arch_exits(self):
        import pytest

        args = argparse.Namespace(arch="nonexistent", tiny=False, detail=False)
        with pytest.raises(SystemExit):
            cmd_count(args)

    def test_all_architectures(self, capsys):
        """Every architecture should be countable."""
        for name in ARCHITECTURES:
            args = argparse.Namespace(arch=name, tiny=True, detail=False)
            cmd_count(args)
            output = capsys.readouterr().out
            assert "parameters" in output
