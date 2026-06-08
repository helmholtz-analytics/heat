from unittest.mock import patch
import argparse
from heat import cli
import io
import contextlib

class TestCLI:
    @patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(info=False))
    def test_cli_help(self, mock_parse_args):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.cli()

        print(stdout.getvalue())
        assert "usage: heat [-h] [-i]" in stdout.getvalue()

    @patch("platform.platform")
    @patch("mpi4py.MPI.Get_library_version")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.current_device")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    def test_platform_info(
        self,
        mock_get_device_properties,
        mock_get_device_name,
        mock_get_default_device,
        mock_device_count,
        mock_cuda_current_device,
        mock_mpi_lib_version,
        mock_platform,
    ):
        mock_platform.return_value = "Test Platform"
        mock_mpi_lib_version.return_value = "Test MPI Library"
        mock_cuda_current_device.return_value = True
        mock_device_count.return_value = 1
        mock_get_default_device.return_value = "cuda:0"
        mock_get_device_name.return_value = "Test Device"
        mock_get_device_properties.return_value.total_memory = 1024**4  # 1TiB

        stdout_stream = io.StringIO()
        with contextlib.redirect_stdout(stdout_stream):
            cli.plaform_info()
        stdout = stdout_stream.getvalue()
        print(stdout)
        assert "HeAT: Helmholtz Analytics Toolkit" in stdout
        assert "Platform: Test Platform" in stdout
        assert "mpi4py Version:" in stdout
        assert "MPI Library Version: Test MPI Library" in stdout
        assert "Torch Version:" in stdout
        assert "CUDA Available: True" in stdout
        assert "Device count: 1" in stdout
        assert "Default device: cuda:0" in stdout
        assert "Device name: Test Device" in stdout
        assert "Device memory: 1024.0 GiB" in stdout
