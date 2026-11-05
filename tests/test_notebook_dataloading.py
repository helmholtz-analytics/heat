"""Execute the Iris data loading tutorial on one or more MPI ranks."""

from collections import Counter
from contextlib import ExitStack
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import heat as ht
import pytest
import torch


NOTEBOOK = Path(__file__).resolve().parents[1] / "doc/source/tutorials/notebooks/Dataloading.ipynb"
MIN_TEST_ACCURACY = 0.85


def test_notebook_structure():
    """Execute tutorial cells through px and cluster management in the kernel."""
    notebook = json.loads(NOTEBOOK.read_text())
    assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 5)
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    local_cells = []
    for cell in notebook["cells"]:
        assert isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], list)
        assert all(isinstance(line, str) for line in cell["source"])
        if cell["cell_type"] == "code":
            count = cell["execution_count"]
            assert count is None or (isinstance(count, int) and count >= 0)
            assert isinstance(cell["outputs"], list)
            assert cell["metadata"]["tags"] in (["tutorial"], ["local-only"])
            source = "".join(cell["source"])
            if cell["metadata"]["tags"] == ["local-only"]:
                assert not source.startswith("%%px")
                local_cells.append(cell["id"])
            else:
                assert source.startswith("%%px\n")
                source = source.removeprefix("%%px\n")
            compile(source, f"{NOTEBOOK}:{cell['id']}", "exec")
        else:
            assert cell["cell_type"] == "markdown"
    assert local_cells == ["execution-setup", "execution-cleanup"]


@pytest.mark.parametrize("external", [False, True])
def test_cluster_connection_and_cleanup(external):
    """Stop notebook-owned clusters while preserving externally started engines."""
    cells = {
        cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]
    }
    ipp = MagicMock()
    client = ipp.Client.return_value
    client.ids = list(range(4))
    cluster = ipp.Cluster.return_value
    cluster.start_and_connect_sync.return_value = client
    source = cells["execution-setup"].replace(
        "use_external_engines = False", f"use_external_engines = {external}"
    )
    namespace = {}
    with patch.dict("sys.modules", {"ipyparallel": ipp}):
        exec(compile(source, "execution-setup", "exec"), namespace)
        if external:
            ipp.Cluster.assert_not_called()
            ipp.Client.assert_called_once_with(profile="default")
            client.wait_for_engines.assert_not_called()
            client.activate.assert_called_once_with()
        else:
            ipp.Client.assert_not_called()
            ipp.Cluster.assert_called_once_with(engines="mpi", n=4)
            cluster.start_and_connect_sync.assert_called_once_with(activate=True)
        exec(compile(cells["execution-cleanup"], "execution-cleanup", "exec"), namespace)
        # Repeating cleanup must be harmless.
        exec(compile(cells["execution-cleanup"], "execution-cleanup", "exec"), namespace)
    client.close.assert_called_once_with()
    client.shutdown.assert_not_called()
    if external:
        cluster.stop_cluster_sync.assert_not_called()
    else:
        cluster.stop_cluster_sync.assert_called_once_with()
    assert namespace["rc"] is None and namespace["cluster"] is None


@pytest.mark.parametrize("external", [False, True])
def test_cluster_setup_failure(external):
    """Release the client on setup failure without shutting down external engines."""
    cells = {
        cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]
    }
    ipp = MagicMock()
    client = ipp.Client.return_value
    cluster = ipp.Cluster.return_value
    cluster.start_and_connect_sync.return_value = client
    client.__getitem__.return_value.execute.side_effect = RuntimeError("Engine setup failed")
    source = cells["execution-setup"].replace(
        "use_external_engines = False", f"use_external_engines = {external}"
    )
    with patch.dict("sys.modules", {"ipyparallel": ipp}):
        with pytest.raises(RuntimeError, match="Engine setup failed"):
            exec(compile(source, "execution-setup", "exec"), {})
    client.close.assert_called_once_with()
    client.shutdown.assert_not_called()
    if external:
        ipp.Cluster.assert_not_called()
        cluster.stop_cluster_sync.assert_not_called()
    else:
        cluster.stop_cluster_sync.assert_called_once_with()


@pytest.fixture(scope="module", params=["cpu", "gpu"])
def tutorial_run(request):
    """Run tutorial bodies on pytest's MPI ranks and independently record losses."""
    try:
        ht.sanitize_device(request.param)
        available = True
    except ValueError:
        available = False
    if not all(ht.MPI_WORLD.allgather(available)):
        pytest.skip(f"{request.param} is not available on every MPI rank")
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn.datasets")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {}
    batch_losses = []
    figures = []
    original_forward = torch.nn.CrossEntropyLoss.forward
    previous_threads = torch.get_num_threads()

    def record_loss(self, logits, targets):
        loss = original_forward(self, logits, targets)
        batch_losses.append((loss.detach().item(), targets.shape[0]))
        return loss

    def record_plot():
        figures.append(plt.gcf())
        plt.close("all")

    try:
        with ExitStack() as stack:
            stack.enter_context(
                torch.random.fork_rng(devices=list(range(torch.cuda.device_count())))
            )
            stack.enter_context(patch.object(plt, "show", side_effect=record_plot))
            stack.enter_context(patch("IPython.display.display"))
            stack.enter_context(patch.object(torch.nn.CrossEntropyLoss, "forward", record_loss))
            for cell in notebook["cells"]:
                # The test launcher supplies the ranks; do not start nested MPI engines.
                if cell["cell_type"] == "code" and cell["metadata"]["tags"] == ["tutorial"]:
                    source = "".join(cell["source"]).removeprefix("%%px\n")
                    if cell["id"] == "select-device":
                        source = source.replace('device = "cpu"', f'device = "{request.param}"')
                    exec(compile(source, f"{NOTEBOOK}:{cell['id']}", "exec"), namespace)
    finally:
        torch.set_num_threads(previous_threads)
        plt.close("all")
    return namespace, batch_losses, figures


def test_split_alignment_and_scaling(tutorial_run):
    """Preserve every labelled Iris row and fit scaling only on training data."""
    ns, _, _ = tutorial_run
    assert ns["X"].dtype == ht.float32
    assert ns["y"].dtype == ht.int64
    train_count = int(ns["n_samples"] * ns["train_fraction"])
    test_count = ns["n_samples"] - train_count
    assert ns["X_train"].shape == (train_count, ns["n_features"])
    assert ns["X_test"].shape == (test_count, ns["n_features"])
    arrays = [ns[name] for name in ("X", "y", "X_train", "X_test", "y_train", "y_test")]
    assert all(array.split == 0 and array.device == ns["heat_device"] for array in arrays)

    def rows(features, labels):
        return Counter((tuple(row), label) for row, label in zip(features.numpy(), labels.numpy()))

    assert rows(ns["X_train"], ns["y_train"]) + rows(ns["X_test"], ns["y_test"]) == rows(
        ns["X"], ns["y"]
    )
    repeated = ht.utils.data.create_train_val_split(
        ns["X"], ns["y"], p=ns["train_fraction"], seed=ns["seed"]
    )
    for actual, name in zip(repeated, ("X_train", "X_test", "y_train", "y_test")):
        assert ht.equal(actual, ns[name])
    assert ht.allclose(ns["scaler"].mean_, ht.mean(ns["X_train"], axis=0))
    for name in ("train", "test"):
        expected = ns["scaler"].transform(ns[f"X_{name}"])
        assert ht.equal(ns[f"X_{name}_scaled"], expected)
        assert expected.split == 0
        assert expected.dtype == ht.float32
        assert expected.device == ns["heat_device"]


def test_batch_tensors_and_labels(tutorial_run):
    """Return aligned, complete PyTorch samples and reset the preview's shuffle epoch."""
    ns, _, _ = tutorial_run
    features, labels = ns["batch_features"], ns["batch_labels"]
    assert isinstance(features, torch.Tensor) and features.dtype == torch.float32
    assert isinstance(labels, torch.Tensor) and labels.dtype == torch.int64
    batch_size = min(ns["batch_size"], math.ceil(ns["X_train"].shape[0] / ht.MPI_WORLD.size))
    assert features.shape == (batch_size, ns["n_features"])
    assert labels.shape == (batch_size,)
    assert features.device == labels.device == ns["torch_device"]
    all_features = torch.from_numpy(ns["X_train_scaled"].numpy()).to(ns["torch_device"])
    all_labels = torch.from_numpy(ns["y_train"].numpy()).to(ns["torch_device"])
    for row, label in zip(features, labels):
        matches = (all_features == row).all(dim=1) & (all_labels == label)
        assert matches.any()
    assert ns["train_loader"].epoch == ns["epochs"]


def test_sample_weighted_loss_and_padding(tutorial_run):
    """Check recorded losses against batch observations, including incomplete batches."""
    ns, batch_losses, _ = tutorial_run
    batches_per_epoch = len(ns["train_loader"])
    assert len(batch_losses) == ns["epochs"] * batches_per_epoch
    expected_samples = math.ceil(ns["X_train"].shape[0] / ht.MPI_WORLD.size) * ht.MPI_WORLD.size
    assert ns["samples_per_epoch"] == [expected_samples] * ns["epochs"]
    losses = ns["loss_history"]
    assert all(math.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]
    for epoch, actual in enumerate(losses):
        start = epoch * batches_per_epoch
        observations = batch_losses[start : start + batches_per_epoch]
        total = ht.MPI_WORLD.allreduce(sum(loss * count for loss, count in observations))
        count = ht.MPI_WORLD.allreduce(sum(count for _, count in observations))
        assert actual == pytest.approx(total / count)


def test_accuracy_counts_original_samples(tutorial_run):
    """Evaluate every original row once and reach useful held-out accuracy."""
    ns, _, _ = tutorial_run
    assert not ns["model"].training
    assert ns["train_count"] == ns["y_train"].shape[0]
    assert ns["test_count"] == ns["y_test"].shape[0]
    assert ns["test_accuracy"] >= MIN_TEST_ACCURACY
    for name in ("train", "test"):
        features = torch.from_numpy(ns[f"X_{name}_scaled"].numpy()).to(ns["torch_device"])
        labels = torch.from_numpy(ns[f"y_{name}"].numpy()).to(ns["torch_device"])
        with torch.no_grad():
            predictions = ns["network"](features).argmax(dim=1)
        expected = (predictions == labels).sum().item()
        assert ns[f"{name}_correct"] == expected
        assert ns[f"{name}_accuracy"] == pytest.approx(expected / labels.shape[0])


def test_synchronized_weights_and_plots(tutorial_run):
    """Keep one shared model and display the two tutorial plots only on rank zero."""
    ns, _, figures = tutorial_run
    for parameter in ns["model"].parameters():
        assert parameter.device == ns["torch_device"]
        copies = ht.MPI_WORLD.allgather(parameter.detach().tolist())
        assert all(copy == copies[0] for copy in copies)
    histories = ht.MPI_WORLD.allgather(ns["loss_history"])
    assert all(history == histories[0] for history in histories)
    assert len(figures) == (2 if ht.MPI_WORLD.rank == 0 else 0)
