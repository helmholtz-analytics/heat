import heat as ht
import torch
import unittest
import heat.nn.functional as F


class ConvNet(ht.nn.Module):
    """Simple CNN model for testing."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = ht.nn.Conv2d(1, 6, 3)
        self.conv2 = ht.nn.Conv2d(6, 16, 3)
        self.fc1 = ht.nn.Linear(16 * 6 * 6, 120)
        self.fc2 = ht.nn.Linear(120, 84)
        self.fc3 = ht.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ShuffleTestDataset(ht.utils.data.Dataset):
    """Test dataset with custom shuffle methods."""

    def __init__(self, array, ishuffle):
        super(ShuffleTestDataset, self).__init__(array, ishuffle=ishuffle)

    def __getitem__(self, item):
        return self.data[item]

    def Ishuffle(self):
        if not self.test_set:
            ht.utils.data.dataset_ishuffle(self, attrs=[["data", None]])

    def Shuffle(self):
        if not self.test_set:
            ht.utils.data.dataset_shuffle(self, attrs=[["data", None]])


class TestDataParallel(unittest.TestCase):
    """Test suite for DataParallel functionality."""

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(1)

    def setUp(self):
        """Common setup for all tests."""
        self.tolerance = 1e-4
        self.batch_size = 2
        self.data_shape = (2 * ht.MPI_WORLD.size, 1, 32, 32)
        self.label_shape = (2, 10)
        self.learning_rate = 0.001
        self.num_epochs = 2

    def _create_dataloader(self, dataset_class=ht.utils.data.Dataset, ishuffle=False):
        """Helper to create dataset and dataloader."""
        data = ht.random.rand(*self.data_shape, split=0)
        dataset = dataset_class(data, ishuffle=ishuffle)
        dataloader = ht.utils.data.datatools.DataLoader(
            dataset=dataset, batch_size=self.batch_size
        )
        return dataloader, data

    def _train_and_validate_parameters(self, ht_model, dp_optimizer, labels, dataloader):
        """Helper to train model and validate parameter synchronization."""
        loss_fn = torch.nn.MSELoss()

        for _ in range(self.num_epochs):
            for batch_data in dataloader:
                self.assertEqual(batch_data.shape[0], self.batch_size)
                dp_optimizer.zero_grad()
                ht_outputs = ht_model(batch_data)
                loss_fn(ht_outputs, labels).backward()
                dp_optimizer.step()

            # Verify parameters are synchronized across processes
            self._verify_parameter_sync(ht_model)

    def _verify_parameter_sync(self, ht_model):
        """Helper to verify all parameters are synchronized across processes."""
        for p in ht_model.parameters():
            p0dim = p.shape[0]
            hld = ht.resplit(ht.array(p, is_split=0)).larray.clone()
            hld_list = [
                hld[i * p0dim : (i + 1) * p0dim] for i in range(ht.MPI_WORLD.size - 1)
            ]
            for i in range(1, len(hld_list)):
                self.assertTrue(
                    torch.allclose(
                        hld_list[0], hld_list[i], rtol=self.tolerance, atol=self.tolerance
                    )
                )

    def _move_to_device(self, model):
        """Helper to move model to GPU if available."""
        if str(ht.get_device()).startswith("gpu"):
            model.to(ht.get_device().torch_device)

    def test_dataloader_invalid_input(self):
        """Test DataLoader raises TypeError for invalid input."""
        with self.assertRaises(TypeError):
            ht.utils.data.datatools.DataLoader("asdf")

    def test_dp_optimizer_invalid_input(self):
        """Test DataParallelOptimizer raises TypeError for invalid input."""
        model = ConvNet()
        optimizer = ht.optim.SGD(model.parameters(), lr=self.learning_rate)
        with self.assertRaises(TypeError):
            ht.optim.DataParallelOptimizer(optimizer, "asdf")

    def test_data_parallel_invalid_optimizer(self):
        """Test DataParallel raises TypeError for invalid optimizer."""
        model = ConvNet()
        data = ht.random.rand(*self.data_shape, split=0)
        with self.assertRaises(TypeError):
            ht.nn.DataParallel(model, data.comm, "asdf")

    def test_data_parallel_blocking_with_custom_dataset(self):
        """Test DataParallel with blocking updates and custom shuffle dataset."""
        model = ConvNet()
        optimizer = ht.optim.SGD(model.parameters(), lr=self.learning_rate)
        dp_optimizer = ht.optim.DataParallelOptimizer(optimizer, True)

        labels = torch.randn(
            self.label_shape, device=ht.get_device().torch_device
        )
        dataloader, data = self._create_dataloader(
            dataset_class=ShuffleTestDataset, ishuffle=True
        )
        self.assertEqual(len(dataloader), 1)

        ht_model = ht.nn.DataParallel(
            model, data.comm, dp_optimizer, blocking_parameter_updates=True
        )
        self._move_to_device(ht_model)

        self._train_and_validate_parameters(ht_model, dp_optimizer, labels, dataloader)

    def test_data_parallel_non_blocking_without_shuffle(self):
        """Test DataParallel with non-blocking updates and no shuffle."""
        model = ConvNet()
        optimizer = ht.optim.SGD(model.parameters(), lr=self.learning_rate)
        dp_optimizer = ht.optim.DataParallelOptimizer(optimizer, False)

        labels = torch.randn(
            self.label_shape, device=ht.get_device().torch_device
        )
        dataloader, data = self._create_dataloader(
            dataset_class=ht.utils.data.Dataset, ishuffle=False
        )

        ht_model = ht.nn.DataParallel(
            model, data.comm, dp_optimizer, blocking_parameter_updates=False
        )
        self._move_to_device(ht_model)

        self._train_and_validate_parameters(ht_model, dp_optimizer, labels, dataloader)

    def test_data_parallel_non_blocking_with_shuffle(self):
        """Test DataParallel with non-blocking updates and shuffle enabled."""
        model = ConvNet()
        optimizer = ht.optim.SGD(model.parameters(), lr=self.learning_rate)
        dp_optimizer = ht.optim.DataParallelOptimizer(optimizer, False)

        labels = torch.randn(
            self.label_shape, device=ht.get_device().torch_device
        )
        dataloader, data = self._create_dataloader(
            dataset_class=ht.utils.data.Dataset, ishuffle=True
        )

        ht_model = ht.nn.DataParallel(
            model, data.comm, dp_optimizer, blocking_parameter_updates=False
        )
        self._move_to_device(ht_model)

        self._train_and_validate_parameters(ht_model, dp_optimizer, labels, dataloader)

    def test_data_parallel_warning_with_list_optimizer(self):
        """Test DataParallel emits warning when passed list of optimizers."""
        model = ConvNet()
        optimizer = ht.optim.SGD(model.parameters(), lr=self.learning_rate)
        dp_optimizer = ht.optim.DataParallelOptimizer(optimizer, False)
        data = ht.random.rand(*self.data_shape, split=0)

        with self.assertWarns(Warning):
            ht_model = ht.nn.DataParallel(
                model,
                ht.MPI_WORLD,
                [dp_optimizer, dp_optimizer],
                blocking_parameter_updates=False,
            )
        self.assertTrue(ht_model.blocking_parameter_updates)
