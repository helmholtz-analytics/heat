import unittest
from unittest.mock import patch

import heat as ht
import torch
from torchvision import datasets, transforms

from heat.testing.basic_test import TestCase
from heat.utils.data.mnist import MNISTDataset


def _mock_mnist_init(
    self,
    root: str,
    train: bool = True,
    transform=None,
    target_transform=None,
    download: bool = True,
):
    """Replacement for torchvision.datasets.MNIST.__init__ generating synthetic tensors."""
    self.root = root
    self.train = train
    self.transform = transform
    self.target_transform = target_transform

    # Fixed seed so all MPI processes generate the identical global dataset
    torch.manual_seed(42)
    num_samples = 120 if train else 40
    self.data = torch.randint(0, 256, (num_samples, 28, 28), dtype=torch.uint8)
    self.targets = torch.randint(0, 10, (num_samples,), dtype=torch.long)


class TestMNISTDataset(TestCase):
    def setUp(self):
        super().setUp()
        self.patcher = patch.object(
            datasets.MNIST,
            "__init__",
            new=_mock_mnist_init,
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        super().tearDown()

    def test_invalid_split(self):
        """Verify that splits other than 0 or None raise ValueError."""
        with self.assertRaises(ValueError):
            MNISTDataset(root="fake_dir", split=1)
        with self.assertRaises(ValueError):
            MNISTDataset(root="fake_dir", split=2)
        with self.assertRaises(ValueError):
            MNISTDataset(root="fake_dir", split=-1)

    def test_init_train_split_zero(self):
        """Test default distributed training set initialization (split=0)."""
        dset = MNISTDataset(root="fake_dir", train=True, split=0)

        self.assertTrue(dset.train)
        self.assertFalse(dset.test_set)
        self.assertFalse(dset.partial_dataset)
        self.assertEqual(dset.comm, dset.htdata.comm)
        self.assertEqual(dset.htdata.split, 0)
        self.assertEqual(dset.httargets.split, 0)

        comm_size = dset.comm.size
        expected_len = 120 // comm_size
        self.assertEqual(dset._cut_slice, slice(expected_len))
        self.assertEqual(dset.lcl_half, expected_len // 2)
        self.assertEqual(dset.data.shape, (expected_len, 28, 28))
        self.assertEqual(dset.targets.shape, (expected_len,))
        self.assertEqual(len(dset), expected_len)

    def test_init_split_none(self):
        """Test initialization with split=None (data duplicated across processes)."""
        dset = MNISTDataset(root="fake_dir", train=True, split=None)

        self.assertIsNone(dset.htdata.split)
        self.assertIsNone(dset.httargets.split)
        self.assertIsNone(dset._cut_slice)
        self.assertEqual(dset.lcl_half, 120 // 2)
        self.assertEqual(dset.data.shape, (120, 28, 28))
        self.assertEqual(dset.targets.shape, (120,))
        self.assertEqual(len(dset), 120)

    def test_init_test_set(self):
        """Test that test_set=True forces split=None even if split=0 is given."""
        dset = MNISTDataset(root="fake_dir", train=False, test_set=True, split=0)

        self.assertTrue(dset.test_set)
        self.assertIsNone(dset.htdata.split)
        self.assertIsNone(dset.httargets.split)
        self.assertIsNone(dset._cut_slice)
        self.assertEqual(dset.lcl_half, 40 // 2)
        self.assertEqual(dset.data.shape, (40, 28, 28))
        self.assertEqual(dset.targets.shape, (40,))
        self.assertEqual(len(dset), 40)

    def test_getitem_with_transforms(self):
        """Test __getitem__ with and without torchvision transforms."""
        # Without transform: returns a PIL Image and integer target
        dset_raw = MNISTDataset(root="fake_dir", train=True, split=0)
        img_raw, target_raw = dset_raw[0]
        self.assertTrue(hasattr(img_raw, "size"))
        self.assertIsInstance(target_raw, int)

        # With transform: returns torch.Tensor and modified target
        transform = transforms.ToTensor()
        target_transform = lambda t: t + 100
        dset_transformed = MNISTDataset(
            root="fake_dir",
            train=True,
            transform=transform,
            target_transform=target_transform,
            split=0,
        )
        img_t, target_t = dset_transformed[0]
        self.assertIsInstance(img_t, torch.Tensor)
        self.assertEqual(img_t.shape, (1, 28, 28))
        self.assertEqual(target_t, int(dset_transformed.targets[0]) + 100)

    def test_shuffle(self):
        """Test dataset shuffle and test_set no-op behavior."""
        dset = MNISTDataset(root="fake_dir", train=True, split=0)
        expected_shape = dset.data.shape
        expected_target_shape = dset.targets.shape

        # Verify Shuffle runs and keeps valid local shapes
        dset.Shuffle()
        self.assertEqual(dset.data.shape, expected_shape)
        self.assertEqual(dset.targets.shape, expected_target_shape)

        # For test_set=True, Shuffle and Ishuffle are no-ops
        dset_test = MNISTDataset(root="fake_dir", train=False, test_set=True)
        orig_data = dset_test.data.clone()
        dset_test.Shuffle()
        self.assertTrue(torch.equal(dset_test.data, orig_data))
        dset_test.Ishuffle()
        self.assertTrue(torch.equal(dset_test.data, orig_data))

    def test_dataloader_integration(self):
        """Test batch iteration using Heat DataLoader."""
        transform = transforms.ToTensor()
        dset = MNISTDataset(root="fake_dir", train=True, transform=transform, split=0)
        batch_size = 4
        dl = ht.utils.data.DataLoader(dataset=dset, batch_size=batch_size)

        total_samples = 0
        for batch_data, batch_targets in dl:
            self.assertEqual(batch_data.shape[1:], (1, 28, 28))
            self.assertEqual(batch_targets.shape[0], batch_data.shape[0])
            total_samples += batch_data.shape[0]

        self.assertEqual(total_samples, len(dset))

    def test_dataloader_with_ishuffle(self):
        """Test DataLoader workflow when ishuffle flag is enabled across epochs."""
        transform = transforms.ToTensor()
        dset = MNISTDataset(
            root="fake_dir", train=True, transform=transform, split=0, ishuffle=True
        )
        self.assertTrue(dset.ishuffle)
        batch_size = 4
        dl = ht.utils.data.DataLoader(dataset=dset, batch_size=batch_size)

        num_epochs = 2
        for epoch in range(num_epochs):
            if epoch == num_epochs - 1:
                dl.last_epoch = True

            epoch_samples = 0
            for batch_data, batch_targets in dl:
                self.assertEqual(batch_data.shape[1:], (1, 28, 28))
                self.assertEqual(batch_targets.shape[0], batch_data.shape[0])
                epoch_samples += batch_data.shape[0]
            self.assertEqual(epoch_samples, len(dset))
