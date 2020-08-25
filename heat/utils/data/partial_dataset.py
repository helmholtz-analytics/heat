import os
import h5py
import math
import numpy as np
import queue
import threading
import torch
import itertools
import time
import warnings
from torch.utils import data as torch_data
from typing import Callable, List, Iterator, Union

from ...core import dndarray
from ...core import io
from ...core.communication import MPICommunication
from ...core.communication import MPI_WORLD
from ...core.communication import MPI
from . import datatools


def queue_thread(q: queue.Queue):
    while True:
        items = q.get()
        if isinstance(items, tuple):
            func = items[0]
            args = items[1:]
            func(*args)
        else:
            items()
        q.task_done()


class PartialDataset(torch_data.Dataset):
    # todo: getitem, len
    def __init__(
        self,
        file: str,
        comm: MPICommunication = MPI_WORLD,
        dataset_names: Union[str, List[str]] = "data",
        # available_memory: int = None,
        transforms: List[Callable] = None,
        # list of transform operations for what will be returned by getitem
        ishuffle: bool = True,
        np_buffer: bool = True,
        np_buffer_dataset_names: Union[str, List[str]] = "data",
        use_gpu: bool = True,
        validate_set: bool = False,
        # folder=None,
    ):
        super(PartialDataset, self).__init__()
        self.ishuffle = False
        self.file = file
        self.comm = comm
        self.transforms = transforms
        self.gpu = True if torch.cuda.device_count() > 0 and use_gpu else False
        self.torch_device = "cpu"
        if torch.cuda.is_available() and use_gpu:
            dev_id = MPI_WORLD.rank % torch.cuda.device_count()
            self.torch_device = torch.device("cuda:" + str(dev_id))
            torch.cuda.set_device(MPI_WORLD.rank % torch.cuda.device_count())

        self.partial_dataset = True
        f = h5py.File(file, "r", driver="mpio", comm=comm.handle)
        # too much data for the process
        # datasize_to_load = available_memory // 2.  # only load data equal to half the memory size
        # todo: supporting h5...
        fkeys = list(f.keys())

        sz = f[fkeys[0]].len()
        for k in fkeys[1:]:
            # ensure that all of the datasets are the same length
            if f[k].len() != sz:
                raise ValueError(f"all datasets in {file} must be the same length")
        self.total_size = sz
        # how many indices will go onto each process (len)
        self.lcl_full_sz = sz // comm.size
        # load data that is half of of the available memory
        self.local_data_start = comm.rank * self.lcl_full_sz
        self.local_data_end = (comm.rank + 1) * self.lcl_full_sz

        if validate_set:
            # if its the validation set then load the whole dataset for each process
            self.lcl_full_sz = sz
            self.local_data_start = 0
            self.local_data_end = sz

        self.local_length = self.local_data_end - self.local_data_start

        # temp values for small scale testing
        self.load_initial = 7000 if 7000 <= self.lcl_full_sz else self.lcl_full_sz - 1000
        self.load_len = 1000  # int(local_data_end / 3)
        self.loads_needed = math.ceil(self.lcl_full_sz / self.load_len)
        self.loads_remaining = self.loads_needed

        self.load_start = self.local_data_start
        self.load_end = self.local_data_start + self.load_initial

        # data being loaded from dataset_names parameter
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        self.dataset_names = dataset_names
        self.np_buff_flag = np_buffer
        self.np_datasets = (
            np_buffer_dataset_names
            if isinstance(np_buffer_dataset_names, list)
            else [np_buffer_dataset_names]
        )
        self.dataset_order = []
        for d in dataset_names:
            hld = f[d][self.load_start : self.load_end]
            self.__setattr__(d, hld)
        self.load_start = self.load_end
        self.load_end += self.load_len
        f.close()
        self.load_thread = None
        self.epoch_end = False
        # need the number of loads required for an epoch
        self.loading_queue = queue.Queue()
        self.loading_condition = threading.Condition()
        threading.Thread(target=queue_thread, args=[self.loading_queue], daemon=True).start()
        self.convert_queue = queue.Queue()
        threading.Thread(target=queue_thread, args=[self.convert_queue], daemon=True).start()

    def Shuffle(self):
        """
        Send half of the local data to the process ``self.comm.rank + 1`` if available, else wrap around. After
        receiving the new data, shuffle the local tensor.
        """
        # datatools.dataset_shuffle(dataset=self, attrs=self.shuffle_list)
        return NotImplementedError

    def Ishuffle(self):
        """
        Send half of the local data to the process ``self.comm.rank + 1`` if available, else wrap around. After
        receiving the new data, shuffle the local tensor.
        """
        # datatools.dataset_ishuffle(dataset=self, attrs=self.shuffle_list)
        return NotImplementedError

    def __getitem__(self, index: Union[int, slice, List[int], torch.Tensor]) -> torch.Tensor:
        # this function needs to be designed such that the data is in the 0th dimension and the indexes called
        #   are only in the 0th dim!
        raise NotImplementedError("__getitem__ must be overwritten")

    def __len__(self) -> int:
        return self.total_size


class LoadingDataLoaderIter(object):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self._dataset_kind = loader.DataLoader._dataset_kind
        self._IterableDataset_len_called = loader.DataLoader._IterableDataset_len_called
        self._auto_collation = loader.DataLoader._auto_collation
        self._drop_last = loader.DataLoader.drop_last
        self._index_sampler = loader.DataLoader._index_sampler
        self._num_workers = loader.DataLoader.num_workers
        self._pin_memory = loader.DataLoader.pin_memory and torch.cuda.is_available()
        self._timeout = loader.DataLoader.timeout
        self._collate_fn = loader.DataLoader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_().item()
        self._num_yielded = 0
        self.batch_size = loader.DataLoader.batch_size
        self.comm = self.dataset.comm
        rand_samp_list = torch.randperm(self.dataset.load_initial).tolist()

        # todo: support other samplers: for now its only random!!!
        if isinstance(self.dataset, PartialDataset) and self.dataset.partial_dataset:
            self.f = h5py.File(self.dataset.file, "r", driver="mpio", comm=self.comm.handle)
            self.used_indices = []
            self.ready_batches = []
            mod_batch = self.dataset.load_len % self.batch_size
            if mod_batch != 0:
                self.dataset.load_len += self.batch_size - mod_batch
                self.dataset.load_end = self.dataset.load_start + self.dataset.load_len
            # generate all indices
            index_list = []
            idx_repeats = math.ceil(self.dataset.lcl_full_sz / self.dataset.load_initial)
            for _ in range(idx_repeats):
                index_list.extend(torch.randperm(self.dataset.load_initial).tolist())
            # start the conversion
            self.dataset.convert_queue.put((self.thread_convert_all, index_list))
            self.length = len(index_list) // self.batch_size - 1
            self.loads_left = self.dataset.loads_needed

            if not self._drop_last and len(index_list) % self.batch_size != 0:
                # todo: implement drop last
                self.length += 1
            self.dataset.loading_queue.put(self.thread_replace_converted_batches)
        else:
            self.rand_samp_list = rand_samp_list
            self.length = len(self._sampler_iter)

        self._dataset_fetcher = torch_data.dataloader._DatasetKind.create_fetcher(
            self._dataset_kind,
            loader.DataLoader.dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )

    def __len__(self):
        return self.length

    def _next_data(self):
        if not self.dataset.partial_dataset:
            index = next(self._sampler_iter)  # may raise StopIteration
            data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
            # if self._pin_memory:
            #     data = _utils.pin_memory.pin_memory(data)
            return data
        # else:
        if self._num_yielded == self.__len__():
            self.f.close()
            raise StopIteration
        asd = 0
        while len(self.ready_batches) < 1:
            time.sleep(0.1)
            asd += 1
        # print("next data after wait", self._num_yielded, asd * 0.1)
        batch = self.ready_batches.pop(0)
        for b in range(len(batch)):
            if batch[b].device != self.dataset.torch_device:
                batch[b] = batch[b].to(self.dataset.torch_device)
        return batch

    def __next__(self):
        # shamelessly stolen from torch
        data = self._next_data()
        self._num_yielded += 1
        if (
            self._dataset_kind == torch_data.dataloader._DatasetKind.Iterable
            and self._IterableDataset_len_called is not None
            and self._num_yielded > self._IterableDataset_len_called
        ):
            warn_msg = (
                "Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                "samples have been fetched. "
            ).format(self._dataset, self._IterableDataset_len_called, self._num_yielded)
            if self._num_workers > 0:
                warn_msg += (
                    "For multiprocessing data-loading, this could be caused by not properly configuring the "
                    "IterableDataset replica at each worker. Please see "
                    "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples."
                )
            warnings.warn(warn_msg)
        return data

    def __iter__(self):
        return self

    def thread_convert_all(self, index_list):
        if isinstance(index_list, int):
            index_list = [index_list]
        converted_items = []
        for ind in index_list:
            single_item = list(self.dataset[ind])
            # have the item, need to convert from numpy to torch
            for ii in range(len(single_item)):
                # have all torch stuff here
                # do transforms
                if self.dataset.transforms[ii] is not None:
                    single_item[ii] = self.dataset.transforms[ii](single_item[ii])
            converted_items.append(single_item)
            self.used_indices.append(ind)
            if len(converted_items) == self.batch_size:
                if len(self.used_indices) == self.dataset.load_len and self.loads_left > 0:
                    with self.dataset.loading_condition:
                        self.dataset.loading_condition.notify()
                        self.dataset.loading_condition.wait()
                batch = self._collate_fn(converted_items)
                try:
                    for b in range(len(batch)):
                        self.ready_batches[0][b] = self.ready_batches[0][b].to(
                            self.dataset.torch_device
                        )
                except IndexError:
                    pass
                self.ready_batches.append(batch)
                converted_items = []

    def thread_replace_converted_batches(self):
        ll = self.loads_left
        for _ in range(ll):
            for d in self.dataset.dataset_names:
                hld = self.f[d][self.dataset.load_start : self.dataset.load_end]
                self.__setattr__("hold" + d, hld)
            self.dataset.load_start = self.dataset.load_end
            self.dataset.load_end += self.dataset.load_len

            # wait for lock1 *from* convert thread
            while self.__getattribute__("hold" + self.dataset.dataset_names[0]).__len__() > 0:
                with self.dataset.loading_condition:
                    self.dataset.loading_condition.wait()
                    for d in self.dataset.dataset_names:
                        new = self.__getattribute__("hold" + d)
                        dset = self.dataset.__getattribute__(d)
                        new_top = new[: len(self.used_indices)]
                        lnew = len(new_top)
                        dset[self.used_indices][:lnew] = new_top
                        self.dataset.__setattr__(d, dset)
                        self.dataset.__setattr__("hold" + d, new[lnew:])
                    # give up lock / notify convert thread
                    self.used_indices = []
                    self.dataset.loading_condition.notify()
            self.loads_left -= 1
