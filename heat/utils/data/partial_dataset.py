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
        available_memory: int = None,
        transforms: List[
            Callable
        ] = None,  # list of transform operations for what will be returned by getitem
        ishuffle: bool = True,
        np_buffer: bool = True,
        np_buffer_dataset_names: Union[str, List[str]] = "data",
        use_gpu: bool = True,
        # folder=None,
    ):
        super(PartialDataset, self).__init__()
        self.ishuffle = ishuffle
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
        # todo: only supporting h5 for now...
        fkeys = list(f.keys())

        sz = f[fkeys[0]].len()
        for k in fkeys[1:]:
            # ensure that all of the datasets are the same length
            if f[k].len() != sz:
                raise ValueError(f"all datasets in {file} must be the same length")
        self.total_size = sz
        # how many indices will go onto each process (len)
        # self.lcl_full_sz = sz // comm.size
        self.lcl_full_sz = 7000
        # load data that is half of of the available memory
        self.local_data_start = comm.rank * self.lcl_full_sz
        # self.local_data_end = (
        #    (comm.rank + 1) * self.lcl_full_sz if comm.rank != comm.size - 1 else self.total_size
        # )
        self.local_data_end = (comm.rank + 1) * self.lcl_full_sz
        self.local_length = self.local_data_end - self.local_data_start

        # temp values for small scale testing
        self.load_initial = 5000
        self.load_len = 1000  # int(local_data_end / 3)

        self.load_start = self.local_data_start
        self.load_end = self.local_data_start + self.load_initial
        self.next_start = self.load_end

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
            # load datasets from file
            if not np_buffer or d not in np_buffer_dataset_names:
                hld = torch.tensor(f[d][self.load_start : self.load_end])
            else:
                # this is loading the data to a np buffer
                hld = f[d][self.load_start : self.load_end]
            self.__setattr__(d, hld)
        self.load_start = self.load_end
        self.load_end += self.load_len
        f.close()
        self.load_thread = None
        self.epoch_end = False
        # need the number of loads required for an epoch
        self.loads_required = math.ceil((self.lcl_full_sz - self.load_initial) / self.load_len)
        self.loads_remaining = self.loads_required
        self.loading_queue = queue.Queue()
        self.loading_condition = threading.Condition()
        threading.Thread(target=queue_thread, args=[self.loading_queue], daemon=True).start()
        self.convert_queue = queue.Queue()
        # self.convert_condition = threading.Condition()
        threading.Thread(target=queue_thread, args=[self.convert_queue], daemon=True).start()

    def Shuffle(self):
        """
        Send half of the local data to the process ``self.comm.rank + 1`` if available, else wrap around. After
        receiving the new data, shuffle the local tensor.
        """
        datatools.dataset_shuffle(dataset=self, attrs=self.shuffle_list)

    def Ishuffle(self):
        """
        Send half of the local data to the process ``self.comm.rank + 1`` if available, else wrap around. After
        receiving the new data, shuffle the local tensor.
        """
        datatools.dataset_ishuffle(dataset=self, attrs=self.shuffle_list)

    def getitem_index_helper(self, index):
        # function for adding the indexes given to the getitem function to the next indices stuff.
        # once this equals the load_len then
        if not self.partial_dataset:
            return

        if isinstance(index, torch.Tensor):
            # todo: abstract this
            rem = self._test_remander_getitem(index)
            self.getitem_num += len(index)
        elif isinstance(index, list):
            rem = self._test_remander_getitem(index)
            self.getitem_num += len(index)
        elif isinstance(index, slice):
            rng = list(range(start=index.start, stop=index.stop, step=index.step))
            rem = self._test_remander_getitem(rng)
            self.getitem_num += len(rng)
        elif isinstance(index, int):
            self.next_indices.add(index)
            self.getitem_num += 1
            rem = 0
        else:
            raise TypeError(f"index must be int, slice, list, or torch.Tensor, not {type(index)}")

        if self.getitem_num >= self.load_len:
            # trigger the receive function and the next load
            self.insert_group(self.getitem_num, self.next_indices.copy())
            self.load_next_group()
            # reset getitem_num
            self.getitem_num = rem

            self.next_indices = self.next_indices_overflow.copy()
            self.next_indices_overflow = set()
            # self.next_indices[:rem] = self.next_indices_overflow[:rem]

    def _test_remander_getitem(self, index):
        rem = self.getitem_num + len(index) - self.load_len
        if rem > 0:
            # need to add the remainder to the overflow
            self.next_indices_overflow.update(index[-1 * rem :])
        else:
            rem = 0
        self.next_indices.update(index[: -1 * rem] if rem > 0 else index)
        return rem

    def __getitem__(self, index: Union[int, slice, List[int], torch.Tensor]) -> torch.Tensor:
        # this function needs to be designed such that the data is in the 0th dimension and the indexes called
        #   are only in the 0th dim!
        raise NotImplementedError("__getitem__ must be overwritten! (see examples)")

    def __len__(self) -> int:
        return self.total_size


class LoadingDataLoaderIter(object):  # torch_data.dataloader._BaseDataLoaderIter):
    def __init__(self, loader, pre_load_batches: int = 4):
        # this is the HeAT DataLoader not torch!
        #       the torch DataLoader is at load.DataLoader
        # super(LoadingDataLoaderIter, self).__init__(loader=loader.DataLoader)
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
            self.loading_lock = threading.Condition()

            mod_batch = self.dataset.load_len % self.batch_size
            if mod_batch != 0:
                self.dataset.load_len += self.batch_size - mod_batch
                self.dataset.load_end = self.dataset.load_start + self.dataset.load_len
            # generate all indices
            index_list = []
            idx_repeats = math.ceil(self.dataset.lcl_full_sz / self.dataset.load_initial)
            for _ in range(idx_repeats):
                index_list.extend(rand_samp_list)
            # start the conversion
            self.dataset.convert_queue.put((self.thread_convert_all, index_list))
            self.length = len(index_list) // self.batch_size - 1
            print(
                "length",
                self.length,
                len(index_list),
                self.batch_size,
                self.dataset.load_len,
                len(self.used_indices),
            )

            if not self._drop_last and len(index_list) % self.batch_size != 0:
                # todo: implement drop last!
                self.length += 1
            self.dataset.loading_queue.put(self.thread_replace_converted_batches)
            self.notify_overwrite = False
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
            self.dataset.loading_queue.put(self.thread_load_next_dataset)
            raise StopIteration
        # print('next_data', len(self.ready_batches), self._num_yielded)
        while len(self.ready_batches) < 1:
            # print('\t', len(self.ready_batches), self._num_yielded)
            time.sleep(0.2)
        # print("next data after wait", len(self.ready_batches), self._num_yielded)
        # return self.ready_batches[self._num_yielded]
        return self.ready_batches.pop(0)

    def __next__(self):
        # shamelessly stolen from torch
        data = self._next_data()
        # print("finished next_data")
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
        self.dataset.loading_condition.acquire()

        converted_items = []
        h = 0
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
            # print('converted items len', len(converted_items))
            if len(converted_items) == self.batch_size:
                print(len(self.used_indices), self.dataset.load_len)
                if len(self.used_indices) == self.dataset.load_len:
                    # print("in release in conversion")
                    self.notify_overwrite = True
                    # self.dataset.loading_condition.notify()
                    self.dataset.loading_condition.release()

                batch = self._collate_fn(converted_items)
                for b in range(len(batch)):
                    batch[b] = batch[b].to(self.dataset.torch_device)
                self.ready_batches.append(batch)
                h += 1
                converted_items = []

                if self.notify_overwrite:
                    # print("waiting")
                    # wait for the *from* the loading thread
                    self.notify_overwrite = False
                    self.dataset.loading_condition.acquire()

        self.dataset.loading_condition.release()

    def thread_replace_converted_batches(self):

        while self.dataset.load_end + self.comm.size < self.dataset.local_data_end:
            print(
                "\t\tload batches",
                self.dataset.load_end + self.comm.size,
                self.dataset.local_data_end,
            )
            for d in self.dataset.dataset_names:
                if not self.dataset.np_buff_flag or d not in self.dataset.np_datasets:
                    hld = torch.tensor(self.f[d][self.dataset.load_start : self.dataset.load_end])
                else:
                    hld = self.f[d][self.dataset.load_start : self.dataset.load_end]
                self.__setattr__("hold" + d, hld)
            self.dataset.load_start = self.dataset.load_end
            self.dataset.load_end += self.dataset.load_len

            # todo: efficiency?? wait for lock1 *from* convert thread
            # with self.dataset.loading_condition:
            # print("in loading condition")
            self.dataset.loading_condition.acquire()
            for d in self.dataset.dataset_names:
                new = self.__getattribute__("hold" + d)
                dset = self.dataset.__getattribute__(d)
                # if isinstance(dset, torch.Tensor) and str(dset.device)[:3] == "gpu":
                #     new.to(dset.device)
                print("dset stuff", len(self.used_indices), new.shape)
                dset[self.used_indices] = new[: len(self.used_indices)]
                self.dataset.__setattr__(d, dset)
            # todo: give up lock / notify convert thread
            self.used_indices = []
            self.dataset.loading_condition.release()
            # time.sleep(0.5)
            # print("end of converted batches")

    def thread_load_next_dataset(self):
        # print("loading next dataset")
        # f = h5py.File(self.dataset.file, "r", driver="mpio", comm=self.comm.handle)
        # wrap at end of file (max difference is the number of processes)
        if self.dataset.load_start + self.comm.size >= self.dataset.total_size:
            self.dataset.load_start = 0
        # load dataset for next epoch
        self.dataset.load_end = self.dataset.load_start + self.dataset.load_initial
        for d in self.dataset.dataset_names:
            if not self.dataset.np_buff_flag or d not in self.dataset.np_datasets:
                hld = torch.tensor(self.f[d][self.dataset.load_start : self.dataset.load_end])
                self.__setattr__(d, hld)
            else:
                self.__setattr__(d, self.f[d][self.dataset.load_start : self.dataset.load_end])
        self.dataset.load_start = self.dataset.load_end
        self.dataset.load_end += self.dataset.load_len
        self.f.close()
