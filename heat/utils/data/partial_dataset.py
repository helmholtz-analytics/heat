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
        #q.task_done()
        #func = items[0]
        #args = items[1:]
        #func(*args)
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
        #self.lcl_full_sz = sz // comm.size
        self.lcl_full_sz = 10000
        # load data that is half of of the available memory
        self.local_data_start = comm.rank * self.lcl_full_sz
        #self.local_data_end = (
        #    (comm.rank + 1) * self.lcl_full_sz if comm.rank != comm.size - 1 else self.total_size
        #)
        self.local_data_end = (comm.rank + 1) * self.lcl_full_sz
        self.local_length = self.local_data_end - self.local_data_start

        # temp values for small scale testing
        self.load_initial = 5000
        self.load_len = 1000  # int(local_data_end / 3)

        self.load_start = self.local_data_start
        # self.load_end = self.load_start + 
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
            # this should set the dataset attributes like data, by getting the data from the file
            #   up to the local data length
            if not np_buffer or d not in np_buffer_dataset_names:
                hld = torch.tensor(f[d][self.load_start : self.load_end])
                self.__setattr__(d, hld)
            else:
                # this is loading the data to a buffer
                # todo: how much data should be loaded here???
                self.__setattr__(d, f[d][self.load_start : self.load_end])
            #     self.__setattr__("conv" + d, [])
            # self.__setattr__("next_" + d, None)
        self.load_start = self.load_end
        self.load_end += self.load_len
        f.close()
        self.load_thread = None
        self.epoch_end = False
        # need the number of loads required for an epoch
        self.loads_required = math.ceil(self.lcl_full_sz / self.load_len)
        self.loads_remaining = self.loads_required
        # self.next_indices = set()
        # self.next_indices_overflow = set()
        # self.getitem_num = 0
        self.loading_queue = queue.Queue()
        self.loading_condition = threading.Condition()
        self.loading_thread = threading.Thread(
            target=queue_thread, args=[self.loading_queue], daemon=True
        ).start()
        self.convert_queue = queue.Queue()
        # self.convert_condition = threading.Condition()
        threading.Thread(target=queue_thread, args=[self.convert_queue], daemon=True).start()

    # end init ================================================================================

    # def convert_all_items_to_batches(self, items, load_list, batch_size):
    #     if not isinstance(items, (torch.Tensor, list, tuple)):
    #         raise TypeError("This function is to put all of the items into the queue")
    #     for i in items:
    #         if i in load_list:
    #             # todo: put the loading somewhere
    #             pass
    #         self.thread_convert_items_to_batches(i, batch_size)
    #
    # def thread_convert_items_to_batches(self, items, batch_size=None, reset_batch_number=False):
    #     if isinstance(items, int):
    #         items = [items]
    #
    #     converted_items = []
    #     for i in items:
    #         single_item = list(self[i])
    #         # have the item, need to convert from numpy to torch
    #         for ii in range(len(single_item)):
    #             if self.getitem_conversion[ii] is not None:
    #                 # note: self.getitem_conversion should have the function to convert
    #                 #   the np array for the proper element
    #                 single_item[ii] = self.getitem_conversion[ii](i)
    #             # have all torch stuff here
    #             # need to do the transform now
    #         converted_items.append(single_item)
    #     self.converted_items_list.extend(converted_items)
    #     # print(len(converted_items))
    #     if batch_size is not None:
    #         print("\nbatch number addition", self.num_bch_conv)
    #         # add the batches to the converted batches dictionary
    #         if len(self.converted_items_list) // batch_size >= 1:
    #             # if len(self.converted_items_list) % batch_size == 0:
    #             bs = len(self.converted_items_list) // batch_size
    #             for b in range(bs):
    #                 self.converted_batches[self.num_bch_conv] = self.converted_items_list[
    #                     :batch_size
    #                 ]
    #                 self.num_bch_conv += 1
    #                 self.converted_items_list = self.converted_items_list[batch_size:]
    #         else:
    #             if self.num_bch_conv not in self.converted_batches.keys():
    #                 self.converted_batches[self.num_bch_conv] = []
    #             elif len(self.converted_batches[self.num_bch_conv]) == batch_size:
    #                 self.num_bch_conv += 1
    #                 self.converted_batches[self.num_bch_conv] = []
    #             self.converted_batches[self.num_bch_conv].extend(self.converted_items_list)
    #         self.converted_items_list = []
    #
    # def load_next_group(self):
    #     # nonblocking calls to start the loading of the next dataset
    #     # need to spin off a thread to do this...
    #     if self.loads_remaining == 0:
    #         return
    #     next_end = self.next_start + self.load_len
    #     wrap = False
    #     rem = 0
    #     if next_end > self.length:
    #         rem = next_end - self.length
    #         next_end = self.length
    #         wrap = True
    #     f = h5py.File(self.file, "r")
    #     print("loading", self.next_start, next_end, rem, self.getitem_num)
    #     nxt_start = self.next_start
    #     self.io_queue.put((self._thread_loading_from_file, f, wrap, nxt_start, next_end, rem))
    #     self.num_file_loads += 1
    #     self.next_start = next_end if next_end < self.length else rem
    #
    # def _thread_loading_from_file(self, f, wrap, next_start, next_end, rem):
    #     for d in self.dataset_names:
    #         nxt = f[d][next_start:next_end]
    #         if d not in self.np_datasets:
    #             nxt = torch.tensor(nxt)  # todo: device?? -> CPU
    #         print("loading", self.next_start, next_end, rem, self.getitem_num)
    #         if wrap:
    #             nxt2 = f[d][0:rem]  # rem + 1?
    #             if not self.np_buff_flag or d not in self.np_datasets:
    #                 nxt2 = torch.tensor(nxt2)
    #                 nxt = torch.cat((nxt, nxt2), dim=0)
    #             else:
    #                 nxt = np.concatenate((nxt, nxt2), axis=0)
    #                 del nxt2
    #             # print("wraping")
    #         self.__setattr__("next_" + d, nxt)
    #         del nxt
    #     self.next_group_ready = True
    #     print("finished thread loading", next_start, next_end)
    #
    # def _thread_insert_group(self, entries, nxt_inds):
    #     # insert into numpy dataset if its there
    #     #   else: put into the torch tensor -> in this case need to send to GPU
    #     print("start insert group")
    #     if entries > self.load_len:
    #         entries = self.load_len
    #     nxt_inds = list(nxt_inds)
    #     nxt_inds = nxt_inds[:entries]
    #     for d in self.dataset_names:
    #         hld = self.__getattribute__(d)
    #         nxt_tens = self.__getattribute__("next_" + d)[: len(nxt_inds)]
    #         hld[nxt_inds] = nxt_tens
    #
    #         self.__setattr__(d, hld)
    #     print("finish insert group")
    #
    # def insert_group(self, entries, next_inds):
    #     # todo: block the main thread here?
    #     if self.loads_remaining == 0:
    #         return
    #     # insert the new data into the target dataset/s with
    #     self.io_queue.put((self._thread_insert_group, entries, next_inds))
    #     self.loads_remaining -= 1

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
            self.used_indices = []
            self.loads_overwritten = 0
            self.ready_batches = []
            # self.loading_lock = threading.Condition()

            mod_batch = self.dataset.load_len % self.batch_size
            if mod_batch != 0:
                self.dataset.load_len += (self.batch_size - mod_batch)
                self.dataset.load_end = self.dataset.load_start + self.dataset.load_len
            # start loading the data
            self.dataset.loading_queue.put(self.thread_replace_converted_batches)
            # generate all indices
            index_list = []
            idx_repeats = math.ceil(self.dataset.lcl_full_sz / self.dataset.load_initial)
            for _ in range(idx_repeats):
                index_list.extend(rand_samp_list)
            # start the conversion
            self.dataset.convert_queue.put((self.thread_convert_all, index_list))
            self.length = len(index_list) // self.batch_size
            print("length", self.length, len(index_list), self.batch_size, self.dataset.load_len)

            if not self._drop_last and len(index_list) % self.batch_size != 0:
                # todo: implement drop last!
                self.length += 1
            # self.write_lock = threading.Lock()
            # n = self.dataset.local_length
            # wait_point = n // self.batch_size
            # rem = n % self.batch_size
            # if rem and self._drop_last:
            #     wait_point += 1
            # self.wait_points = [wait_point * (i + 1) for i in range(self.dataset.loads_required)]
            # # slice the dataset so that it is only full batches
            # rng_sampl_lists = range(1, self.dataset.lcl_full_sz // n)
            # self.rand_samp_list = torch.randperm(n)[: rem * -1].tolist()
            # # self.wait_points = [wait_point]
            # # # slice the dataset so that it is only full batches
            # # self.rand_samp_list = torch.randperm(n)[:wait_point].tolist()
            # for _ in rng_sampl_lists:
            #     # self.wait_points.append(wait_point * (i + 1))
            #     self.rand_samp_list.extend(torch.randperm(n)[: rem * -1].tolist())
            # # print(len(self.rand_samp_list), self.batch_size, self.dataset.lcl_full_sz)
            # self.num_batches = 5  # sum(self.wait_points)
            # self.inds = []
            #
            # # if the number of batches to be loaded equates to more data than is loaded then make it smaller
            # if pre_load_batches * self.batch_size > self.dataset.local_length:
            #     pre_load_batches = self.dataset.local_length // self.batch_size
            # self.pre_loaded_batches = pre_load_batches
            # reset = True
            # for xi in range(self.pre_loaded_batches):
            #     batch = self.rand_samp_list[xi * self.batch_size: (xi + 1) * self.batch_size]
            #     # todo: send next_inds to data placement loader, still todo????
            #     self.dataset.io_queue.put(
            #         (
            #             self.dataset.thread_convert_items_to_batches,
            #             batch.copy(),
            #             self.batch_size,
            #             reset,
            #         )
            #     )
            #     reset = False
            #     self.inds.append(batch)
            #     # load the first batch to device here
            #     self.dataset.convert_queue.put(
            #         (self.thread_transform_dict_device, self.dataset, xi, self._collate_fn)
            #     )
            #
            # self.batches_sent_to_transform = pre_load_batches
            # self.iter_loaded_batches = pre_load_batches
            # # these batches are only SENT to the thread to convert them
            # self.length = len(self.rand_samp_list) // self.batch_size
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
        while len(self.ready_batches) < 1:
            time.sleep(0.2)
        return self.ready_batches.pop(0)

    def __next__(self):
        # shamelessly stolen from torch
        data = self._next_data()
        # todo: add the last batch to the getitem_index_helper after the last data is used...??
        # if self.batch_num_to_return == self.num_batches:
        #     self.dataset.getitem_index_helper(self.rand_samp_list[self.batch_num_to_return * self.batch_size :])

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
                notify_wait = False
                if len(self.used_indices) == self.dataset.load_len:
                    notify_wait = True
                    self.dataset.loading_condition.notifyAll()
                    self.dataset.loading_condition.release()

                #batch = self._collate_fn(converted_items).to(self.dataset.torch_device)
                batch = self._collate_fn(converted_items)
                for b in range(len(batch)):
                    batch[b] = batch[b].to(self.dataset.torch_device)
                self.ready_batches.append(batch)
                converted_items = []

                if notify_wait:
                    print("waiting")
                    # wait for the *from* the loading thread
                    self.dataset.loading_condition.acquire()
                    self.dataset.loading_condition.wait()

    def thread_replace_converted_batches(self):
        f = h5py.File(self.dataset.file, "r", driver="mpio", comm=self.comm.handle)

        if self.dataset.next_start >= self.dataset.local_data_end:
            print('wrap at end')
            # wrap at end of file
            if self.dataset.next_start + self.comm.size >= self.dataset.total_size:
                self.dataset.next_start = 0
            # load dataset for next epoch
            self.dataset.load_end = self.dataset.load_start + self.dataset.load_initial
            for d in self.dataset.dataset_names:
                if not self.dataset.np_buff_flag or d not in self.dataset.np_datasets:
                    hld = torch.tensor(f[d][self.dataset.load_start : self.dataset.load_end])
                    self.__setattr__(d, hld)
                else:
                    self.__setattr__(d, f[d][self.dataset.load_start : self.dataset.load_end])
            f.close()
            self.dataset.next_start = self.dataset.load_end
            self.dataset.load_end += self.dataset.load_len
            return

        # load the next data
        for d in self.dataset.dataset_names:
            # this should set the dataset attributes like data, by getting the data from the file
            #   up to the local data length
            if not self.dataset.np_buff_flag or d not in self.dataset.np_datasets:
                # todo: load to CPU first? will this matter?
                hld = torch.tensor(f[d][self.dataset.load_start : self.dataset.load_end])
            else:
                hld = f[d][self.dataset.load_start : self.dataset.load_end]
            self.__setattr__("hold" + d, hld)
        f.close()
        self.dataset.next_start = self.dataset.load_end
        self.dataset.load_end += self.dataset.load_len

        # wait for lock1 *from* convert thread
        self.dataset.loading_condition.acquire()
        print("replace wait", len(self.used_indices))
        self.dataset.loading_condition.wait()
        # todo: acquire lock2 for setting
        for d in self.dataset.dataset_names:
            new = self.__getattribute__("hold" + d)
            dset = self.dataset.__getattribute__(d)
            if isinstance(dset, torch.Tensor) and str(dset.device)[:3] == "gpu":
                new = new.to(dset.device)
            dset[self.used_indices] = new
            self.dataset.__setattr__(d, dset)
        # todo: give up lock / notify convert thread
        self.used_indices = []
        self.dataset.loading_condition.notifyAll()
        print("releasing")
        self.dataset.loading_condition.release()
        # recursive call to load the next batch
        #self.thread_replace_converted_batches()
        self.dataset.loading_queue.put(self.thread_replace_converted_batches)
