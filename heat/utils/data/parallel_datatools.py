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
from . import datatools


def queue_thread(q: queue.Queue):
    while True:
        items = q.get()
        func = items[0]
        args = items[1:]
        func(*args)
        q.task_done()


class PartialDataset(torch_data.Dataset):
    # todo: getitem, len
    def __init__(
        self,
        file: str,
        single_data_element_shape: Union[int, tuple],
        comm: MPICommunication = MPI_WORLD,
        dataset_names: Union[str, List[str]] = "data",
        available_memory: int = None,
        transform: Callable = None,
        target_transform: Callable = None,
        ishuffle: bool = True,
        np_buffer: bool = True,
        np_buffer_dataset_names: Union[str, List[str]] = "data",
        # folder=None,
    ):
        super(PartialDataset, self).__init__()
        self.file = file
        self.transform = transform
        self.target_transform = target_transform
        self.gpu = True if torch.cuda.device_count() > 0 else False
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # doing the file stuff first, folder to come later?
        # file_size = os.path.getsize(file)
        # file_size_per_pr = file_size / float(comm.size)

        # note: this is for a linux system!
        # if available_memory is None:
        #    available_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        # print(available_memory)

        # todo: add this back in for the datasets which can be loaded into memory
        # if file_size_per_pr < available_memory * 0.75:  # todo: what value should this be?
        #     self.partial_dataset = False
        #     self.ishuffle = ishuffle
        #     # need to fall back on the normal dataset stuff
        #     self.shuffle_list = [["ht" + dataset_names[0], dataset_names[0]]]
        #     data = io.load_hdf5(file, dataset_names[0], comm=comm.handle, split=0)
        #     sz = len(data)
        #     self.__setattr__("ht" + dataset_names[0], data)
        #     min_data_split = data.gshape[data.split] // data.comm.size
        #     self.lcl_half = min_data_split // 2
        #     arb_slice = slice(min_data_split)
        #     self._cut_slice = arb_slice
        #     self.__setattr__(dataset_names[0], data._DNDarray__array[arb_slice])
        #     self.length = sz
        #     if len(dataset_names) > 1:
        #         for d in dataset_names[1:]:
        #             data = io.load_hdf5(file, d, comm=comm, split=0)
        #             self.__setattr__("ht" + d, data)
        #             self.__setattr__(d, data._DNDarray__array)
        #             self.shuffle_list.append(["ht" + d, d])
        # else:
        if isinstance(single_data_element_shape, (tuple, list)):
            self.single_shape = list(single_data_element_shape)
        else:
            self.single_shape = [single_data_element_shape, single_data_element_shape]
        self.ishuffle = None
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
        self.length = sz
        # print("lcl_full_sz original", sz //comm.size)
        self.lcl_full_sz = (
            3000
        )  # sz // comm.size  # how many indices will go onto each process (len)
        # self.lcl_full_sz = 100000
        # load data that is half of of the available memory
        self.local_data_start = comm.rank * self.lcl_full_sz
        # local_data_end = int(
        #    (((0.25 * available_memory) / file_size_per_pr) * self.lcl_full_sz)
        #    + self.local_data_start
        # )
        # self.load_len = ((0.10 * available_memory) / file_size_per_pr) * self.lcl_full_sz
        # print(self.local_data_start, local_data_end, self.load_len)
        # temp values for small scale testing
        local_data_end = self.local_data_start + 3000
        self.load_len = 500  # int(local_data_end / 3)
        self.local_length = local_data_end - self.local_data_start

        self.next_start = local_data_end
        self.pr_data_end = self.local_data_start + self.lcl_full_sz

        # data being loaded from dataset_names parameter
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        self.dataset_names = dataset_names
        self.np_buffer = np_buffer
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
                hld = torch.tensor(f[d][self.local_data_start : local_data_end])
                self.__setattr__(d, hld)
            else:
                # hld = torch.zeros([local_data_end - self.local_data_start] + self.single_shape)
                # todo: how much data should be loaded here???
                self.__setattr__(d, f[d][self.local_data_start : local_data_end])
                self.__setattr__("conv" + d, [])
            self.__setattr__("next_" + d, None)
        f.close()
        self.load_thread = None
        self.epoch_end = False
        self.next_group_ready = False
        # need the number of loads required for an epoch
        self.loads_required = math.ceil(
            (self.lcl_full_sz - (local_data_end - self.local_data_start)) / self.load_len
        )
        self.loads_remaining = self.loads_required
        self.next_indices = set()
        self.next_indices_overflow = set()
        self.getitem_num = 0
        self.io_queue = queue.Queue()
        threading.Thread(target=queue_thread, args=[self.io_queue], daemon=True).start()
        self.convert_queue = queue.Queue()
        threading.Thread(target=queue_thread, args=[self.convert_queue], daemon=True).start()
        self.batch_loading_condition = threading.Condition()
        self.getitem_conversion = [None] * len(self.dataset_names)
        # todo: needs to be overwritten by user from the getitem script
        self.transform_list = [transform, None]
        if self.np_buffer:
            self.last_converted_batch = 0
            self.num_bch_conv = 0
            self.num_file_loads = 0
            self.converted_items_list = []
            self.converted_batches = {}
            self.transformed_batches = {}
        else:
            self.last_converted_batch = None
            self.num_bch_conv = None
            self.num_file_loads = None
            self.converted_batches = None
            self.converted_items_list = None
            self.transformed_batches = None

    def thread_convert_items_to_batches(self, items, batch_size=None):
        if isinstance(items, int):
            items = [items]

        converted_items = []
        for i in items:
            single_item = list(self[i])
            # have the item, need to convert from numpy to torch
            for ii in range(len(single_item)):
                if self.getitem_conversion[ii] is not None:
                    # note: self.getitem_conversion should have the function to convert
                    #   the np array for the proper element
                    single_item[ii] = self.getitem_conversion[ii](i)
                # have all torch stuff here
                # need to do the transform now
            converted_items.append(single_item)
        self.converted_items_list.extend(converted_items)
        # print(len(converted_items))
        if batch_size is not None:
            # add the batches to the converted batches dictionary
            if len(self.converted_items_list) // batch_size >= 1:
                # if len(self.converted_items_list) % batch_size == 0:
                bs = len(self.converted_items_list) // batch_size
                for b in range(bs):
                    self.converted_batches[self.num_bch_conv] = self.converted_items_list[
                        :batch_size
                    ]
                    self.num_bch_conv += 1
                    self.converted_items_list = self.converted_items_list[batch_size:]
            else:
                if self.num_bch_conv not in self.converted_batches.keys():
                    self.converted_batches[self.num_bch_conv] = []
                self.converted_batches[self.num_bch_conv].extend(self.converted_items_list)
            self.converted_items_list = []

    def load_next_group(self):
        # nonblocking calls to start the loading of the next dataset
        # need to spin off a thread to do this...
        if self.loads_remaining == 0:
            return
        next_end = self.next_start + self.load_len
        wrap = False
        rem = 0
        if next_end > self.length:
            rem = next_end - self.length
            next_end = self.length
            wrap = True
        f = h5py.File(self.file, "r")
        print("loading", self.next_start, next_end, rem, self.getitem_num)
        nxt_start = self.next_start
        self.io_queue.put((self._thread_loading_from_file, f, wrap, nxt_start, next_end, rem))
        self.num_file_loads += 1
        self.next_start = next_end if next_end < self.length else rem

    def _thread_loading_from_file(self, f, wrap, next_start, next_end, rem):
        for d in self.dataset_names:
            nxt = f[d][next_start:next_end]
            if d not in self.np_datasets:
                nxt = torch.tensor(nxt)  # todo: device?? -> CPU
            print("loading", self.next_start, next_end, rem, self.getitem_num)
            if wrap:
                nxt2 = f[d][0:rem]  # rem + 1?
                if not self.np_buffer or d not in self.np_datasets:
                    nxt2 = torch.tensor(nxt2)
                    nxt = torch.cat((nxt, nxt2), dim=0)
                else:
                    nxt = np.concatenate((nxt, nxt2), axis=0)
                    del nxt2
                # print("wraping")
            self.__setattr__("next_" + d, nxt)
            del nxt
        self.next_group_ready = True
        print("finished thread loading", next_start, next_end)

    def _thread_insert_group(self, entries, nxt_inds):
        # insert into numpy dataset if its there
        #   else: put into the torch tensor -> in this case need to send to GPU
        print("start insert group")
        if entries > self.load_len:
            entries = self.load_len
        nxt_inds = list(nxt_inds)
        nxt_inds = nxt_inds[:entries]
        for d in self.dataset_names:
            hld = self.__getattribute__(d)
            nxt_tens = self.__getattribute__("next_" + d)[: len(nxt_inds)]
            hld[nxt_inds] = nxt_tens

            self.__setattr__(d, hld)
        print("finish insert group")

    def insert_group(self, entries, next_inds):
        # todo: block the main thread here?
        if self.loads_remaining == 0:
            return
        # insert the new data into the target dataset/s with
        self.io_queue.put((self._thread_insert_group, entries, next_inds))
        self.loads_remaining -= 1

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
        return self.local_length


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

        # todo: support other samplers: for now its only random!!!
        if isinstance(self.dataset, PartialDataset) and self.dataset.partial_dataset:
            n = self.dataset.local_length
            wait_point = n // self.batch_size
            rem = n % self.batch_size
            if rem and self._drop_last:
                wait_point += 1
            self.wait_points = [wait_point * (i + 1) for i in range(self.dataset.loads_required)]
            # slice the dataset so that it is only full batches
            rng_sampl_lists = range(1, self.dataset.lcl_full_sz // n)
            self.rand_samp_list = torch.randperm(n)[: rem * -1].tolist()
            # self.wait_points = [wait_point]
            # # slice the dataset so that it is only full batches
            # self.rand_samp_list = torch.randperm(n)[:wait_point].tolist()
            for _ in rng_sampl_lists:
                # self.wait_points.append(wait_point * (i + 1))
                self.rand_samp_list.extend(torch.randperm(n)[: rem * -1].tolist())
            # print(len(self.rand_samp_list), self.batch_size, self.dataset.lcl_full_sz)
            self.num_batches = 5  # sum(self.wait_points)
            self.inds = []

            # if the number of batches to be loaded equates to more data than is loaded then make it smaller
            if pre_load_batches * self.batch_size > self.dataset.local_length:
                pre_load_batches = self.dataset.local_length // self.batch_size
            self.pre_loaded_batches = pre_load_batches

            self.batches_sent_to_transform = pre_load_batches
            self.iter_loaded_batches = pre_load_batches
            # these batches are only SENT to the thread to convert them
            self.length = len(self.rand_samp_list) // self.batch_size
        else:
            self.rand_samp_list = torch.randperm(len(self.dataset)).tolist()
            self.length = len(self._sampler_iter)

        self._dataset_fetcher = torch_data.dataloader._DatasetKind.create_fetcher(
            self._dataset_kind,
            loader.DataLoader.dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )

        self.batch_num_to_return = 0

    def __len__(self):
        return self.length

    @staticmethod
    def thread_transform_dict_device(dataset, target_batch, collate_func):
        if dataset.np_buffer is None:
            raise NotImplementedError(
                "this shouldnt be reached! function is only for partial datasets"
            )
        if target_batch in dataset.transformed_batches.keys():
            return
        w = 0.0
        # print("waiting in transform", target_batch)
        while target_batch not in dataset.converted_batches.keys():
            # print(dataset.converted_batches.keys())
            time.sleep(0.01)
            w += 0.01
        print("batch:", target_batch, "transform wait:", w)
        batch = dataset.converted_batches[target_batch].copy()
        # del dataset.converted_batches[target_batch]
        # batch is a list of tensors here
        for b in range(len(batch)):
            if isinstance(batch[b], (tuple, list)):
                batch[b] = list(batch[b])
                for bi in range(len(batch[b])):
                    if dataset.transform_list[bi] is not None:
                        batch[b][bi] = dataset.transform_list[bi](batch[b][bi])
                    batch[b][bi] = batch[b][bi].to(dataset.torch_device)
            elif isinstance(batch[b], torch.Tensor):
                if len(dataset.transform_list) == 1:
                    batch[b] = dataset.transform_list[0](batch[b])
                batch[b] = batch[b].to(dataset.torch_device)
            else:
                raise TypeError(
                    f"batches should be either torch tensors or a list/tuple of tensors, "
                    f"currently {type(batch)}"
                )
        # collate fn and put it on the target device
        dataset.transformed_batches[target_batch] = collate_func(batch)
        print(
            "created", target_batch, " in transformed batches", dataset.transformed_batches.keys()
        )
        del dataset.converted_batches[target_batch]

    def _next_data(self):
        t0 = time.perf_counter()
        # print("in next data")
        if not self.dataset.np_buffer:
            # shamelessly stolen from torch: torch_data.utils._SingleProcessDataLoaderIter
            index = next(self._sampler_iter)  # may raise StopIteration
            data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
            if self._pin_memory:
                # todo: where is this function??
                data = torch_data._utils.pin_memory.pin_memory(data)
                # torch.utils.data._utils.pin_memory.pin_memory(data)
            return data
        # ===================================== partial datasets ===========================================
        num_to_ret = self.batch_num_to_return
        num_to_delete = num_to_ret - 1 if num_to_ret > 0 else None
        num_to_transform = num_to_ret + self.pre_loaded_batches - 1  # todo: more than 1 in front?
        num_to_convert = num_to_ret + self.pre_loaded_batches

        print("\tnext data", num_to_delete, num_to_ret, num_to_transform, num_to_convert)
        self.num_batches = 5
        if num_to_ret == self.num_batches:
            raise StopIteration
        if num_to_convert < self.num_batches:
            # load the items from the numpy array to a torch tensor
            print("adding convert number to queue", num_to_convert)
            self.dataset.io_queue.put(
                (
                    self.dataset.thread_convert_items_to_batches,
                    self.rand_samp_list[
                        num_to_convert * self.batch_size : (num_to_convert + 1) * self.batch_size
                    ],
                    self.batch_size,
                )
            )

        if num_to_delete is not None:
            print("deleting batch :", num_to_delete)
            prev_batch_inds = self.rand_samp_list[
                num_to_delete * self.batch_size : num_to_ret * self.batch_size
            ]
            # getitem helper for the previous batch indexes
            self.dataset.getitem_index_helper(prev_batch_inds)
            # delete the previous batch
            del self.dataset.transformed_batches[num_to_delete]

        # check if the next batch needs transforming, if its less then the number of batches
        if (
            num_to_transform < self.num_batches
            and num_to_transform <= self.batches_sent_to_transform
        ):
            print(num_to_transform, self.batches_sent_to_transform)
            self.dataset.convert_queue.put(
                (
                    self.thread_transform_dict_device,
                    self.dataset,
                    num_to_transform,
                    self._collate_fn,
                )
            )
            self.batches_sent_to_transform += 1
        # check if the data is there
        w1 = 0.0
        print("waiting for", num_to_ret, self.dataset.transformed_batches.keys())
        while num_to_ret not in self.dataset.transformed_batches.keys():
            # todo: locking / waiting
            time.sleep(0.01)
            w1 += 0.01
        print("next data:", num_to_ret, "time:", time.perf_counter() - t0, "\n")
        return self.dataset.transformed_batches[num_to_ret]
        # return [None, None]

    def __next__(self):
        # shamelessly stolen from torch
        data = self._next_data()
        # todo: add the last batch to the getitem_index_helper after the last data is used...??
        # if self.batch_num_to_return == self.num_batches:
        #     self.dataset.getitem_index_helper(self.rand_samp_list[self.batch_num_to_return * self.batch_size :])
        self.batch_num_to_return += 1  # for next time this is called!

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
        if self.dataset.partial_dataset:
            # do the first few loads before next is called
            for xi in range(self.pre_loaded_batches):
                batch = self.rand_samp_list[xi * self.batch_size : (xi + 1) * self.batch_size]
                # todo: send next_inds to data placement loader, still todo????
                self.dataset.io_queue.put(
                    (self.dataset.thread_convert_items_to_batches, batch.copy(), self.batch_size)
                )
                self.inds.append(batch)
                # load the first batch to device here
                self.dataset.convert_queue.put(
                    (self.thread_transform_dict_device, self.dataset, xi, self._collate_fn)
                )
        return self
