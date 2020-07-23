import os
import h5py
import math
import numpy as np
import queue
import threading
import torch
import itertools
import time
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
        #if available_memory is None:
        #    available_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        #print(available_memory)

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
        self.lcl_full_sz = sz // comm.size  # how many indices will go onto each process (len)
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
        local_data_end = self.local_data_start + 1000
        self.load_len = 512  # int(local_data_end / 3)
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
        for d in dataset_names:
            # this should set the dataset attributes like data, by getting the data from the file
            #   up to the local data length
            if not np_buffer or d not in np_buffer_dataset_names:
                hld = torch.tensor(f[d][self.local_data_start : local_data_end])
            else:
                hld = torch.zeros([local_data_end - self.local_data_start] + self.single_shape)
                # todo: how much data should be loaded here???
                self.__setattr__("np" + d, f[d][self.local_data_start : local_data_end])

            self.__setattr__(d, hld.to(self.torch_device))
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
        self.last_converted_batch = 0 if self.np_buffer else None
        self.batch_loading_condition = threading.Condition()

    def load_item_transform(self, item):
        # need to either do a standard load here if not overwritten
        # this should be overwritten in the case that there is a requirement to load the data with numpy first
        # if not self.np_buffer:
        #     return item
        return item

    # def convert_items(self, items, dataset, target=False):
    # self.convert_queue.put((self._thread_insert_group, items, dataset, target))

    def thread_convert_items(self, items, target=False):
        # items must be a list or a torch tensor
        for dataset_lp in self.np_datasets:
            # get the numpy dataset
            np_data = self.__getattribute__("np" + dataset_lp)[items]
            t_data = torch.zeros([len(np_data)] + self.single_shape)
            rem_list = []
            for i in range(len(items)):
                try:
                    # convert the item to a torch tensor:
                    hold = self.load_item_transform(item=items[i], image=np_data[i])
                    if target:
                        t_data[i] = self.target_transform(hold)
                    else:
                        hld1 = self.transform(hold)
                        t_data[i] = hld1
                except ValueError:
                    # todo: investigate further when and how often this happens
                    rem_list.append(i)
            #   write to the torch set in the items
            dat = self.__getattribute__(dataset_lp)
            keep_list = list(range(len(items)))
            if len(rem_list) > 0:
                for ridx in reversed(rem_list):
                    del keep_list[ridx]
                    del items[ridx]
            # if GPU is there, put t_data on the GPUs
            t_data = t_data[keep_list].to(self.torch_device)
            dat[items] = t_data
            # todo: required?
            self.__setattr__(dataset_lp, dat)
        self.last_converted_batch += 1

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
        self.next_start = next_end if next_end < self.length else rem

    def _thread_loading_from_file(self, f, wrap, next_start, next_end, rem):
        for d in self.dataset_names:
            nxt = f[d][next_start:next_end]
            if not self.np_buffer or d not in self.np_datasets:
                nxt = torch.tensor(nxt)
            # print('loading', self.next_start, next_end, rem, self.getitem_num)
            if wrap:
                nxt2 = f[d][0:rem]  # rem + 1?
                if not self.np_buffer or d not in self.np_datasets:
                    nxt2 = torch.tensor(nxt2)
                    nxt = torch.cat((nxt, nxt2), dim=0)
                else:
                    nxt = np.concatenate((nxt, nxt2), axis=0)
                print("wraping")
                # nxt2 = torch.tensor(f[d][0:rem])  # rem + 1?
                # nxt = torch.cat((nxt, nxt2), dim=0)
            self.__setattr__("next_" + d, nxt)
        self.next_group_ready = True

    def _thread_insert_group(self, entries, nxt_inds):
        # insert into numpy dataset if its there
        #   else: put into the torch tensor -> in this case need to send to GPU
        if entries > self.load_len:
            entries = self.load_len
        nxt_inds = list(nxt_inds)
        # self.next_indices = torch.empty_like(nxt_inds)
        nxt_inds = nxt_inds[:entries]
        for d in self.dataset_names:
            if d in self.np_datasets:
                hld = self.__getattribute__("np" + d)
            else:
                # todo: get from gpu??
                hld = self.__getattribute__(d)
            # print(len(nxt_inds), entries, hld.shape, self.loads_remaining)
            if d not in self.np_datasets:
                nxt_tens = self.__getattribute__("next_" + d)
                nxt_tens = nxt_tens[: len(nxt_inds)].to(self.torch_device)
            else:
                nxt_tens = self.__getattribute__("next_" + d)[: len(nxt_inds)]
            hld[nxt_inds] = nxt_tens

            # todo: is the required?
            if d in self.np_datasets:
                self.__setattr__("np" + d, hld)
            else:
                # gpu insert
                hld = hld.to(self.torch_device)
                self.__setattr__(d, hld)

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


class LoadingBatchSampler(torch_data.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super(LoadingBatchSampler, self).__init__(sampler, batch_size, drop_last)
        if self.sampler.replacement:
            raise TypeError("Replacement drawing not implemented for loading sampler")
        # self.old_next = self.__next__
        self.cnt = 0

    def __iter__(self):
        dset = self.sampler.data_source
        batch = []
        samp_iter = self.sampler.__iter__()

        if not dset.np_buffer and not dset.partial_dataset:
            for idx in samp_iter:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch

        else:  # either np buffer or partial dataset
            iter_clones1, iter_clones2 = itertools.tee(samp_iter)
            if dset.np_buffer:
                for xi in range(3):
                    next_inds = [0] * self.batch_size
                    for idx in range(self.batch_size):
                        next_inds[idx] = next(iter_clones2)
                        # todo: send next_inds to data placement loader
                    dset.convert_queue.put((dset.thread_convert_items, next_inds.copy(), False))
            next_inds = []
            prev_batch_inds = []
            for idx in iter_clones1:
                batch.append(idx)
                try:
                    next_inds.append(next(iter_clones2))
                except StopIteration:
                    pass
                # todo: send the previous batch to the threads
                if len(batch) == self.batch_size:
                    self.cnt += 1
                    # do the conversion on the current batch
                    if dset.partial_dataset and len(next_inds) > 0:
                        # todo: wait for the setting to be done
                        next_inds = self._conversion_helper(dset, next_inds)

                    # todo: do the getitem on the previous batch
                    if len(prev_batch_inds) == 0:
                        # if there is no previous batch:
                        #   yield the batch, and save the batch to the previous batch inds
                        yield batch
                    if dset.partial_dataset and len(prev_batch_inds) > 0:
                        # there is a previous batch:
                        #   do the getitem helper on the previous batch to only overwrite that
                        dset.getitem_index_helper(batch)
                    # todo: send next_inds to the data placement loader
                    prev_batch_inds = batch.copy()
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                # dont need to worry about the next inds here, that iterator is exhausted
                yield batch
            if dset.partial_dataset and len(batch) > 0:
                dset.getitem_index_helper(batch)

    def _conversion_helper(self, dset, next_inds):
        if len(next_inds) > 0:
            dset.convert_queue.put((dset.thread_convert_items, next_inds.copy(), False))
        next_inds = []
        # wait for this batch to be done from the other thread
        xx = 0
        while dset.last_converted_batch < self.cnt:
            time.sleep(0.01)
            xx += 1
        print(f"waited for {xx} loops")
        return next_inds
