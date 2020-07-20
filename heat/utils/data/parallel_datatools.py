import os
import h5py
import math
import queue
import threading
import torch
from torch.utils import data as torch_data
from typing import Callable, List, Iterator, Union

from ...core import dndarray
from ...core import io
from ...core.communication import MPICommunication
from ...core.communication import MPI_WORLD
from . import datatools


def io_thread(q: queue.Queue):
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
        comm: MPICommunication = MPI_WORLD,
        dataset_names: Union[str, List[str]] = "data",
        available_memory: int = None,
        transform: Callable = None,
        target_transform: Callable = None,
        ishuffle: bool = True
        # folder=None,
    ):
        super(PartialDataset, self).__init__()
        self.file = file
        self.transform = transform
        self.target_transform = target_transform
        # doing the file stuff first, folder to come later?
        file_size = os.path.getsize(file)
        file_size_per_pr = file_size / float(comm.size)

        # note: this is for a linux system!
        if available_memory is None:
            available_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

        # todo: add this back in for the datasets which can be loaded into memory
        if file_size_per_pr < available_memory * 0.75:  # todo: what value should this be?
            self.partial_dataset = False
            self.ishuffle = ishuffle
            # need to fall back on the normal dataset stuff
            self.shuffle_list = [["ht" + dataset_names[0], dataset_names[0]]]
            data = io.load_hdf5(file, dataset_names[0], comm=comm, split=0)
            sz = len(data)
            self.__setattr__("ht" + dataset_names[0], data)
            min_data_split = data.gshape[data.split] // data.comm.size
            self.lcl_half = min_data_split // 2
            arb_slice = slice(min_data_split)
            self._cut_slice = arb_slice
            self.__setattr__(dataset_names[0], data._DNDarray__array[arb_slice])
            self.length = sz
            if len(dataset_names) > 1:
                for d in dataset_names[1:]:
                    data = io.load_hdf5(file, d, comm=comm, split=0)
                    self.__setattr__("ht" + d, data)
                    self.__setattr__(d, data._DNDarray__array)
                    self.shuffle_list.append(["ht" + d, d])
        else:
            self.ishuffle = None
            self.partial_dataset = True
            f = h5py.File(file, "r", driver="mpio", comm=comm)
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
            self.lcl_full_sz = sz // comm.size  # how many indices will go onto each process (len)
            # load data that is half of of the available memory
            self.local_data_start = comm.rank * self.lcl_full_sz
            # local_data_end = int(
            #     (((0.50 * available_memory) / file_size_per_pr) * self.lcl_full_sz)
            #     + self.local_data_start
            # )
            # self.load_len = ((0.10 * available_memory) / file_size_per_pr) * self.lcl_full_sz
            # temp values for small scale testing
            local_data_end = self.local_data_start + 20000
            self.load_len = 10000
            self.local_length = local_data_end - self.local_data_start

            self.next_start = local_data_end
            self.pr_data_end = self.local_data_start + self.lcl_full_sz

            # data being loaded from dataset_names parameter
            if isinstance(dataset_names, str):
                dataset_names = [dataset_names]
            self.dataset_names = dataset_names
            for d in dataset_names:
                # this should set the dataset attributes like data, by getting the data from the file
                #   up to the local data length
                self.__setattr__(d, torch.tensor(f[d][self.local_data_start : local_data_end]))
                self.__setattr__("next_" + d, None)
                # todo: set proper device
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
            threading.Thread(target=io_thread, args=[self.io_queue], daemon=True).start()

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
            nxt = torch.tensor(f[d][next_start:next_end])
            # print('loading', self.next_start, next_end, rem, self.getitem_num)
            if wrap:
                print("wraping")
                nxt2 = torch.tensor(f[d][0:rem])  # rem + 1?
                nxt = torch.cat((nxt, nxt2), dim=0)
            self.__setattr__("next_" + d, nxt)
        self.next_group_ready = True

    def _thread_insert_group(self, entries, nxt_inds):
        if entries > self.load_len:
            entries = self.load_len
        nxt_inds = list(nxt_inds)
        # self.next_indices = torch.empty_like(nxt_inds)
        nxt_inds = nxt_inds[:entries]
        for d in self.dataset_names:
            hld = self.__getattribute__(d)
            # print(len(nxt_inds), entries, hld.shape, self.loads_remaining)
            hld[nxt_inds] = self.__getattribute__("next_" + d)[: len(nxt_inds)]
            self.__setattr__(d, hld)

    def insert_group(self, entries, next_inds):
        # todo: block the main thread here?
        # print("e", entries, self.loads_remaining)
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


# class LoadingBatchSampler(torch_data.BatchSampler):
#     def __init__(self, sampler, batch_size, drop_last):
#         super(LoadingBatchSampler, self).__init__(sampler, batch_size, drop_last)
#         if self.sampler.replacement:
#             raise TypeError("Replacement drawing not implemented for loading sampler")
#         # self.old_next = self.__next__
#         self.cnt = 0

# def __iter__(self):
#     batch = []
#     # print('start iter', self.sampler.data_source.local_data_start)
#     # self.sampler.data_source.getitem_num = 0
#     for idx in self.sampler:
#         batch.append(idx)
#         if len(batch) == self.batch_size:
#             # if self.sampler.data_source.partial_dataset:
#             #     self.sampler.data_source.getitem_index_helper(batch)
#             yield batch
#             batch = []
#     if len(batch) > 0 and not self.drop_last:
#         yield batch
#     # if self.sampler.data_source.partial_dataset and len(batch) > 0:
#     #     self.sampler.data_source.getitem_index_helper(batch)
#     #     self.sampler.data_source.insert_group(
#     #         self.sampler.data_source.getitem_num, self.sampler.data_source.next_indices
#     #     )
#         # todo: add the next load if it isnt the end...
#         # self.dataset.load_next_group()
#     # print('end iter', self.sampler.data_source.next_start)

# def __next__(self):
