import h5py
import numpy as np
import base64
import itertools
import torch
from torch.utils import data as torch_data
from typing import Callable, List, Iterator, Union

from ...core import dndarray
from ...core.communication import MPICommunication
from . import parallel_datatools

__all__ = ["DataLoader", "Dataset", "dataset_shuffle"]


class DataLoader:
    """
    Data Loader. The combines either a ``DNDarray`` or a torch ``Dataset`` with a sampler. It will provide an iterable
    over the local dataset and it will have a ``shuffle()`` function which calls the ``shuffle()`` function of the
    given Dataset. If a HeAT DNDarray is given a general Dataset will be created.

    Currently, the DataLoader supports only map-style datasets with single-process loading, and users the random
    batch sampler. The rest of the DataLoader functionality mentioned in :func:`torch.utils.data.dataloader` applies.

    Arguments:
        data : Dataset or DNDarray
            dataset from which to load the data.
        batch_size : int, optional
            how many samples per batch to load (default: ``1``).
        num_workers : int, optional
            how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn : callable, optional
            merges a list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory : bool, optional
            If ``True``, the data loader will copy Tensors into CUDA pinned memory before returning them.  If your
            data elements are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type, see
            the example below.
        drop_last : bool, optional
            set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
            the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
            the last batch will be smaller. (default: ``False``)
        timeout : int or float, optional
            if positive, the timeout value for collecting a batch from workers. Should always be non-negative.
            (default: ``0``)
        worker_init_fn : callable, optional
            If not ``None``, this will be called on each worker subprocess with the worker id
            (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: ``None``)
        lcl_dataset : torch.Dataset
            a PyTorch dataset from which the data will be returned by the created iterator
        transform : Callable
            transform to be given to Dataset creation if a Dataset is created

    Attributes
    ----------
    dataset : torch.data.utils.data.Dataset or heat.Dataset
        the dataset created from the local data
    DataLoader : torch.utils.data.DataLoader
        the local DataLoader object. Used in the creation of the iterable and the length
    _first_iter : bool
        flag indicating if the iterator created is the first one. If it is not, then the data will be shuffled before
        the iterator is created

    TODO: add data annotation
    """

    def __init__(
        self,
        data=None,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: Callable = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: Union[int, float] = 0,
        worker_init_fn: Callable = None,
        lcl_dataset: Union[torch_data.Dataset, parallel_datatools.PartialDataset] = None,
        transform: Callable = None,
    ):
        if isinstance(data, dndarray.DNDarray) and lcl_dataset is not None:
            self.dataset = Dataset(array=data, transform=transform)
        elif lcl_dataset:
            self.dataset = lcl_dataset
        else:
            raise TypeError(
                f"data must be a DNDarray or lcl_dataset must be given, data is currently: {type(data)}"
            )
        self.ishuffle = self.dataset.ishuffle
        # this is effectively setting ``shuffle`` to True
        # rand_sampler = torch_data.RandomSampler(self.dataset)
        # sampler = parallel_datatools.LoadingBatchSampler(rand_sampler, batch_size, drop_last)
        self.DataLoader = torch_data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
        self._first_iter = True
        self.last_epoch = False

    def __iter__(self) -> Iterator:
        # need a new iterator for each epoch
        if not self.dataset.partial_dataset:
            print("here1")
            self._full_dataset_shuffle_iter()
            return self.DataLoader.__iter__()
        # todo: do the loading of the next groups here
        self.dataset.loads_remaining = self.dataset.loads_required
        # need to start the first loader
        self.dataset.load_next_group()
        # print(self.dataset.loads_required)
        iters = [
            parallel_datatools.LoadingDataLoaderIter(self).__iter__()
            for _ in range(self.dataset.loads_required)
        ]
        return itertools.chain.from_iterable(iters)

    def __len__(self) -> int:
        return len(self.DataLoader)

    def _full_dataset_shuffle_iter(self):
        if not self.ishuffle:
            if self._first_iter:
                self._first_iter = False
            else:
                # shuffle after the first epoch but before the iterator is generated
                self.dataset.Shuffle()
        else:
            # start the shuffling for the next iteration
            if not self.last_epoch:
                self.dataset.Ishuffle()

            if self._first_iter:
                self._first_iter = False
            else:
                dataset_irecv(self.dataset)


class Dataset(torch_data.Dataset):
    """
    An abstract class representing a given dataset. This inherits from torch.utils.data.Dataset.

    This class is a general example for what should be done to create a Dataset. When creating a dataset all of the
    standard attributes should be set, the ``__getitem__``, ``__len__``, and ``shuffle`` functions must be defined.

        - ``__getitem__`` : how an item is given to the network
        - ``__len__`` : the number of data elements to be given to the network in total
        - ``Shuffle()`` : how the data should be shuffled between the processes. The function shown below is for a dataset
            composed of only data and without targets. The function :func:`dataset_shuffle` abstracts this. For this function
            only the dataset and a list of attributes to shuffle are given.
        - (optional) ``Ishuffle()`` : A non-blocking version of ``Shuffle()``, this is handled in the abstract function
            :func:`dataset_ishuffle`. It works similarly to :func:`dataset_shuffle`.

    As the amount of data across processes can be non-uniform, the dataset class will slice off the remaining elements
    on whichever processes have more data than the others. This should only be 1 element.
    The shuffle function will shuffle all of the data on the process.

    It is recommended that for DNDarrays, the split is either 0 or None

    Parameters
    ----------
    array : DNDarray
        DNDarray for which to great the dataset
    transform : Callable
        transformation to call before a data item is returned
    ishuffle : bool (optional)
        flag indicating whether to use non-blocking communications for shuffling the data between epochs
        Note: if True, the ``Ishuffle()`` function must be defined within the class
        Default: False

    Attributes
    ----------
    These are the required attributes. Optional attributed are whatever are required for the Dataset
    (see :class:`heat.utils.data.mnist.py`)
    htdata : DNDarray
        full data
    _cut_slice : slice
        slice to cut off the last element to get a uniform amount of data on each process
    comm : MPICommunicator
        communication object used to send the data between processes
    lcl_half : int
        half of the number of data elements on the process
    data : torch.Tensor
        the local data to be used in training
    transform : Callable
        transform to be called during the getitem function
    ishuffle : bool
        flag indicating if non-blocking communications are used for shuffling the data between epochs

    TODO: type annotation for array
    """

    def __init__(self, array, transform: Callable = None, ishuffle: bool = False):
        self.partial_dataset = False
        self.htdata = array
        self.comm = array.comm
        # create a slice to create a uniform amount of data on each process
        min_data_split = array.gshape[array.split] // array.comm.size
        self.lcl_half = min_data_split // 2
        arb_slice = [slice(None)] * array.ndim
        arb_slice[array.split] = slice(min_data_split)
        self._cut_slice = tuple(arb_slice)
        self.data = array._DNDarray__array[self._cut_slice]
        self.transform = transform
        self.ishuffle = ishuffle

    def __getitem__(self, index: Union[int, slice, tuple, list, torch.Tensor]) -> torch.Tensor:
        if self.transform:
            return self.transform(self.data[index])
        return self.data[index]

    def __len__(self) -> int:
        return self.data.shape[0]

    def Shuffle(self):
        """
        Send half of the local data to the process ``self.comm.rank + 1`` if available, else wrap around. After
        receiving the new data, shuffle the local tensor.
        """
        dataset_shuffle(dataset=self, attrs=[["htdata", "data"]])

    def Ishuffle(self):
        """
        Send half of the local data to the process ``self.comm.rank + 1`` if available, else wrap around. After
        receiving the new data, shuffle the local tensor.
        """
        dataset_ishuffle(dataset=self, attrs=[["htdata", "data"]])


def dataset_shuffle(dataset: Union[Dataset, torch_data.Dataset], attrs: List[list]):
    """
    Shuffle the given attributes of a dataset across multiple processes. This will send half of the data to rank + 1.
    Once the new data is received, it will be shuffled into the existing data on the process.

    This function will be called by the DataLoader automatically if ``dataset.ishuffle = False``.

    Parameters
    ----------
    dataset : Dataset
    attrs : List[List[str, str], ... ]
        List of lists each of which contains 2 strings. The strings are the handles corresponding to the Dataset
        attributes corresponding to the global data DNDarray and the local data of that array, i.e. [["htdata, "data"],]
        would shuffle the htdata around and set the correct amount of data for the ``dataset.data`` attribute. For
        multiple parameters multiple lists are required. I.e. [["htdata", "data"], ["httargets", "targets"]]

    Notes
    -----
    ``dataset.comm`` must be defined for this function to work.
    """
    # attrs should have the form [[heat array, sliced array], [...], ...]
    #       i.e. [['htdata', 'data]]
    # assume that all of the attrs have the same dim0 shape as the local data
    prm = torch.randperm(dataset.htdata._DNDarray__array.shape[0])
    comm = dataset.comm
    for att in attrs:
        ld = getattr(dataset, att[0])._DNDarray__array
        snd = ld[: dataset.lcl_half].clone()
        snd_shape, snd_dtype, snd_dev = snd.shape, snd.dtype, snd.device
        dest = comm.rank + 1 if comm.rank + 1 != comm.size else 0
        # send the top half of the data to the next process
        send_wait = comm.Isend(snd, dest=dest)
        del snd
        new_data = torch.empty(snd_shape, dtype=snd_dtype, device=snd_dev)
        src = comm.rank - 1 if comm.rank != 0 else comm.size - 1
        rcv_w = comm.Irecv(new_data, source=src)
        send_wait.wait()
        rcv_w.wait()
        # todo: put the rcv'ed stuff on GPU if available
        getattr(dataset, att[0])._DNDarray__array[: dataset.lcl_half] = new_data
        getattr(dataset, att[0])._DNDarray__array = getattr(dataset, att[0])._DNDarray__array[prm]
        setattr(dataset, att[1], getattr(dataset, att[0])._DNDarray__array[dataset._cut_slice])


def dataset_ishuffle(dataset: Union[Dataset, torch_data.Dataset], attrs: List[list]):
    """
    Shuffle the given attributes of a dataset across multiple processes, using non-blocking communications.
    This will send half of the data to rank + 1. The data must be received by the :func:`dataset_irecv` function.

    This function will be called by the DataLoader automatically if ``dataset.ishuffle = True``. This is set either
    during the definition of the class of its initialization by a given paramete.

    Parameters
    ----------
    dataset : Dataset
    attrs : List[List[str, str], ... ]
        List of lists each of which contains 2 strings. The strings are the handles corresponding to the Dataset
        attributes corresponding to the global data DNDarray and the local data of that array, i.e. [["htdata, "data"],]
        would shuffle the htdata around and set the correct amount of data for the ``dataset.data`` attribute. For
        multiple parameters multiple lists are required. I.e. [["htdata", "data"], ["httargets", "targets"]]

    Notes
    -----
    ``dataset.comm`` must be defined for this function to work.
    """
    # attrs should have the form [[heat array, sliced array], [...], ...]
    #       i.e. [['htdata', 'data]]
    # assume that all of the attrs have the same dim0 shape as the local data
    comm = dataset.comm
    ret_list = []
    for att in attrs:
        ld = getattr(dataset, att[0])._DNDarray__array
        snd = ld[: dataset.lcl_half].clone()
        snd_shape, snd_dtype, snd_dev = snd.shape, snd.dtype, snd.device
        dest = comm.rank + 1 if comm.rank + 1 != comm.size else 0
        # send the top half of the data to the next process
        send_wait = comm.Isend(snd, dest=dest, tag=99999)
        new_data = torch.empty(snd_shape, dtype=snd_dtype, device=snd_dev)
        src = comm.rank - 1 if comm.rank != 0 else comm.size - 1
        wait = comm.Irecv(new_data, source=src, tag=99999)
        ret_list.append([att, wait, new_data])
        send_wait.wait()
        del snd
    setattr(dataset, "rcv_list", ret_list)


def dataset_irecv(dataset: Union[Dataset, torch_data.Dataset]):
    """
    Receive the data sent by the :func:`dataset_ishuffle` function. This will wait for the data and then shuffle the
    data into the existing data on the process

    This function will be called by the DataLoader automatically if ``dataset.ishuffle = True``. This is set either
    during the definition of the class of its initialization by a given paramete.

    Parameters
    ----------
    dataset : Dataset

    Notes
    -----
    ``dataset.comm`` must be defined for this function to work.
    """
    setattr(dataset, "shuffle_prm", torch.randperm(dataset.htdata._DNDarray__array.shape[0]))
    rcv_list = getattr(dataset, "rcv_list")
    prm = getattr(dataset, "shuffle_prm")
    for rcv in rcv_list:
        rcv[1].wait()
        # todo: put the rcv'ed stuff on GPU if available
        getattr(dataset, rcv[0][0])._DNDarray__array[: dataset.lcl_half] = rcv[2]
        getattr(dataset, rcv[0][0])._DNDarray__array = getattr(dataset, rcv[0][0])._DNDarray__array[
            prm
        ]
        setattr(
            dataset, rcv[0][1], getattr(dataset, rcv[0][0])._DNDarray__array[dataset._cut_slice]
        )


def merge_files_imagenet_tfrecord(folder_name, output_folder=None):
    """
    merge multiple preprocessed imagenet TFRecord files together,
    result is one HDF5 file with all of the images stacked in the 0th dimension

    Parameters
    ----------
    folder_name : str, optional*
        folder location of the files to join, either filenames or folder_names must not be None
    output_folder : str, optional
        location to create the output files. Defaults to current directory

    Notes
    -----
    Metadata for both the created files (`imagenet_merged.h5` and `imagenet_merged_validation.h5`):

    The datasets are the combination of all of the images in the Image-net 2012 dataset.
    The data is split into training and validation.

    imagenet_merged.h5 -> training
    imagenet_merged_validation.h5 -> validation

    both files have the same internal structure:
    - file
            * "images" : encoded ASCII string of the decoded RGB JPEG image.
                    - to decode: `torch.as_tensor(bytearray(base64.binascii.a2b_base64(string_repr.encode('ascii'))), dtype=torch.uint8)`
                    - note: the images must be reshaped using: `.reshape(file["metadata"]["image/height"], file["metadata"]["image/height"], 3)`
                            (3 is the number of channels, all images are RGB)
            * "metadata" : the metadata for each image quotes are the titles for each column
                    0. "image/height"
                    1. "image/width"
                    2. "image/channels"
                    3. "image/class/label"
                    4. "image/object/bbox/xmin"
                    5. "image/object/bbox/xmax"
                    6. "image/object/bbox/ymin"
                    7. "image/object/bbox/ymax"
                    8. "image/object/bbox/label"
            * "file_info" : string information related to each image
                    0. "image/format"
                    1. "image/filename"
                    2. "image/class/synset"
                    3. "image/class/text"


    The dataset was created using the preprocessed data from the script:
            https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_imagenet.sh

    """
    import tensorflow as tf
    import os

    """
    labels:
        image/encoded: string containing JPEG encoded image in RGB colorspace
        image/height: integer, image height in pixels
        image/width: integer, image width in pixels
        image/colorspace: string, specifying the colorspace, always 'RGB'
        image/channels: integer, specifying the number of channels, always 3
        image/format: string, specifying the format, always 'JPEG'
        image/filename: string containing the basename of the image file
                e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
        image/class/label: integer specifying the index in a classification layer.
                The label ranges from [1, 1000] where 0 is not used.
        image/class/synset: string specifying the unique ID of the label, e.g. 'n01440764'
        image/class/text: string specifying the human-readable version of the label
                e.g. 'red fox, Vulpes vulpes'
        image/object/bbox/xmin: list of integers specifying the 0+ human annotated bounding boxes
        image/object/bbox/xmax: list of integers specifying the 0+ human annotated bounding boxes
        image/object/bbox/ymin: list of integers specifying the 0+ human annotated bounding boxes
        image/object/bbox/ymax: list of integers specifying the 0+ human annotated bounding boxes
        image/object/bbox/label: integer specifying the index in a classification
                layer. The label ranges from [1, 1000] where 0 is not used. Note this is
                always identical to the image label."""
    # get the number of files from the contents of the folder
    if folder_name is not None:
        train_names = [folder_name + f for f in os.listdir(folder_name) if f.startswith("train")]
        val_names = [folder_name + f for f in os.listdir(folder_name) if f.startswith("val")]
    train_names.sort()
    val_names.sort()
    num_train = len(train_names)
    num_val = len(val_names)

    def _find_output_name_and_stsp(num_names):
        start = 0
        stop = num_names + 1
        output_name_lcl = output_folder
        output_name_lcl += "imagenet_merged.h5"
        return start, stop, output_name_lcl

    train_start, train_stop, output_name_lcl_train = _find_output_name_and_stsp(num_train)
    val_start, val_stop, output_name_lcl_val = _find_output_name_and_stsp(num_val)
    output_name_lcl_val = output_name_lcl_val[:-3] + "_validation.h5"

    # create the output files
    train_lcl_file = h5py.File(output_name_lcl_train, "w")
    dt = h5py.string_dtype(encoding="ascii")
    train_lcl_file.create_dataset("images", (2502,), chunks=(1251,), maxshape=(None,), dtype=dt)
    train_lcl_file.create_dataset("metadata", (2502, 9), chunks=(1251, 9), maxshape=(None, 9))
    train_lcl_file.create_dataset(
        "file_info", (2502, 4), chunks=(1251, 4), maxshape=(None, 4), dtype="S10"
    )

    val_lcl_file = h5py.File(output_name_lcl_val, "w")
    val_lcl_file.create_dataset("images", (50000,), chunks=True, maxshape=(None,), dtype=dt)
    val_lcl_file.create_dataset("metadata", (50000, 9), chunks=True, maxshape=(None, 9))
    val_lcl_file.create_dataset(
        "file_info", (50000, 4), chunks=True, maxshape=(None, 4), dtype="S10"
    )

    def __single_file_load(src):
        # load a file and read it to a numpy array
        dataset = tf.data.TFRecordDataset(filenames=[src])
        imgs = []
        img_meta = [[] for _ in range(9)]
        file_arr = [[] for _ in range(4)]
        for raw_example in iter(dataset):
            parsed = tf.train.Example.FromString(raw_example.numpy())
            img_str = parsed.features.feature["image/encoded"].bytes_list.value[0]
            img = tf.image.decode_jpeg(img_str, channels=3).numpy()
            string_repr = base64.binascii.b2a_base64(img).decode("ascii")
            imgs.append(string_repr)
            # to decode: np.frombuffer(base64.binascii.a2b_base64(string_repr.encode('ascii')))
            img_meta[0].append(
                tf.cast(
                    parsed.features.feature["image/height"].int64_list.value[0], tf.float32
                ).numpy()
            )
            img_meta[1].append(
                tf.cast(
                    parsed.features.feature["image/width"].int64_list.value[0], tf.float32
                ).numpy()
            )
            img_meta[2].append(
                tf.cast(
                    parsed.features.feature["image/channels"].int64_list.value[0], tf.float32
                ).numpy()
            )
            img_meta[3].append(parsed.features.feature["image/class/label"].int64_list.value[0] - 1)
            try:
                bbxmin = parsed.features.feature["image/object/bbox/xmin"].float_list.value[0]
                bbxmax = parsed.features.feature["image/object/bbox/xmax"].float_list.value[0]
                bbymin = parsed.features.feature["image/object/bbox/ymin"].float_list.value[0]
                bbymax = parsed.features.feature["image/object/bbox/ymax"].float_list.value[0]
                bblabel = parsed.features.feature["image/object/bbox/label"].int64_list.value[0] - 1
            except IndexError:
                bbxmin = 0.0
                bbxmax = img_meta[1][-1]
                bbymin = 0.0
                bbymax = img_meta[0][-1]
                bblabel = -2

            img_meta[4].append(np.float(bbxmin))
            img_meta[5].append(np.float(bbxmax))
            img_meta[6].append(np.float(bbymin))
            img_meta[7].append(np.float(bbymax))
            img_meta[8].append(bblabel)

            file_arr[0].append(parsed.features.feature["image/format"].bytes_list.value[0])
            file_arr[1].append(parsed.features.feature["image/filename"].bytes_list.value[0])
            file_arr[2].append(parsed.features.feature["image/class/synset"].bytes_list.value[0])
            file_arr[3].append(
                np.array(parsed.features.feature["image/class/text"].bytes_list.value[0])
            )
        # need to transpose because of the way that numpy understands nested lists
        img_meta = np.array(img_meta, dtype=np.float64).T
        file_arr = np.array(file_arr).T
        return imgs, img_meta, file_arr

    def __write_datasets(img_outl, img_metal, file_arrl, past_sizel, file):
        file["images"].resize((past_sizel + len(img_outl),))
        file["images"][past_sizel : len(img_outl) + past_sizel] = img_outl
        file["metadata"].resize((past_sizel + img_metal.shape[0], 9))
        file["metadata"][past_sizel : img_metal.shape[0] + past_sizel] = img_metal
        file["file_info"].resize((past_sizel + img_metal.shape[0], 4))
        file["file_info"][past_sizel : img_metal.shape[0] + past_sizel] = file_arrl

    def __load_multiple_files(train_names, train_start, train_stop, file):
        loc_files = train_names[train_start:train_stop]
        img_out, img_meta, file_arr = None, None, None
        past_size, i = 0, 0
        for f in loc_files:  # train
            # print(f)
            # this is where the data is created for
            imgs, img_metaf, file_arrf = __single_file_load(f)
            # create a larger ndarray with the results
            if img_out is not None:
                img_out.extend(imgs)
            else:
                img_out = imgs
            img_meta = np.vstack((img_meta, img_metaf)) if img_meta is not None else img_metaf
            file_arr = np.vstack((file_arr, file_arrf)) if file_arr is not None else file_arrf
            # when 2 files are read, write to the output file
            if i % 2 == 1:
                print(past_size)
                __write_datasets(img_out, img_meta, file_arr, past_size, file)
                past_size += len(img_out)
                img_out, img_meta, file_arr = None, None, None
                del imgs, img_metaf, file_arrf
            i += 1

        if img_out is not None:
            __write_datasets(img_out, img_meta, file_arr, past_size, file)

    __load_multiple_files(train_names, train_start, train_stop, train_lcl_file)
    __load_multiple_files(val_names, val_start, val_stop, val_lcl_file)

    #  add the label names to the datasets
    img_list = [1, 2, 4, 7, 10, 11, 12, 13, 14]
    file_list = [5, 6, 8, 9]
    feature_list = [
        "image/encoded",
        "image/height",
        "image/width",
        "image/colorspace",
        "image/channels",
        "image/format",
        "image/filename",
        "image/class/label",
        "image/class/synset",
        "image/class/text",
        "image/object/bbox/xmin",
        "image/object/bbox/xmax",
        "image/object/bbox/ymin",
        "image/object/bbox/ymax",
        "image/object/bbox/label",
    ]

    train_lcl_file["metadata"].attrs["column_names"] = [feature_list[l] for l in img_list]
    train_lcl_file["file_info"].attrs["column_names"] = [feature_list[l] for l in file_list]
    val_lcl_file["metadata"].attrs["column_names"] = [feature_list[l] for l in img_list]
    val_lcl_file["file_info"].attrs["column_names"] = [feature_list[l] for l in file_list]
