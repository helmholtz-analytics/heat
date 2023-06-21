"""
This file contains functions which may be useful for certain datatypes, but are not test in the heat framework
This file contains standalone utilities for data preparation which may be useful
The functions contained within are not tested, nor actively supported
"""

import base64
import numpy as np
import os
import struct


def dali_tfrecord2idx(train_dir, train_idx_dir, val_dir, val_idx_dir):
    """
    WARNING: This function likely requires adjustments and it is by no means a final product !!!
    this file contains standalone utilities for data preparation which may be useful
    this function contained within are not tested, nor actively supported

    prepare TFRecords indexes for use with DALI. It will produce indexes for all files in the
    given ``train_dir`` and ``val_dir`` directories
    """
    for tv in [train_dir, val_dir]:
        dir_list = os.listdir(tv)
        out = train_idx_dir if tv == train_dir else val_idx_dir
        for file in dir_list:
            with open(file, "rb") as f, open(out + file, "w") as idx:
                while True:
                    current = f.tell()
                    try:
                        # length
                        byte_len = f.read(8)
                        if len(byte_len) == 0:
                            break
                        # crc
                        f.read(4)
                        proto_len = struct.unpack("q", byte_len)[0]
                        # proto
                        f.read(proto_len)
                        # crc
                        f.read(4)
                        idx.write(str(current) + " " + str(f.tell() - current) + "\n")
                    except Exception:
                        print("Not a valid TFRecord file")
                        break


def merge_files_imagenet_tfrecord(folder_name, output_folder=None):
    """
    WARNING: This function likely requires adjustments and it is by no means a final product !!!
    this file contains standalone utilities for data preparation which may be useful
    this function contained within are not tested, nor actively supported

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
    import h5py
    import tensorflow as tf

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
    train_names = [folder_name + f for f in os.listdir(folder_name) if f.startswith("train")].sort()
    val_names = [folder_name + f for f in os.listdir(folder_name) if f.startswith("val")].sort()
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
    output_name_lcl_val = f"{output_name_lcl_val[:-3]}_validation.h5"

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

    train_lcl_file["metadata"].attrs["column_names"] = [feature_list[im] for im in img_list]
    train_lcl_file["file_info"].attrs["column_names"] = [feature_list[im] for im in file_list]
    val_lcl_file["metadata"].attrs["column_names"] = [feature_list[im] for im in img_list]
    val_lcl_file["file_info"].attrs["column_names"] = [feature_list[im] for im in file_list]
