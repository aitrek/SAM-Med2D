import os
import json
import glob
import shutil

import cv2
import numpy as np

from typing import List
from utils import get_boxes_from_mask


def create_dataset(ori_dataset_root: str, dst_dataset_root: str):
    ori_data_dir = os.path.join(ori_dataset_root, "data")
    ori_label_dir = os.path.join(ori_dataset_root, "label")
    dst_data_dir = os.path.join(dst_dataset_root, "data")
    dst_label_dir = os.path.join(dst_dataset_root, "label")
    os.makedirs(dst_label_dir)

    # copy data
    shutil.copytree(ori_data_dir, dst_data_dir)

    # generate masks
    mask_paths = glob.glob(os.path.join(ori_label_dir, "*.npy"))
    for mask_path in mask_paths:
        mask = np.load(mask_path)
        basename = os.path.basename(mask_path)

        i = 1
        for val in np.unique(mask):
            if val == 0:
                continue
            tmp_mask = np.zeros(shape=mask.shape, dtype=np.uint8)
            tmp_mask[mask == val] = 255
            try:
                get_boxes_from_mask(tmp_mask, max_pixel=0)
            except:
                print(f"fail: {mask_path}, val: {val}")
                continue

            cv2.imwrite(
                os.path.join(dst_label_dir, basename[:-4] + f"_{i:05d}.png"),
                tmp_mask
            )

            i += 1


def get_label_paths(img_path: str, label_dir: str, break_max: int = 10) -> List[str]:
    paths = []
    i = 0
    break_num = 0
    while True:
        basename = os.path.basename(img_path)
        label_basename = basename[:-4] + f"_{i:05d}.png"
        label_path = os.path.join(label_dir, label_basename)
        if os.path.exists(label_path):
            paths.append(label_path)
            i += 1
            break_num = 0
        else:
            i += 1
            break_num += 1
            if break_num > break_max:
                break
    return paths


def create_test_json(data_root: str, split_path: str, seed: int, test_size: float):
    # data_dir = os.path.join(data_root, "data")
    label_dir = os.path.join(data_root, "label")

    with open(split_path) as f:
        split_data = json.load(f)

    # img_paths = glob.glob(os.path.join(data_dir, "*.png"))
    test_data = split_data["test"]
    json_data = {}
    for img_path, _ in test_data:
        for label_path in get_label_paths(img_path, label_dir):
            json_data[os.path.relpath(label_path, data_root)] = \
                os.path.relpath(img_path, data_root)

    with open(os.path.join(data_root, "label2image_test.json"), "w") as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    # create_test_json("/Users/zhaojq/Datasets/CPM15")
    # exit()
    ori_root = "/root/autodl-tmp/datasets/SAM_nuclei_preprocessed/ALL_Multi"
    dst_root = "/root/autodl-tmp/datasets/SAM_nuclei_preprocessed/ALL_Multi_BL"
    seed = 42
    test_size = 0.1

    # create json file
    for name in os.listdir(ori_root):
        ori_dataset_dir = os.path.join(ori_root, name)
        if os.path.isdir(ori_dataset_dir):
            print(ori_dataset_dir)
            split_path = os.path.join(ori_dataset_dir,
                                      f"split_seed-{seed}_test_size-{test_size}.json")
            dst_dataset_dir = os.path.join(dst_root, name)
            create_dataset(ori_dataset_dir, dst_dataset_dir)
            create_test_json(dst_dataset_dir, split_path, seed, test_size)
